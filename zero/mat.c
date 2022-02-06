#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>
#include <gkyl_mat.h>
#include <gkyl_ref_count.h>
#include <gkyl_util.h>

#include <stdbool.h>

#ifdef GKYL_HAVE_CUDA
# include <cublas_v2.h>
#endif

// BLAS and LAPACKE includes
#ifdef GKYL_USING_FRAMEWORK_ACCELERATE
# include <Accelerate/Accelerate.h>
#else
// On non-Darwin platforms use OpenBLAS
# include <cblas.h>
# include <lapacke.h>
#endif

#include <assert.h>
#include <string.h>

/** Map Gkyl flags to CBLAS flags */
static int cblas_trans_flags[] = {
  [GKYL_NO_TRANS] = CblasNoTrans,
  [GKYL_TRANS] = CblasTrans,
  [GKYL_CONJ_TRANS] = CblasConjTrans
};

struct gkyl_nmat_mem {
  bool on_gpu; // flag to indicate if we are on GPU

  size_t num, nrows; // numer of RHSs and rows in matrix

  // data needed in batched LU solves on host
  long *ipiv_ho; // host-side pivot vector

  // data needed in batched LU solves on device
  int *ipiv_cu; // device-side pivot vector
  int *infos_cu; // device-side info flags
  int *infos_ho; // host-side info flags
};

/** Helper functions to determine sizes needed in BLAS/LAPACKE routines */
struct mat_sizes { size_t nr, nc; };

static inline struct mat_sizes
get_mat_sizes(enum gkyl_mat_trans trans, const struct gkyl_mat *A)
{
  if (trans == GKYL_NO_TRANS)
    return (struct mat_sizes) { .nr = A->nr, .nc = A->nc };
  return (struct mat_sizes) { .nr = A->nc, .nc = A->nr };
}

struct gkyl_mat*
gkyl_mat_new(size_t nr, size_t nc, double val)
{
  struct gkyl_mat *m = gkyl_malloc(sizeof(struct gkyl_mat));
  m->data = gkyl_malloc(sizeof(double[nr*nc]));
  m->nr = nr; m->nc = nc;
  for (size_t i=0; i<nr*nc; ++i) m->data[i] = val;
  return m;
}

struct gkyl_mat*
gkyl_mat_clone(const struct gkyl_mat *in)
{
  struct gkyl_mat *m = gkyl_malloc(sizeof(struct gkyl_mat));
  m->data = gkyl_malloc(sizeof(double[in->nr*in->nc]));
  m->nc = in->nc; m->nr = in->nr;
  size_t tot = sizeof(double[in->nr*in->nc]);
  memcpy(m->data, in->data, tot);
  return m;
}

struct gkyl_mat*
gkyl_mat_diag(struct gkyl_mat *mat, double val)
{
  gkyl_mat_clear(mat, 0.0);
  for (size_t i=0; i<GKYL_MIN(mat->nr, mat->nc); ++i)
    gkyl_mat_set(mat, i, i, val);
  return mat;
}

void
gkyl_mat_show(const char *name, FILE *fp, const struct gkyl_mat *mat)
{
  fprintf(fp, "%s : matrix( ", name);

  for (int i=0; i<mat->nr-1; ++i) {
    fprintf(fp, "[");
    for (int j=0; j<mat->nc-1; ++j) {
      fprintf(fp, "%lg, ", gkyl_mat_get(mat,i,j));
    }
    fprintf(fp, "%lg ", gkyl_mat_get(mat,i,mat->nc-1));
    fprintf(fp, "], ");
  }

  fprintf(fp, "[");
  for (int j=0; j<mat->nc-1; ++j) {
    fprintf(fp, "%lg, ", gkyl_mat_get(mat,mat->nr-1,j));
  }
  fprintf(fp, "%lg ", gkyl_mat_get(mat,mat->nr-1,mat->nc-1));
  fprintf(fp, "] ");
  
  fprintf(fp, " )\n");
}

struct gkyl_mat*
gkyl_mat_mm(double alpha, double beta,
  enum gkyl_mat_trans transa, const struct gkyl_mat *A,
  enum gkyl_mat_trans transb, const struct gkyl_mat *B, struct gkyl_mat *C)
{
  // determine matrix sizes
  struct mat_sizes sza = get_mat_sizes(transa, A);
  struct mat_sizes szb = get_mat_sizes(transb, B);
  struct mat_sizes szc = get_mat_sizes(GKYL_NO_TRANS, C);

  // intermediate size
  size_t k = sza.nc; // same as szb.nr
  size_t lda = transa == GKYL_NO_TRANS ? C->nr : k;
  size_t ldb = transb == GKYL_NO_TRANS ? k : C->nc;
  size_t ldc = C->nr;
  
  assert( (sza.nr == szc.nr) && (sza.nc == k) && (szb.nr == k) && (szb.nc == szc.nc) );

  // call BLAS routine to perform matrix-matrix multiply
  cblas_dgemm(CblasColMajor,
    cblas_trans_flags[transa],
    cblas_trans_flags[transb],
    C->nr, C->nc, k,
    alpha,
    A->data, lda,
    B->data, ldb,
    beta, C->data, ldc);

  return C;
}

bool
gkyl_mat_linsolve_lu(struct gkyl_mat *A, struct gkyl_mat *x, void* ipiv)
{
  assert( A->nr == A->nc );

#ifdef GKYL_USING_FRAMEWORK_ACCELERATE
  // On Darwin need to use old clapack interface. Of course Apple has
  // to do everything "different"
  __CLPK_integer info;
  __CLPK_integer n = A->nr;
  __CLPK_integer nrhs = x->nc;
  __CLPK_integer lda = A->nr;
  __CLPK_integer ldb = A->nr;
  dgesv_(&n, &nrhs, A->data, &lda, ipiv, x->data, &ldb, &info);
#else
  // on non-Darwin platforms modern LAPACKE interface is available
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR,
    A->nr, x->nc, A->data, A->nr, ipiv, x->data, A->nr);
#endif
  
  return info == 0 ? true : false;
}

void
gkyl_mat_release(struct gkyl_mat *mat)
{
  if (mat) {
    gkyl_free(mat->data);
    gkyl_free(mat);
  }
}

static void
nmat_free(const struct gkyl_ref_count *ref)
{
  struct gkyl_nmat *mat = container_of(ref, struct gkyl_nmat, ref_count);
  if (GKYL_IS_CU_ALLOC(mat->flags)) {
    gkyl_cu_free(mat->data);
    gkyl_cu_free(mat->mptr);
    gkyl_cu_free(mat->on_dev);
  }
  else {
    gkyl_free(mat->data);
    gkyl_free(mat->mptr);
  }
  gkyl_free(mat);  
}

struct gkyl_nmat*
gkyl_nmat_new(size_t num, size_t nr, size_t nc)
{
  struct gkyl_nmat *mat = gkyl_malloc(sizeof(struct gkyl_nmat));
  mat->num = num; mat->nr = nr; mat->nc = nc;
  mat->flags = 0;
  mat->data = gkyl_malloc(sizeof(double[num*nr*nc]));  
  mat->mptr = gkyl_malloc(num*sizeof(double*));
  for (size_t i=0; i<num; ++i)
    mat->mptr[i] = mat->data+nr*nc*i;
  mat->on_dev = mat; // on CPU this is a self-reference
  mat->ref_count = gkyl_ref_count_init(nmat_free);

  return mat;
}

struct gkyl_nmat*
gkyl_nmat_copy(struct gkyl_nmat *dest, const struct gkyl_nmat *src)
{
  assert( dest->num == src->num && dest->nr == src->nr && dest->nc == src->nc );

  bool dest_is_cu_dev = gkyl_nmat_is_cu_dev(dest);
  bool src_is_cu_dev = gkyl_nmat_is_cu_dev(src);

  size_t nby = src->num*src->nr*src->nc*sizeof(double);

  if (src_is_cu_dev) {
    // source is on device
    if (dest_is_cu_dev)
      gkyl_cu_memcpy(dest->data, src->data, nby, GKYL_CU_MEMCPY_D2D);
    else
      gkyl_cu_memcpy(dest->data, src->data, nby, GKYL_CU_MEMCPY_D2H);
  }
  else {
    // source is on host
    if (dest_is_cu_dev)
      gkyl_cu_memcpy(dest->data, src->data, nby, GKYL_CU_MEMCPY_H2D);
    else
      memcpy(dest->data, src->data, nby);
  }
  
  return dest;
}

bool
gkyl_nmat_is_cu_dev(const struct gkyl_nmat *mat)
{
  return GKYL_IS_CU_ALLOC(mat->flags);
}

struct gkyl_nmat*
gkyl_nmat_acquire(const struct gkyl_nmat *mat)
{
  gkyl_ref_count_inc(&mat->ref_count);
  return (struct gkyl_nmat*) mat;
}

gkyl_nmat_mem*
gkyl_nmat_linsolve_lu_alloc(size_t num, size_t nrow)
{
  gkyl_nmat_mem *mem = gkyl_malloc(sizeof(*mem));

  mem->on_gpu = false;
  mem->num = num;
  mem->nrows = nrow;
  
  mem->ipiv_ho = gkyl_malloc(sizeof(long[nrow]));

  return mem;
}

gkyl_nmat_mem *
gkyl_nmat_linsolve_lu_alloc_cu_dev(size_t num, size_t nrow)
{
  gkyl_nmat_mem *mem = gkyl_malloc(sizeof(*mem));

  mem->on_gpu = true;
  mem->num = num;
  mem->nrows = nrow;
  
  mem->ipiv_cu = gkyl_cu_malloc(num*nrow*sizeof(int));
  mem->infos_cu = gkyl_cu_malloc(num*sizeof(int));
  mem->infos_ho = gkyl_malloc(num*sizeof(int));

  return mem;
}

void
gkyl_nmat_linsolve_lu_release(gkyl_nmat_mem *mem)
{
  if (mem->on_gpu) {
    gkyl_cu_free(mem->ipiv_cu);
    gkyl_cu_free(mem->infos_cu);
    gkyl_free(mem->infos_ho);
  }
  else {
    gkyl_free(mem->ipiv_ho);
  }
  gkyl_free(mem);
}

static bool
ho_nmat_linsolve_lu(gkyl_nmat_mem *mem, struct gkyl_nmat *A, struct gkyl_nmat *x)
{
  size_t num = A->num;
  assert( num <= x->num );
  assert(mem->on_gpu == false);
  assert(mem->num == A->num);
  assert(mem->nrows == A->nr);

  bool status = true;

  for (size_t i=0; i<num; ++i) {
    struct gkyl_mat Ai = gkyl_nmat_get(A,i);
    struct gkyl_mat xi = gkyl_nmat_get(x,i);
    status = gkyl_mat_linsolve_lu( &Ai, &xi, mem->ipiv_ho );
    if (!status) break;
  }

  return status;
}

static bool
cu_nmat_linsolve_lu(gkyl_nmat_mem *mem, struct gkyl_nmat *A, struct gkyl_nmat *x)
{
#ifdef GKYL_HAVE_CUDA
  assert(mem->on_gpu);
  assert(mem->num == A->num);
  assert(mem->nrows == A->nr);
  
  cublasHandle_t cuh = 0; cublasCreate_v2(&cuh);

  bool status = true;
  size_t num = A->num, nr = A->nr, nrhs = x->nc;
  size_t lda = nr, ldb = nr;
  cublasStatus_t cu_stat;  
  
  int *ipiv = mem->ipiv_cu;
  int *infos = mem->infos_cu;
  int *infos_h = mem->infos_ho;

  // compute LU decomp
  cu_stat = cublasDgetrfBatched(cuh, nr, A->mptr, lda, ipiv, infos, num);
  if (cu_stat != CUBLAS_STATUS_SUCCESS) {
    status = false;
    goto cleanup;
  }
  
  // copy info back to host and check if there were any errors
  gkyl_cu_memcpy(infos_h, infos, num*sizeof(int), GKYL_CU_MEMCPY_D2H);
  for (size_t i=0; i<num; ++i)
    if (infos_h[i] != 0) {
      status = false;
      goto cleanup;
    }

  // solve linear systems using back-subst of already LU decomposed
  // matrices
  int info;
  // ugly cast below is needed due to signature of CUBLAS method
  // (CUBLAS sig is correct, though it is inconsistent with the LU
  // decomp sig)
  cublasDgetrsBatched(cuh, CUBLAS_OP_N, nr, nrhs, (const double*const*) A->mptr,
    lda, ipiv, x->mptr, ldb, &info, num);
  if (info != 0) {
    status = false;
    goto cleanup;
  }
  
  cleanup:
  cublasDestroy_v2(cuh);
  
  return status;

#else  
  return false;
#endif  
}

bool
gkyl_nmat_linsolve_lu(struct gkyl_nmat *A, struct gkyl_nmat *x)
{
  bool status = false;
  
  if (!gkyl_nmat_is_cu_dev(A) && !gkyl_nmat_is_cu_dev(x)) {
    gkyl_nmat_mem *mem = gkyl_nmat_linsolve_lu_alloc(A->num, A->nr);
    status = ho_nmat_linsolve_lu(mem, A, x);
    gkyl_nmat_linsolve_lu_release(mem);
  }
  
  if (gkyl_nmat_is_cu_dev(A) && gkyl_nmat_is_cu_dev(x)) {
    gkyl_nmat_mem *mem = gkyl_nmat_linsolve_lu_alloc_cu_dev(A->num, A->nr);
    status = cu_nmat_linsolve_lu(mem, A, x);
    gkyl_nmat_linsolve_lu_release(mem);
  }
  
  return status;
}

bool
gkyl_nmat_linsolve_lu_pa(gkyl_nmat_mem *mem, struct gkyl_nmat *A, struct gkyl_nmat *x)
{
  bool status = false;
  
  if (!gkyl_nmat_is_cu_dev(A) && !gkyl_nmat_is_cu_dev(x))
    status = ho_nmat_linsolve_lu(mem, A, x);
  
  if (gkyl_nmat_is_cu_dev(A) && gkyl_nmat_is_cu_dev(x))
    status = cu_nmat_linsolve_lu(mem, A, x);
  
  return status;  
}

void
gkyl_nmat_release(struct gkyl_nmat *mat)
{
  if (mat)
    gkyl_ref_count_dec(&mat->ref_count);
}

// CUDA specific code

#ifdef GKYL_HAVE_CUDA

struct gkyl_nmat*
gkyl_nmat_cu_dev_new(size_t num, size_t nr, size_t nc)
{
  struct gkyl_nmat *mat = gkyl_malloc(sizeof(struct gkyl_nmat));
  mat->num = num; mat->nr = nr; mat->nc = nc;

  mat->flags = 0;
  GKYL_SET_CU_ALLOC(mat->flags);
  mat->data = gkyl_cu_malloc(sizeof(double[num*nr*nc]));
  mat->mptr = gkyl_cu_malloc(num*sizeof(double*));
  mat->ref_count = gkyl_ref_count_init(nmat_free);

  double **mptr_h = gkyl_malloc(num*sizeof(double*));
  // create pointers to various matrices and copy to device
  for (size_t i=0; i<num; ++i)
    mptr_h[i] = mat->data+nr*nc*i;
  gkyl_cu_memcpy(mat->mptr, mptr_h, num*sizeof(double*), GKYL_CU_MEMCPY_H2D);
  gkyl_free(mptr_h);  

  // create a clone of struct mat->on_dev that lives on device, so
  // that the whole mat->on_dev struct can be passed to a device
  // kernel
  mat->on_dev = gkyl_cu_malloc(sizeof(struct gkyl_nmat));
  gkyl_cu_memcpy(mat->on_dev, mat, sizeof(struct gkyl_nmat), GKYL_CU_MEMCPY_H2D);
  
  // set device-side data pointer in mat->on_dev to mat->data 
  // (which is the host-side pointer to the device data)
  gkyl_cu_memcpy(&((mat->on_dev)->data), &mat->data, sizeof(double*), GKYL_CU_MEMCPY_H2D);

  // set device-side mptr pointer in mat->on_dev to mat->mptr 
  // (which is the host-side pointer to the device mptr)
  gkyl_cu_memcpy(&((mat->on_dev)->mptr), &mat->mptr, sizeof(double**), GKYL_CU_MEMCPY_H2D);

  return mat;
}

#else

struct gkyl_nmat*
gkyl_nmat_cu_dev_new(size_t num, size_t nr, size_t nc)
{
  assert(false);
  return 0;
}

#endif // CUDA specific code
