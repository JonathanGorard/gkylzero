#include <gkyl_mat.h>
#include <gkyl_bc_twistshift.h>
#include <gkyl_bc_twistshift_priv.h>
#include <gkyl_array.h>

struct gkyl_bc_twistshift*
gkyl_bc_twistshift_new(int dir, int do_dir, int shift_dir, enum gkyl_edge_loc edge,
  const struct gkyl_range *local_range_ext, const struct gkyl_range *local_range_update, const int *num_ghosts, const struct gkyl_basis *basis,
  const struct gkyl_rect_grid *grid, int cdim,
  const struct gkyl_array *yshift, const int *ndonors, const int *cells_do, bool use_gpu) {

  // Allocate space for new updater.
  struct gkyl_bc_twistshift *up = gkyl_malloc(sizeof(struct gkyl_bc_twistshift));

  up->dir = dir;
  up->do_dir = do_dir;
  up->shift_dir = shift_dir;
  up->edge = edge;
  up->basis = basis;
  up->grid = grid;
  up->ndonors = ndonors;
  up->cells_do = cells_do;
  up->use_gpu = use_gpu;
  up->local_range_ext = local_range_ext;
  up->local_range_update = local_range_update; // rename this range

  // Choose the kernels that do the subcell and full cell integrals
  up->kernels = gkyl_malloc(sizeof(struct gkyl_bc_twistshift_kernels));
#ifdef GKYL_HAVE_CUDA
  if (use_gpu) {
    up->kernels_cu = gkyl_cu_malloc(sizeof(struct gkyl_bc_twistshift_kernels));
    gkyl_bc_twistshift_choose_kernels_cu(basis, cdim, up->kernels_cu);
  } else {
    gkyl_bc_twistshift_choose_kernels(basis, cdim, up->kernels);
    up->kernels_cu = up->kernels;
  }
#else
  gkyl_bc_twistshift_choose_kernels(basis, cdim, up->kernels);
  up->kernels_cu = up->kernels;
#endif


  // Allocate matrices and vectors, initialize if needed
  int total_donor_mats = 0;
  for(int i = 0; i<grid->cells[0]; i++)
    total_donor_mats +=ndonors[i];


  up->matsdo_ho = gkyl_nmat_new(total_donor_mats, basis->num_basis, basis->num_basis);

  for (size_t n=0; n<up->matsdo_ho->num; ++n) {
    struct gkyl_mat m = gkyl_nmat_get(up->matsdo_ho, n);
    for (size_t j=0; j<up->matsdo_ho->nc; ++j)
      for (size_t i=0; i<up->matsdo_ho->nr; ++i)
        gkyl_mat_set(&m, i, j, 0.0);
  }
#ifdef GKYL_HAVE_CUDA
  up->matsdo = gkyl_nmat_cu_dev_new(total_donor_mats, basis->num_basis, basis->num_basis);
  gkyl_nmat_copy(up->matsdo, up->matsdo_ho);
  up->vecstar = gkyl_nmat_cu_dev_new(up->matsdo->num, up->matsdo->nr, 1);
  up->vecsdo = gkyl_nmat_cu_dev_new(up->matsdo->num, up->matsdo->nr, 1);
#else
  up->matsdo = up->matsdo_ho;
  up->vecstar = gkyl_nmat_new(up->matsdo->num, up->matsdo->nr, 1);
  up->vecsdo = gkyl_nmat_new(up->matsdo->num, up->matsdo->nr, 1);
#endif



  // Create necessary ranges
  up->yrange = gkyl_malloc(sizeof(struct gkyl_range));
  up->xrange = gkyl_malloc(sizeof(struct gkyl_range));
  // Create the deflated range only need to loop over the shift dir and extraneous dirs (vpar,mu)
  up->remDir = (int*) gkyl_malloc(sizeof(int) * up->local_range_update->ndim);
  for(int i=0;i<up->local_range_update->ndim;i++)
    up->remDir[i]=1; // remove all dirs by default
  up->remDir[up->shift_dir] = 0; // keep only the donor dir 
  for(int i = up->dir + 1; i<up->local_range_update->ndim; i++) // keep the extaneous dirs (vpar and mu) assumed to be after z
    up->remDir[i] = 0;
  // Setup for deflated range for looping over x
  up->remDir_do = (int*) gkyl_malloc(sizeof(int) * up->local_range_update->ndim);
  for(int i=0;i<up->local_range_update->ndim;i++)
    up->remDir_do[i]=1; // remove all dirs y default
  up->remDir_do[up->do_dir] = 0; // keep x only
  // Choose locations in other dirs. Some will be reset in loop
  up->locDir = (int*) gkyl_malloc(sizeof(int) * up->local_range_update->ndim);
  up->locDir_do = (int*) gkyl_malloc(sizeof(int) * up->local_range_update->ndim);
  for(int i=0;i<up->local_range_update->ndim;i++){
    up->locDir[i] = up->local_range_update->lower[i];
    up->locDir_do[i]=up->local_range_update->lower[i];
  }
  if(up->local_range_update->ndim > 2){
    if(up->edge == GKYL_LOWER_EDGE){
      up->locDir[up->dir] = up->local_range_update->lower[up->dir];
      up->locDir_do[up->dir] = up->local_range_update->upper[up->dir] - 1 ;
    }
    else if(up->edge == GKYL_UPPER_EDGE){
      up->locDir[up->dir] = up->local_range_update->upper[up->dir];
      up->locDir_do[up->dir] = up->local_range_update->lower[up->dir] + 1 ;
    }
  }
  gkyl_range_deflate(up->yrange, up->local_range_update, up->remDir, up->locDir);


  return up;
}

void gkyl_bc_twistshift_integral_xlimdg(struct gkyl_bc_twistshift *up,
  double sFac, const double *xLimLo, const double *xLimUp, double yLimLo, double yLimUp,
  double dyDo, double yOff, const double *ySh, int cellidx, int doidx) {
  
  size_t linidx = 0;
  for(int i = 0; i<cellidx-1; i++)
    linidx += up->ndonors[i];
  
  linidx += doidx;
  struct gkyl_mat tsmat = gkyl_nmat_get(up->matsdo_ho, linidx);
  up->kernels->xlimdg(sFac, xLimLo, xLimUp, yLimLo, yLimUp, dyDo, yOff, ySh, &tsmat);
}

void gkyl_bc_twistshift_integral_ylimdg(struct gkyl_bc_twistshift *up,
  double sFac, double xLimLo, double xLimUp, const double *yLimLo, const double *yLimUp,
  double dyDo, double yOff, const double *ySh, int cellidx, int doidx) {
  size_t linidx = 0;
  for(int i = 0; i<cellidx-1; i++)
    linidx += up->ndonors[i];
  
  linidx += doidx;
  struct gkyl_mat tsmat = gkyl_nmat_get(up->matsdo_ho, linidx);
  up->kernels->ylimdg(sFac, xLimLo, xLimUp, yLimLo, yLimUp, dyDo, yOff, ySh, &tsmat);
}

void gkyl_bc_twistshift_integral_fullcelllimdg(struct gkyl_bc_twistshift *up,
  double dyDo, double yOff, const double *ySh, int cellidx, int doidx) {
  size_t linidx = 0;
  for(int i = 0; i<cellidx-1; i++)
    linidx += up->ndonors[i];
  
  linidx += doidx;
  struct gkyl_mat tsmat = gkyl_nmat_get(up->matsdo_ho, linidx);
  up->kernels->fullcell(dyDo, yOff, ySh, &tsmat);
}

void gkyl_bc_twistshift_copy_matsdo(struct gkyl_bc_twistshift *up)
{
  #ifdef GKYL_HAVE_CUDA
  gkyl_nmat_copy(up->matsdo, up->matsdo_ho);
  #endif
}

void gkyl_bc_twistshift_advance(struct gkyl_bc_twistshift *up, struct gkyl_array *fdo, struct gkyl_array *ftar)
{
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, up->yrange);
  struct gkyl_range_iter iterx;
  int lin_idx = 0;
  int do_idx[up->local_range_update->ndim];
  int tar_idx[up->local_range_update->ndim];
  int lin_vecdo_idx = 0;
  int lin_tar_idx = 0;
  int last_yidx = 0;
  int dim_diff = up->local_range_update->ndim - up->yrange->ndim;
  if(up->local_range_update->ndim > 2){
    do_idx[up->dir] = up->locDir_do[up->dir];
    tar_idx[up->dir] = up->locDir[up->dir];
  }

  while (gkyl_range_iter_next(&iter)) {
    if(iter.idx[0] == last_yidx){
      lin_idx = (iter.idx[0] - 1)*up->matsdo->num;
    }
    up->locDir_do[up->shift_dir] = iter.idx[0];
    if(up->local_range_update->ndim>3){
      for(int i = up->dir + 1; i < up->local_range_update->ndim; i++){
        up->locDir_do[i] = iter.idx[i - dim_diff];
        do_idx[i] = iter.idx[i - dim_diff];
        tar_idx[i] = iter.idx[i - dim_diff];
      }
    }
    gkyl_range_deflate(up->xrange, up->local_range_update, up->remDir_do, up->locDir_do);
    gkyl_range_iter_init(&iterx, up->xrange);
    lin_vecdo_idx = 0;
    while (gkyl_range_iter_next(&iterx)) {
      for(int i = 0; i < up->ndonors[iterx.idx[0]-1];i++){
        do_idx[up->do_dir] = iterx.idx[0];
        do_idx[up->shift_dir] = up->cells_do[lin_idx];

        struct gkyl_mat gkyl_mat_itr = gkyl_nmat_get(up->vecsdo, lin_vecdo_idx);
        long loc = gkyl_range_idx(up->local_range_update, do_idx);
        const double *fdo_itr = gkyl_array_cfetch(fdo, loc);
        #ifdef GKYL_HAVE_CUDA
        cudaMemcpy(gkyl_mat_itr.data, fdo_itr, up->vecsdo->nr*sizeof(double),GKYL_CU_MEMCPY_D2D);
        #else
        for(int ib = 0; ib < up->vecsdo->nr; ib++){
          gkyl_mat_set(&gkyl_mat_itr, ib, 0, fdo_itr[ib]);
        }
        #endif
        lin_vecdo_idx += 1;
        lin_idx += 1; 
      }
    }

    gkyl_nmat_mv(1.0, 0.0, GKYL_NO_TRANS, up->matsdo, up->vecsdo, up->vecstar);

    gkyl_range_deflate(up->xrange, up->local_range_update, up->remDir_do, up->locDir_do);
    gkyl_range_iter_init(&iterx, up->xrange);
    lin_tar_idx = 0;
    while (gkyl_range_iter_next(&iterx)) {
      tar_idx[up->do_dir] = iterx.idx[0];
      tar_idx[up->shift_dir] = iter.idx[0];

      long loc = gkyl_range_idx(up->local_range_update, tar_idx);
      double *ftar_itr = gkyl_array_fetch(ftar, loc);
      for(int n=0; n<ftar->ncomp; n++){
        #ifdef GKYL_HAVE_CUDA
        gkyl_bc_twistshift_clear_cu(ftar_itr,n);
        #else
        ftar_itr[n] = 0;
        #endif
      }
      for(int i = 0; i < up->ndonors[iterx.idx[0]-1]; i++){
        struct gkyl_mat temp = gkyl_nmat_get(up->vecstar,lin_tar_idx);
        for(int n=0; n<up->matsdo->nr; n++){
          #ifdef GKYL_HAVE_CUDA
          gkyl_bc_twistshift_inc_cu(ftar_itr, n, &temp);
          #else
          ftar_itr[n] += gkyl_mat_get(&temp, n, 0);
          #endif
        }
        lin_tar_idx +=1;
      }
    }
    last_yidx = iter.idx[0];
  }
}

void gkyl_bc_twistshift_release(struct gkyl_bc_twistshift *up) {
  // Release memory associated with this updater.
#ifdef GKYL_HAVE_CUDA
  gkyl_nmat_release(up->matsdo_ho);
  if (up->use_gpu)
    gkyl_cu_free(up->kernels_cu);
#endif
  gkyl_free(up->kernels);
  gkyl_free(up->yrange);
  gkyl_free(up->xrange);
  gkyl_free(up->remDir);
  gkyl_free(up->locDir);
  gkyl_free(up->remDir_do);
  gkyl_free(up->locDir_do);
  gkyl_nmat_release(up->matsdo);
  gkyl_nmat_release(up->vecsdo);
  gkyl_nmat_release(up->vecstar);
  gkyl_free(up);
}
