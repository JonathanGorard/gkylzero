#include <gkyl_array_integrate.h>
#include <gkyl_array_integrate_priv.h>
#include <gkyl_alloc.h>
#include <assert.h>

struct gkyl_array_integrate*
gkyl_array_integrate_new(const struct gkyl_rect_grid *grid, const struct gkyl_basis *basis,
  int num_comp, enum gkyl_array_integrate_op op, bool use_gpu)
{
#ifdef GKYL_HAVE_CUDA
  if (use_gpu)
    return gkyl_array_integrate_cu_dev_new(grid, basis, num_comp, op);
#endif

  // Allocate space for new updater.
  struct gkyl_array_integrate *up = gkyl_malloc(sizeof(struct gkyl_array_integrate));

  up->num_basis = basis->num_basis;
  up->num_comp = num_comp;
  up->use_gpu = use_gpu;
  for (int d=0; d<grid->ndim; ++d) up->dxSq[d] = grid->dx[d]*grid->dx[d];

  assert(basis->poly_order > 0); // Need to check normalization for p=0.

  int ndim = basis->ndim;
  up->vol = 1.0;
  if (op == GKYL_ARRAY_INTEGRATE_OP_GRAD_SQ) {
    for (unsigned d=0; d<ndim; ++d)
      up->vol *= (1./(2.*grid->dx[d]));
    up->vol *= 12.;
  } else if (op == GKYL_ARRAY_INTEGRATE_OP_GRADPERP_SQ) {
    assert(ndim > 1);
    for (unsigned d=0; d<ndim; ++d)
      up->vol *= grid->dx[d]/2.;
    for (unsigned d=0; d<2; ++d)
      up->vol *= 1./(grid->dx[d]*grid->dx[d]);
    up->vol *= 12.;
  } else if (op == GKYL_ARRAY_INTEGRATE_OP_EPS_GRADPERP_SQ) {
    assert(ndim > 1);
    for (unsigned d=0; d<ndim; ++d)
      up->vol *= grid->dx[d]/2.;
  } else {
    for (unsigned d=0; d<ndim; ++d)
      up->vol *= op == GKYL_ARRAY_INTEGRATE_OP_SQ? grid->dx[d]/2.0 : grid->dx[d]/sqrt(2.0);
  }

  // Choose the kernel that performs the desired operation within the integral.
  gkyl_array_integrate_choose_kernel(op, basis, up);

  return up;
}

void gkyl_array_integrate_advance(gkyl_array_integrate *up, const struct gkyl_array *fin,
  double factor, const struct gkyl_array *weight, const struct gkyl_range *range, double *out)
{
#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu) {
    gkyl_array_integrate_advance_cu(up, fin, factor, weight, range, out);
    return;
  }
#endif

  for (int k=0; k<up->num_comp; k++) out[k] = 0;

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);
  while (gkyl_range_iter_next(&iter)) {

    long linidx = gkyl_range_idx(range, iter.idx);
    const double *fin_d = gkyl_array_cfetch(fin, linidx);
    const double *wei_d = gkyl_array_cfetch(weight, linidx);

    up->kernel(up->dxSq, up->vol, up->num_comp, up->num_basis, wei_d, fin_d, out);
  }

  for (int k=0; k<up->num_comp; k++) out[k] *= factor;
}

void gkyl_array_integrate_release(gkyl_array_integrate *up)
{
  // Release memory associated with this updater.
#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu)
    gkyl_cu_free(up->on_dev);
#endif
  gkyl_free(up);
}
