#include <assert.h>

#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_ops_priv.h>
#include <gkyl_dg_gk_neut_hamil.h>
#include <gkyl_dg_gk_neut_hamil_priv.h>
#include <gkyl_util.h>

gkyl_dg_gk_neut_hamil*
gkyl_dg_gk_neut_hamil_new(const struct gkyl_rect_grid *phase_grid,
  const struct gkyl_basis *conf_basis, const struct gkyl_basis *vel_basis, bool use_gpu)
{
#ifdef GKYL_HAVE_CUDA
  if(use_gpu) {
    return gkyl_dg_gk_neut_hamil_cu_dev_new(phase_grid,
      conf_basis, vel_basis);
  } 
#endif     
  gkyl_dg_gk_neut_hamil *up = gkyl_malloc(sizeof(*up));

  up->phase_grid = *phase_grid;

  int cdim = conf_basis->ndim;
  int poly_order = conf_basis->poly_order;
  enum gkyl_basis_type b_type = conf_basis->b_type;
  up->poly_order = poly_order;

  int vdim = vel_basis->ndim;

  up->calc_hamil = choose_kern(b_type, cdim, vdim, poly_order);

  up->flags = 0;
  GKYL_CLEAR_CU_ALLOC(up->flags);
  up->on_dev = up; // self-reference on host
  
  return up;
}

void gkyl_gk_neut_hamil_calc(struct gkyl_dg_gk_neut_hamil *up, 
  const struct gkyl_array* gij, struct gkyl_array* hamil)
{
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(gamma)) {
    return gkyl_gk_neut_hamil_calc_cu(up, gij, hamil);
  }
#endif

  // Cell center array
  double xc[GKYL_MAX_DIM];  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, phase_range);
  while (gkyl_range_iter_next(&iter)) {
    gkyl_rect_grid_cell_center(&up->phase_grid, iter.idx, xc);
    long loc_conf = gkyl_range_idx(conf_range, iter.idx);
    long loc_phase = gkyl_range_idx(phase_range, iter.idx);

    double *gij_d = gkyl_array_cfetch(gij, loc_conf);
    double *hamil_d = gkyl_array_fetch(hamil, loc_phase);

    up->calc_hamil(xc, up->phase_grid.dx, gij_d, hamil_d);
  }
}

void gkyl_dg_gk_neut_hamil_release(gkyl_dg_gk_neut_hamil *up)
{
  gkyl_free(up);
}