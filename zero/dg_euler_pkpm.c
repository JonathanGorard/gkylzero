#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>
#include <gkyl_dg_euler_pkpm.h>
#include <gkyl_dg_euler_pkpm_priv.h>
#include <gkyl_util.h>

// "Choose Kernel" based on cdim and polyorder
#define CK(lst,cdim,poly_order) lst[cdim-1].kernels[poly_order]

void 
gkyl_euler_pkpm_free(const struct gkyl_ref_count *ref)
{
  struct gkyl_dg_eqn *base = container_of(ref, struct gkyl_dg_eqn, ref_count);

  if (gkyl_dg_eqn_is_cu_dev(base)) {
    // free inner on_dev object
    struct dg_euler_pkpm *euler_pkpm = container_of(base->on_dev, struct dg_euler_pkpm, eqn);
    gkyl_cu_free(euler_pkpm);
  }  
  
  struct dg_euler_pkpm *euler_pkpm = container_of(base, struct dg_euler_pkpm, eqn);
  gkyl_free(euler_pkpm);
}

void
gkyl_euler_pkpm_set_auxfields(const struct gkyl_dg_eqn *eqn, struct gkyl_dg_euler_pkpm_auxfields auxin)
{
#ifdef GKYL_HAVE_CUDA
  if (gkyl_array_is_cu_dev(auxin.u_i)) {
    gkyl_euler_pkpm_set_auxfields_cu(eqn->on_dev, auxin);
    return;
  }
#endif

  struct dg_euler_pkpm *euler_pkpm = container_of(eqn, struct dg_euler_pkpm, eqn);
  euler_pkpm->auxfields.u_i = auxin.u_i;
  euler_pkpm->auxfields.div_p = auxin.div_p;
  euler_pkpm->auxfields.vth_sq = auxin.vth_sq;
}

struct gkyl_dg_eqn*
gkyl_dg_euler_pkpm_new(const struct gkyl_basis* cbasis, const struct gkyl_range* conf_range, bool use_gpu)
{
#ifdef GKYL_HAVE_CUDA
  if(use_gpu) {
    return gkyl_dg_euler_pkpm_cu_dev_new(cbasis, conf_range);
  } 
#endif
  struct dg_euler_pkpm *euler_pkpm = gkyl_malloc(sizeof(struct dg_euler_pkpm));

  int cdim = cbasis->ndim;
  int poly_order = cbasis->poly_order;

  const gkyl_dg_euler_pkpm_vol_kern_list *vol_kernels;
  const gkyl_dg_euler_pkpm_surf_kern_list *surf_x_kernels, *surf_y_kernels, *surf_z_kernels;

  switch (cbasis->b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      vol_kernels = ser_vol_kernels;
      surf_x_kernels = ser_surf_x_kernels;
      surf_y_kernels = ser_surf_y_kernels;
      surf_z_kernels = ser_surf_z_kernels;
      break;

    default:
      assert(false);
      break;    
  }  
    
  euler_pkpm->eqn.num_equations = 4;
  euler_pkpm->eqn.surf_term = surf;
  euler_pkpm->eqn.boundary_surf_term = boundary_surf;

  euler_pkpm->eqn.vol_term = CK(vol_kernels, cdim, poly_order);

  euler_pkpm->surf[0] = CK(surf_x_kernels, cdim, poly_order);
  if (cdim>1)
    euler_pkpm->surf[1] = CK(surf_y_kernels, cdim, poly_order);
  if (cdim>2)
    euler_pkpm->surf[2] = CK(surf_z_kernels, cdim, poly_order);

  // ensure non-NULL pointers 
  for (int i=0; i<cdim; ++i) assert(euler_pkpm->surf[i]);

  euler_pkpm->auxfields.u_i = 0;  
  euler_pkpm->auxfields.div_p = 0;
  euler_pkpm->auxfields.vth_sq = 0;
  euler_pkpm->conf_range = *conf_range;
  
  euler_pkpm->eqn.flags = 0;
  GKYL_CLEAR_CU_ALLOC(euler_pkpm->eqn.flags);
  euler_pkpm->eqn.ref_count = gkyl_ref_count_init(gkyl_euler_pkpm_free);
  euler_pkpm->eqn.on_dev = &euler_pkpm->eqn; // CPU eqn obj points to itself
  
  return &euler_pkpm->eqn;
}

#ifndef GKYL_HAVE_CUDA

struct gkyl_dg_eqn*
gkyl_dg_euler_pkpm_cu_dev_new(const struct gkyl_basis* cbasis, const struct gkyl_range* conf_range)
{
  assert(false);
  return 0;
}

#endif