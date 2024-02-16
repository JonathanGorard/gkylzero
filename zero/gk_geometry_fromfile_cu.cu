/* -*- c++ -*- */
extern "C" {
#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_basis.h>
#include <gkyl_math.h>
#include <gkyl_util.h>
#include <gkyl_gk_geometry.h>
#include <gkyl_gk_geometry_fromfile.h>

}
// CPU interface to create and track a GPU object
struct gk_geometry*
gkyl_gk_geometry_fromfile_cu_dev_new(struct gk_geometry* geo_host, const struct gkyl_rect_grid* grid, const struct gkyl_range *local, const struct gkyl_range* local_ext, const struct gkyl_range *global, const struct gkyl_range* global_ext, 
  const struct gkyl_basis* basis)
{
  struct gk_geometry *up =(struct gk_geometry*) gkyl_malloc(sizeof(struct gk_geometry));

  up->basis = *basis;
  up->local = *local;
  up->local_ext = *local_ext;
  up->global = *global;
  up->global_ext = *global_ext;
  up->grid = *grid;


  // Copy the host-side initialized geometry object to the device
  struct gkyl_array *mc2p_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 3*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *bmag_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *g_ij_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 6*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *dxdz_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 9*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *dzdx_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 9*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *jacobgeo_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *jacobgeo_inv_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *gij_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 6*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *b_i_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, 3*up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *cmag_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *jacobtot_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *jacobtot_inv_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *bmag_inv_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *bmag_inv_sq_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *gxxj_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *gxyj_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *gyyj_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *gxzj_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);
  struct gkyl_array *eps2_dev = gkyl_array_cu_dev_new(GKYL_DOUBLE, up->basis.num_basis, up->local_ext.volume);


  gkyl_array_copy(mc2p_dev, geo_host->mc2p);
  gkyl_array_copy(bmag_dev, geo_host->bmag);
  gkyl_array_copy(g_ij_dev, geo_host->g_ij);
  gkyl_array_copy(dxdz_dev, geo_host->dxdz);
  gkyl_array_copy(dzdx_dev, geo_host->dzdx);
  gkyl_array_copy(jacobgeo_dev , geo_host->jacobgeo);
  gkyl_array_copy(jacobgeo_inv_dev, geo_host->jacobgeo_inv);
  gkyl_array_copy(gij_dev, geo_host->gij);
  gkyl_array_copy(b_i_dev, geo_host->b_i);
  gkyl_array_copy(cmag_dev, geo_host->cmag);
  gkyl_array_copy(jacobtot_dev, geo_host->jacobtot);
  gkyl_array_copy(jacobtot_inv_dev, geo_host->jacobtot_inv);
  gkyl_array_copy(bmag_inv_dev, geo_host->bmag_inv);
  gkyl_array_copy(bmag_inv_sq_dev, geo_host->bmag_inv_sq);
  gkyl_array_copy(gxxj_dev, geo_host->gxxj);
  gkyl_array_copy(gxyj_dev, geo_host->gxyj);
  gkyl_array_copy(gyyj_dev, geo_host->gyyj);
  gkyl_array_copy(gxzj_dev, geo_host->gxzj);
  gkyl_array_copy(eps2_dev, geo_host->eps2);

  // this is for the memcpy below
  up->mc2p  = mc2p_dev->on_dev;
  up->bmag  = bmag_dev->on_dev;
  up->g_ij  = g_ij_dev->on_dev;
  up->dxdz  = dxdz_dev->on_dev;
  up->dzdx  = dzdx_dev->on_dev;
  up->jacobgeo  = jacobgeo_dev->on_dev;
  up->jacobgeo_inv = jacobgeo_inv_dev->on_dev;
  up->gij  = gij_dev->on_dev;
  up->b_i  = b_i_dev->on_dev;
  up->cmag  =  cmag_dev->on_dev;
  up->jacobtot  = jacobtot_dev->on_dev;
  up->jacobtot_inv = jacobtot_inv_dev->on_dev;
  up->bmag_inv  = bmag_inv_dev->on_dev;
  up->bmag_inv_sq = bmag_inv_sq_dev->on_dev;
  up->gxxj  = gxxj_dev->on_dev;
  up->gxyj  = gxyj_dev->on_dev;
  up->gyyj  = gyyj_dev->on_dev;
  up->gxzj  = gxzj_dev->on_dev;
  up->eps2  = eps2_dev->on_dev;

  up->flags = 0;
  GKYL_SET_CU_ALLOC(up->flags);
  up->ref_count = gkyl_ref_count_init(gkyl_gk_geometry_free);

  // Initialize the device geometry object
  struct gk_geometry *up_cu = (struct gk_geometry*) gkyl_cu_malloc(sizeof(struct gk_geometry));
  gkyl_cu_memcpy(up_cu, up, sizeof(struct gk_geometry), GKYL_CU_MEMCPY_H2D);
  up->on_dev = up_cu;

  // geometry object should store host pointer
  up->mc2p  = mc2p_dev;
  up->bmag  = bmag_dev;
  up->g_ij  = g_ij_dev;
  up->dxdz  = dxdz_dev;
  up->dzdx  = dzdx_dev;
  up->jacobgeo  = jacobgeo_dev;
  up->jacobgeo_inv = jacobgeo_inv_dev;
  up->gij  = gij_dev;
  up->b_i  = b_i_dev;
  up->cmag  =  cmag_dev;
  up->jacobtot  = jacobtot_dev;
  up->jacobtot_inv = jacobtot_inv_dev;
  up->bmag_inv  = bmag_inv_dev;
  up->bmag_inv_sq = bmag_inv_sq_dev;
  up->gxxj  = gxxj_dev;
  up->gxyj  = gxyj_dev;
  up->gyyj  = gyyj_dev;
  up->gxzj  = gxzj_dev;
  up->eps2  = eps2_dev;
  
  return up;
}

