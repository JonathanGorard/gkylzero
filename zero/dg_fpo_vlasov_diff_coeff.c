#include <gkyl_range.h>
#include <gkyl_util.h>
#include <gkyl_alloc.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_ops_priv.h>
#include <gkyl_dg_fpo_vlasov_diff_coeff.h>
#include <gkyl_dg_fpo_vlasov_diff_coeff_priv.h>

void gkyl_calc_fpo_diff_coeff_recovery(const struct gkyl_rect_grid *grid, 
  struct gkyl_basis pbasis, const struct gkyl_range *range, const struct gkyl_range *conf_range, 
  const struct gkyl_array *gamma, const struct gkyl_array *fpo_g, const struct gkyl_array *fpo_g_surf, 
  const struct gkyl_array *fpo_dgdv_surf, const struct gkyl_array *fpo_d2gdv2_surf, struct gkyl_array *fpo_diff_coeff)
{
  int pdim = pbasis.ndim;
  int vdim = 3;
  int cdim = pdim - vdim; 

  int poly_order = pbasis.poly_order;

  fpo_diff_coeff_diag_t diff_coeff_diag_recovery_stencil[3][3];
  fpo_diff_coeff_cross_t diff_coeff_cross_recovery_stencil[3][3][9];

  // Fetch kernels in each direction
  // off-diagonal terms have to be fetched later
  for (int d1=0; d1<vdim; ++d1) {
    for (int idx=0; idx<3; ++idx)  {
      diff_coeff_diag_recovery_stencil[d1][idx] = 
        choose_ser_fpo_diff_coeff_diag_recovery_kern(d1, cdim, poly_order, idx);
    }
    for (int d2=0; d2<vdim; ++d2) {
      if (d1 != d2) {
        for (int idx=0; idx<9; ++idx) {
          diff_coeff_cross_recovery_stencil[d1][d2][idx] = 
            choose_ser_fpo_diff_coeff_cross_recovery_kern(d1, d2, cdim, poly_order, idx);
        }
      }
    }
  }

  // Indices in each direction
  int idxl[GKYL_MAX_DIM], idxc[GKYL_MAX_DIM], idxr[GKYL_MAX_DIM], conf_idxc[GKYL_MAX_DIM];
  int idx_edge[GKYL_MAX_DIM], idx_skin[GKYL_MAX_DIM];
  int edge;

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, range);

  while (gkyl_range_iter_next(&iter)) {
    gkyl_copy_int_arr(pdim, iter.idx, idxc);
    gkyl_copy_int_arr(cdim, iter.idx, conf_idxc);

    long linc = gkyl_range_idx(range, idxc);
    long conf_linc = gkyl_range_idx(conf_range, conf_idxc);

    const double *fpo_dgdv_surf_c = gkyl_array_cfetch(fpo_dgdv_surf, linc);
    const double *fpo_d2gdv2_surf_c = gkyl_array_cfetch(fpo_d2gdv2_surf, linc);
    double *fpo_diff_coeff_c = gkyl_array_fetch(fpo_diff_coeff, linc);

    const double *gamma_c = gkyl_array_cfetch(gamma, conf_linc);

    // Check if we are at in edge in each direction since we need to handle
    // edges differently for diagonal and off-diagonal terms.
    bool is_edge_in_dir[3];
    bool is_edge = false;
    for (int p=cdim; p<pdim; ++p) {
      is_edge_in_dir[p-cdim] =  (idxc[p] == range->lower[p] || idxc[p] == range->upper[p]);
      is_edge = is_edge || is_edge_in_dir[p-cdim];
    }

    // Iterate over velocity space directions
    for (int d1=0; d1<vdim; ++d1) {
      int dir1 = d1 + cdim;

      // Diagonal terms of the diffusion tensor.
      // Always a 1D, 3-cell stencil.
      const long sz_dim = 3;
      long offsets[sz_dim] = {0};
      int update_dir[] = {dir1};

      bool is_edge_upper[1], is_edge_lower[1];
      is_edge_lower[0] = idxc[dir1] == range->lower[dir1]; 
      is_edge_upper[0] = idxc[dir1] == range->upper[dir1];

      // Create offsets from center cell to stencil and index into kernel list.
      create_offsets(1, is_edge_lower, is_edge_upper, update_dir, range, offsets);
      int keri = idx_to_inloup_ker(1, idxc, update_dir, range->upper);

      const double* fpo_g_stencil[sz_dim];
      int idx[sz_dim][GKYL_MAX_DIM];
      int in_grid = 1;
      for (int i=0; i<sz_dim; ++i) {
        gkyl_range_inv_idx(range, linc+offsets[i], idx[i]);
        if (!(idx[i][dir1] < range->lower[dir1] || idx[i][dir1] > range->upper[dir1])) {
          fpo_g_stencil[i] = gkyl_array_cfetch(fpo_g, linc+offsets[i]);
        }
      }
      
      diff_coeff_diag_recovery_stencil[d1][keri](grid->dx, gamma_c[0], 
        fpo_g_stencil, fpo_d2gdv2_surf_c, fpo_diff_coeff_c);

      for (int d2=0; d2<vdim; ++d2) {
        if (d1 == d2) continue;
        int dir2 = d2+cdim;

        // Off-diagonal terms of the diffusion tensor.
        // Offsets that would be outside the grid will point to center cell.
        // Always 2D and we need 9 cell stencil for 2D recovery.
        const long sz_dim = 9;
        long offsets[sz_dim] = {0};
        int update_dirs[] = {dir1, dir2};

        bool is_edge_lower[2], is_edge_upper[2];
        for (int i=0; i<2; ++i) {
          is_edge_lower[i] = idxc[update_dirs[i]] == range->lower[update_dirs[i]]; 
          is_edge_upper[i] = idxc[update_dirs[i]] == range->upper[update_dirs[i]]; 
        }
        bool is_corner = is_edge_in_dir[d1] & is_edge_in_dir[d2];

        create_offsets(2, is_edge_lower, is_edge_upper, update_dirs, range, offsets);

        // Index into kernel list
        int keri = idx_to_inloup_ker(2, idxc, update_dirs, range->upper);

        const double *fpo_g_stencil[sz_dim], *fpo_g_surf_stencil[sz_dim];
        int idx[sz_dim][GKYL_MAX_DIM];
        int in_grid = 1;
        for (int i=0; i<sz_dim; ++i) {
          gkyl_range_inv_idx(range, linc+offsets[i], idx[i]);

          for (int d=0; d<2; ++d) {
            int dir = update_dirs[d];
            if (idx[i][dir] < range->lower[dir] || idx[i][dir] > range->upper[dir]) {
              in_grid = 0;
            }
          }

          if (in_grid) {
            fpo_g_stencil[i] = gkyl_array_cfetch(fpo_g, linc+offsets[i]);
            fpo_g_surf_stencil[i] = gkyl_array_cfetch(fpo_g_surf, linc+offsets[i]);           
          }
          in_grid = 1;
        }

        diff_coeff_cross_recovery_stencil[d1][d2][keri](grid->dx, gamma_c[0], fpo_g_stencil,
          fpo_g_surf_stencil, fpo_dgdv_surf_c, fpo_diff_coeff_c);
      }
    }
  }
}
