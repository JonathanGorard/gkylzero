#pragma once

#include <gkyl_array.h>
#include <gkyl_basis.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_util.h>
#include <gkyl_fpo_vlasov_kernels.h>

GKYL_CU_DH
static void
create_offsets(const int num_up_dirs, const int update_dirs[2], 
  const struct gkyl_range *range, const int idxc[GKYL_MAX_DIM], long offsets[9])
{
  
  // Check if we're at an upper or lower edge in each direction
  bool is_edge_upper[2], is_edge_lower[2];
  for (int i=0; i<num_up_dirs; ++i) {
    is_edge_lower[i] = idxc[update_dirs[i]] == range->lower[update_dirs[i]];
    is_edge_upper[i] = idxc[update_dirs[i]] == range->upper[update_dirs[i]];
  }

  // Construct the offsets *only* in the directions being updated.
  // No need to load the neighbors that are not needed for the update.
  int lower_offset[GKYL_MAX_DIM] = {0};
  int upper_offset[GKYL_MAX_DIM] = {0};
  for (int d=0; d<num_up_dirs; ++d) {
    int dir = update_dirs[d];
    lower_offset[dir] = -1 + is_edge_lower[d];
    upper_offset[dir] = 1 - is_edge_upper[d];
  }  

  // box spanning stencil
  struct gkyl_range box3;
  gkyl_range_init(&box3, range->ndim, lower_offset, upper_offset);
  struct gkyl_range_iter iter3;
  gkyl_range_iter_init(&iter3, &box3);
  // construct list of offsets
  int count = 0;
  while (gkyl_range_iter_next(&iter3))
    offsets[count++] = gkyl_range_offset(range, iter3.idx);

}

GKYL_CU_DH
static int 
idx_to_inloup_ker(int dim, const int *idx, const int *dirs, const int *num_cells) {
  int iout = 0;

  for (int d=0; d<dim; ++d) {
    if (idx[dirs[d]] == 1) {
      iout = 2*iout+(int)(pow(3,d)+0.5);
    } else if (idx[dirs[d]] == num_cells[dirs[d]]) {
      iout = 2*iout+(int)(pow(3,d)+0.5)+1;
    }
  }
  return iout;
}
// Kernel function pointers
typedef void (*fpo_diff_coeff_diag_t)(const double *dxv, const double *gamma,
    const double* fpo_g_stencil[3], const double* fpo_d2gdv2_surf,
    double *diff_coeff);

typedef void (*fpo_diff_coeff_cross_t)(const double *dxv, const double *gamma,
    const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9],
    const double* fpo_dgdv_surf, double *diff_coeff);

typedef void (*fpo_diff_coeff_surf_t)(const double *diff_coeff_L, 
    const double *diff_coeff_R, double *diff_coeff_surf_R);


// For use in kernel tables
typedef struct { fpo_diff_coeff_diag_t kernels[3]; } gkyl_dg_diff_coeff_diag_kern_list;
typedef struct { gkyl_dg_diff_coeff_diag_kern_list list[3]; } gkyl_dg_fpo_diff_coeff_diag_stencil_list;
typedef struct { fpo_diff_coeff_cross_t kernels[9]; } gkyl_dg_diff_coeff_cross_kern_list;
typedef struct { gkyl_dg_diff_coeff_cross_kern_list list[3]; } gkyl_dg_fpo_diff_coeff_cross_stencil_list;
typedef struct { fpo_diff_coeff_surf_t kernels[3]; } gkyl_dg_fpo_diff_coeff_surf_kern_list;

// diffusion coefficient diagonal term kernel lists
GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_diag_stencil_list ser_fpo_diff_coeff_diag_1x3v_vx_kernels = {
  {
    {NULL, NULL, NULL},
    {fpo_diff_coeff_diag_1x3v_vx_ser_p1_invx, fpo_diff_coeff_diag_1x3v_vx_ser_p1_lovx, fpo_diff_coeff_diag_1x3v_vx_ser_p1_upvx},
    {fpo_diff_coeff_diag_1x3v_vx_ser_p2_invx, fpo_diff_coeff_diag_1x3v_vx_ser_p2_lovx, fpo_diff_coeff_diag_1x3v_vx_ser_p2_upvx}
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_diag_stencil_list ser_fpo_diff_coeff_diag_1x3v_vy_kernels = {
  {
    {NULL, NULL, NULL},
    {fpo_diff_coeff_diag_1x3v_vy_ser_p1_invy, fpo_diff_coeff_diag_1x3v_vy_ser_p1_lovy, fpo_diff_coeff_diag_1x3v_vy_ser_p1_upvy},
    {fpo_diff_coeff_diag_1x3v_vy_ser_p2_invy, fpo_diff_coeff_diag_1x3v_vy_ser_p2_lovy, fpo_diff_coeff_diag_1x3v_vy_ser_p2_upvy}
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_diag_stencil_list ser_fpo_diff_coeff_diag_1x3v_vz_kernels = {
  {
    {NULL, NULL, NULL},
    {fpo_diff_coeff_diag_1x3v_vz_ser_p1_invz, fpo_diff_coeff_diag_1x3v_vz_ser_p1_lovz, fpo_diff_coeff_diag_1x3v_vz_ser_p1_upvz},
    {fpo_diff_coeff_diag_1x3v_vz_ser_p2_invz, fpo_diff_coeff_diag_1x3v_vz_ser_p2_lovz, fpo_diff_coeff_diag_1x3v_vz_ser_p2_upvz}
  }
};

// diffusion coefficient off-diagonal term kernel lists
GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vxvy_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_invx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_lovx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_upvx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_invx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_invx_upvy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_lovx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_lovx_upvy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_upvx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p1_upvx_upvy},
    {fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_invx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_lovx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_upvx_invy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_invx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_invx_upvy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_lovx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_lovx_upvy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_upvx_lovy, fpo_diff_coeff_cross_1x3v_vxvy_ser_p2_upvx_upvy},
  },
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vxvz_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_invx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_lovx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_upvx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_invx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_invx_upvz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_lovx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_lovx_upvz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_upvx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p1_upvx_upvz},
    {fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_invx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_lovx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_upvx_invz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_invx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_invx_upvz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_lovx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_lovx_upvz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_upvx_lovz, fpo_diff_coeff_cross_1x3v_vxvz_ser_p2_upvx_upvz},
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vyvx_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_invy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_invy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_invy_upvx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_lovy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_upvy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_lovy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_upvy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_lovy_upvx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p1_upvy_upvx},
    {fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_invy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_invy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_invy_upvx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_lovy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_upvy_invx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_lovy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_upvy_lovx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_lovy_upvx, fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_upvy_upvx},
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vyvz_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_invy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_lovy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_upvy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_invy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_invy_upvz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_lovy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_lovy_upvz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_upvy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_upvy_upvz},
    {fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_invy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_lovy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_upvy_invz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_invy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_invy_upvz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_lovy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_lovy_upvz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_upvy_lovz, fpo_diff_coeff_cross_1x3v_vyvz_ser_p2_upvy_upvz},
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vzvx_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_invz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_invz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_invz_upvx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_lovz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_upvz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_lovz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_upvz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_lovz_upvx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_upvz_upvx},
    {fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_invz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_invz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_invz_upvx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_lovz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_upvz_invx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_lovz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_upvz_lovx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_lovz_upvx, fpo_diff_coeff_cross_1x3v_vzvx_ser_p2_upvz_upvx},
  }
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_cross_stencil_list ser_fpo_diff_coeff_cross_1x3v_vzvy_kernels = {
  {
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_invz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_invz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_invz_upvy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_lovz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_upvz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_lovz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_upvz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_lovz_upvy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p1_upvz_upvy},
    {fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_invz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_invz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_invz_upvy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_lovz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_upvz_invy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_lovz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_upvz_lovy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_lovz_upvy, fpo_diff_coeff_cross_1x3v_vzvy_ser_p2_upvz_upvy},
  }
};

// diffusion coefficient surface projection kernel lists
GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_surf_kern_list ser_fpo_diff_coeff_surf_1x3v_vx_kernels = {
  NULL, fpo_diff_coeff_surf_1x3v_vx_ser_p1, fpo_diff_coeff_surf_1x3v_vx_ser_p2
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_surf_kern_list ser_fpo_diff_coeff_surf_1x3v_vy_kernels = {
  NULL, fpo_diff_coeff_surf_1x3v_vy_ser_p1, fpo_diff_coeff_surf_1x3v_vy_ser_p2
};

GKYL_CU_D
static const gkyl_dg_fpo_diff_coeff_surf_kern_list ser_fpo_diff_coeff_surf_1x3v_vz_kernels = {
  NULL, fpo_diff_coeff_surf_1x3v_vz_ser_p1, fpo_diff_coeff_surf_1x3v_vz_ser_p2
};


GKYL_CU_D
static const fpo_diff_coeff_cross_t
choose_ser_fpo_diff_coeff_cross_recovery_kern(int d1, int d2, int cdim, int poly_order, int stencil_idx)
{
  int lin_idx = d1*3 + d2;
  switch (lin_idx) {
    case 1: 
      return ser_fpo_diff_coeff_cross_1x3v_vxvy_kernels.list[poly_order].kernels[stencil_idx];
    case 2:
      return ser_fpo_diff_coeff_cross_1x3v_vxvz_kernels.list[poly_order].kernels[stencil_idx];
    case 3:
      return ser_fpo_diff_coeff_cross_1x3v_vyvx_kernels.list[poly_order].kernels[stencil_idx];   
    case 5:
      return ser_fpo_diff_coeff_cross_1x3v_vyvz_kernels.list[poly_order].kernels[stencil_idx];   
    case 6:
      return ser_fpo_diff_coeff_cross_1x3v_vzvx_kernels.list[poly_order].kernels[stencil_idx];   
    case 7:
      return ser_fpo_diff_coeff_cross_1x3v_vzvy_kernels.list[poly_order].kernels[stencil_idx];   
    default:
      return NULL;
  }
};

GKYL_CU_D
static const fpo_diff_coeff_diag_t
choose_ser_fpo_diff_coeff_diag_recovery_kern(int d, int cdim, int poly_order, int stencil_idx)
{
  switch (d){
    case 0:
      return ser_fpo_diff_coeff_diag_1x3v_vx_kernels.list[poly_order].kernels[stencil_idx];
    case 1:
      return ser_fpo_diff_coeff_diag_1x3v_vy_kernels.list[poly_order].kernels[stencil_idx];
    case 2:
      return ser_fpo_diff_coeff_diag_1x3v_vz_kernels.list[poly_order].kernels[stencil_idx];
    default:
      return NULL;
  }
};

GKYL_CU_D
static const fpo_diff_coeff_surf_t
choose_ser_fpo_diff_coeff_surf_recovery_kern(int d, int cdim, int poly_order)
{
  switch (d){
    case 0:
      return ser_fpo_diff_coeff_surf_1x3v_vx_kernels.kernels[poly_order];
    case 1:
      return ser_fpo_diff_coeff_surf_1x3v_vy_kernels.kernels[poly_order];
    case 2:
      return ser_fpo_diff_coeff_surf_1x3v_vz_kernels.kernels[poly_order];
    default:
      return NULL;
  }
}