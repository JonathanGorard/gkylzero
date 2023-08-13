// Private header: not for direct use
#pragma once

#include <math.h>

#include <gkyl_array.h>
#include <gkyl_basis.h>
#include <gkyl_euler_pkpm_kernels.h>
#include <gkyl_range.h>
#include <gkyl_util.h>
#include <assert.h>

typedef int (*pkpm_set_t)(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, 
  const double *vlasov_pkpm_moms, const double *euler_pkpm, 
  const double *pkpm_div_ppar);

typedef void (*pkpm_surf_set_t)(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, 
  const double *vlasov_pkpm_moms, const double *euler_pkpm, 
  const double *p_ij, const int *cell_avg_prim);

typedef void (*pkpm_copy_t)(int count, struct gkyl_nmat *x, double* GKYL_RESTRICT prim);

typedef void (*pkpm_pressure_t)(const double *bvar, const double *vlasov_pkpm_moms, 
  double* GKYL_RESTRICT p_ij);

typedef void (*pkpm_accel_t)(const double *dxv, 
  const double *bvar_l, const double *bvar_c, const double *bvar_r, 
  const double *prim_surf_l, const double *prim_surf_c, const double *prim_surf_r, 
  const double *prim_c, const double *nu_c, 
  double* GKYL_RESTRICT pkpm_lax, double* GKYL_RESTRICT pkpm_accel); 

typedef void (*pkpm_int_t)(const double *vlasov_pkpm_moms, 
  const double *euler_pkpm, const double* prim, 
  double* GKYL_RESTRICT int_pkpm_vars); 

typedef void (*pkpm_source_t)(const double* qmem, 
  const double *vlasov_pkpm_moms, const double *euler_pkpm, 
  double* GKYL_RESTRICT out);

typedef void (*pkpm_io_t)(const double *vlasov_pkpm_moms, 
  const double *euler_pkpm, const double* p_ij, 
  const double* prim, const double* pkpm_accel, 
  double* GKYL_RESTRICT fluid_io, double* GKYL_RESTRICT pkpm_vars_io); 

// for use in kernel tables
typedef struct { pkpm_set_t kernels[3]; } gkyl_dg_pkpm_set_kern_list;
typedef struct { pkpm_surf_set_t kernels[3]; } gkyl_dg_pkpm_surf_set_kern_list;
typedef struct { pkpm_copy_t kernels[3]; } gkyl_dg_pkpm_copy_kern_list;
typedef struct { pkpm_copy_t kernels[3]; } gkyl_dg_pkpm_surf_copy_kern_list;
typedef struct { pkpm_pressure_t kernels[3]; } gkyl_dg_pkpm_pressure_kern_list;
typedef struct { pkpm_accel_t kernels[3]; } gkyl_dg_pkpm_accel_kern_list;
typedef struct { pkpm_int_t kernels[3]; } gkyl_dg_pkpm_int_kern_list;
typedef struct { pkpm_source_t kernels[3]; } gkyl_dg_pkpm_source_kern_list;
typedef struct { pkpm_io_t kernels[3]; } gkyl_dg_pkpm_io_kern_list;

struct gkyl_dg_calc_pkpm_vars {
  struct gkyl_rect_grid conf_grid; // Configuration space grid for cell spacing and cell center
  int cdim; // Configuration space dimensionality
  int poly_order; // polynomial order (determines whether we solve linear system or use basis_inv method)
  struct gkyl_range mem_range; // Configuration space range for linear solve

  struct gkyl_nmat *As, *xs; // matrices for LHS and RHS
  gkyl_nmat_mem *mem; // memory for use in batched linear solve
  int Ncomp; // number of components in the linear solve (6 variables being solved for)
  struct gkyl_nmat *As_surf, *xs_surf; // matrices for LHS and RHS of surface variable solve
  gkyl_nmat_mem *mem_surf; // memory for use in batched linear solve of surface variables
  int Ncomp_surf; // number of components in the surface linear solve (2*cdim*3 + 2*cdim variables being solved for)

  pkpm_set_t pkpm_set;  // kernel for setting matrices for linear solve
  pkpm_surf_set_t pkpm_surf_set;  // kernel for setting matrices for linear solve of surface variables
  pkpm_copy_t pkpm_copy; // kernel for copying solution to output 
  pkpm_copy_t pkpm_surf_copy; // kernel for copying solution to output surface variables
  pkpm_pressure_t pkpm_pressure; // kernel for computing pressure
  pkpm_accel_t pkpm_accel[3]; // kernel for computing pkpm acceleration and Lax variables
  pkpm_int_t pkpm_int; // kernel for computing integrated pkpm variables
  pkpm_source_t pkpm_source; // kernel for computing pkpm source update
  pkpm_io_t pkpm_io; // kernel for constructing I/O arrays for pkpm diagnostics

  uint32_t flags;
  struct gkyl_dg_calc_pkpm_vars *on_dev; // pointer to itself or device data
};

// Set matrices for computing pkpm primitive vars, e.g., ux,uy,uz (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_set_kern_list ser_pkpm_set_kernels[] = {
  { NULL, pkpm_vars_set_1x_ser_p1, pkpm_vars_set_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_set_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_set_3x_ser_p1, NULL }, // 2
};

// Set matrices for computing pkpm primitive vars, e.g., ux,uy,uz (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_set_kern_list ten_pkpm_set_kernels[] = {
  { NULL, pkpm_vars_set_1x_ser_p1, pkpm_vars_set_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_set_2x_ser_p1, pkpm_vars_set_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_set_3x_ser_p1, NULL }, // 2
};

// Set matrices for computing surface pkpm primitive vars, e.g., surface expansion of ux,uy,uz (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_surf_set_kern_list ser_pkpm_surf_set_kernels[] = {
  { NULL, pkpm_vars_surf_set_1x_ser_p1, pkpm_vars_surf_set_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_surf_set_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_surf_set_3x_ser_p1, NULL }, // 2
};

// Set matrices for computing surface pkpm primitive vars, e.g., surface expansion of ux,uy,uz (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_surf_set_kern_list ten_pkpm_surf_set_kernels[] = {
  { NULL, pkpm_vars_surf_set_1x_ser_p1, pkpm_vars_surf_set_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_surf_set_2x_ser_p1, pkpm_vars_surf_set_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_surf_set_3x_ser_p1, NULL }, // 2
};

// Copy solution for pkpm primitive vars, e.g., ux,uy,uz (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_copy_kern_list ser_pkpm_copy_kernels[] = {
  { NULL, pkpm_vars_copy_1x_ser_p1, pkpm_vars_copy_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_copy_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_copy_3x_ser_p1, NULL }, // 2
};

// Copy solution for pkpm primitive vars, e.g., ux,uy,uz (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_copy_kern_list ten_pkpm_copy_kernels[] = {
  { NULL, pkpm_vars_copy_1x_ser_p1, pkpm_vars_copy_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_copy_2x_ser_p1, pkpm_vars_copy_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_copy_3x_ser_p1, NULL }, // 2
};

// Copy solution for surface pkpm primitive vars, e.g., surface expansion of ux,uy,uz (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_surf_copy_kern_list ser_pkpm_surf_copy_kernels[] = {
  { NULL, pkpm_vars_surf_copy_1x_ser_p1, pkpm_vars_surf_copy_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_surf_copy_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_surf_copy_3x_ser_p1, NULL }, // 2
};

// Copy solution for surface pkpm primitive vars, e.g., surface expansion of ux,uy,uz (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_surf_copy_kern_list ten_pkpm_surf_copy_kernels[] = {
  { NULL, pkpm_vars_surf_copy_1x_ser_p1, pkpm_vars_surf_copy_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_surf_copy_2x_ser_p1, pkpm_vars_surf_copy_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_surf_copy_3x_ser_p1, NULL }, // 2
};

// PKPM Pressure (p_ij = (p_par - p_perp)b_i b_j + p_perp g_ij) (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_pressure_kern_list ser_pkpm_pressure_kernels[] = {
  { NULL, pkpm_vars_pressure_1x_ser_p1, pkpm_vars_pressure_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_pressure_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_pressure_3x_ser_p1, NULL }, // 2
};

// PKPM Pressure (p_ij = (p_ij = (p_par - p_perp)b_i b_j + p_perp g_ij) (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_pressure_kern_list ten_pkpm_pressure_kernels[] = {
  { NULL, pkpm_vars_pressure_1x_ser_p1, pkpm_vars_pressure_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_pressure_2x_ser_p1, pkpm_vars_pressure_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_pressure_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in x) (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ser_pkpm_accel_x_kernels[] = {
  { NULL, pkpm_vars_accel_x_1x_ser_p1, pkpm_vars_accel_x_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_accel_x_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_accel_x_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in y) (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ser_pkpm_accel_y_kernels[] = {
  { NULL, NULL, NULL }, // 0
  { NULL, pkpm_vars_accel_y_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_accel_y_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in z) (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ser_pkpm_accel_z_kernels[] = {
  { NULL, NULL, NULL }, // 0
  { NULL, NULL, NULL }, // 1
  { NULL, pkpm_vars_accel_z_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in x) (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ten_pkpm_accel_x_kernels[] = {
  { NULL, pkpm_vars_accel_x_1x_ser_p1, pkpm_vars_accel_x_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_accel_x_2x_ser_p1, pkpm_vars_accel_x_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_accel_x_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in y) (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ten_pkpm_accel_y_kernels[] = {
  { NULL, NULL, NULL }, // 0
  { NULL, pkpm_vars_accel_y_2x_ser_p1, pkpm_vars_accel_y_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_accel_y_3x_ser_p1, NULL }, // 2
};

// PKPM acceleration variables, e.g., div(b) and bb:grad(u), 
// and Lax penalization (lambda_i = |u_i| + sqrt(3*T_ii/m)) (in z) (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_accel_kern_list ten_pkpm_accel_z_kernels[] = {
  { NULL, NULL, NULL }, // 0
  { NULL, NULL, NULL }, // 1
  { NULL, pkpm_vars_accel_z_3x_ser_p1, NULL }, // 2
};

// PKPM integrated variables integral (rho, p_parallel, p_perp, rhoux^2, rhouy^2, rhouz^2) (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_int_kern_list ser_pkpm_int_kernels[] = {
  { NULL, pkpm_vars_integrated_1x_ser_p1, pkpm_vars_integrated_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_integrated_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_integrated_3x_ser_p1, NULL }, // 2
};

// PKPM integrated variables integral (rho, p_parallel, p_perp, rhoux^2, rhouy^2, rhouz^2) (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_int_kern_list ten_pkpm_int_kernels[] = {
  { NULL, pkpm_vars_integrated_1x_ser_p1, pkpm_vars_integrated_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_integrated_2x_ser_p1, pkpm_vars_integrated_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_integrated_3x_ser_p1, NULL }, // 2
};

// PKPM explicit source solve (Serendipity kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_source_kern_list ser_pkpm_source_kernels[] = {
  { NULL, euler_pkpm_source_1x_ser_p1, euler_pkpm_source_1x_ser_p2 }, // 0
  { NULL, euler_pkpm_source_2x_ser_p1, NULL }, // 1
  { NULL, euler_pkpm_source_3x_ser_p1, NULL }, // 2
};

// PKPM explicit source solve (Tensor kernels)
GKYL_CU_D
static const gkyl_dg_pkpm_source_kern_list ten_pkpm_source_kernels[] = {
  { NULL, euler_pkpm_source_1x_ser_p1, euler_pkpm_source_1x_ser_p2 }, // 0
  { NULL, euler_pkpm_source_2x_ser_p1, euler_pkpm_source_2x_tensor_p2 }, // 1
  { NULL, euler_pkpm_source_3x_ser_p1, NULL }, // 2
};

// PKPM io variables (Serendipity kernels)
// Conserved fluid variables: [rho, rho ux, rho uy, rho uz, Pxx + rho ux^2, Pxy + rho ux uy, Pxz + rho ux uz, Pyy + rho uy^2, Pyz + rho uy uz, Pzz + rho uz^2]
// PKPM primitive and acceleration variables:  
// [ux, uy, uz, T_perp/m, m/T_perp, div(b), 1/rho div(p_par b), T_perp/m div(b), bb : grad(u), 
// vperp configuration space characteristics = bb : grad(u) - div(u) - 2 nu]
GKYL_CU_D
static const gkyl_dg_pkpm_io_kern_list ser_pkpm_io_kernels[] = {
  { NULL, pkpm_vars_io_1x_ser_p1, pkpm_vars_io_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_io_2x_ser_p1, NULL }, // 1
  { NULL, pkpm_vars_io_3x_ser_p1, NULL }, // 2
};

// PKPM io variables (Tensor kernels)
// Conserved fluid variables: [rho, rho ux, rho uy, rho uz, Pxx + rho ux^2, Pxy + rho ux uy, Pxz + rho ux uz, Pyy + rho uy^2, Pyz + rho uy uz, Pzz + rho uz^2]
// PKPM primitive and acceleration variables:  
// [ux, uy, uz, T_perp/m, m/T_perp, div(b), 1/rho div(p_par b), T_perp/m div(b), bb : grad(u), 
// vperp configuration space characteristics = bb : grad(u) - div(u) - 2 nu]
GKYL_CU_D
static const gkyl_dg_pkpm_io_kern_list ten_pkpm_io_kernels[] = {
  { NULL, pkpm_vars_io_1x_ser_p1, pkpm_vars_io_1x_ser_p2 }, // 0
  { NULL, pkpm_vars_io_2x_ser_p1, pkpm_vars_io_2x_tensor_p2 }, // 1
  { NULL, pkpm_vars_io_3x_ser_p1, NULL }, // 2
};

GKYL_CU_D
static pkpm_set_t
choose_pkpm_set_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_set_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_set_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_surf_set_t
choose_pkpm_surf_set_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_surf_set_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_surf_set_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_copy_t
choose_pkpm_copy_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_copy_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_copy_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_copy_t
choose_pkpm_surf_copy_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_surf_copy_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_surf_copy_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_pressure_t
choose_pkpm_pressure_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_pressure_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_pressure_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_accel_t
choose_pkpm_accel_kern(int dir, enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      if (dir == 0)
        return ser_pkpm_accel_x_kernels[cdim-1].kernels[poly_order];
      else if (dir == 1)
        return ser_pkpm_accel_y_kernels[cdim-1].kernels[poly_order];
      else if (dir == 2)
        return ser_pkpm_accel_z_kernels[cdim-1].kernels[poly_order];
      else
        return NULL;
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      if (dir == 0)
        return ten_pkpm_accel_x_kernels[cdim-1].kernels[poly_order];
      else if (dir == 1)
        return ten_pkpm_accel_y_kernels[cdim-1].kernels[poly_order];
      else if (dir == 2)
        return ten_pkpm_accel_z_kernels[cdim-1].kernels[poly_order];
      else
        return NULL;
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_int_t
choose_pkpm_int_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_int_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_int_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_source_t
choose_pkpm_source_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_source_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_source_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}

GKYL_CU_D
static pkpm_io_t
choose_pkpm_io_kern(enum gkyl_basis_type b_type, int cdim, int poly_order)
{
  switch (b_type) {
    case GKYL_BASIS_MODAL_SERENDIPITY:
      return ser_pkpm_io_kernels[cdim-1].kernels[poly_order];
      break;
    case GKYL_BASIS_MODAL_TENSOR:
      return ten_pkpm_io_kernels[cdim-1].kernels[poly_order];
      break;
    default:
      assert(false);
      break;  
  }
}
