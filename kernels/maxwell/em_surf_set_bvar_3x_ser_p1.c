#include <gkyl_mat.h> 
#include <gkyl_maxwell_kernels.h> 
#include <gkyl_binop_mul_ser.h> 
#include <gkyl_basis_ser_2x_p1_inv.h> 
GKYL_CU_DH void em_surf_set_bvar_3x_ser_p1(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, const double *BB_surf, int* cell_avg_magB2_surf) 
{ 
  // count:   integer to indicate which matrix being fetched. 
  // A:       preallocated LHS matrix. 
  // rhs:     preallocated RHS vector. 
  // BB_surf: Surface B_i B_j [BxBx_xl, BxBx_xr, ByBy_xl, ByBy_xr, BzBz_xl, BzBz_xr, BxBy_xl, BxBy_xr, BxBz_xl, BxBz_xr, 
  //                           BxBx_yl, BxBx_yr, ByBy_yl, ByBy_yr, BzBz_yl, BzBz_yr, BxBy_yl, BxBy_yr, ByBz_yl, ByBz_yr, 
  //                           BxBx_zl, BxBx_zr, ByBy_zl, ByBy_zr, BzBz_zl, BzBz_zr, BxBz_zl, BxBz_zr, ByBz_zl, ByBz_zr]. 
  // cell_avg_magB2_surf:      Output flag for cell average if 1/|B|^2 at a surface only used cell averages. 

  struct gkyl_mat rhs_bxbx_xl = gkyl_nmat_get(rhs, count); 
  struct gkyl_mat rhs_bxbx_xr = gkyl_nmat_get(rhs, count+1); 
  struct gkyl_mat rhs_bxby_xl = gkyl_nmat_get(rhs, count+2); 
  struct gkyl_mat rhs_bxby_xr = gkyl_nmat_get(rhs, count+3); 
  struct gkyl_mat rhs_bxbz_xl = gkyl_nmat_get(rhs, count+4); 
  struct gkyl_mat rhs_bxbz_xr = gkyl_nmat_get(rhs, count+5); 
  gkyl_mat_clear(&rhs_bxbx_xl, 0.0); 
  gkyl_mat_clear(&rhs_bxbx_xr, 0.0); 
  gkyl_mat_clear(&rhs_bxby_xl, 0.0); 
  gkyl_mat_clear(&rhs_bxby_xr, 0.0); 
  gkyl_mat_clear(&rhs_bxbz_xl, 0.0); 
  gkyl_mat_clear(&rhs_bxbz_xr, 0.0); 
  const double *Bx_sq_xl = &BB_surf[0]; 
  const double *Bx_sq_xr = &BB_surf[4]; 
  const double *By_sq_xl = &BB_surf[8]; 
  const double *By_sq_xr = &BB_surf[12]; 
  const double *Bz_sq_xl = &BB_surf[16]; 
  const double *Bz_sq_xr = &BB_surf[20]; 
  const double *B_x_B_y_xl = &BB_surf[24]; 
  const double *B_x_B_y_xr = &BB_surf[28]; 
  const double *B_x_B_z_xl = &BB_surf[32]; 
  const double *B_x_B_z_xr = &BB_surf[36]; 
  int *cell_avg_magB2_xl = &cell_avg_magB2_surf[0]; 
  int *cell_avg_magB2_xr = &cell_avg_magB2_surf[1]; 
 
  struct gkyl_mat rhs_byby_yl = gkyl_nmat_get(rhs, count+6); 
  struct gkyl_mat rhs_byby_yr = gkyl_nmat_get(rhs, count+7); 
  struct gkyl_mat rhs_bxby_yl = gkyl_nmat_get(rhs, count+8); 
  struct gkyl_mat rhs_bxby_yr = gkyl_nmat_get(rhs, count+9); 
  struct gkyl_mat rhs_bybz_yl = gkyl_nmat_get(rhs, count+10); 
  struct gkyl_mat rhs_bybz_yr = gkyl_nmat_get(rhs, count+11); 
  gkyl_mat_clear(&rhs_byby_yl, 0.0); 
  gkyl_mat_clear(&rhs_byby_yr, 0.0); 
  gkyl_mat_clear(&rhs_bxby_yl, 0.0); 
  gkyl_mat_clear(&rhs_bxby_yr, 0.0); 
  gkyl_mat_clear(&rhs_bybz_yl, 0.0); 
  gkyl_mat_clear(&rhs_bybz_yr, 0.0); 
  const double *Bx_sq_yl = &BB_surf[40]; 
  const double *Bx_sq_yr = &BB_surf[44]; 
  const double *By_sq_yl = &BB_surf[48]; 
  const double *By_sq_yr = &BB_surf[52]; 
  const double *Bz_sq_yl = &BB_surf[56]; 
  const double *Bz_sq_yr = &BB_surf[60]; 
  const double *B_x_B_y_yl = &BB_surf[64]; 
  const double *B_x_B_y_yr = &BB_surf[68]; 
  const double *B_y_B_z_yl = &BB_surf[72]; 
  const double *B_y_B_z_yr = &BB_surf[76]; 
  int *cell_avg_magB2_yl = &cell_avg_magB2_surf[2]; 
  int *cell_avg_magB2_yr = &cell_avg_magB2_surf[3]; 
 
  struct gkyl_mat rhs_bzbz_zl = gkyl_nmat_get(rhs, count+12); 
  struct gkyl_mat rhs_bzbz_zr = gkyl_nmat_get(rhs, count+13); 
  struct gkyl_mat rhs_bxbz_zl = gkyl_nmat_get(rhs, count+14); 
  struct gkyl_mat rhs_bxbz_zr = gkyl_nmat_get(rhs, count+15); 
  struct gkyl_mat rhs_bybz_zl = gkyl_nmat_get(rhs, count+16); 
  struct gkyl_mat rhs_bybz_zr = gkyl_nmat_get(rhs, count+17); 
  gkyl_mat_clear(&rhs_bzbz_zl, 0.0); 
  gkyl_mat_clear(&rhs_bzbz_zr, 0.0); 
  gkyl_mat_clear(&rhs_bxbz_zl, 0.0); 
  gkyl_mat_clear(&rhs_bxbz_zr, 0.0); 
  gkyl_mat_clear(&rhs_bybz_zl, 0.0); 
  gkyl_mat_clear(&rhs_bybz_zr, 0.0); 
  const double *Bx_sq_zl = &BB_surf[80]; 
  const double *Bx_sq_zr = &BB_surf[84]; 
  const double *By_sq_zl = &BB_surf[88]; 
  const double *By_sq_zr = &BB_surf[92]; 
  const double *Bz_sq_zl = &BB_surf[96]; 
  const double *Bz_sq_zr = &BB_surf[100]; 
  const double *B_x_B_z_zl = &BB_surf[104]; 
  const double *B_x_B_z_zr = &BB_surf[108]; 
  const double *B_y_B_z_zl = &BB_surf[112]; 
  const double *B_y_B_z_zr = &BB_surf[116]; 
  int *cell_avg_magB2_zl = &cell_avg_magB2_surf[4]; 
  int *cell_avg_magB2_zr = &cell_avg_magB2_surf[5]; 
 
  double magB2_xl[4] = {0.0}; 
  double magB2_xr[4] = {0.0}; 
  double magB2_yl[4] = {0.0}; 
  double magB2_yr[4] = {0.0}; 
  double magB2_zl[4] = {0.0}; 
  double magB2_zr[4] = {0.0}; 
  magB2_xl[0] = Bx_sq_xl[0] + By_sq_xl[0] + Bz_sq_xl[0]; 
  magB2_xr[0] = Bx_sq_xr[0] + By_sq_xr[0] + Bz_sq_xr[0]; 
  magB2_yl[0] = Bx_sq_yl[0] + By_sq_yl[0] + Bz_sq_yl[0]; 
  magB2_yr[0] = Bx_sq_yr[0] + By_sq_yr[0] + Bz_sq_yr[0]; 
  magB2_zl[0] = Bx_sq_zl[0] + By_sq_zl[0] + Bz_sq_zl[0]; 
  magB2_zr[0] = Bx_sq_zr[0] + By_sq_zr[0] + Bz_sq_zr[0]; 
  magB2_xl[1] = Bx_sq_xl[1] + By_sq_xl[1] + Bz_sq_xl[1]; 
  magB2_xr[1] = Bx_sq_xr[1] + By_sq_xr[1] + Bz_sq_xr[1]; 
  magB2_yl[1] = Bx_sq_yl[1] + By_sq_yl[1] + Bz_sq_yl[1]; 
  magB2_yr[1] = Bx_sq_yr[1] + By_sq_yr[1] + Bz_sq_yr[1]; 
  magB2_zl[1] = Bx_sq_zl[1] + By_sq_zl[1] + Bz_sq_zl[1]; 
  magB2_zr[1] = Bx_sq_zr[1] + By_sq_zr[1] + Bz_sq_zr[1]; 
  magB2_xl[2] = Bx_sq_xl[2] + By_sq_xl[2] + Bz_sq_xl[2]; 
  magB2_xr[2] = Bx_sq_xr[2] + By_sq_xr[2] + Bz_sq_xr[2]; 
  magB2_yl[2] = Bx_sq_yl[2] + By_sq_yl[2] + Bz_sq_yl[2]; 
  magB2_yr[2] = Bx_sq_yr[2] + By_sq_yr[2] + Bz_sq_yr[2]; 
  magB2_zl[2] = Bx_sq_zl[2] + By_sq_zl[2] + Bz_sq_zl[2]; 
  magB2_zr[2] = Bx_sq_zr[2] + By_sq_zr[2] + Bz_sq_zr[2]; 
  magB2_xl[3] = Bx_sq_xl[3] + By_sq_xl[3] + Bz_sq_xl[3]; 
  magB2_xr[3] = Bx_sq_xr[3] + By_sq_xr[3] + Bz_sq_xr[3]; 
  magB2_yl[3] = Bx_sq_yl[3] + By_sq_yl[3] + Bz_sq_yl[3]; 
  magB2_yr[3] = Bx_sq_yr[3] + By_sq_yr[3] + Bz_sq_yr[3]; 
  magB2_zl[3] = Bx_sq_zl[3] + By_sq_zl[3] + Bz_sq_zl[3]; 
  magB2_zr[3] = Bx_sq_zr[3] + By_sq_zr[3] + Bz_sq_zr[3]; 
  // If |B|^2 < 0 at control points along a surface, only use cell average to get 1/|B|^2. 
  // Each surface is checked independently. 
  int cell_avg_xl = 0;
  double bxbx_xl[4] = {0.0}; 
  double bxby_xl[4] = {0.0}; 
  double bxbz_xl[4] = {0.0}; 
 
  int cell_avg_xr = 0;
  double bxbx_xr[4] = {0.0}; 
  double bxby_xr[4] = {0.0}; 
  double bxbz_xr[4] = {0.0}; 
 
  int cell_avg_yl = 0;
  double byby_yl[4] = {0.0}; 
  double bxby_yl[4] = {0.0}; 
  double bybz_yl[4] = {0.0}; 
 
  int cell_avg_yr = 0;
  double byby_yr[4] = {0.0}; 
  double bxby_yr[4] = {0.0}; 
  double bybz_yr[4] = {0.0}; 
 
  int cell_avg_zl = 0;
  double bzbz_zl[4] = {0.0}; 
  double bxbz_zl[4] = {0.0}; 
  double bybz_zl[4] = {0.0}; 
 
  int cell_avg_zr = 0;
  double bzbz_zr[4] = {0.0}; 
  double bxbz_zr[4] = {0.0}; 
  double bybz_zr[4] = {0.0}; 
 
  if (1.5*magB2_xl[3]-0.8660254037844386*magB2_xl[2]-0.8660254037844386*magB2_xl[1]+0.5*magB2_xl[0] < 0.0) cell_avg_xl = 1; 
  if (1.5*magB2_xr[3]-0.8660254037844386*magB2_xr[2]-0.8660254037844386*magB2_xr[1]+0.5*magB2_xr[0] < 0.0) cell_avg_xr = 1; 
  if (1.5*magB2_yl[3]-0.8660254037844386*magB2_yl[2]-0.8660254037844386*magB2_yl[1]+0.5*magB2_yl[0] < 0.0) cell_avg_yl = 1; 
  if (1.5*magB2_yr[3]-0.8660254037844386*magB2_yr[2]-0.8660254037844386*magB2_yr[1]+0.5*magB2_yr[0] < 0.0) cell_avg_yr = 1; 
  if (1.5*magB2_zl[3]-0.8660254037844386*magB2_zl[2]-0.8660254037844386*magB2_zl[1]+0.5*magB2_zl[0] < 0.0) cell_avg_zl = 1; 
  if (1.5*magB2_zr[3]-0.8660254037844386*magB2_zr[2]-0.8660254037844386*magB2_zr[1]+0.5*magB2_zr[0] < 0.0) cell_avg_zr = 1; 
  if ((-1.5*magB2_xl[3])-0.8660254037844386*magB2_xl[2]+0.8660254037844386*magB2_xl[1]+0.5*magB2_xl[0] < 0.0) cell_avg_xl = 1; 
  if ((-1.5*magB2_xr[3])-0.8660254037844386*magB2_xr[2]+0.8660254037844386*magB2_xr[1]+0.5*magB2_xr[0] < 0.0) cell_avg_xr = 1; 
  if ((-1.5*magB2_yl[3])-0.8660254037844386*magB2_yl[2]+0.8660254037844386*magB2_yl[1]+0.5*magB2_yl[0] < 0.0) cell_avg_yl = 1; 
  if ((-1.5*magB2_yr[3])-0.8660254037844386*magB2_yr[2]+0.8660254037844386*magB2_yr[1]+0.5*magB2_yr[0] < 0.0) cell_avg_yr = 1; 
  if ((-1.5*magB2_zl[3])-0.8660254037844386*magB2_zl[2]+0.8660254037844386*magB2_zl[1]+0.5*magB2_zl[0] < 0.0) cell_avg_zl = 1; 
  if ((-1.5*magB2_zr[3])-0.8660254037844386*magB2_zr[2]+0.8660254037844386*magB2_zr[1]+0.5*magB2_zr[0] < 0.0) cell_avg_zr = 1; 
  if ((-1.5*magB2_xl[3])+0.8660254037844386*magB2_xl[2]-0.8660254037844386*magB2_xl[1]+0.5*magB2_xl[0] < 0.0) cell_avg_xl = 1; 
  if ((-1.5*magB2_xr[3])+0.8660254037844386*magB2_xr[2]-0.8660254037844386*magB2_xr[1]+0.5*magB2_xr[0] < 0.0) cell_avg_xr = 1; 
  if ((-1.5*magB2_yl[3])+0.8660254037844386*magB2_yl[2]-0.8660254037844386*magB2_yl[1]+0.5*magB2_yl[0] < 0.0) cell_avg_yl = 1; 
  if ((-1.5*magB2_yr[3])+0.8660254037844386*magB2_yr[2]-0.8660254037844386*magB2_yr[1]+0.5*magB2_yr[0] < 0.0) cell_avg_yr = 1; 
  if ((-1.5*magB2_zl[3])+0.8660254037844386*magB2_zl[2]-0.8660254037844386*magB2_zl[1]+0.5*magB2_zl[0] < 0.0) cell_avg_zl = 1; 
  if ((-1.5*magB2_zr[3])+0.8660254037844386*magB2_zr[2]-0.8660254037844386*magB2_zr[1]+0.5*magB2_zr[0] < 0.0) cell_avg_zr = 1; 
  if (1.5*magB2_xl[3]+0.8660254037844386*magB2_xl[2]+0.8660254037844386*magB2_xl[1]+0.5*magB2_xl[0] < 0.0) cell_avg_xl = 1; 
  if (1.5*magB2_xr[3]+0.8660254037844386*magB2_xr[2]+0.8660254037844386*magB2_xr[1]+0.5*magB2_xr[0] < 0.0) cell_avg_xr = 1; 
  if (1.5*magB2_yl[3]+0.8660254037844386*magB2_yl[2]+0.8660254037844386*magB2_yl[1]+0.5*magB2_yl[0] < 0.0) cell_avg_yl = 1; 
  if (1.5*magB2_yr[3]+0.8660254037844386*magB2_yr[2]+0.8660254037844386*magB2_yr[1]+0.5*magB2_yr[0] < 0.0) cell_avg_yr = 1; 
  if (1.5*magB2_zl[3]+0.8660254037844386*magB2_zl[2]+0.8660254037844386*magB2_zl[1]+0.5*magB2_zl[0] < 0.0) cell_avg_zl = 1; 
  if (1.5*magB2_zr[3]+0.8660254037844386*magB2_zr[2]+0.8660254037844386*magB2_zr[1]+0.5*magB2_zr[0] < 0.0) cell_avg_zr = 1; 
 
  cell_avg_magB2_xl[0] = cell_avg_xl; 
  cell_avg_magB2_xr[0] = cell_avg_xr; 
  cell_avg_magB2_yl[0] = cell_avg_yl; 
  cell_avg_magB2_yr[0] = cell_avg_yr; 
  cell_avg_magB2_zl[0] = cell_avg_zl; 
  cell_avg_magB2_zr[0] = cell_avg_zr; 
 
  double magB2_inv_xl[4] = {0.0}; 

  if (cell_avg_xl) { 
  magB2_inv_xl[0] = 4.0/magB2_xl[0]; 
  } else { 
  ser_2x_p1_inv(magB2_xl, magB2_inv_xl); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_xl, Bx_sq_xl, bxbx_xl); 
  binop_mul_2d_ser_p1(magB2_inv_xl, B_x_B_y_xl, bxby_xl); 
  binop_mul_2d_ser_p1(magB2_inv_xl, B_x_B_z_xl, bxbz_xl); 
 
  double magB2_inv_xr[4] = {0.0}; 

  if (cell_avg_xr) { 
  magB2_inv_xr[0] = 4.0/magB2_xr[0]; 
  } else { 
  ser_2x_p1_inv(magB2_xr, magB2_inv_xr); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_xr, Bx_sq_xr, bxbx_xr); 
  binop_mul_2d_ser_p1(magB2_inv_xr, B_x_B_y_xr, bxby_xr); 
  binop_mul_2d_ser_p1(magB2_inv_xr, B_x_B_z_xr, bxbz_xr); 
 
  double magB2_inv_yl[4] = {0.0}; 

  if (cell_avg_yl) { 
  magB2_inv_yl[0] = 4.0/magB2_yl[0]; 
  } else { 
  ser_2x_p1_inv(magB2_yl, magB2_inv_yl); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_yl, By_sq_yl, byby_yl); 
  binop_mul_2d_ser_p1(magB2_inv_yl, B_x_B_y_yl, bxby_yl); 
  binop_mul_2d_ser_p1(magB2_inv_yl, B_y_B_z_yl, bybz_yl); 
 
  double magB2_inv_yr[4] = {0.0}; 

  if (cell_avg_yr) { 
  magB2_inv_yr[0] = 4.0/magB2_yr[0]; 
  } else { 
  ser_2x_p1_inv(magB2_yr, magB2_inv_yr); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_yr, By_sq_yr, byby_yr); 
  binop_mul_2d_ser_p1(magB2_inv_yr, B_x_B_y_yr, bxby_yr); 
  binop_mul_2d_ser_p1(magB2_inv_yr, B_y_B_z_yr, bybz_yr); 
 
  double magB2_inv_zl[4] = {0.0}; 

  if (cell_avg_zl) { 
  magB2_inv_zl[0] = 4.0/magB2_zl[0]; 
  } else { 
  ser_2x_p1_inv(magB2_zl, magB2_inv_zl); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_zl, Bz_sq_zl, bzbz_zl); 
  binop_mul_2d_ser_p1(magB2_inv_zl, B_x_B_z_zl, bxbz_zl); 
  binop_mul_2d_ser_p1(magB2_inv_zl, B_y_B_z_zl, bybz_zl); 
 
  double magB2_inv_zr[4] = {0.0}; 

  if (cell_avg_zr) { 
  magB2_inv_zr[0] = 4.0/magB2_zr[0]; 
  } else { 
  ser_2x_p1_inv(magB2_zr, magB2_inv_zr); 
  } 
  binop_mul_2d_ser_p1(magB2_inv_zr, Bz_sq_zr, bzbz_zr); 
  binop_mul_2d_ser_p1(magB2_inv_zr, B_x_B_z_zr, bxbz_zr); 
  binop_mul_2d_ser_p1(magB2_inv_zr, B_y_B_z_zr, bybz_zr); 
 
  gkyl_mat_set(&rhs_bxbx_xl,0,0,bxbx_xl[0]); 
  gkyl_mat_set(&rhs_bxbx_xr,0,0,bxbx_xr[0]); 
  gkyl_mat_set(&rhs_bxby_xl,0,0,bxby_xl[0]); 
  gkyl_mat_set(&rhs_bxby_xr,0,0,bxby_xr[0]); 
  gkyl_mat_set(&rhs_bxbz_xl,0,0,bxbz_xl[0]); 
  gkyl_mat_set(&rhs_bxbz_xr,0,0,bxbz_xr[0]); 
 
  gkyl_mat_set(&rhs_byby_yl,0,0,byby_yl[0]); 
  gkyl_mat_set(&rhs_byby_yr,0,0,byby_yr[0]); 
  gkyl_mat_set(&rhs_bxby_yl,0,0,bxby_yl[0]); 
  gkyl_mat_set(&rhs_bxby_yr,0,0,bxby_yr[0]); 
  gkyl_mat_set(&rhs_bybz_yl,0,0,bybz_yl[0]); 
  gkyl_mat_set(&rhs_bybz_yr,0,0,bybz_yr[0]); 
 
  gkyl_mat_set(&rhs_bzbz_zl,0,0,bzbz_zl[0]); 
  gkyl_mat_set(&rhs_bzbz_zr,0,0,bzbz_zr[0]); 
  gkyl_mat_set(&rhs_bxbz_zl,0,0,bxbz_zl[0]); 
  gkyl_mat_set(&rhs_bxbz_zr,0,0,bxbz_zr[0]); 
  gkyl_mat_set(&rhs_bybz_zl,0,0,bybz_zl[0]); 
  gkyl_mat_set(&rhs_bybz_zr,0,0,bybz_zr[0]); 
 
  gkyl_mat_set(&rhs_bxbx_xl,1,0,bxbx_xl[1]); 
  gkyl_mat_set(&rhs_bxbx_xr,1,0,bxbx_xr[1]); 
  gkyl_mat_set(&rhs_bxby_xl,1,0,bxby_xl[1]); 
  gkyl_mat_set(&rhs_bxby_xr,1,0,bxby_xr[1]); 
  gkyl_mat_set(&rhs_bxbz_xl,1,0,bxbz_xl[1]); 
  gkyl_mat_set(&rhs_bxbz_xr,1,0,bxbz_xr[1]); 
 
  gkyl_mat_set(&rhs_byby_yl,1,0,byby_yl[1]); 
  gkyl_mat_set(&rhs_byby_yr,1,0,byby_yr[1]); 
  gkyl_mat_set(&rhs_bxby_yl,1,0,bxby_yl[1]); 
  gkyl_mat_set(&rhs_bxby_yr,1,0,bxby_yr[1]); 
  gkyl_mat_set(&rhs_bybz_yl,1,0,bybz_yl[1]); 
  gkyl_mat_set(&rhs_bybz_yr,1,0,bybz_yr[1]); 
 
  gkyl_mat_set(&rhs_bzbz_zl,1,0,bzbz_zl[1]); 
  gkyl_mat_set(&rhs_bzbz_zr,1,0,bzbz_zr[1]); 
  gkyl_mat_set(&rhs_bxbz_zl,1,0,bxbz_zl[1]); 
  gkyl_mat_set(&rhs_bxbz_zr,1,0,bxbz_zr[1]); 
  gkyl_mat_set(&rhs_bybz_zl,1,0,bybz_zl[1]); 
  gkyl_mat_set(&rhs_bybz_zr,1,0,bybz_zr[1]); 
 
  gkyl_mat_set(&rhs_bxbx_xl,2,0,bxbx_xl[2]); 
  gkyl_mat_set(&rhs_bxbx_xr,2,0,bxbx_xr[2]); 
  gkyl_mat_set(&rhs_bxby_xl,2,0,bxby_xl[2]); 
  gkyl_mat_set(&rhs_bxby_xr,2,0,bxby_xr[2]); 
  gkyl_mat_set(&rhs_bxbz_xl,2,0,bxbz_xl[2]); 
  gkyl_mat_set(&rhs_bxbz_xr,2,0,bxbz_xr[2]); 
 
  gkyl_mat_set(&rhs_byby_yl,2,0,byby_yl[2]); 
  gkyl_mat_set(&rhs_byby_yr,2,0,byby_yr[2]); 
  gkyl_mat_set(&rhs_bxby_yl,2,0,bxby_yl[2]); 
  gkyl_mat_set(&rhs_bxby_yr,2,0,bxby_yr[2]); 
  gkyl_mat_set(&rhs_bybz_yl,2,0,bybz_yl[2]); 
  gkyl_mat_set(&rhs_bybz_yr,2,0,bybz_yr[2]); 
 
  gkyl_mat_set(&rhs_bzbz_zl,2,0,bzbz_zl[2]); 
  gkyl_mat_set(&rhs_bzbz_zr,2,0,bzbz_zr[2]); 
  gkyl_mat_set(&rhs_bxbz_zl,2,0,bxbz_zl[2]); 
  gkyl_mat_set(&rhs_bxbz_zr,2,0,bxbz_zr[2]); 
  gkyl_mat_set(&rhs_bybz_zl,2,0,bybz_zl[2]); 
  gkyl_mat_set(&rhs_bybz_zr,2,0,bybz_zr[2]); 
 
  gkyl_mat_set(&rhs_bxbx_xl,3,0,bxbx_xl[3]); 
  gkyl_mat_set(&rhs_bxbx_xr,3,0,bxbx_xr[3]); 
  gkyl_mat_set(&rhs_bxby_xl,3,0,bxby_xl[3]); 
  gkyl_mat_set(&rhs_bxby_xr,3,0,bxby_xr[3]); 
  gkyl_mat_set(&rhs_bxbz_xl,3,0,bxbz_xl[3]); 
  gkyl_mat_set(&rhs_bxbz_xr,3,0,bxbz_xr[3]); 
 
  gkyl_mat_set(&rhs_byby_yl,3,0,byby_yl[3]); 
  gkyl_mat_set(&rhs_byby_yr,3,0,byby_yr[3]); 
  gkyl_mat_set(&rhs_bxby_yl,3,0,bxby_yl[3]); 
  gkyl_mat_set(&rhs_bxby_yr,3,0,bxby_yr[3]); 
  gkyl_mat_set(&rhs_bybz_yl,3,0,bybz_yl[3]); 
  gkyl_mat_set(&rhs_bybz_yr,3,0,bybz_yr[3]); 
 
  gkyl_mat_set(&rhs_bzbz_zl,3,0,bzbz_zl[3]); 
  gkyl_mat_set(&rhs_bzbz_zr,3,0,bzbz_zr[3]); 
  gkyl_mat_set(&rhs_bxbz_zl,3,0,bxbz_zl[3]); 
  gkyl_mat_set(&rhs_bxbz_zr,3,0,bxbz_zr[3]); 
  gkyl_mat_set(&rhs_bybz_zl,3,0,bybz_zl[3]); 
  gkyl_mat_set(&rhs_bybz_zr,3,0,bybz_zr[3]); 
 
} 