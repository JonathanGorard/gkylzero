#include <gkyl_mat.h> 
#include <gkyl_maxwell_kernels.h> 
#include <gkyl_basis_ser_1x_p1_sqrt_with_sign.h> 
GKYL_CU_DH void em_surf_copy_bvar_2x_ser_p1(int count, struct gkyl_nmat *x, const double *em, int* cell_avg_magB2_surf, double* GKYL_RESTRICT bvar_surf) 
{ 
  // count:               Integer to indicate which matrix being fetched. 
  // x:                   Input solution vector. 
  // em:                  Input electromagnetic fields. 
  // cell_avg_magB2_surf: Output flag for cell average if 1/|B|^2 at a surface only used cell averages. 
  // bvar_surf:           Output magnetic field unit tensor and unit vector at surfaces. 
  //                      [bx_xl, bx_xr, bxbx_xl, bxbx_xr, bxby_xl, bxby_xr, bxbz_xl, bxbz_xr, 
  //                       by_yl, by_yr, byby_yl, byby_yr, bxby_yl, bxby_yr, bybz_yl, bybz_yr, 
  //                       bz_zl, bz_zr, bzbz_zl, bzbz_zr, bxbz_zl, bxbz_zr, bybz_zl, bybz_zr] 
 
  struct gkyl_mat x_bxbx_xl = gkyl_nmat_get(x, count); 
  struct gkyl_mat x_bxbx_xr = gkyl_nmat_get(x, count+1); 
  struct gkyl_mat x_bxby_xl = gkyl_nmat_get(x, count+2); 
  struct gkyl_mat x_bxby_xr = gkyl_nmat_get(x, count+3); 
  struct gkyl_mat x_bxbz_xl = gkyl_nmat_get(x, count+4); 
  struct gkyl_mat x_bxbz_xr = gkyl_nmat_get(x, count+5); 
 
  double *bx_xl = &bvar_surf[0]; 
  double *bx_xr = &bvar_surf[2]; 
  double *bxbx_xl = &bvar_surf[4]; 
  double *bxbx_xr = &bvar_surf[6]; 
  double *bxby_xl = &bvar_surf[8]; 
  double *bxby_xr = &bvar_surf[10]; 
  double *bxbz_xl = &bvar_surf[12]; 
  double *bxbz_xr = &bvar_surf[14]; 
  int *cell_avg_magB2_xl = &cell_avg_magB2_surf[0]; 
  int *cell_avg_magB2_xr = &cell_avg_magB2_surf[1]; 
 
  struct gkyl_mat x_byby_yl = gkyl_nmat_get(x, count+6); 
  struct gkyl_mat x_byby_yr = gkyl_nmat_get(x, count+7); 
  struct gkyl_mat x_bxby_yl = gkyl_nmat_get(x, count+8); 
  struct gkyl_mat x_bxby_yr = gkyl_nmat_get(x, count+9); 
  struct gkyl_mat x_bybz_yl = gkyl_nmat_get(x, count+10); 
  struct gkyl_mat x_bybz_yr = gkyl_nmat_get(x, count+11); 
 
  double *by_yl = &bvar_surf[16]; 
  double *by_yr = &bvar_surf[18]; 
  double *byby_yl = &bvar_surf[20]; 
  double *byby_yr = &bvar_surf[22]; 
  double *bxby_yl = &bvar_surf[24]; 
  double *bxby_yr = &bvar_surf[26]; 
  double *bybz_yl = &bvar_surf[28]; 
  double *bybz_yr = &bvar_surf[30]; 
  int *cell_avg_magB2_yl = &cell_avg_magB2_surf[2]; 
  int *cell_avg_magB2_yr = &cell_avg_magB2_surf[3]; 
 
  bxbx_xl[0] = gkyl_mat_get(&x_bxbx_xl,0,0); 
  bxbx_xr[0] = gkyl_mat_get(&x_bxbx_xr,0,0); 
  bxby_xl[0] = gkyl_mat_get(&x_bxby_xl,0,0); 
  bxby_xr[0] = gkyl_mat_get(&x_bxby_xr,0,0); 
  bxbz_xl[0] = gkyl_mat_get(&x_bxbz_xl,0,0); 
  bxbz_xr[0] = gkyl_mat_get(&x_bxbz_xr,0,0); 
 
  byby_yl[0] = gkyl_mat_get(&x_byby_yl,0,0); 
  byby_yr[0] = gkyl_mat_get(&x_byby_yr,0,0); 
  bxby_yl[0] = gkyl_mat_get(&x_bxby_yl,0,0); 
  bxby_yr[0] = gkyl_mat_get(&x_bxby_yr,0,0); 
  bybz_yl[0] = gkyl_mat_get(&x_bybz_yl,0,0); 
  bybz_yr[0] = gkyl_mat_get(&x_bybz_yr,0,0); 
 
  bxbx_xl[1] = gkyl_mat_get(&x_bxbx_xl,1,0); 
  bxbx_xr[1] = gkyl_mat_get(&x_bxbx_xr,1,0); 
  bxby_xl[1] = gkyl_mat_get(&x_bxby_xl,1,0); 
  bxby_xr[1] = gkyl_mat_get(&x_bxby_xr,1,0); 
  bxbz_xl[1] = gkyl_mat_get(&x_bxbz_xl,1,0); 
  bxbz_xr[1] = gkyl_mat_get(&x_bxbz_xr,1,0); 
 
  byby_yl[1] = gkyl_mat_get(&x_byby_yl,1,0); 
  byby_yr[1] = gkyl_mat_get(&x_byby_yr,1,0); 
  bxby_yl[1] = gkyl_mat_get(&x_bxby_yl,1,0); 
  bxby_yr[1] = gkyl_mat_get(&x_bxby_yr,1,0); 
  bybz_yl[1] = gkyl_mat_get(&x_bybz_yl,1,0); 
  bybz_yr[1] = gkyl_mat_get(&x_bybz_yr,1,0); 
 
  const double *B_x = &em[12]; 
  const double *B_y = &em[16]; 
  const double *B_z = &em[20]; 
 
  int cell_avg_xl = 0;
  int cell_avg_xr = 0;
  if (0.7071067811865475*bxbx_xl[0]-0.7071067811865475*bxbx_xl[1] < 0.0) cell_avg_xl = 1; 
  if (0.7071067811865475*bxbx_xr[0]-0.7071067811865475*bxbx_xr[1] < 0.0) cell_avg_xr = 1; 
  if (0.7071067811865475*bxbx_xl[1]+0.7071067811865475*bxbx_xl[0] < 0.0) cell_avg_xl = 1; 
  if (0.7071067811865475*bxbx_xr[1]+0.7071067811865475*bxbx_xr[0] < 0.0) cell_avg_xr = 1; 
 
  if (cell_avg_xl || cell_avg_magB2_xl[0]) { 
    bxbx_xl[1] = 0.0; 
    bxby_xl[1] = 0.0; 
    bxbz_xl[1] = 0.0; 
    // If bxbx, bxby, or bxbz < 0.0 at the lower x surface quadrature points, 
    // set cell_avg_magB2_xl to be true in case it was not true before. 
    cell_avg_magB2_xl[0] = 1; 
  } 
 
  if (cell_avg_xr || cell_avg_magB2_xr[0]) { 
    bxbx_xr[1] = 0.0; 
    bxby_xr[1] = 0.0; 
    bxbz_xr[1] = 0.0; 
    // If bxbx, bxby, or bxbz < 0.0 at the upper x surface quadrature points, 
    // set cell_avg_magB2_xr to be true in case it was not true before. 
    cell_avg_magB2_xr[0] = 1; 
  } 
 
  int cell_avg_yl = 0;
  int cell_avg_yr = 0;
  if (0.7071067811865475*byby_yl[0]-0.7071067811865475*byby_yl[1] < 0.0) cell_avg_yl = 1; 
  if (0.7071067811865475*byby_yr[0]-0.7071067811865475*byby_yr[1] < 0.0) cell_avg_yr = 1; 
  if (0.7071067811865475*byby_yl[1]+0.7071067811865475*byby_yl[0] < 0.0) cell_avg_yl = 1; 
  if (0.7071067811865475*byby_yr[1]+0.7071067811865475*byby_yr[0] < 0.0) cell_avg_yr = 1; 
 
  if (cell_avg_yl || cell_avg_magB2_yl[0]) { 
    byby_yl[1] = 0.0; 
    bxby_yl[1] = 0.0; 
    bybz_yl[1] = 0.0; 
    // If byby, bxby, or bybz < 0.0 at the lower y surface quadrature points, 
    // set cell_avg_magB2_yl to be true in case it was not true before. 
    cell_avg_magB2_yl[0] = 1; 
  } 
 
  if (cell_avg_yr || cell_avg_magB2_yr[0]) { 
    byby_yr[1] = 0.0; 
    bxby_yr[1] = 0.0; 
    bybz_yr[1] = 0.0; 
    // If byby, bxby, or bybz < 0.0 at the upper y surface quadrature points, 
    // set cell_avg_magB2_yr to be true in case it was not true before. 
    cell_avg_magB2_yr[0] = 1; 
  } 
 
  // Calculate b_i = B_i/|B| by taking square root of B_i^2/|B|^2 at quadrature points. 
  // Uses the sign of B_i at quadrature points *on the surface* to get the correct sign of b_i *on the surface*. 
  // Note: positivity check already happened, so only uses cell average if needed to avoid imaginary values of b_hat. 
  double B_x_xl[2] = {0.0}; 
  double B_x_xr[2] = {0.0}; 
 
  B_x_xl[0] = 0.7071067811865475*B_x[0]-1.224744871391589*B_x[1]; 
  B_x_xl[1] = 0.7071067811865475*B_x[2]-1.224744871391589*B_x[3]; 
  B_x_xr[0] = 1.224744871391589*B_x[1]+0.7071067811865475*B_x[0]; 
  B_x_xr[1] = 1.224744871391589*B_x[3]+0.7071067811865475*B_x[2]; 
  ser_1x_p1_sqrt_with_sign(B_x_xl, bxbx_xl, bx_xl); 
  ser_1x_p1_sqrt_with_sign(B_x_xr, bxbx_xr, bx_xr); 
 
  double B_y_yl[2] = {0.0}; 
  double B_y_yr[2] = {0.0}; 
 
  B_y_yl[0] = 0.7071067811865475*B_y[0]-1.224744871391589*B_y[2]; 
  B_y_yl[1] = 0.7071067811865475*B_y[1]-1.224744871391589*B_y[3]; 
  B_y_yr[0] = 1.224744871391589*B_y[2]+0.7071067811865475*B_y[0]; 
  B_y_yr[1] = 1.224744871391589*B_y[3]+0.7071067811865475*B_y[1]; 
  ser_1x_p1_sqrt_with_sign(B_y_yl, byby_yl, by_yl); 
  ser_1x_p1_sqrt_with_sign(B_y_yr, byby_yr, by_yr); 
 
} 
 
