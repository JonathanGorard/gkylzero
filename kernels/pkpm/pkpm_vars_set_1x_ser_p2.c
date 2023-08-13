#include <gkyl_mat.h> 
#include <gkyl_euler_pkpm_kernels.h> 
GKYL_CU_DH int pkpm_vars_set_1x_ser_p2(int count, struct gkyl_nmat *A, struct gkyl_nmat *rhs, 
  const double *vlasov_pkpm_moms, const double *euler_pkpm, const double *pkpm_div_ppar) 
{ 
  // count:            integer to indicate which matrix being fetched. 
  // A:                preallocated LHS matrix. 
  // rhs:              preallocated RHS vector. 
  // vlasov_pkpm_moms: [rho, p_parallel, p_perp], Moments computed from kinetic equation in pkpm model.
  // euler_pkpm:       [rho ux, rho uy, rho uz], Fluid input state vector.
  // pkpm_div_ppar:    div(p_par b) computed from kinetic equation for consistency.

  struct gkyl_mat A_ux = gkyl_nmat_get(A, count); 
  struct gkyl_mat A_uy = gkyl_nmat_get(A, count+1); 
  struct gkyl_mat A_uz = gkyl_nmat_get(A, count+2); 
  struct gkyl_mat A_pkpm_div_ppar = gkyl_nmat_get(A, count+3); 
  struct gkyl_mat A_T_perp_over_m = gkyl_nmat_get(A, count+4); 
  struct gkyl_mat A_T_perp_over_m_inv = gkyl_nmat_get(A, count+5); 
  struct gkyl_mat rhs_ux = gkyl_nmat_get(rhs, count); 
  struct gkyl_mat rhs_uy = gkyl_nmat_get(rhs, count+1); 
  struct gkyl_mat rhs_uz = gkyl_nmat_get(rhs, count+2); 
  struct gkyl_mat rhs_pkpm_div_ppar = gkyl_nmat_get(rhs, count+3); 
  struct gkyl_mat rhs_T_perp_over_m = gkyl_nmat_get(rhs, count+4); 
  struct gkyl_mat rhs_T_perp_over_m_inv = gkyl_nmat_get(rhs, count+5); 
  // Clear matrix and rhs for each component of primitive variables being solved for 
  gkyl_mat_clear(&A_ux, 0.0); gkyl_mat_clear(&rhs_ux, 0.0); 
  gkyl_mat_clear(&A_uy, 0.0); gkyl_mat_clear(&rhs_uy, 0.0); 
  gkyl_mat_clear(&A_uz, 0.0); gkyl_mat_clear(&rhs_uz, 0.0); 
  gkyl_mat_clear(&A_pkpm_div_ppar, 0.0); gkyl_mat_clear(&rhs_pkpm_div_ppar, 0.0); 
  gkyl_mat_clear(&A_T_perp_over_m, 0.0); gkyl_mat_clear(&rhs_T_perp_over_m, 0.0); 
  gkyl_mat_clear(&A_T_perp_over_m_inv, 0.0); gkyl_mat_clear(&rhs_T_perp_over_m_inv, 0.0); 
  const double *rhoux = &euler_pkpm[0]; 
  const double *rhouy = &euler_pkpm[3]; 
  const double *rhouz = &euler_pkpm[6]; 
  const double *rho = &vlasov_pkpm_moms[0]; 
  const double *p_perp = &vlasov_pkpm_moms[6]; 
  int cell_avg = 0;
  // Check if rho, p_par, or p_perp < 0 at control points. 
  if (1.58113883008419*rho[2]-1.224744871391589*rho[1]+0.7071067811865475*rho[0] < 0.0) cell_avg = 1; 
  if (1.58113883008419*p_perp[2]-1.224744871391589*p_perp[1]+0.7071067811865475*p_perp[0] < 0.0) cell_avg = 1; 
  if (0.7071067811865475*rho[0]-0.7905694150420947*rho[2] < 0.0) cell_avg = 1; 
  if (0.7071067811865475*p_perp[0]-0.7905694150420947*p_perp[2] < 0.0) cell_avg = 1; 
  if (1.58113883008419*rho[2]+1.224744871391589*rho[1]+0.7071067811865475*rho[0] < 0.0) cell_avg = 1; 
  if (1.58113883008419*p_perp[2]+1.224744871391589*p_perp[1]+0.7071067811865475*p_perp[0] < 0.0) cell_avg = 1; 
  double rho_temp[3] = {0.0}; 
  double p_perp_temp[3] = {0.0}; 
  if (cell_avg) { 
    rho_temp[0] = rho[0]; 
    p_perp_temp[0] = p_perp[0]; 
  } else { 
    rho_temp[0] = rho[0]; 
    p_perp_temp[0] = p_perp[0]; 
    rho_temp[1] = rho[1]; 
    p_perp_temp[1] = p_perp[1]; 
    rho_temp[2] = rho[2]; 
    p_perp_temp[2] = p_perp[2]; 
  } 
 
  if (cell_avg) { 
    gkyl_mat_set(&rhs_ux,0,0,rhoux[0]); 
    gkyl_mat_set(&rhs_uy,0,0,rhouy[0]); 
    gkyl_mat_set(&rhs_uz,0,0,rhouz[0]); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,0,0,pkpm_div_ppar[0]); 
    gkyl_mat_set(&rhs_T_perp_over_m,0,0,p_perp[0]); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,0,0,rho[0]); 
    gkyl_mat_set(&rhs_ux,1,0,0.0); 
    gkyl_mat_set(&rhs_uy,1,0,0.0); 
    gkyl_mat_set(&rhs_uz,1,0,0.0); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,1,0,0.0); 
    gkyl_mat_set(&rhs_T_perp_over_m,1,0,0.0); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,1,0,0.0); 
    gkyl_mat_set(&rhs_ux,2,0,0.0); 
    gkyl_mat_set(&rhs_uy,2,0,0.0); 
    gkyl_mat_set(&rhs_uz,2,0,0.0); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,2,0,0.0); 
    gkyl_mat_set(&rhs_T_perp_over_m,2,0,0.0); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,2,0,0.0); 
  } else { 
    gkyl_mat_set(&rhs_ux,0,0,rhoux[0]); 
    gkyl_mat_set(&rhs_uy,0,0,rhouy[0]); 
    gkyl_mat_set(&rhs_uz,0,0,rhouz[0]); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,0,0,pkpm_div_ppar[0]); 
    gkyl_mat_set(&rhs_T_perp_over_m,0,0,p_perp[0]); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,0,0,rho[0]); 
    gkyl_mat_set(&rhs_ux,1,0,rhoux[1]); 
    gkyl_mat_set(&rhs_uy,1,0,rhouy[1]); 
    gkyl_mat_set(&rhs_uz,1,0,rhouz[1]); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,1,0,pkpm_div_ppar[1]); 
    gkyl_mat_set(&rhs_T_perp_over_m,1,0,p_perp[1]); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,1,0,rho[1]); 
    gkyl_mat_set(&rhs_ux,2,0,rhoux[2]); 
    gkyl_mat_set(&rhs_uy,2,0,rhouy[2]); 
    gkyl_mat_set(&rhs_uz,2,0,rhouz[2]); 
    gkyl_mat_set(&rhs_pkpm_div_ppar,2,0,pkpm_div_ppar[2]); 
    gkyl_mat_set(&rhs_T_perp_over_m,2,0,p_perp[2]); 
    gkyl_mat_set(&rhs_T_perp_over_m_inv,2,0,rho[2]); 
  } 
 
  double temp_rho = 0.0; 
  double temp_p_perp = 0.0; 
  temp_rho = 0.7071067811865475*rho_temp[0]; 
  gkyl_mat_set(&A_ux,0,0,temp_rho); 
  gkyl_mat_set(&A_uy,0,0,temp_rho); 
  gkyl_mat_set(&A_uz,0,0,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,0,0,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,0,0,temp_rho); 

  temp_p_perp = 0.7071067811865475*p_perp_temp[0]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,0,0,temp_p_perp); 

  temp_rho = 0.7071067811865475*rho_temp[1]; 
  gkyl_mat_set(&A_ux,0,1,temp_rho); 
  gkyl_mat_set(&A_uy,0,1,temp_rho); 
  gkyl_mat_set(&A_uz,0,1,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,0,1,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,0,1,temp_rho); 

  temp_p_perp = 0.7071067811865475*p_perp_temp[1]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,0,1,temp_p_perp); 

  temp_rho = 0.7071067811865475*rho_temp[2]; 
  gkyl_mat_set(&A_ux,0,2,temp_rho); 
  gkyl_mat_set(&A_uy,0,2,temp_rho); 
  gkyl_mat_set(&A_uz,0,2,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,0,2,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,0,2,temp_rho); 

  temp_p_perp = 0.7071067811865475*p_perp_temp[2]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,0,2,temp_p_perp); 

  temp_rho = 0.7071067811865475*rho_temp[1]; 
  gkyl_mat_set(&A_ux,1,0,temp_rho); 
  gkyl_mat_set(&A_uy,1,0,temp_rho); 
  gkyl_mat_set(&A_uz,1,0,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,1,0,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,1,0,temp_rho); 

  temp_p_perp = 0.7071067811865475*p_perp_temp[1]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,1,0,temp_p_perp); 

  temp_rho = 0.6324555320336759*rho_temp[2]+0.7071067811865475*rho_temp[0]; 
  gkyl_mat_set(&A_ux,1,1,temp_rho); 
  gkyl_mat_set(&A_uy,1,1,temp_rho); 
  gkyl_mat_set(&A_uz,1,1,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,1,1,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,1,1,temp_rho); 

  temp_p_perp = 0.6324555320336759*p_perp_temp[2]+0.7071067811865475*p_perp_temp[0]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,1,1,temp_p_perp); 

  temp_rho = 0.6324555320336759*rho_temp[1]; 
  gkyl_mat_set(&A_ux,1,2,temp_rho); 
  gkyl_mat_set(&A_uy,1,2,temp_rho); 
  gkyl_mat_set(&A_uz,1,2,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,1,2,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,1,2,temp_rho); 

  temp_p_perp = 0.6324555320336759*p_perp_temp[1]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,1,2,temp_p_perp); 

  temp_rho = 0.7071067811865475*rho_temp[2]; 
  gkyl_mat_set(&A_ux,2,0,temp_rho); 
  gkyl_mat_set(&A_uy,2,0,temp_rho); 
  gkyl_mat_set(&A_uz,2,0,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,2,0,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,2,0,temp_rho); 

  temp_p_perp = 0.7071067811865475*p_perp_temp[2]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,2,0,temp_p_perp); 

  temp_rho = 0.6324555320336759*rho_temp[1]; 
  gkyl_mat_set(&A_ux,2,1,temp_rho); 
  gkyl_mat_set(&A_uy,2,1,temp_rho); 
  gkyl_mat_set(&A_uz,2,1,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,2,1,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,2,1,temp_rho); 

  temp_p_perp = 0.6324555320336759*p_perp_temp[1]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,2,1,temp_p_perp); 

  temp_rho = 0.4517539514526256*rho_temp[2]+0.7071067811865475*rho_temp[0]; 
  gkyl_mat_set(&A_ux,2,2,temp_rho); 
  gkyl_mat_set(&A_uy,2,2,temp_rho); 
  gkyl_mat_set(&A_uz,2,2,temp_rho); 
  gkyl_mat_set(&A_pkpm_div_ppar,2,2,temp_rho); 
  gkyl_mat_set(&A_T_perp_over_m,2,2,temp_rho); 

  temp_p_perp = 0.4517539514526256*p_perp_temp[2]+0.7071067811865475*p_perp_temp[0]; 
  gkyl_mat_set(&A_T_perp_over_m_inv,2,2,temp_p_perp); 

  return cell_avg;
} 
