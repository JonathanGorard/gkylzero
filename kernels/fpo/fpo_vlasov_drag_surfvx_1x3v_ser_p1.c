#include <gkyl_fpo_vlasov_kernels.h> 
#include <gkyl_basis_ser_4x_p1_upwind_quad_to_modal.h> 


GKYL_CU_DH double fpo_vlasov_drag_surfvx_1x3v_ser_p1(const double* dxv, 
  const double *alpha_surf_L, const double *alpha_surf_R,
  const double *sgn_alpha_surf_L, const double *sgn_alpha_surf_R,
  const int *const_sgn_alpha_L, const int *const_sgn_alpha_R,
  const double *fL, const double *fC, const double *fR, double* GKYL_RESTRICT out) 
{ 
  // dxv[NDIM]: Cell spacing in each direction. 
  // alpha_surf_L,R: Surface expansion of drag coefficient on left,right boundary of center cell. 
  // sgn_alpha_L,R: sign(alpha_surf_l,r) at quadrature points. 
  // const_sgn_alpha_L,R: Boolean array true if sign(alpha_surf_l,r) is only one sign. 
  // fL, fC, fR: Distribution function in left, center, and right cells. 
  // out: Incremented output. 


  // Index into drag coefficient surface expansion arrays 
  const double *drag_coeff_surf_L = &alpha_surf_L[0]; 
  const double *drag_coeff_surf_R = &alpha_surf_R[0]; 
  const double *sgn_drag_coeff_surf_L = &sgn_alpha_surf_L[0]; 
  const double *sgn_drag_coeff_surf_R = &sgn_alpha_surf_R[0]; 
  const int *const_sgn_drag_coeff_L = &const_sgn_alpha_L[0]; 
  const int *const_sgn_drag_coeff_R = &const_sgn_alpha_R[0]; 
  double dv1 = 2.0/dxv[1]; 


  double fUp_L[8] = {0.0}; 
  if (const_sgn_drag_coeff_L[0] == 1) { 
    if (sgn_drag_coeff_surf_L[0] == 1.0) { 
  fUp_L[0] = 1.5811388300841895*fL[16]+1.224744871391589*fL[2]+0.7071067811865475*fL[0]; 
  fUp_L[1] = 1.5811388300841898*fL[17]+1.224744871391589*fL[5]+0.7071067811865475*fL[1]; 
  fUp_L[2] = 1.5811388300841898*fL[18]+1.224744871391589*fL[7]+0.7071067811865475*fL[3]; 
  fUp_L[3] = 1.5811388300841898*fL[19]+1.224744871391589*fL[9]+0.7071067811865475*fL[4]; 
  fUp_L[4] = 1.5811388300841895*fL[20]+1.224744871391589*fL[11]+0.7071067811865475*fL[6]; 
  fUp_L[5] = 1.5811388300841895*fL[21]+1.224744871391589*fL[12]+0.7071067811865475*fL[8]; 
  fUp_L[6] = 1.5811388300841895*fL[22]+1.224744871391589*fL[14]+0.7071067811865475*fL[10]; 
  fUp_L[7] = 1.5811388300841898*fL[23]+1.224744871391589*fL[15]+0.7071067811865475*fL[13]; 
    } else { 
  fUp_L[0] = 1.5811388300841895*fC[16]-1.224744871391589*fC[2]+0.7071067811865475*fC[0]; 
  fUp_L[1] = 1.5811388300841898*fC[17]-1.224744871391589*fC[5]+0.7071067811865475*fC[1]; 
  fUp_L[2] = 1.5811388300841898*fC[18]-1.224744871391589*fC[7]+0.7071067811865475*fC[3]; 
  fUp_L[3] = 1.5811388300841898*fC[19]-1.224744871391589*fC[9]+0.7071067811865475*fC[4]; 
  fUp_L[4] = 1.5811388300841895*fC[20]-1.224744871391589*fC[11]+0.7071067811865475*fC[6]; 
  fUp_L[5] = 1.5811388300841895*fC[21]-1.224744871391589*fC[12]+0.7071067811865475*fC[8]; 
  fUp_L[6] = 1.5811388300841895*fC[22]-1.224744871391589*fC[14]+0.7071067811865475*fC[10]; 
  fUp_L[7] = 1.5811388300841898*fC[23]-1.224744871391589*fC[15]+0.7071067811865475*fC[13]; 
   } 
  } else { 
  double fL_r[8] = {0.0}; 
  double fC_l[8] = {0.0}; 
  double sgn_drag_coeff_Up_L[8] = {0.0}; 
  ser_4x_p1_upwind_quad_to_modal(sgn_drag_coeff_surf_L, sgn_drag_coeff_Up_L); 

  fL_r[0] = 1.5811388300841895*fL[16]+1.224744871391589*fL[2]+0.7071067811865475*fL[0]; 
  fL_r[1] = 1.5811388300841898*fL[17]+1.224744871391589*fL[5]+0.7071067811865475*fL[1]; 
  fL_r[2] = 1.5811388300841898*fL[18]+1.224744871391589*fL[7]+0.7071067811865475*fL[3]; 
  fL_r[3] = 1.5811388300841898*fL[19]+1.224744871391589*fL[9]+0.7071067811865475*fL[4]; 
  fL_r[4] = 1.5811388300841895*fL[20]+1.224744871391589*fL[11]+0.7071067811865475*fL[6]; 
  fL_r[5] = 1.5811388300841895*fL[21]+1.224744871391589*fL[12]+0.7071067811865475*fL[8]; 
  fL_r[6] = 1.5811388300841895*fL[22]+1.224744871391589*fL[14]+0.7071067811865475*fL[10]; 
  fL_r[7] = 1.5811388300841898*fL[23]+1.224744871391589*fL[15]+0.7071067811865475*fL[13]; 

  fC_l[0] = 1.5811388300841895*fC[16]-1.224744871391589*fC[2]+0.7071067811865475*fC[0]; 
  fC_l[1] = 1.5811388300841898*fC[17]-1.224744871391589*fC[5]+0.7071067811865475*fC[1]; 
  fC_l[2] = 1.5811388300841898*fC[18]-1.224744871391589*fC[7]+0.7071067811865475*fC[3]; 
  fC_l[3] = 1.5811388300841898*fC[19]-1.224744871391589*fC[9]+0.7071067811865475*fC[4]; 
  fC_l[4] = 1.5811388300841895*fC[20]-1.224744871391589*fC[11]+0.7071067811865475*fC[6]; 
  fC_l[5] = 1.5811388300841895*fC[21]-1.224744871391589*fC[12]+0.7071067811865475*fC[8]; 
  fC_l[6] = 1.5811388300841895*fC[22]-1.224744871391589*fC[14]+0.7071067811865475*fC[10]; 
  fC_l[7] = 1.5811388300841898*fC[23]-1.224744871391589*fC[15]+0.7071067811865475*fC[13]; 

  fUp_L[0] = 0.1767766952966368*fL_r[7]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[7]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*fL_r[6]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[6]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*fL_r[5]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[5]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*fL_r[4]*sgn_drag_coeff_Up_L[4]-0.1767766952966368*fC_l[4]*sgn_drag_coeff_Up_L[4]+0.1767766952966368*fL_r[3]*sgn_drag_coeff_Up_L[3]-0.1767766952966368*fC_l[3]*sgn_drag_coeff_Up_L[3]+0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[2]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[2]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[1]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[1]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[0]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[0]+0.5*fL_r[0]+0.5*fC_l[0]; 
  fUp_L[1] = 0.1767766952966368*fL_r[6]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[6]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[6]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[6]*fC_l[7]+0.1767766952966368*fL_r[3]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[3]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*sgn_drag_coeff_Up_L[3]*fL_r[5]-0.1767766952966368*sgn_drag_coeff_Up_L[3]*fC_l[5]+0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[4]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[4]+0.1767766952966368*sgn_drag_coeff_Up_L[2]*fL_r[4]-0.1767766952966368*sgn_drag_coeff_Up_L[2]*fC_l[4]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[1]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[1]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[1]+0.5*fL_r[1]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[1]+0.5*fC_l[1]; 
  fUp_L[2] = 0.1767766952966368*fL_r[5]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[5]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[5]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[5]*fC_l[7]+0.1767766952966368*fL_r[3]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[3]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[3]*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[3]*fC_l[6]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[4]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[4]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[4]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[4]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[2]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[2]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[2]+0.5*fL_r[2]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[2]+0.5*fC_l[2]; 
  fUp_L[3] = 0.1767766952966368*fL_r[4]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[4]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[4]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[4]*fC_l[7]+0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[2]*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[2]*fC_l[6]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[5]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[5]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[3]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[3]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[3]+0.5*fL_r[3]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[3]+0.5*fC_l[3]; 
  fUp_L[4] = 0.1767766952966368*fL_r[3]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[3]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[3]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[3]*fC_l[7]+0.1767766952966368*fL_r[5]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[5]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[5]*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[5]*fC_l[6]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[4]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[4]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[4]+0.5*fL_r[4]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[4]+0.5*fC_l[4]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[2]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[2]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[2]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[2]; 
  fUp_L[5] = 0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[2]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[2]*fC_l[7]+0.1767766952966368*fL_r[4]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[4]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[4]*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[4]*fC_l[6]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[5]+0.5*fL_r[5]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[5]+0.5*fC_l[5]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[3]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[3]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[3]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[3]; 
  fUp_L[6] = 0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[7]+0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[6]+0.5*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[6]+0.5*fC_l[6]+0.1767766952966368*fL_r[4]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[4]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*sgn_drag_coeff_Up_L[4]*fL_r[5]-0.1767766952966368*sgn_drag_coeff_Up_L[4]*fC_l[5]+0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[3]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[3]+0.1767766952966368*sgn_drag_coeff_Up_L[2]*fL_r[3]-0.1767766952966368*sgn_drag_coeff_Up_L[2]*fC_l[3]; 
  fUp_L[7] = 0.1767766952966368*fL_r[0]*sgn_drag_coeff_Up_L[7]-0.1767766952966368*fC_l[0]*sgn_drag_coeff_Up_L[7]+0.1767766952966368*sgn_drag_coeff_Up_L[0]*fL_r[7]+0.5*fL_r[7]-0.1767766952966368*sgn_drag_coeff_Up_L[0]*fC_l[7]+0.5*fC_l[7]+0.1767766952966368*fL_r[1]*sgn_drag_coeff_Up_L[6]-0.1767766952966368*fC_l[1]*sgn_drag_coeff_Up_L[6]+0.1767766952966368*sgn_drag_coeff_Up_L[1]*fL_r[6]-0.1767766952966368*sgn_drag_coeff_Up_L[1]*fC_l[6]+0.1767766952966368*fL_r[2]*sgn_drag_coeff_Up_L[5]-0.1767766952966368*fC_l[2]*sgn_drag_coeff_Up_L[5]+0.1767766952966368*sgn_drag_coeff_Up_L[2]*fL_r[5]-0.1767766952966368*sgn_drag_coeff_Up_L[2]*fC_l[5]+0.1767766952966368*fL_r[3]*sgn_drag_coeff_Up_L[4]-0.1767766952966368*fC_l[3]*sgn_drag_coeff_Up_L[4]+0.1767766952966368*sgn_drag_coeff_Up_L[3]*fL_r[4]-0.1767766952966368*sgn_drag_coeff_Up_L[3]*fC_l[4]; 

  } 
  double fUp_R[8] = {0.0}; 
  if (const_sgn_drag_coeff_R[0] == 1) { 
    if (sgn_drag_coeff_surf_R[0] == 1.0) { 
  fUp_R[0] = 1.5811388300841895*fC[16]+1.224744871391589*fC[2]+0.7071067811865475*fC[0]; 
  fUp_R[1] = 1.5811388300841898*fC[17]+1.224744871391589*fC[5]+0.7071067811865475*fC[1]; 
  fUp_R[2] = 1.5811388300841898*fC[18]+1.224744871391589*fC[7]+0.7071067811865475*fC[3]; 
  fUp_R[3] = 1.5811388300841898*fC[19]+1.224744871391589*fC[9]+0.7071067811865475*fC[4]; 
  fUp_R[4] = 1.5811388300841895*fC[20]+1.224744871391589*fC[11]+0.7071067811865475*fC[6]; 
  fUp_R[5] = 1.5811388300841895*fC[21]+1.224744871391589*fC[12]+0.7071067811865475*fC[8]; 
  fUp_R[6] = 1.5811388300841895*fC[22]+1.224744871391589*fC[14]+0.7071067811865475*fC[10]; 
  fUp_R[7] = 1.5811388300841898*fC[23]+1.224744871391589*fC[15]+0.7071067811865475*fC[13]; 
    } else { 
  fUp_R[0] = 1.5811388300841895*fR[16]-1.224744871391589*fR[2]+0.7071067811865475*fR[0]; 
  fUp_R[1] = 1.5811388300841898*fR[17]-1.224744871391589*fR[5]+0.7071067811865475*fR[1]; 
  fUp_R[2] = 1.5811388300841898*fR[18]-1.224744871391589*fR[7]+0.7071067811865475*fR[3]; 
  fUp_R[3] = 1.5811388300841898*fR[19]-1.224744871391589*fR[9]+0.7071067811865475*fR[4]; 
  fUp_R[4] = 1.5811388300841895*fR[20]-1.224744871391589*fR[11]+0.7071067811865475*fR[6]; 
  fUp_R[5] = 1.5811388300841895*fR[21]-1.224744871391589*fR[12]+0.7071067811865475*fR[8]; 
  fUp_R[6] = 1.5811388300841895*fR[22]-1.224744871391589*fR[14]+0.7071067811865475*fR[10]; 
  fUp_R[7] = 1.5811388300841898*fR[23]-1.224744871391589*fR[15]+0.7071067811865475*fR[13]; 
   } 
  } else { 
  double fC_r[8] = {0.0}; 
  double fR_l[8] = {0.0}; 
  double sgn_drag_coeff_Up_R[8] = {0.0}; 
  ser_4x_p1_upwind_quad_to_modal(sgn_drag_coeff_surf_R, sgn_drag_coeff_Up_R); 

  fC_r[0] = 1.5811388300841895*fC[16]+1.224744871391589*fC[2]+0.7071067811865475*fC[0]; 
  fC_r[1] = 1.5811388300841898*fC[17]+1.224744871391589*fC[5]+0.7071067811865475*fC[1]; 
  fC_r[2] = 1.5811388300841898*fC[18]+1.224744871391589*fC[7]+0.7071067811865475*fC[3]; 
  fC_r[3] = 1.5811388300841898*fC[19]+1.224744871391589*fC[9]+0.7071067811865475*fC[4]; 
  fC_r[4] = 1.5811388300841895*fC[20]+1.224744871391589*fC[11]+0.7071067811865475*fC[6]; 
  fC_r[5] = 1.5811388300841895*fC[21]+1.224744871391589*fC[12]+0.7071067811865475*fC[8]; 
  fC_r[6] = 1.5811388300841895*fC[22]+1.224744871391589*fC[14]+0.7071067811865475*fC[10]; 
  fC_r[7] = 1.5811388300841898*fC[23]+1.224744871391589*fC[15]+0.7071067811865475*fC[13]; 

  fR_l[0] = 1.5811388300841895*fR[16]-1.224744871391589*fR[2]+0.7071067811865475*fR[0]; 
  fR_l[1] = 1.5811388300841898*fR[17]-1.224744871391589*fR[5]+0.7071067811865475*fR[1]; 
  fR_l[2] = 1.5811388300841898*fR[18]-1.224744871391589*fR[7]+0.7071067811865475*fR[3]; 
  fR_l[3] = 1.5811388300841898*fR[19]-1.224744871391589*fR[9]+0.7071067811865475*fR[4]; 
  fR_l[4] = 1.5811388300841895*fR[20]-1.224744871391589*fR[11]+0.7071067811865475*fR[6]; 
  fR_l[5] = 1.5811388300841895*fR[21]-1.224744871391589*fR[12]+0.7071067811865475*fR[8]; 
  fR_l[6] = 1.5811388300841895*fR[22]-1.224744871391589*fR[14]+0.7071067811865475*fR[10]; 
  fR_l[7] = 1.5811388300841898*fR[23]-1.224744871391589*fR[15]+0.7071067811865475*fR[13]; 

  fUp_R[0] = -(0.1767766952966368*fR_l[7]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[7]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*fR_l[6]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[6]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*fR_l[5]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[5]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*fR_l[4]*sgn_drag_coeff_Up_R[4]+0.1767766952966368*fC_r[4]*sgn_drag_coeff_Up_R[4]-0.1767766952966368*fR_l[3]*sgn_drag_coeff_Up_R[3]+0.1767766952966368*fC_r[3]*sgn_drag_coeff_Up_R[3]-0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[2]+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[2]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[1]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[1]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[0]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[0]+0.5*fR_l[0]+0.5*fC_r[0]; 
  fUp_R[1] = -(0.1767766952966368*fR_l[6]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[6]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[6]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[6]*fC_r[7]-0.1767766952966368*fR_l[3]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[3]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*sgn_drag_coeff_Up_R[3]*fR_l[5]+0.1767766952966368*sgn_drag_coeff_Up_R[3]*fC_r[5]-0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[4]+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[4]-0.1767766952966368*sgn_drag_coeff_Up_R[2]*fR_l[4]+0.1767766952966368*sgn_drag_coeff_Up_R[2]*fC_r[4]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[1]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[1]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[1]+0.5*fR_l[1]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[1]+0.5*fC_r[1]; 
  fUp_R[2] = -(0.1767766952966368*fR_l[5]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[5]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[5]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[5]*fC_r[7]-0.1767766952966368*fR_l[3]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[3]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[3]*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[3]*fC_r[6]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[4]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[4]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[4]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[4]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[2]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[2]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[2]+0.5*fR_l[2]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[2]+0.5*fC_r[2]; 
  fUp_R[3] = -(0.1767766952966368*fR_l[4]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[4]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[4]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[4]*fC_r[7]-0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[2]*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[2]*fC_r[6]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[5]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[5]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[3]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[3]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[3]+0.5*fR_l[3]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[3]+0.5*fC_r[3]; 
  fUp_R[4] = -(0.1767766952966368*fR_l[3]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[3]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[3]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[3]*fC_r[7]-0.1767766952966368*fR_l[5]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[5]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[5]*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[5]*fC_r[6]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[4]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[4]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[4]+0.5*fR_l[4]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[4]+0.5*fC_r[4]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[2]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[2]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[2]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[2]; 
  fUp_R[5] = -(0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[2]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[2]*fC_r[7]-0.1767766952966368*fR_l[4]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[4]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[4]*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[4]*fC_r[6]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[5]+0.5*fR_l[5]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[5]+0.5*fC_r[5]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[3]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[3]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[3]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[3]; 
  fUp_R[6] = -(0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[7]-0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[6]+0.5*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[6]+0.5*fC_r[6]-0.1767766952966368*fR_l[4]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[4]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*sgn_drag_coeff_Up_R[4]*fR_l[5]+0.1767766952966368*sgn_drag_coeff_Up_R[4]*fC_r[5]-0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[3]+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[3]-0.1767766952966368*sgn_drag_coeff_Up_R[2]*fR_l[3]+0.1767766952966368*sgn_drag_coeff_Up_R[2]*fC_r[3]; 
  fUp_R[7] = -(0.1767766952966368*fR_l[0]*sgn_drag_coeff_Up_R[7])+0.1767766952966368*fC_r[0]*sgn_drag_coeff_Up_R[7]-0.1767766952966368*sgn_drag_coeff_Up_R[0]*fR_l[7]+0.5*fR_l[7]+0.1767766952966368*sgn_drag_coeff_Up_R[0]*fC_r[7]+0.5*fC_r[7]-0.1767766952966368*fR_l[1]*sgn_drag_coeff_Up_R[6]+0.1767766952966368*fC_r[1]*sgn_drag_coeff_Up_R[6]-0.1767766952966368*sgn_drag_coeff_Up_R[1]*fR_l[6]+0.1767766952966368*sgn_drag_coeff_Up_R[1]*fC_r[6]-0.1767766952966368*fR_l[2]*sgn_drag_coeff_Up_R[5]+0.1767766952966368*fC_r[2]*sgn_drag_coeff_Up_R[5]-0.1767766952966368*sgn_drag_coeff_Up_R[2]*fR_l[5]+0.1767766952966368*sgn_drag_coeff_Up_R[2]*fC_r[5]-0.1767766952966368*fR_l[3]*sgn_drag_coeff_Up_R[4]+0.1767766952966368*fC_r[3]*sgn_drag_coeff_Up_R[4]-0.1767766952966368*sgn_drag_coeff_Up_R[3]*fR_l[4]+0.1767766952966368*sgn_drag_coeff_Up_R[3]*fC_r[4]; 

  } 
  double GhatL[8] = {0.0}; 
  double GhatR[8] = {0.0}; 
  GhatL[0] = 0.3535533905932737*(drag_coeff_surf_L[7]*fUp_L[7]+drag_coeff_surf_L[6]*fUp_L[6]+drag_coeff_surf_L[5]*fUp_L[5]+drag_coeff_surf_L[4]*fUp_L[4]+drag_coeff_surf_L[3]*fUp_L[3]+drag_coeff_surf_L[2]*fUp_L[2]+drag_coeff_surf_L[1]*fUp_L[1]+drag_coeff_surf_L[0]*fUp_L[0]); 
  GhatL[1] = 0.3535533905932737*(drag_coeff_surf_L[6]*fUp_L[7]+fUp_L[6]*drag_coeff_surf_L[7]+drag_coeff_surf_L[3]*fUp_L[5]+fUp_L[3]*drag_coeff_surf_L[5]+drag_coeff_surf_L[2]*fUp_L[4]+fUp_L[2]*drag_coeff_surf_L[4]+drag_coeff_surf_L[0]*fUp_L[1]+fUp_L[0]*drag_coeff_surf_L[1]); 
  GhatL[2] = 0.3535533905932737*(drag_coeff_surf_L[5]*fUp_L[7]+fUp_L[5]*drag_coeff_surf_L[7]+drag_coeff_surf_L[3]*fUp_L[6]+fUp_L[3]*drag_coeff_surf_L[6]+drag_coeff_surf_L[1]*fUp_L[4]+fUp_L[1]*drag_coeff_surf_L[4]+drag_coeff_surf_L[0]*fUp_L[2]+fUp_L[0]*drag_coeff_surf_L[2]); 
  GhatL[3] = 0.3535533905932737*(drag_coeff_surf_L[4]*fUp_L[7]+fUp_L[4]*drag_coeff_surf_L[7]+drag_coeff_surf_L[2]*fUp_L[6]+fUp_L[2]*drag_coeff_surf_L[6]+drag_coeff_surf_L[1]*fUp_L[5]+fUp_L[1]*drag_coeff_surf_L[5]+drag_coeff_surf_L[0]*fUp_L[3]+fUp_L[0]*drag_coeff_surf_L[3]); 
  GhatL[4] = 0.3535533905932737*(drag_coeff_surf_L[3]*fUp_L[7]+fUp_L[3]*drag_coeff_surf_L[7]+drag_coeff_surf_L[5]*fUp_L[6]+fUp_L[5]*drag_coeff_surf_L[6]+drag_coeff_surf_L[0]*fUp_L[4]+fUp_L[0]*drag_coeff_surf_L[4]+drag_coeff_surf_L[1]*fUp_L[2]+fUp_L[1]*drag_coeff_surf_L[2]); 
  GhatL[5] = 0.3535533905932737*(drag_coeff_surf_L[2]*fUp_L[7]+fUp_L[2]*drag_coeff_surf_L[7]+drag_coeff_surf_L[4]*fUp_L[6]+fUp_L[4]*drag_coeff_surf_L[6]+drag_coeff_surf_L[0]*fUp_L[5]+fUp_L[0]*drag_coeff_surf_L[5]+drag_coeff_surf_L[1]*fUp_L[3]+fUp_L[1]*drag_coeff_surf_L[3]); 
  GhatL[6] = 0.3535533905932737*(drag_coeff_surf_L[1]*fUp_L[7]+fUp_L[1]*drag_coeff_surf_L[7]+drag_coeff_surf_L[0]*fUp_L[6]+fUp_L[0]*drag_coeff_surf_L[6]+drag_coeff_surf_L[4]*fUp_L[5]+fUp_L[4]*drag_coeff_surf_L[5]+drag_coeff_surf_L[2]*fUp_L[3]+fUp_L[2]*drag_coeff_surf_L[3]); 
  GhatL[7] = 0.3535533905932737*(drag_coeff_surf_L[0]*fUp_L[7]+fUp_L[0]*drag_coeff_surf_L[7]+drag_coeff_surf_L[1]*fUp_L[6]+fUp_L[1]*drag_coeff_surf_L[6]+drag_coeff_surf_L[2]*fUp_L[5]+fUp_L[2]*drag_coeff_surf_L[5]+drag_coeff_surf_L[3]*fUp_L[4]+fUp_L[3]*drag_coeff_surf_L[4]); 

  GhatR[0] = 0.3535533905932737*(drag_coeff_surf_R[7]*fUp_R[7]+drag_coeff_surf_R[6]*fUp_R[6]+drag_coeff_surf_R[5]*fUp_R[5]+drag_coeff_surf_R[4]*fUp_R[4]+drag_coeff_surf_R[3]*fUp_R[3]+drag_coeff_surf_R[2]*fUp_R[2]+drag_coeff_surf_R[1]*fUp_R[1]+drag_coeff_surf_R[0]*fUp_R[0]); 
  GhatR[1] = 0.3535533905932737*(drag_coeff_surf_R[6]*fUp_R[7]+fUp_R[6]*drag_coeff_surf_R[7]+drag_coeff_surf_R[3]*fUp_R[5]+fUp_R[3]*drag_coeff_surf_R[5]+drag_coeff_surf_R[2]*fUp_R[4]+fUp_R[2]*drag_coeff_surf_R[4]+drag_coeff_surf_R[0]*fUp_R[1]+fUp_R[0]*drag_coeff_surf_R[1]); 
  GhatR[2] = 0.3535533905932737*(drag_coeff_surf_R[5]*fUp_R[7]+fUp_R[5]*drag_coeff_surf_R[7]+drag_coeff_surf_R[3]*fUp_R[6]+fUp_R[3]*drag_coeff_surf_R[6]+drag_coeff_surf_R[1]*fUp_R[4]+fUp_R[1]*drag_coeff_surf_R[4]+drag_coeff_surf_R[0]*fUp_R[2]+fUp_R[0]*drag_coeff_surf_R[2]); 
  GhatR[3] = 0.3535533905932737*(drag_coeff_surf_R[4]*fUp_R[7]+fUp_R[4]*drag_coeff_surf_R[7]+drag_coeff_surf_R[2]*fUp_R[6]+fUp_R[2]*drag_coeff_surf_R[6]+drag_coeff_surf_R[1]*fUp_R[5]+fUp_R[1]*drag_coeff_surf_R[5]+drag_coeff_surf_R[0]*fUp_R[3]+fUp_R[0]*drag_coeff_surf_R[3]); 
  GhatR[4] = 0.3535533905932737*(drag_coeff_surf_R[3]*fUp_R[7]+fUp_R[3]*drag_coeff_surf_R[7]+drag_coeff_surf_R[5]*fUp_R[6]+fUp_R[5]*drag_coeff_surf_R[6]+drag_coeff_surf_R[0]*fUp_R[4]+fUp_R[0]*drag_coeff_surf_R[4]+drag_coeff_surf_R[1]*fUp_R[2]+fUp_R[1]*drag_coeff_surf_R[2]); 
  GhatR[5] = 0.3535533905932737*(drag_coeff_surf_R[2]*fUp_R[7]+fUp_R[2]*drag_coeff_surf_R[7]+drag_coeff_surf_R[4]*fUp_R[6]+fUp_R[4]*drag_coeff_surf_R[6]+drag_coeff_surf_R[0]*fUp_R[5]+fUp_R[0]*drag_coeff_surf_R[5]+drag_coeff_surf_R[1]*fUp_R[3]+fUp_R[1]*drag_coeff_surf_R[3]); 
  GhatR[6] = 0.3535533905932737*(drag_coeff_surf_R[1]*fUp_R[7]+fUp_R[1]*drag_coeff_surf_R[7]+drag_coeff_surf_R[0]*fUp_R[6]+fUp_R[0]*drag_coeff_surf_R[6]+drag_coeff_surf_R[4]*fUp_R[5]+fUp_R[4]*drag_coeff_surf_R[5]+drag_coeff_surf_R[2]*fUp_R[3]+fUp_R[2]*drag_coeff_surf_R[3]); 
  GhatR[7] = 0.3535533905932737*(drag_coeff_surf_R[0]*fUp_R[7]+fUp_R[0]*drag_coeff_surf_R[7]+drag_coeff_surf_R[1]*fUp_R[6]+fUp_R[1]*drag_coeff_surf_R[6]+drag_coeff_surf_R[2]*fUp_R[5]+fUp_R[2]*drag_coeff_surf_R[5]+drag_coeff_surf_R[3]*fUp_R[4]+fUp_R[3]*drag_coeff_surf_R[4]); 

  out[0] += (0.35355339059327373*GhatL[0]-0.35355339059327373*GhatR[0])*dv1; 
  out[1] += (0.35355339059327373*GhatL[1]-0.35355339059327373*GhatR[1])*dv1; 
  out[2] += (-(0.6123724356957945*GhatR[0])-0.6123724356957945*GhatL[0])*dv1; 
  out[3] += (0.35355339059327373*GhatL[2]-0.35355339059327373*GhatR[2])*dv1; 
  out[4] += (0.35355339059327373*GhatL[3]-0.35355339059327373*GhatR[3])*dv1; 
  out[5] += (-(0.6123724356957945*GhatR[1])-0.6123724356957945*GhatL[1])*dv1; 
  out[6] += (0.35355339059327373*GhatL[4]-0.35355339059327373*GhatR[4])*dv1; 
  out[7] += (-(0.6123724356957945*GhatR[2])-0.6123724356957945*GhatL[2])*dv1; 
  out[8] += (0.35355339059327373*GhatL[5]-0.35355339059327373*GhatR[5])*dv1; 
  out[9] += (-(0.6123724356957945*GhatR[3])-0.6123724356957945*GhatL[3])*dv1; 
  out[10] += (0.35355339059327373*GhatL[6]-0.35355339059327373*GhatR[6])*dv1; 
  out[11] += (-(0.6123724356957945*GhatR[4])-0.6123724356957945*GhatL[4])*dv1; 
  out[12] += (-(0.6123724356957945*GhatR[5])-0.6123724356957945*GhatL[5])*dv1; 
  out[13] += (0.35355339059327373*GhatL[7]-0.35355339059327373*GhatR[7])*dv1; 
  out[14] += (-(0.6123724356957945*GhatR[6])-0.6123724356957945*GhatL[6])*dv1; 
  out[15] += (-(0.6123724356957945*GhatR[7])-0.6123724356957945*GhatL[7])*dv1; 
  out[16] += (0.7905694150420948*GhatL[0]-0.7905694150420948*GhatR[0])*dv1; 
  out[17] += (0.7905694150420949*GhatL[1]-0.7905694150420949*GhatR[1])*dv1; 
  out[18] += (0.7905694150420949*GhatL[2]-0.7905694150420949*GhatR[2])*dv1; 
  out[19] += (0.7905694150420949*GhatL[3]-0.7905694150420949*GhatR[3])*dv1; 
  out[20] += (0.7905694150420948*GhatL[4]-0.7905694150420948*GhatR[4])*dv1; 
  out[21] += (0.7905694150420948*GhatL[5]-0.7905694150420948*GhatR[5])*dv1; 
  out[22] += (0.7905694150420948*GhatL[6]-0.7905694150420948*GhatR[6])*dv1; 
  out[23] += (0.7905694150420949*GhatL[7]-0.7905694150420949*GhatR[7])*dv1; 
  double cflFreq = fmax(fabs(drag_coeff_surf_L[0]), fabs(drag_coeff_surf_R[0])); 

  return 0.5303300858899105*dv1*cflFreq; 
}
