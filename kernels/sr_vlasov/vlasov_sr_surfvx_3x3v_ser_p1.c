#include <gkyl_vlasov_sr_kernels.h> 
#include <gkyl_basis_ser_6x_p1_surfx4_eval_quad.h> 
#include <gkyl_basis_ser_6x_p1_upwind_quad_to_modal.h> 
GKYL_CU_DH void vlasov_sr_surfvx_3x3v_ser_p1(const double *w, const double *dxv, const double *p_over_gamma, const double *qmem, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w:         Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // p_over_gamma:      p/gamma (velocity).
  // qmem:      q/m*EM fields.
  // fl/fc/fr:  Input Distribution function in left/center/right cells 
  // out:       Output distribution function in center cell 
  const double dv10 = 2/dxv[3]; 
  const double dv1 = dxv[3], wv1 = w[3]; 
  const double dv2 = dxv[4], wv2 = w[4]; 
  const double dv3 = dxv[5], wv3 = w[5]; 
  const double *E0 = &qmem[0]; 
  const double *p0_over_gamma = &p_over_gamma[0]; 
  const double *B0 = &qmem[24]; 
  const double *p1_over_gamma = &p_over_gamma[8]; 
  const double *B1 = &qmem[32]; 
  const double *p2_over_gamma = &p_over_gamma[16]; 
  const double *B2 = &qmem[40]; 

  double alpha_l[32] = {0.0}; 
  double alpha_r[32] = {0.0}; 

  alpha_l[0] = 1.224744871391589*B1[0]*p2_over_gamma[1]-1.224744871391589*B2[0]*p1_over_gamma[1]-0.7071067811865475*B1[0]*p2_over_gamma[0]+0.7071067811865475*B2[0]*p1_over_gamma[0]+2.0*E0[0]; 
  alpha_l[1] = 1.224744871391589*B1[1]*p2_over_gamma[1]-1.224744871391589*B2[1]*p1_over_gamma[1]+2.0*E0[1]+0.7071067811865475*p1_over_gamma[0]*B2[1]-0.7071067811865475*p2_over_gamma[0]*B1[1]; 
  alpha_l[2] = 2.0*E0[2]-1.224744871391589*p1_over_gamma[1]*B2[2]+0.7071067811865475*p1_over_gamma[0]*B2[2]+1.224744871391589*p2_over_gamma[1]*B1[2]-0.7071067811865475*p2_over_gamma[0]*B1[2]; 
  alpha_l[3] = 2.0*E0[3]-1.224744871391589*p1_over_gamma[1]*B2[3]+0.7071067811865475*p1_over_gamma[0]*B2[3]+1.224744871391589*p2_over_gamma[1]*B1[3]-0.7071067811865475*p2_over_gamma[0]*B1[3]; 
  alpha_l[4] = 1.224744871391589*B1[0]*p2_over_gamma[4]-1.224744871391589*B2[0]*p1_over_gamma[4]-0.7071067811865475*B1[0]*p2_over_gamma[2]+0.7071067811865475*B2[0]*p1_over_gamma[2]; 
  alpha_l[5] = 1.224744871391589*B1[0]*p2_over_gamma[5]-1.224744871391589*B2[0]*p1_over_gamma[5]-0.7071067811865475*B1[0]*p2_over_gamma[3]+0.7071067811865475*B2[0]*p1_over_gamma[3]; 
  alpha_l[6] = 2.0*E0[4]-1.224744871391589*p1_over_gamma[1]*B2[4]+0.7071067811865475*p1_over_gamma[0]*B2[4]+1.224744871391589*p2_over_gamma[1]*B1[4]-0.7071067811865475*p2_over_gamma[0]*B1[4]; 
  alpha_l[7] = 2.0*E0[5]-1.224744871391589*p1_over_gamma[1]*B2[5]+0.7071067811865475*p1_over_gamma[0]*B2[5]+1.224744871391589*p2_over_gamma[1]*B1[5]-0.7071067811865475*p2_over_gamma[0]*B1[5]; 
  alpha_l[8] = 2.0*E0[6]-1.224744871391589*p1_over_gamma[1]*B2[6]+0.7071067811865475*p1_over_gamma[0]*B2[6]+1.224744871391589*p2_over_gamma[1]*B1[6]-0.7071067811865475*p2_over_gamma[0]*B1[6]; 
  alpha_l[9] = 1.224744871391589*B1[1]*p2_over_gamma[4]-1.224744871391589*B2[1]*p1_over_gamma[4]-0.7071067811865475*B1[1]*p2_over_gamma[2]+0.7071067811865475*B2[1]*p1_over_gamma[2]; 
  alpha_l[10] = 1.224744871391589*B1[2]*p2_over_gamma[4]-1.224744871391589*B2[2]*p1_over_gamma[4]-0.7071067811865475*B1[2]*p2_over_gamma[2]+0.7071067811865475*B2[2]*p1_over_gamma[2]; 
  alpha_l[11] = 1.224744871391589*B1[3]*p2_over_gamma[4]-1.224744871391589*B2[3]*p1_over_gamma[4]+0.7071067811865475*p1_over_gamma[2]*B2[3]-0.7071067811865475*p2_over_gamma[2]*B1[3]; 
  alpha_l[12] = 1.224744871391589*B1[1]*p2_over_gamma[5]-1.224744871391589*B2[1]*p1_over_gamma[5]-0.7071067811865475*B1[1]*p2_over_gamma[3]+0.7071067811865475*B2[1]*p1_over_gamma[3]; 
  alpha_l[13] = 1.224744871391589*B1[2]*p2_over_gamma[5]-1.224744871391589*B2[2]*p1_over_gamma[5]-0.7071067811865475*B1[2]*p2_over_gamma[3]+0.7071067811865475*B2[2]*p1_over_gamma[3]; 
  alpha_l[14] = 1.224744871391589*B1[3]*p2_over_gamma[5]-1.224744871391589*B2[3]*p1_over_gamma[5]-0.7071067811865475*B1[3]*p2_over_gamma[3]+0.7071067811865475*B2[3]*p1_over_gamma[3]; 
  alpha_l[15] = 1.224744871391589*B1[0]*p2_over_gamma[7]-1.224744871391589*B2[0]*p1_over_gamma[7]-0.7071067811865475*B1[0]*p2_over_gamma[6]+0.7071067811865475*B2[0]*p1_over_gamma[6]; 
  alpha_l[16] = 2.0*E0[7]-1.224744871391589*p1_over_gamma[1]*B2[7]+0.7071067811865475*p1_over_gamma[0]*B2[7]+1.224744871391589*p2_over_gamma[1]*B1[7]-0.7071067811865475*p2_over_gamma[0]*B1[7]; 
  alpha_l[17] = 1.224744871391589*B1[4]*p2_over_gamma[4]-1.224744871391589*B2[4]*p1_over_gamma[4]+0.7071067811865475*p1_over_gamma[2]*B2[4]-0.7071067811865475*p2_over_gamma[2]*B1[4]; 
  alpha_l[18] = (-1.224744871391589*p1_over_gamma[4]*B2[5])+0.7071067811865475*p1_over_gamma[2]*B2[5]+1.224744871391589*p2_over_gamma[4]*B1[5]-0.7071067811865475*p2_over_gamma[2]*B1[5]; 
  alpha_l[19] = (-1.224744871391589*p1_over_gamma[4]*B2[6])+0.7071067811865475*p1_over_gamma[2]*B2[6]+1.224744871391589*p2_over_gamma[4]*B1[6]-0.7071067811865475*p2_over_gamma[2]*B1[6]; 
  alpha_l[20] = 1.224744871391589*B1[4]*p2_over_gamma[5]-1.224744871391589*B2[4]*p1_over_gamma[5]+0.7071067811865475*p1_over_gamma[3]*B2[4]-0.7071067811865475*p2_over_gamma[3]*B1[4]; 
  alpha_l[21] = 1.224744871391589*B1[5]*p2_over_gamma[5]-1.224744871391589*B2[5]*p1_over_gamma[5]+0.7071067811865475*p1_over_gamma[3]*B2[5]-0.7071067811865475*p2_over_gamma[3]*B1[5]; 
  alpha_l[22] = (-1.224744871391589*p1_over_gamma[5]*B2[6])+0.7071067811865475*p1_over_gamma[3]*B2[6]+1.224744871391589*p2_over_gamma[5]*B1[6]-0.7071067811865475*p2_over_gamma[3]*B1[6]; 
  alpha_l[23] = 1.224744871391589*B1[1]*p2_over_gamma[7]-1.224744871391589*B2[1]*p1_over_gamma[7]-0.7071067811865475*B1[1]*p2_over_gamma[6]+0.7071067811865475*B2[1]*p1_over_gamma[6]; 
  alpha_l[24] = 1.224744871391589*B1[2]*p2_over_gamma[7]-1.224744871391589*B2[2]*p1_over_gamma[7]-0.7071067811865475*B1[2]*p2_over_gamma[6]+0.7071067811865475*B2[2]*p1_over_gamma[6]; 
  alpha_l[25] = 1.224744871391589*B1[3]*p2_over_gamma[7]-1.224744871391589*B2[3]*p1_over_gamma[7]-0.7071067811865475*B1[3]*p2_over_gamma[6]+0.7071067811865475*B2[3]*p1_over_gamma[6]; 
  alpha_l[26] = (-1.224744871391589*p1_over_gamma[4]*B2[7])+0.7071067811865475*p1_over_gamma[2]*B2[7]+1.224744871391589*p2_over_gamma[4]*B1[7]-0.7071067811865475*p2_over_gamma[2]*B1[7]; 
  alpha_l[27] = (-1.224744871391589*p1_over_gamma[5]*B2[7])+0.7071067811865475*p1_over_gamma[3]*B2[7]+1.224744871391589*p2_over_gamma[5]*B1[7]-0.7071067811865475*p2_over_gamma[3]*B1[7]; 
  alpha_l[28] = 1.224744871391589*B1[4]*p2_over_gamma[7]-1.224744871391589*B2[4]*p1_over_gamma[7]-0.7071067811865475*B1[4]*p2_over_gamma[6]+0.7071067811865475*B2[4]*p1_over_gamma[6]; 
  alpha_l[29] = 1.224744871391589*B1[5]*p2_over_gamma[7]-1.224744871391589*B2[5]*p1_over_gamma[7]-0.7071067811865475*B1[5]*p2_over_gamma[6]+0.7071067811865475*B2[5]*p1_over_gamma[6]; 
  alpha_l[30] = 1.224744871391589*B1[6]*p2_over_gamma[7]-1.224744871391589*B2[6]*p1_over_gamma[7]-0.7071067811865475*B1[6]*p2_over_gamma[6]+0.7071067811865475*B2[6]*p1_over_gamma[6]; 
  alpha_l[31] = 1.224744871391589*B1[7]*p2_over_gamma[7]-1.224744871391589*B2[7]*p1_over_gamma[7]+0.7071067811865475*p1_over_gamma[6]*B2[7]-0.7071067811865475*p2_over_gamma[6]*B1[7]; 

  alpha_r[0] = (-1.224744871391589*B1[0]*p2_over_gamma[1])+1.224744871391589*B2[0]*p1_over_gamma[1]-0.7071067811865475*B1[0]*p2_over_gamma[0]+0.7071067811865475*B2[0]*p1_over_gamma[0]+2.0*E0[0]; 
  alpha_r[1] = (-1.224744871391589*B1[1]*p2_over_gamma[1])+1.224744871391589*B2[1]*p1_over_gamma[1]+2.0*E0[1]+0.7071067811865475*p1_over_gamma[0]*B2[1]-0.7071067811865475*p2_over_gamma[0]*B1[1]; 
  alpha_r[2] = 2.0*E0[2]+1.224744871391589*p1_over_gamma[1]*B2[2]+0.7071067811865475*p1_over_gamma[0]*B2[2]-1.224744871391589*p2_over_gamma[1]*B1[2]-0.7071067811865475*p2_over_gamma[0]*B1[2]; 
  alpha_r[3] = 2.0*E0[3]+1.224744871391589*p1_over_gamma[1]*B2[3]+0.7071067811865475*p1_over_gamma[0]*B2[3]-1.224744871391589*p2_over_gamma[1]*B1[3]-0.7071067811865475*p2_over_gamma[0]*B1[3]; 
  alpha_r[4] = (-1.224744871391589*B1[0]*p2_over_gamma[4])+1.224744871391589*B2[0]*p1_over_gamma[4]-0.7071067811865475*B1[0]*p2_over_gamma[2]+0.7071067811865475*B2[0]*p1_over_gamma[2]; 
  alpha_r[5] = (-1.224744871391589*B1[0]*p2_over_gamma[5])+1.224744871391589*B2[0]*p1_over_gamma[5]-0.7071067811865475*B1[0]*p2_over_gamma[3]+0.7071067811865475*B2[0]*p1_over_gamma[3]; 
  alpha_r[6] = 2.0*E0[4]+1.224744871391589*p1_over_gamma[1]*B2[4]+0.7071067811865475*p1_over_gamma[0]*B2[4]-1.224744871391589*p2_over_gamma[1]*B1[4]-0.7071067811865475*p2_over_gamma[0]*B1[4]; 
  alpha_r[7] = 2.0*E0[5]+1.224744871391589*p1_over_gamma[1]*B2[5]+0.7071067811865475*p1_over_gamma[0]*B2[5]-1.224744871391589*p2_over_gamma[1]*B1[5]-0.7071067811865475*p2_over_gamma[0]*B1[5]; 
  alpha_r[8] = 2.0*E0[6]+1.224744871391589*p1_over_gamma[1]*B2[6]+0.7071067811865475*p1_over_gamma[0]*B2[6]-1.224744871391589*p2_over_gamma[1]*B1[6]-0.7071067811865475*p2_over_gamma[0]*B1[6]; 
  alpha_r[9] = (-1.224744871391589*B1[1]*p2_over_gamma[4])+1.224744871391589*B2[1]*p1_over_gamma[4]-0.7071067811865475*B1[1]*p2_over_gamma[2]+0.7071067811865475*B2[1]*p1_over_gamma[2]; 
  alpha_r[10] = (-1.224744871391589*B1[2]*p2_over_gamma[4])+1.224744871391589*B2[2]*p1_over_gamma[4]-0.7071067811865475*B1[2]*p2_over_gamma[2]+0.7071067811865475*B2[2]*p1_over_gamma[2]; 
  alpha_r[11] = (-1.224744871391589*B1[3]*p2_over_gamma[4])+1.224744871391589*B2[3]*p1_over_gamma[4]+0.7071067811865475*p1_over_gamma[2]*B2[3]-0.7071067811865475*p2_over_gamma[2]*B1[3]; 
  alpha_r[12] = (-1.224744871391589*B1[1]*p2_over_gamma[5])+1.224744871391589*B2[1]*p1_over_gamma[5]-0.7071067811865475*B1[1]*p2_over_gamma[3]+0.7071067811865475*B2[1]*p1_over_gamma[3]; 
  alpha_r[13] = (-1.224744871391589*B1[2]*p2_over_gamma[5])+1.224744871391589*B2[2]*p1_over_gamma[5]-0.7071067811865475*B1[2]*p2_over_gamma[3]+0.7071067811865475*B2[2]*p1_over_gamma[3]; 
  alpha_r[14] = (-1.224744871391589*B1[3]*p2_over_gamma[5])+1.224744871391589*B2[3]*p1_over_gamma[5]-0.7071067811865475*B1[3]*p2_over_gamma[3]+0.7071067811865475*B2[3]*p1_over_gamma[3]; 
  alpha_r[15] = (-1.224744871391589*B1[0]*p2_over_gamma[7])+1.224744871391589*B2[0]*p1_over_gamma[7]-0.7071067811865475*B1[0]*p2_over_gamma[6]+0.7071067811865475*B2[0]*p1_over_gamma[6]; 
  alpha_r[16] = 2.0*E0[7]+1.224744871391589*p1_over_gamma[1]*B2[7]+0.7071067811865475*p1_over_gamma[0]*B2[7]-1.224744871391589*p2_over_gamma[1]*B1[7]-0.7071067811865475*p2_over_gamma[0]*B1[7]; 
  alpha_r[17] = (-1.224744871391589*B1[4]*p2_over_gamma[4])+1.224744871391589*B2[4]*p1_over_gamma[4]+0.7071067811865475*p1_over_gamma[2]*B2[4]-0.7071067811865475*p2_over_gamma[2]*B1[4]; 
  alpha_r[18] = 1.224744871391589*p1_over_gamma[4]*B2[5]+0.7071067811865475*p1_over_gamma[2]*B2[5]-1.224744871391589*p2_over_gamma[4]*B1[5]-0.7071067811865475*p2_over_gamma[2]*B1[5]; 
  alpha_r[19] = 1.224744871391589*p1_over_gamma[4]*B2[6]+0.7071067811865475*p1_over_gamma[2]*B2[6]-1.224744871391589*p2_over_gamma[4]*B1[6]-0.7071067811865475*p2_over_gamma[2]*B1[6]; 
  alpha_r[20] = (-1.224744871391589*B1[4]*p2_over_gamma[5])+1.224744871391589*B2[4]*p1_over_gamma[5]+0.7071067811865475*p1_over_gamma[3]*B2[4]-0.7071067811865475*p2_over_gamma[3]*B1[4]; 
  alpha_r[21] = (-1.224744871391589*B1[5]*p2_over_gamma[5])+1.224744871391589*B2[5]*p1_over_gamma[5]+0.7071067811865475*p1_over_gamma[3]*B2[5]-0.7071067811865475*p2_over_gamma[3]*B1[5]; 
  alpha_r[22] = 1.224744871391589*p1_over_gamma[5]*B2[6]+0.7071067811865475*p1_over_gamma[3]*B2[6]-1.224744871391589*p2_over_gamma[5]*B1[6]-0.7071067811865475*p2_over_gamma[3]*B1[6]; 
  alpha_r[23] = (-1.224744871391589*B1[1]*p2_over_gamma[7])+1.224744871391589*B2[1]*p1_over_gamma[7]-0.7071067811865475*B1[1]*p2_over_gamma[6]+0.7071067811865475*B2[1]*p1_over_gamma[6]; 
  alpha_r[24] = (-1.224744871391589*B1[2]*p2_over_gamma[7])+1.224744871391589*B2[2]*p1_over_gamma[7]-0.7071067811865475*B1[2]*p2_over_gamma[6]+0.7071067811865475*B2[2]*p1_over_gamma[6]; 
  alpha_r[25] = (-1.224744871391589*B1[3]*p2_over_gamma[7])+1.224744871391589*B2[3]*p1_over_gamma[7]-0.7071067811865475*B1[3]*p2_over_gamma[6]+0.7071067811865475*B2[3]*p1_over_gamma[6]; 
  alpha_r[26] = 1.224744871391589*p1_over_gamma[4]*B2[7]+0.7071067811865475*p1_over_gamma[2]*B2[7]-1.224744871391589*p2_over_gamma[4]*B1[7]-0.7071067811865475*p2_over_gamma[2]*B1[7]; 
  alpha_r[27] = 1.224744871391589*p1_over_gamma[5]*B2[7]+0.7071067811865475*p1_over_gamma[3]*B2[7]-1.224744871391589*p2_over_gamma[5]*B1[7]-0.7071067811865475*p2_over_gamma[3]*B1[7]; 
  alpha_r[28] = (-1.224744871391589*B1[4]*p2_over_gamma[7])+1.224744871391589*B2[4]*p1_over_gamma[7]-0.7071067811865475*B1[4]*p2_over_gamma[6]+0.7071067811865475*B2[4]*p1_over_gamma[6]; 
  alpha_r[29] = (-1.224744871391589*B1[5]*p2_over_gamma[7])+1.224744871391589*B2[5]*p1_over_gamma[7]-0.7071067811865475*B1[5]*p2_over_gamma[6]+0.7071067811865475*B2[5]*p1_over_gamma[6]; 
  alpha_r[30] = (-1.224744871391589*B1[6]*p2_over_gamma[7])+1.224744871391589*B2[6]*p1_over_gamma[7]-0.7071067811865475*B1[6]*p2_over_gamma[6]+0.7071067811865475*B2[6]*p1_over_gamma[6]; 
  alpha_r[31] = (-1.224744871391589*B1[7]*p2_over_gamma[7])+1.224744871391589*B2[7]*p1_over_gamma[7]+0.7071067811865475*p1_over_gamma[6]*B2[7]-0.7071067811865475*p2_over_gamma[6]*B1[7]; 

  double fUpwindQuad_l[32] = {0.0};
  double fUpwindQuad_r[32] = {0.0};
  double fUpwind_l[32] = {0.0};;
  double fUpwind_r[32] = {0.0};
  double Ghat_l[32] = {0.0}; 
  double Ghat_r[32] = {0.0}; 

  if ((-alpha_l[31])+alpha_l[30]+alpha_l[29]+alpha_l[28]+alpha_l[27]+alpha_l[26]-alpha_l[25]-alpha_l[24]-alpha_l[23]-alpha_l[22]-alpha_l[21]-alpha_l[20]-alpha_l[19]-alpha_l[18]-alpha_l[17]-alpha_l[16]+alpha_l[15]+alpha_l[14]+alpha_l[13]+alpha_l[12]+alpha_l[11]+alpha_l[10]+alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]-alpha_l[5]-alpha_l[4]-alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[0] = ser_6x_p1_surfx4_eval_quad_node_0_r(fl); 
  } else { 
    fUpwindQuad_l[0] = ser_6x_p1_surfx4_eval_quad_node_0_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]+alpha_r[29]+alpha_r[28]+alpha_r[27]+alpha_r[26]-alpha_r[25]-alpha_r[24]-alpha_r[23]-alpha_r[22]-alpha_r[21]-alpha_r[20]-alpha_r[19]-alpha_r[18]-alpha_r[17]-alpha_r[16]+alpha_r[15]+alpha_r[14]+alpha_r[13]+alpha_r[12]+alpha_r[11]+alpha_r[10]+alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]-alpha_r[5]-alpha_r[4]-alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[0] = ser_6x_p1_surfx4_eval_quad_node_0_r(fc); 
  } else { 
    fUpwindQuad_r[0] = ser_6x_p1_surfx4_eval_quad_node_0_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]-alpha_l[29]-alpha_l[28]-alpha_l[27]+alpha_l[26]+alpha_l[25]+alpha_l[24]+alpha_l[23]+alpha_l[22]+alpha_l[21]+alpha_l[20]-alpha_l[19]-alpha_l[18]-alpha_l[17]-alpha_l[16]-alpha_l[15]-alpha_l[14]-alpha_l[13]-alpha_l[12]+alpha_l[11]+alpha_l[10]+alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]+alpha_l[5]-alpha_l[4]-alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[1] = ser_6x_p1_surfx4_eval_quad_node_1_r(fl); 
  } else { 
    fUpwindQuad_l[1] = ser_6x_p1_surfx4_eval_quad_node_1_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]-alpha_r[29]-alpha_r[28]-alpha_r[27]+alpha_r[26]+alpha_r[25]+alpha_r[24]+alpha_r[23]+alpha_r[22]+alpha_r[21]+alpha_r[20]-alpha_r[19]-alpha_r[18]-alpha_r[17]-alpha_r[16]-alpha_r[15]-alpha_r[14]-alpha_r[13]-alpha_r[12]+alpha_r[11]+alpha_r[10]+alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]+alpha_r[5]-alpha_r[4]-alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[1] = ser_6x_p1_surfx4_eval_quad_node_1_r(fc); 
  } else { 
    fUpwindQuad_r[1] = ser_6x_p1_surfx4_eval_quad_node_1_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]-alpha_l[29]-alpha_l[28]+alpha_l[27]-alpha_l[26]+alpha_l[25]+alpha_l[24]+alpha_l[23]-alpha_l[22]-alpha_l[21]-alpha_l[20]+alpha_l[19]+alpha_l[18]+alpha_l[17]-alpha_l[16]-alpha_l[15]+alpha_l[14]+alpha_l[13]+alpha_l[12]-alpha_l[11]-alpha_l[10]-alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]-alpha_l[5]+alpha_l[4]-alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[2] = ser_6x_p1_surfx4_eval_quad_node_2_r(fl); 
  } else { 
    fUpwindQuad_l[2] = ser_6x_p1_surfx4_eval_quad_node_2_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]-alpha_r[29]-alpha_r[28]+alpha_r[27]-alpha_r[26]+alpha_r[25]+alpha_r[24]+alpha_r[23]-alpha_r[22]-alpha_r[21]-alpha_r[20]+alpha_r[19]+alpha_r[18]+alpha_r[17]-alpha_r[16]-alpha_r[15]+alpha_r[14]+alpha_r[13]+alpha_r[12]-alpha_r[11]-alpha_r[10]-alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]-alpha_r[5]+alpha_r[4]-alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[2] = ser_6x_p1_surfx4_eval_quad_node_2_r(fc); 
  } else { 
    fUpwindQuad_r[2] = ser_6x_p1_surfx4_eval_quad_node_2_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]+alpha_l[29]+alpha_l[28]-alpha_l[27]-alpha_l[26]-alpha_l[25]-alpha_l[24]-alpha_l[23]+alpha_l[22]+alpha_l[21]+alpha_l[20]+alpha_l[19]+alpha_l[18]+alpha_l[17]-alpha_l[16]+alpha_l[15]-alpha_l[14]-alpha_l[13]-alpha_l[12]-alpha_l[11]-alpha_l[10]-alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]+alpha_l[5]+alpha_l[4]-alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[3] = ser_6x_p1_surfx4_eval_quad_node_3_r(fl); 
  } else { 
    fUpwindQuad_l[3] = ser_6x_p1_surfx4_eval_quad_node_3_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]+alpha_r[29]+alpha_r[28]-alpha_r[27]-alpha_r[26]-alpha_r[25]-alpha_r[24]-alpha_r[23]+alpha_r[22]+alpha_r[21]+alpha_r[20]+alpha_r[19]+alpha_r[18]+alpha_r[17]-alpha_r[16]+alpha_r[15]-alpha_r[14]-alpha_r[13]-alpha_r[12]-alpha_r[11]-alpha_r[10]-alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]+alpha_r[5]+alpha_r[4]-alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[3] = ser_6x_p1_surfx4_eval_quad_node_3_r(fc); 
  } else { 
    fUpwindQuad_r[3] = ser_6x_p1_surfx4_eval_quad_node_3_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]-alpha_l[29]+alpha_l[28]-alpha_l[27]-alpha_l[26]+alpha_l[25]-alpha_l[24]-alpha_l[23]+alpha_l[22]+alpha_l[21]-alpha_l[20]+alpha_l[19]+alpha_l[18]-alpha_l[17]+alpha_l[16]+alpha_l[15]-alpha_l[14]+alpha_l[13]+alpha_l[12]-alpha_l[11]+alpha_l[10]+alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]-alpha_l[5]-alpha_l[4]+alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[4] = ser_6x_p1_surfx4_eval_quad_node_4_r(fl); 
  } else { 
    fUpwindQuad_l[4] = ser_6x_p1_surfx4_eval_quad_node_4_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]-alpha_r[29]+alpha_r[28]-alpha_r[27]-alpha_r[26]+alpha_r[25]-alpha_r[24]-alpha_r[23]+alpha_r[22]+alpha_r[21]-alpha_r[20]+alpha_r[19]+alpha_r[18]-alpha_r[17]+alpha_r[16]+alpha_r[15]-alpha_r[14]+alpha_r[13]+alpha_r[12]-alpha_r[11]+alpha_r[10]+alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]-alpha_r[5]-alpha_r[4]+alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[4] = ser_6x_p1_surfx4_eval_quad_node_4_r(fc); 
  } else { 
    fUpwindQuad_r[4] = ser_6x_p1_surfx4_eval_quad_node_4_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]+alpha_l[29]-alpha_l[28]+alpha_l[27]-alpha_l[26]-alpha_l[25]+alpha_l[24]+alpha_l[23]-alpha_l[22]-alpha_l[21]+alpha_l[20]+alpha_l[19]+alpha_l[18]-alpha_l[17]+alpha_l[16]-alpha_l[15]+alpha_l[14]-alpha_l[13]-alpha_l[12]-alpha_l[11]+alpha_l[10]+alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]+alpha_l[5]-alpha_l[4]+alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[5] = ser_6x_p1_surfx4_eval_quad_node_5_r(fl); 
  } else { 
    fUpwindQuad_l[5] = ser_6x_p1_surfx4_eval_quad_node_5_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]+alpha_r[29]-alpha_r[28]+alpha_r[27]-alpha_r[26]-alpha_r[25]+alpha_r[24]+alpha_r[23]-alpha_r[22]-alpha_r[21]+alpha_r[20]+alpha_r[19]+alpha_r[18]-alpha_r[17]+alpha_r[16]-alpha_r[15]+alpha_r[14]-alpha_r[13]-alpha_r[12]-alpha_r[11]+alpha_r[10]+alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]+alpha_r[5]-alpha_r[4]+alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[5] = ser_6x_p1_surfx4_eval_quad_node_5_r(fc); 
  } else { 
    fUpwindQuad_r[5] = ser_6x_p1_surfx4_eval_quad_node_5_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]+alpha_l[29]-alpha_l[28]-alpha_l[27]+alpha_l[26]-alpha_l[25]+alpha_l[24]+alpha_l[23]+alpha_l[22]+alpha_l[21]-alpha_l[20]-alpha_l[19]-alpha_l[18]+alpha_l[17]+alpha_l[16]-alpha_l[15]-alpha_l[14]+alpha_l[13]+alpha_l[12]+alpha_l[11]-alpha_l[10]-alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]-alpha_l[5]+alpha_l[4]+alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[6] = ser_6x_p1_surfx4_eval_quad_node_6_r(fl); 
  } else { 
    fUpwindQuad_l[6] = ser_6x_p1_surfx4_eval_quad_node_6_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]+alpha_r[29]-alpha_r[28]-alpha_r[27]+alpha_r[26]-alpha_r[25]+alpha_r[24]+alpha_r[23]+alpha_r[22]+alpha_r[21]-alpha_r[20]-alpha_r[19]-alpha_r[18]+alpha_r[17]+alpha_r[16]-alpha_r[15]-alpha_r[14]+alpha_r[13]+alpha_r[12]+alpha_r[11]-alpha_r[10]-alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]-alpha_r[5]+alpha_r[4]+alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[6] = ser_6x_p1_surfx4_eval_quad_node_6_r(fc); 
  } else { 
    fUpwindQuad_r[6] = ser_6x_p1_surfx4_eval_quad_node_6_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]-alpha_l[29]+alpha_l[28]+alpha_l[27]+alpha_l[26]+alpha_l[25]-alpha_l[24]-alpha_l[23]-alpha_l[22]-alpha_l[21]+alpha_l[20]-alpha_l[19]-alpha_l[18]+alpha_l[17]+alpha_l[16]+alpha_l[15]+alpha_l[14]-alpha_l[13]-alpha_l[12]+alpha_l[11]-alpha_l[10]-alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]+alpha_l[5]+alpha_l[4]+alpha_l[3]-alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[7] = ser_6x_p1_surfx4_eval_quad_node_7_r(fl); 
  } else { 
    fUpwindQuad_l[7] = ser_6x_p1_surfx4_eval_quad_node_7_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]-alpha_r[29]+alpha_r[28]+alpha_r[27]+alpha_r[26]+alpha_r[25]-alpha_r[24]-alpha_r[23]-alpha_r[22]-alpha_r[21]+alpha_r[20]-alpha_r[19]-alpha_r[18]+alpha_r[17]+alpha_r[16]+alpha_r[15]+alpha_r[14]-alpha_r[13]-alpha_r[12]+alpha_r[11]-alpha_r[10]-alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]+alpha_r[5]+alpha_r[4]+alpha_r[3]-alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[7] = ser_6x_p1_surfx4_eval_quad_node_7_r(fc); 
  } else { 
    fUpwindQuad_r[7] = ser_6x_p1_surfx4_eval_quad_node_7_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]+alpha_l[29]-alpha_l[28]-alpha_l[27]-alpha_l[26]-alpha_l[25]+alpha_l[24]-alpha_l[23]+alpha_l[22]-alpha_l[21]+alpha_l[20]+alpha_l[19]-alpha_l[18]+alpha_l[17]+alpha_l[16]+alpha_l[15]+alpha_l[14]-alpha_l[13]+alpha_l[12]+alpha_l[11]-alpha_l[10]+alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]-alpha_l[5]-alpha_l[4]-alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[8] = ser_6x_p1_surfx4_eval_quad_node_8_r(fl); 
  } else { 
    fUpwindQuad_l[8] = ser_6x_p1_surfx4_eval_quad_node_8_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]+alpha_r[29]-alpha_r[28]-alpha_r[27]-alpha_r[26]-alpha_r[25]+alpha_r[24]-alpha_r[23]+alpha_r[22]-alpha_r[21]+alpha_r[20]+alpha_r[19]-alpha_r[18]+alpha_r[17]+alpha_r[16]+alpha_r[15]+alpha_r[14]-alpha_r[13]+alpha_r[12]+alpha_r[11]-alpha_r[10]+alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]-alpha_r[5]-alpha_r[4]-alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[8] = ser_6x_p1_surfx4_eval_quad_node_8_r(fc); 
  } else { 
    fUpwindQuad_r[8] = ser_6x_p1_surfx4_eval_quad_node_8_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]-alpha_l[29]+alpha_l[28]+alpha_l[27]-alpha_l[26]+alpha_l[25]-alpha_l[24]+alpha_l[23]-alpha_l[22]+alpha_l[21]-alpha_l[20]+alpha_l[19]-alpha_l[18]+alpha_l[17]+alpha_l[16]-alpha_l[15]-alpha_l[14]+alpha_l[13]-alpha_l[12]+alpha_l[11]-alpha_l[10]+alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]+alpha_l[5]-alpha_l[4]-alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[9] = ser_6x_p1_surfx4_eval_quad_node_9_r(fl); 
  } else { 
    fUpwindQuad_l[9] = ser_6x_p1_surfx4_eval_quad_node_9_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]-alpha_r[29]+alpha_r[28]+alpha_r[27]-alpha_r[26]+alpha_r[25]-alpha_r[24]+alpha_r[23]-alpha_r[22]+alpha_r[21]-alpha_r[20]+alpha_r[19]-alpha_r[18]+alpha_r[17]+alpha_r[16]-alpha_r[15]-alpha_r[14]+alpha_r[13]-alpha_r[12]+alpha_r[11]-alpha_r[10]+alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]+alpha_r[5]-alpha_r[4]-alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[9] = ser_6x_p1_surfx4_eval_quad_node_9_r(fc); 
  } else { 
    fUpwindQuad_r[9] = ser_6x_p1_surfx4_eval_quad_node_9_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]-alpha_l[29]+alpha_l[28]-alpha_l[27]+alpha_l[26]+alpha_l[25]-alpha_l[24]+alpha_l[23]+alpha_l[22]-alpha_l[21]+alpha_l[20]-alpha_l[19]+alpha_l[18]-alpha_l[17]+alpha_l[16]-alpha_l[15]+alpha_l[14]-alpha_l[13]+alpha_l[12]-alpha_l[11]+alpha_l[10]-alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]-alpha_l[5]+alpha_l[4]-alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[10] = ser_6x_p1_surfx4_eval_quad_node_10_r(fl); 
  } else { 
    fUpwindQuad_l[10] = ser_6x_p1_surfx4_eval_quad_node_10_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]-alpha_r[29]+alpha_r[28]-alpha_r[27]+alpha_r[26]+alpha_r[25]-alpha_r[24]+alpha_r[23]+alpha_r[22]-alpha_r[21]+alpha_r[20]-alpha_r[19]+alpha_r[18]-alpha_r[17]+alpha_r[16]-alpha_r[15]+alpha_r[14]-alpha_r[13]+alpha_r[12]-alpha_r[11]+alpha_r[10]-alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]-alpha_r[5]+alpha_r[4]-alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[10] = ser_6x_p1_surfx4_eval_quad_node_10_r(fc); 
  } else { 
    fUpwindQuad_r[10] = ser_6x_p1_surfx4_eval_quad_node_10_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]+alpha_l[29]-alpha_l[28]+alpha_l[27]+alpha_l[26]-alpha_l[25]+alpha_l[24]-alpha_l[23]-alpha_l[22]+alpha_l[21]-alpha_l[20]-alpha_l[19]+alpha_l[18]-alpha_l[17]+alpha_l[16]+alpha_l[15]-alpha_l[14]+alpha_l[13]-alpha_l[12]-alpha_l[11]+alpha_l[10]-alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]+alpha_l[5]+alpha_l[4]-alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[11] = ser_6x_p1_surfx4_eval_quad_node_11_r(fl); 
  } else { 
    fUpwindQuad_l[11] = ser_6x_p1_surfx4_eval_quad_node_11_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]+alpha_r[29]-alpha_r[28]+alpha_r[27]+alpha_r[26]-alpha_r[25]+alpha_r[24]-alpha_r[23]-alpha_r[22]+alpha_r[21]-alpha_r[20]-alpha_r[19]+alpha_r[18]-alpha_r[17]+alpha_r[16]+alpha_r[15]-alpha_r[14]+alpha_r[13]-alpha_r[12]-alpha_r[11]+alpha_r[10]-alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]+alpha_r[5]+alpha_r[4]-alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[11] = ser_6x_p1_surfx4_eval_quad_node_11_r(fc); 
  } else { 
    fUpwindQuad_r[11] = ser_6x_p1_surfx4_eval_quad_node_11_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]-alpha_l[29]-alpha_l[28]+alpha_l[27]+alpha_l[26]+alpha_l[25]+alpha_l[24]-alpha_l[23]-alpha_l[22]+alpha_l[21]+alpha_l[20]-alpha_l[19]+alpha_l[18]+alpha_l[17]-alpha_l[16]+alpha_l[15]-alpha_l[14]-alpha_l[13]+alpha_l[12]-alpha_l[11]-alpha_l[10]+alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]-alpha_l[5]-alpha_l[4]+alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[12] = ser_6x_p1_surfx4_eval_quad_node_12_r(fl); 
  } else { 
    fUpwindQuad_l[12] = ser_6x_p1_surfx4_eval_quad_node_12_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]-alpha_r[29]-alpha_r[28]+alpha_r[27]+alpha_r[26]+alpha_r[25]+alpha_r[24]-alpha_r[23]-alpha_r[22]+alpha_r[21]+alpha_r[20]-alpha_r[19]+alpha_r[18]+alpha_r[17]-alpha_r[16]+alpha_r[15]-alpha_r[14]-alpha_r[13]+alpha_r[12]-alpha_r[11]-alpha_r[10]+alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]-alpha_r[5]-alpha_r[4]+alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[12] = ser_6x_p1_surfx4_eval_quad_node_12_r(fc); 
  } else { 
    fUpwindQuad_r[12] = ser_6x_p1_surfx4_eval_quad_node_12_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]+alpha_l[29]+alpha_l[28]-alpha_l[27]+alpha_l[26]-alpha_l[25]-alpha_l[24]+alpha_l[23]+alpha_l[22]-alpha_l[21]-alpha_l[20]-alpha_l[19]+alpha_l[18]+alpha_l[17]-alpha_l[16]-alpha_l[15]+alpha_l[14]+alpha_l[13]-alpha_l[12]-alpha_l[11]-alpha_l[10]+alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]+alpha_l[5]-alpha_l[4]+alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[13] = ser_6x_p1_surfx4_eval_quad_node_13_r(fl); 
  } else { 
    fUpwindQuad_l[13] = ser_6x_p1_surfx4_eval_quad_node_13_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]+alpha_r[29]+alpha_r[28]-alpha_r[27]+alpha_r[26]-alpha_r[25]-alpha_r[24]+alpha_r[23]+alpha_r[22]-alpha_r[21]-alpha_r[20]-alpha_r[19]+alpha_r[18]+alpha_r[17]-alpha_r[16]-alpha_r[15]+alpha_r[14]+alpha_r[13]-alpha_r[12]-alpha_r[11]-alpha_r[10]+alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]+alpha_r[5]-alpha_r[4]+alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[13] = ser_6x_p1_surfx4_eval_quad_node_13_r(fc); 
  } else { 
    fUpwindQuad_r[13] = ser_6x_p1_surfx4_eval_quad_node_13_l(fr); 
  } 
  if (alpha_l[31]-alpha_l[30]+alpha_l[29]+alpha_l[28]+alpha_l[27]-alpha_l[26]-alpha_l[25]-alpha_l[24]+alpha_l[23]-alpha_l[22]+alpha_l[21]+alpha_l[20]+alpha_l[19]-alpha_l[18]-alpha_l[17]-alpha_l[16]-alpha_l[15]-alpha_l[14]-alpha_l[13]+alpha_l[12]+alpha_l[11]+alpha_l[10]-alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]-alpha_l[5]+alpha_l[4]+alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[14] = ser_6x_p1_surfx4_eval_quad_node_14_r(fl); 
  } else { 
    fUpwindQuad_l[14] = ser_6x_p1_surfx4_eval_quad_node_14_l(fc); 
  } 
  if (alpha_r[31]-alpha_r[30]+alpha_r[29]+alpha_r[28]+alpha_r[27]-alpha_r[26]-alpha_r[25]-alpha_r[24]+alpha_r[23]-alpha_r[22]+alpha_r[21]+alpha_r[20]+alpha_r[19]-alpha_r[18]-alpha_r[17]-alpha_r[16]-alpha_r[15]-alpha_r[14]-alpha_r[13]+alpha_r[12]+alpha_r[11]+alpha_r[10]-alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]-alpha_r[5]+alpha_r[4]+alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[14] = ser_6x_p1_surfx4_eval_quad_node_14_r(fc); 
  } else { 
    fUpwindQuad_r[14] = ser_6x_p1_surfx4_eval_quad_node_14_l(fr); 
  } 
  if ((-alpha_l[31])+alpha_l[30]-alpha_l[29]-alpha_l[28]-alpha_l[27]-alpha_l[26]+alpha_l[25]+alpha_l[24]-alpha_l[23]+alpha_l[22]-alpha_l[21]-alpha_l[20]+alpha_l[19]-alpha_l[18]-alpha_l[17]-alpha_l[16]+alpha_l[15]+alpha_l[14]+alpha_l[13]-alpha_l[12]+alpha_l[11]+alpha_l[10]-alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]+alpha_l[5]+alpha_l[4]+alpha_l[3]+alpha_l[2]-alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[15] = ser_6x_p1_surfx4_eval_quad_node_15_r(fl); 
  } else { 
    fUpwindQuad_l[15] = ser_6x_p1_surfx4_eval_quad_node_15_l(fc); 
  } 
  if ((-alpha_r[31])+alpha_r[30]-alpha_r[29]-alpha_r[28]-alpha_r[27]-alpha_r[26]+alpha_r[25]+alpha_r[24]-alpha_r[23]+alpha_r[22]-alpha_r[21]-alpha_r[20]+alpha_r[19]-alpha_r[18]-alpha_r[17]-alpha_r[16]+alpha_r[15]+alpha_r[14]+alpha_r[13]-alpha_r[12]+alpha_r[11]+alpha_r[10]-alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]+alpha_r[5]+alpha_r[4]+alpha_r[3]+alpha_r[2]-alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[15] = ser_6x_p1_surfx4_eval_quad_node_15_r(fc); 
  } else { 
    fUpwindQuad_r[15] = ser_6x_p1_surfx4_eval_quad_node_15_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]-alpha_l[29]-alpha_l[28]-alpha_l[27]-alpha_l[26]-alpha_l[25]-alpha_l[24]+alpha_l[23]-alpha_l[22]+alpha_l[21]+alpha_l[20]-alpha_l[19]+alpha_l[18]+alpha_l[17]+alpha_l[16]+alpha_l[15]+alpha_l[14]+alpha_l[13]-alpha_l[12]+alpha_l[11]+alpha_l[10]-alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]-alpha_l[5]-alpha_l[4]-alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[16] = ser_6x_p1_surfx4_eval_quad_node_16_r(fl); 
  } else { 
    fUpwindQuad_l[16] = ser_6x_p1_surfx4_eval_quad_node_16_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]-alpha_r[29]-alpha_r[28]-alpha_r[27]-alpha_r[26]-alpha_r[25]-alpha_r[24]+alpha_r[23]-alpha_r[22]+alpha_r[21]+alpha_r[20]-alpha_r[19]+alpha_r[18]+alpha_r[17]+alpha_r[16]+alpha_r[15]+alpha_r[14]+alpha_r[13]-alpha_r[12]+alpha_r[11]+alpha_r[10]-alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]-alpha_r[5]-alpha_r[4]-alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[16] = ser_6x_p1_surfx4_eval_quad_node_16_r(fc); 
  } else { 
    fUpwindQuad_r[16] = ser_6x_p1_surfx4_eval_quad_node_16_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]+alpha_l[29]+alpha_l[28]+alpha_l[27]-alpha_l[26]+alpha_l[25]+alpha_l[24]-alpha_l[23]+alpha_l[22]-alpha_l[21]-alpha_l[20]-alpha_l[19]+alpha_l[18]+alpha_l[17]+alpha_l[16]-alpha_l[15]-alpha_l[14]-alpha_l[13]+alpha_l[12]+alpha_l[11]+alpha_l[10]-alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]+alpha_l[5]-alpha_l[4]-alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[17] = ser_6x_p1_surfx4_eval_quad_node_17_r(fl); 
  } else { 
    fUpwindQuad_l[17] = ser_6x_p1_surfx4_eval_quad_node_17_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]+alpha_r[29]+alpha_r[28]+alpha_r[27]-alpha_r[26]+alpha_r[25]+alpha_r[24]-alpha_r[23]+alpha_r[22]-alpha_r[21]-alpha_r[20]-alpha_r[19]+alpha_r[18]+alpha_r[17]+alpha_r[16]-alpha_r[15]-alpha_r[14]-alpha_r[13]+alpha_r[12]+alpha_r[11]+alpha_r[10]-alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]+alpha_r[5]-alpha_r[4]-alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[17] = ser_6x_p1_surfx4_eval_quad_node_17_r(fc); 
  } else { 
    fUpwindQuad_r[17] = ser_6x_p1_surfx4_eval_quad_node_17_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]+alpha_l[29]+alpha_l[28]-alpha_l[27]+alpha_l[26]+alpha_l[25]+alpha_l[24]-alpha_l[23]-alpha_l[22]+alpha_l[21]+alpha_l[20]+alpha_l[19]-alpha_l[18]-alpha_l[17]+alpha_l[16]-alpha_l[15]+alpha_l[14]+alpha_l[13]-alpha_l[12]-alpha_l[11]-alpha_l[10]+alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]-alpha_l[5]+alpha_l[4]-alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[18] = ser_6x_p1_surfx4_eval_quad_node_18_r(fl); 
  } else { 
    fUpwindQuad_l[18] = ser_6x_p1_surfx4_eval_quad_node_18_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]+alpha_r[29]+alpha_r[28]-alpha_r[27]+alpha_r[26]+alpha_r[25]+alpha_r[24]-alpha_r[23]-alpha_r[22]+alpha_r[21]+alpha_r[20]+alpha_r[19]-alpha_r[18]-alpha_r[17]+alpha_r[16]-alpha_r[15]+alpha_r[14]+alpha_r[13]-alpha_r[12]-alpha_r[11]-alpha_r[10]+alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]-alpha_r[5]+alpha_r[4]-alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[18] = ser_6x_p1_surfx4_eval_quad_node_18_r(fc); 
  } else { 
    fUpwindQuad_r[18] = ser_6x_p1_surfx4_eval_quad_node_18_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]-alpha_l[29]-alpha_l[28]+alpha_l[27]+alpha_l[26]-alpha_l[25]-alpha_l[24]+alpha_l[23]+alpha_l[22]-alpha_l[21]-alpha_l[20]+alpha_l[19]-alpha_l[18]-alpha_l[17]+alpha_l[16]+alpha_l[15]-alpha_l[14]-alpha_l[13]+alpha_l[12]-alpha_l[11]-alpha_l[10]+alpha_l[9]+alpha_l[8]-alpha_l[7]-alpha_l[6]+alpha_l[5]+alpha_l[4]-alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[19] = ser_6x_p1_surfx4_eval_quad_node_19_r(fl); 
  } else { 
    fUpwindQuad_l[19] = ser_6x_p1_surfx4_eval_quad_node_19_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]-alpha_r[29]-alpha_r[28]+alpha_r[27]+alpha_r[26]-alpha_r[25]-alpha_r[24]+alpha_r[23]+alpha_r[22]-alpha_r[21]-alpha_r[20]+alpha_r[19]-alpha_r[18]-alpha_r[17]+alpha_r[16]+alpha_r[15]-alpha_r[14]-alpha_r[13]+alpha_r[12]-alpha_r[11]-alpha_r[10]+alpha_r[9]+alpha_r[8]-alpha_r[7]-alpha_r[6]+alpha_r[5]+alpha_r[4]-alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[19] = ser_6x_p1_surfx4_eval_quad_node_19_r(fc); 
  } else { 
    fUpwindQuad_r[19] = ser_6x_p1_surfx4_eval_quad_node_19_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]+alpha_l[29]-alpha_l[28]+alpha_l[27]+alpha_l[26]+alpha_l[25]-alpha_l[24]+alpha_l[23]+alpha_l[22]-alpha_l[21]+alpha_l[20]+alpha_l[19]-alpha_l[18]+alpha_l[17]-alpha_l[16]+alpha_l[15]-alpha_l[14]+alpha_l[13]-alpha_l[12]-alpha_l[11]+alpha_l[10]-alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]-alpha_l[5]-alpha_l[4]+alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[20] = ser_6x_p1_surfx4_eval_quad_node_20_r(fl); 
  } else { 
    fUpwindQuad_l[20] = ser_6x_p1_surfx4_eval_quad_node_20_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]+alpha_r[29]-alpha_r[28]+alpha_r[27]+alpha_r[26]+alpha_r[25]-alpha_r[24]+alpha_r[23]+alpha_r[22]-alpha_r[21]+alpha_r[20]+alpha_r[19]-alpha_r[18]+alpha_r[17]-alpha_r[16]+alpha_r[15]-alpha_r[14]+alpha_r[13]-alpha_r[12]-alpha_r[11]+alpha_r[10]-alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]-alpha_r[5]-alpha_r[4]+alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[20] = ser_6x_p1_surfx4_eval_quad_node_20_r(fc); 
  } else { 
    fUpwindQuad_r[20] = ser_6x_p1_surfx4_eval_quad_node_20_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]-alpha_l[29]+alpha_l[28]-alpha_l[27]+alpha_l[26]-alpha_l[25]+alpha_l[24]-alpha_l[23]-alpha_l[22]+alpha_l[21]-alpha_l[20]+alpha_l[19]-alpha_l[18]+alpha_l[17]-alpha_l[16]-alpha_l[15]+alpha_l[14]-alpha_l[13]+alpha_l[12]-alpha_l[11]+alpha_l[10]-alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]+alpha_l[5]-alpha_l[4]+alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[21] = ser_6x_p1_surfx4_eval_quad_node_21_r(fl); 
  } else { 
    fUpwindQuad_l[21] = ser_6x_p1_surfx4_eval_quad_node_21_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]-alpha_r[29]+alpha_r[28]-alpha_r[27]+alpha_r[26]-alpha_r[25]+alpha_r[24]-alpha_r[23]-alpha_r[22]+alpha_r[21]-alpha_r[20]+alpha_r[19]-alpha_r[18]+alpha_r[17]-alpha_r[16]-alpha_r[15]+alpha_r[14]-alpha_r[13]+alpha_r[12]-alpha_r[11]+alpha_r[10]-alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]+alpha_r[5]-alpha_r[4]+alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[21] = ser_6x_p1_surfx4_eval_quad_node_21_r(fc); 
  } else { 
    fUpwindQuad_r[21] = ser_6x_p1_surfx4_eval_quad_node_21_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]-alpha_l[29]+alpha_l[28]+alpha_l[27]-alpha_l[26]-alpha_l[25]+alpha_l[24]-alpha_l[23]+alpha_l[22]-alpha_l[21]+alpha_l[20]-alpha_l[19]+alpha_l[18]-alpha_l[17]-alpha_l[16]-alpha_l[15]-alpha_l[14]+alpha_l[13]-alpha_l[12]+alpha_l[11]-alpha_l[10]+alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]-alpha_l[5]+alpha_l[4]+alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[22] = ser_6x_p1_surfx4_eval_quad_node_22_r(fl); 
  } else { 
    fUpwindQuad_l[22] = ser_6x_p1_surfx4_eval_quad_node_22_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]-alpha_r[29]+alpha_r[28]+alpha_r[27]-alpha_r[26]-alpha_r[25]+alpha_r[24]-alpha_r[23]+alpha_r[22]-alpha_r[21]+alpha_r[20]-alpha_r[19]+alpha_r[18]-alpha_r[17]-alpha_r[16]-alpha_r[15]-alpha_r[14]+alpha_r[13]-alpha_r[12]+alpha_r[11]-alpha_r[10]+alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]-alpha_r[5]+alpha_r[4]+alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[22] = ser_6x_p1_surfx4_eval_quad_node_22_r(fc); 
  } else { 
    fUpwindQuad_r[22] = ser_6x_p1_surfx4_eval_quad_node_22_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]+alpha_l[29]-alpha_l[28]-alpha_l[27]-alpha_l[26]+alpha_l[25]-alpha_l[24]+alpha_l[23]-alpha_l[22]+alpha_l[21]-alpha_l[20]-alpha_l[19]+alpha_l[18]-alpha_l[17]-alpha_l[16]+alpha_l[15]+alpha_l[14]-alpha_l[13]+alpha_l[12]+alpha_l[11]-alpha_l[10]+alpha_l[9]-alpha_l[8]+alpha_l[7]-alpha_l[6]+alpha_l[5]+alpha_l[4]+alpha_l[3]-alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[23] = ser_6x_p1_surfx4_eval_quad_node_23_r(fl); 
  } else { 
    fUpwindQuad_l[23] = ser_6x_p1_surfx4_eval_quad_node_23_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]+alpha_r[29]-alpha_r[28]-alpha_r[27]-alpha_r[26]+alpha_r[25]-alpha_r[24]+alpha_r[23]-alpha_r[22]+alpha_r[21]-alpha_r[20]-alpha_r[19]+alpha_r[18]-alpha_r[17]-alpha_r[16]+alpha_r[15]+alpha_r[14]-alpha_r[13]+alpha_r[12]+alpha_r[11]-alpha_r[10]+alpha_r[9]-alpha_r[8]+alpha_r[7]-alpha_r[6]+alpha_r[5]+alpha_r[4]+alpha_r[3]-alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[23] = ser_6x_p1_surfx4_eval_quad_node_23_r(fc); 
  } else { 
    fUpwindQuad_r[23] = ser_6x_p1_surfx4_eval_quad_node_23_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]-alpha_l[29]+alpha_l[28]+alpha_l[27]+alpha_l[26]-alpha_l[25]+alpha_l[24]+alpha_l[23]+alpha_l[22]+alpha_l[21]-alpha_l[20]+alpha_l[19]+alpha_l[18]-alpha_l[17]-alpha_l[16]+alpha_l[15]+alpha_l[14]-alpha_l[13]-alpha_l[12]+alpha_l[11]-alpha_l[10]-alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]-alpha_l[5]-alpha_l[4]-alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[24] = ser_6x_p1_surfx4_eval_quad_node_24_r(fl); 
  } else { 
    fUpwindQuad_l[24] = ser_6x_p1_surfx4_eval_quad_node_24_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]-alpha_r[29]+alpha_r[28]+alpha_r[27]+alpha_r[26]-alpha_r[25]+alpha_r[24]+alpha_r[23]+alpha_r[22]+alpha_r[21]-alpha_r[20]+alpha_r[19]+alpha_r[18]-alpha_r[17]-alpha_r[16]+alpha_r[15]+alpha_r[14]-alpha_r[13]-alpha_r[12]+alpha_r[11]-alpha_r[10]-alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]-alpha_r[5]-alpha_r[4]-alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[24] = ser_6x_p1_surfx4_eval_quad_node_24_r(fc); 
  } else { 
    fUpwindQuad_r[24] = ser_6x_p1_surfx4_eval_quad_node_24_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]+alpha_l[29]-alpha_l[28]-alpha_l[27]+alpha_l[26]+alpha_l[25]-alpha_l[24]-alpha_l[23]-alpha_l[22]-alpha_l[21]+alpha_l[20]+alpha_l[19]+alpha_l[18]-alpha_l[17]-alpha_l[16]-alpha_l[15]-alpha_l[14]+alpha_l[13]+alpha_l[12]+alpha_l[11]-alpha_l[10]-alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]+alpha_l[5]-alpha_l[4]-alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[25] = ser_6x_p1_surfx4_eval_quad_node_25_r(fl); 
  } else { 
    fUpwindQuad_l[25] = ser_6x_p1_surfx4_eval_quad_node_25_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]+alpha_r[29]-alpha_r[28]-alpha_r[27]+alpha_r[26]+alpha_r[25]-alpha_r[24]-alpha_r[23]-alpha_r[22]-alpha_r[21]+alpha_r[20]+alpha_r[19]+alpha_r[18]-alpha_r[17]-alpha_r[16]-alpha_r[15]-alpha_r[14]+alpha_r[13]+alpha_r[12]+alpha_r[11]-alpha_r[10]-alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]+alpha_r[5]-alpha_r[4]-alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[25] = ser_6x_p1_surfx4_eval_quad_node_25_r(fc); 
  } else { 
    fUpwindQuad_r[25] = ser_6x_p1_surfx4_eval_quad_node_25_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]+alpha_l[29]-alpha_l[28]+alpha_l[27]-alpha_l[26]+alpha_l[25]-alpha_l[24]-alpha_l[23]+alpha_l[22]+alpha_l[21]-alpha_l[20]-alpha_l[19]-alpha_l[18]+alpha_l[17]-alpha_l[16]-alpha_l[15]+alpha_l[14]-alpha_l[13]-alpha_l[12]-alpha_l[11]+alpha_l[10]+alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]-alpha_l[5]+alpha_l[4]-alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[26] = ser_6x_p1_surfx4_eval_quad_node_26_r(fl); 
  } else { 
    fUpwindQuad_l[26] = ser_6x_p1_surfx4_eval_quad_node_26_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]+alpha_r[29]-alpha_r[28]+alpha_r[27]-alpha_r[26]+alpha_r[25]-alpha_r[24]-alpha_r[23]+alpha_r[22]+alpha_r[21]-alpha_r[20]-alpha_r[19]-alpha_r[18]+alpha_r[17]-alpha_r[16]-alpha_r[15]+alpha_r[14]-alpha_r[13]-alpha_r[12]-alpha_r[11]+alpha_r[10]+alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]-alpha_r[5]+alpha_r[4]-alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[26] = ser_6x_p1_surfx4_eval_quad_node_26_r(fc); 
  } else { 
    fUpwindQuad_r[26] = ser_6x_p1_surfx4_eval_quad_node_26_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]-alpha_l[29]+alpha_l[28]-alpha_l[27]-alpha_l[26]-alpha_l[25]+alpha_l[24]+alpha_l[23]-alpha_l[22]-alpha_l[21]+alpha_l[20]-alpha_l[19]-alpha_l[18]+alpha_l[17]-alpha_l[16]+alpha_l[15]-alpha_l[14]+alpha_l[13]+alpha_l[12]-alpha_l[11]+alpha_l[10]+alpha_l[9]-alpha_l[8]-alpha_l[7]+alpha_l[6]+alpha_l[5]+alpha_l[4]-alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[27] = ser_6x_p1_surfx4_eval_quad_node_27_r(fl); 
  } else { 
    fUpwindQuad_l[27] = ser_6x_p1_surfx4_eval_quad_node_27_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]-alpha_r[29]+alpha_r[28]-alpha_r[27]-alpha_r[26]-alpha_r[25]+alpha_r[24]+alpha_r[23]-alpha_r[22]-alpha_r[21]+alpha_r[20]-alpha_r[19]-alpha_r[18]+alpha_r[17]-alpha_r[16]+alpha_r[15]-alpha_r[14]+alpha_r[13]+alpha_r[12]-alpha_r[11]+alpha_r[10]+alpha_r[9]-alpha_r[8]-alpha_r[7]+alpha_r[6]+alpha_r[5]+alpha_r[4]-alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[27] = ser_6x_p1_surfx4_eval_quad_node_27_r(fc); 
  } else { 
    fUpwindQuad_r[27] = ser_6x_p1_surfx4_eval_quad_node_27_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]+alpha_l[29]+alpha_l[28]-alpha_l[27]-alpha_l[26]+alpha_l[25]+alpha_l[24]+alpha_l[23]-alpha_l[22]-alpha_l[21]-alpha_l[20]-alpha_l[19]-alpha_l[18]-alpha_l[17]+alpha_l[16]+alpha_l[15]-alpha_l[14]-alpha_l[13]-alpha_l[12]-alpha_l[11]-alpha_l[10]-alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]-alpha_l[5]-alpha_l[4]+alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[28] = ser_6x_p1_surfx4_eval_quad_node_28_r(fl); 
  } else { 
    fUpwindQuad_l[28] = ser_6x_p1_surfx4_eval_quad_node_28_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]+alpha_r[29]+alpha_r[28]-alpha_r[27]-alpha_r[26]+alpha_r[25]+alpha_r[24]+alpha_r[23]-alpha_r[22]-alpha_r[21]-alpha_r[20]-alpha_r[19]-alpha_r[18]-alpha_r[17]+alpha_r[16]+alpha_r[15]-alpha_r[14]-alpha_r[13]-alpha_r[12]-alpha_r[11]-alpha_r[10]-alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]-alpha_r[5]-alpha_r[4]+alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[28] = ser_6x_p1_surfx4_eval_quad_node_28_r(fc); 
  } else { 
    fUpwindQuad_r[28] = ser_6x_p1_surfx4_eval_quad_node_28_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]-alpha_l[29]-alpha_l[28]+alpha_l[27]-alpha_l[26]-alpha_l[25]-alpha_l[24]-alpha_l[23]+alpha_l[22]+alpha_l[21]+alpha_l[20]-alpha_l[19]-alpha_l[18]-alpha_l[17]+alpha_l[16]-alpha_l[15]+alpha_l[14]+alpha_l[13]+alpha_l[12]-alpha_l[11]-alpha_l[10]-alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]+alpha_l[5]-alpha_l[4]+alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[29] = ser_6x_p1_surfx4_eval_quad_node_29_r(fl); 
  } else { 
    fUpwindQuad_l[29] = ser_6x_p1_surfx4_eval_quad_node_29_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]-alpha_r[29]-alpha_r[28]+alpha_r[27]-alpha_r[26]-alpha_r[25]-alpha_r[24]-alpha_r[23]+alpha_r[22]+alpha_r[21]+alpha_r[20]-alpha_r[19]-alpha_r[18]-alpha_r[17]+alpha_r[16]-alpha_r[15]+alpha_r[14]+alpha_r[13]+alpha_r[12]-alpha_r[11]-alpha_r[10]-alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]+alpha_r[5]-alpha_r[4]+alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[29] = ser_6x_p1_surfx4_eval_quad_node_29_r(fc); 
  } else { 
    fUpwindQuad_r[29] = ser_6x_p1_surfx4_eval_quad_node_29_l(fr); 
  } 
  if ((-alpha_l[31])-alpha_l[30]-alpha_l[29]-alpha_l[28]-alpha_l[27]+alpha_l[26]-alpha_l[25]-alpha_l[24]-alpha_l[23]-alpha_l[22]-alpha_l[21]-alpha_l[20]+alpha_l[19]+alpha_l[18]+alpha_l[17]+alpha_l[16]-alpha_l[15]-alpha_l[14]-alpha_l[13]-alpha_l[12]+alpha_l[11]+alpha_l[10]+alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]-alpha_l[5]+alpha_l[4]+alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[30] = ser_6x_p1_surfx4_eval_quad_node_30_r(fl); 
  } else { 
    fUpwindQuad_l[30] = ser_6x_p1_surfx4_eval_quad_node_30_l(fc); 
  } 
  if ((-alpha_r[31])-alpha_r[30]-alpha_r[29]-alpha_r[28]-alpha_r[27]+alpha_r[26]-alpha_r[25]-alpha_r[24]-alpha_r[23]-alpha_r[22]-alpha_r[21]-alpha_r[20]+alpha_r[19]+alpha_r[18]+alpha_r[17]+alpha_r[16]-alpha_r[15]-alpha_r[14]-alpha_r[13]-alpha_r[12]+alpha_r[11]+alpha_r[10]+alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]-alpha_r[5]+alpha_r[4]+alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[30] = ser_6x_p1_surfx4_eval_quad_node_30_r(fc); 
  } else { 
    fUpwindQuad_r[30] = ser_6x_p1_surfx4_eval_quad_node_30_l(fr); 
  } 
  if (alpha_l[31]+alpha_l[30]+alpha_l[29]+alpha_l[28]+alpha_l[27]+alpha_l[26]+alpha_l[25]+alpha_l[24]+alpha_l[23]+alpha_l[22]+alpha_l[21]+alpha_l[20]+alpha_l[19]+alpha_l[18]+alpha_l[17]+alpha_l[16]+alpha_l[15]+alpha_l[14]+alpha_l[13]+alpha_l[12]+alpha_l[11]+alpha_l[10]+alpha_l[9]+alpha_l[8]+alpha_l[7]+alpha_l[6]+alpha_l[5]+alpha_l[4]+alpha_l[3]+alpha_l[2]+alpha_l[1]+alpha_l[0] > 0) { 
    fUpwindQuad_l[31] = ser_6x_p1_surfx4_eval_quad_node_31_r(fl); 
  } else { 
    fUpwindQuad_l[31] = ser_6x_p1_surfx4_eval_quad_node_31_l(fc); 
  } 
  if (alpha_r[31]+alpha_r[30]+alpha_r[29]+alpha_r[28]+alpha_r[27]+alpha_r[26]+alpha_r[25]+alpha_r[24]+alpha_r[23]+alpha_r[22]+alpha_r[21]+alpha_r[20]+alpha_r[19]+alpha_r[18]+alpha_r[17]+alpha_r[16]+alpha_r[15]+alpha_r[14]+alpha_r[13]+alpha_r[12]+alpha_r[11]+alpha_r[10]+alpha_r[9]+alpha_r[8]+alpha_r[7]+alpha_r[6]+alpha_r[5]+alpha_r[4]+alpha_r[3]+alpha_r[2]+alpha_r[1]+alpha_r[0] > 0) { 
    fUpwindQuad_r[31] = ser_6x_p1_surfx4_eval_quad_node_31_r(fc); 
  } else { 
    fUpwindQuad_r[31] = ser_6x_p1_surfx4_eval_quad_node_31_l(fr); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  ser_6x_p1_upwind_quad_to_modal(fUpwindQuad_l, fUpwind_l); 
  ser_6x_p1_upwind_quad_to_modal(fUpwindQuad_r, fUpwind_r); 

  Ghat_l[0] = 0.1767766952966368*(alpha_l[31]*fUpwind_l[31]+alpha_l[30]*fUpwind_l[30]+alpha_l[29]*fUpwind_l[29]+alpha_l[28]*fUpwind_l[28]+alpha_l[27]*fUpwind_l[27]+alpha_l[26]*fUpwind_l[26]+alpha_l[25]*fUpwind_l[25]+alpha_l[24]*fUpwind_l[24]+alpha_l[23]*fUpwind_l[23]+alpha_l[22]*fUpwind_l[22]+alpha_l[21]*fUpwind_l[21]+alpha_l[20]*fUpwind_l[20]+alpha_l[19]*fUpwind_l[19]+alpha_l[18]*fUpwind_l[18]+alpha_l[17]*fUpwind_l[17]+alpha_l[16]*fUpwind_l[16]+alpha_l[15]*fUpwind_l[15]+alpha_l[14]*fUpwind_l[14]+alpha_l[13]*fUpwind_l[13]+alpha_l[12]*fUpwind_l[12]+alpha_l[11]*fUpwind_l[11]+alpha_l[10]*fUpwind_l[10]+alpha_l[9]*fUpwind_l[9]+alpha_l[8]*fUpwind_l[8]+alpha_l[7]*fUpwind_l[7]+alpha_l[6]*fUpwind_l[6]+alpha_l[5]*fUpwind_l[5]+alpha_l[4]*fUpwind_l[4]+alpha_l[3]*fUpwind_l[3]+alpha_l[2]*fUpwind_l[2]+alpha_l[1]*fUpwind_l[1]+alpha_l[0]*fUpwind_l[0]); 
  Ghat_l[1] = 0.1767766952966368*(alpha_l[30]*fUpwind_l[31]+fUpwind_l[30]*alpha_l[31]+alpha_l[25]*fUpwind_l[29]+fUpwind_l[25]*alpha_l[29]+alpha_l[24]*fUpwind_l[28]+fUpwind_l[24]*alpha_l[28]+alpha_l[22]*fUpwind_l[27]+fUpwind_l[22]*alpha_l[27]+alpha_l[19]*fUpwind_l[26]+fUpwind_l[19]*alpha_l[26]+alpha_l[15]*fUpwind_l[23]+fUpwind_l[15]*alpha_l[23]+alpha_l[14]*fUpwind_l[21]+fUpwind_l[14]*alpha_l[21]+alpha_l[13]*fUpwind_l[20]+fUpwind_l[13]*alpha_l[20]+alpha_l[11]*fUpwind_l[18]+fUpwind_l[11]*alpha_l[18]+alpha_l[10]*fUpwind_l[17]+fUpwind_l[10]*alpha_l[17]+alpha_l[8]*fUpwind_l[16]+fUpwind_l[8]*alpha_l[16]+alpha_l[5]*fUpwind_l[12]+fUpwind_l[5]*alpha_l[12]+alpha_l[4]*fUpwind_l[9]+fUpwind_l[4]*alpha_l[9]+alpha_l[3]*fUpwind_l[7]+fUpwind_l[3]*alpha_l[7]+alpha_l[2]*fUpwind_l[6]+fUpwind_l[2]*alpha_l[6]+alpha_l[0]*fUpwind_l[1]+fUpwind_l[0]*alpha_l[1]); 
  Ghat_l[2] = 0.1767766952966368*(alpha_l[29]*fUpwind_l[31]+fUpwind_l[29]*alpha_l[31]+alpha_l[25]*fUpwind_l[30]+fUpwind_l[25]*alpha_l[30]+alpha_l[23]*fUpwind_l[28]+fUpwind_l[23]*alpha_l[28]+alpha_l[21]*fUpwind_l[27]+fUpwind_l[21]*alpha_l[27]+alpha_l[18]*fUpwind_l[26]+fUpwind_l[18]*alpha_l[26]+alpha_l[15]*fUpwind_l[24]+fUpwind_l[15]*alpha_l[24]+alpha_l[14]*fUpwind_l[22]+fUpwind_l[14]*alpha_l[22]+alpha_l[12]*fUpwind_l[20]+fUpwind_l[12]*alpha_l[20]+alpha_l[11]*fUpwind_l[19]+fUpwind_l[11]*alpha_l[19]+alpha_l[9]*fUpwind_l[17]+fUpwind_l[9]*alpha_l[17]+alpha_l[7]*fUpwind_l[16]+fUpwind_l[7]*alpha_l[16]+alpha_l[5]*fUpwind_l[13]+fUpwind_l[5]*alpha_l[13]+alpha_l[4]*fUpwind_l[10]+fUpwind_l[4]*alpha_l[10]+alpha_l[3]*fUpwind_l[8]+fUpwind_l[3]*alpha_l[8]+alpha_l[1]*fUpwind_l[6]+fUpwind_l[1]*alpha_l[6]+alpha_l[0]*fUpwind_l[2]+fUpwind_l[0]*alpha_l[2]); 
  Ghat_l[3] = 0.1767766952966368*(alpha_l[28]*fUpwind_l[31]+fUpwind_l[28]*alpha_l[31]+alpha_l[24]*fUpwind_l[30]+fUpwind_l[24]*alpha_l[30]+alpha_l[23]*fUpwind_l[29]+fUpwind_l[23]*alpha_l[29]+alpha_l[20]*fUpwind_l[27]+fUpwind_l[20]*alpha_l[27]+alpha_l[17]*fUpwind_l[26]+fUpwind_l[17]*alpha_l[26]+alpha_l[15]*fUpwind_l[25]+fUpwind_l[15]*alpha_l[25]+alpha_l[13]*fUpwind_l[22]+fUpwind_l[13]*alpha_l[22]+alpha_l[12]*fUpwind_l[21]+fUpwind_l[12]*alpha_l[21]+alpha_l[10]*fUpwind_l[19]+fUpwind_l[10]*alpha_l[19]+alpha_l[9]*fUpwind_l[18]+fUpwind_l[9]*alpha_l[18]+alpha_l[6]*fUpwind_l[16]+fUpwind_l[6]*alpha_l[16]+alpha_l[5]*fUpwind_l[14]+fUpwind_l[5]*alpha_l[14]+alpha_l[4]*fUpwind_l[11]+fUpwind_l[4]*alpha_l[11]+alpha_l[2]*fUpwind_l[8]+fUpwind_l[2]*alpha_l[8]+alpha_l[1]*fUpwind_l[7]+fUpwind_l[1]*alpha_l[7]+alpha_l[0]*fUpwind_l[3]+fUpwind_l[0]*alpha_l[3]); 
  Ghat_l[4] = 0.1767766952966368*(alpha_l[27]*fUpwind_l[31]+fUpwind_l[27]*alpha_l[31]+alpha_l[22]*fUpwind_l[30]+fUpwind_l[22]*alpha_l[30]+alpha_l[21]*fUpwind_l[29]+fUpwind_l[21]*alpha_l[29]+alpha_l[20]*fUpwind_l[28]+fUpwind_l[20]*alpha_l[28]+alpha_l[16]*fUpwind_l[26]+fUpwind_l[16]*alpha_l[26]+alpha_l[14]*fUpwind_l[25]+fUpwind_l[14]*alpha_l[25]+alpha_l[13]*fUpwind_l[24]+fUpwind_l[13]*alpha_l[24]+alpha_l[12]*fUpwind_l[23]+fUpwind_l[12]*alpha_l[23]+alpha_l[8]*fUpwind_l[19]+fUpwind_l[8]*alpha_l[19]+alpha_l[7]*fUpwind_l[18]+fUpwind_l[7]*alpha_l[18]+alpha_l[6]*fUpwind_l[17]+fUpwind_l[6]*alpha_l[17]+alpha_l[5]*fUpwind_l[15]+fUpwind_l[5]*alpha_l[15]+alpha_l[3]*fUpwind_l[11]+fUpwind_l[3]*alpha_l[11]+alpha_l[2]*fUpwind_l[10]+fUpwind_l[2]*alpha_l[10]+alpha_l[1]*fUpwind_l[9]+fUpwind_l[1]*alpha_l[9]+alpha_l[0]*fUpwind_l[4]+fUpwind_l[0]*alpha_l[4]); 
  Ghat_l[5] = 0.1767766952966368*(alpha_l[26]*fUpwind_l[31]+fUpwind_l[26]*alpha_l[31]+alpha_l[19]*fUpwind_l[30]+fUpwind_l[19]*alpha_l[30]+alpha_l[18]*fUpwind_l[29]+fUpwind_l[18]*alpha_l[29]+alpha_l[17]*fUpwind_l[28]+fUpwind_l[17]*alpha_l[28]+alpha_l[16]*fUpwind_l[27]+fUpwind_l[16]*alpha_l[27]+alpha_l[11]*fUpwind_l[25]+fUpwind_l[11]*alpha_l[25]+alpha_l[10]*fUpwind_l[24]+fUpwind_l[10]*alpha_l[24]+alpha_l[9]*fUpwind_l[23]+fUpwind_l[9]*alpha_l[23]+alpha_l[8]*fUpwind_l[22]+fUpwind_l[8]*alpha_l[22]+alpha_l[7]*fUpwind_l[21]+fUpwind_l[7]*alpha_l[21]+alpha_l[6]*fUpwind_l[20]+fUpwind_l[6]*alpha_l[20]+alpha_l[4]*fUpwind_l[15]+fUpwind_l[4]*alpha_l[15]+alpha_l[3]*fUpwind_l[14]+fUpwind_l[3]*alpha_l[14]+alpha_l[2]*fUpwind_l[13]+fUpwind_l[2]*alpha_l[13]+alpha_l[1]*fUpwind_l[12]+fUpwind_l[1]*alpha_l[12]+alpha_l[0]*fUpwind_l[5]+fUpwind_l[0]*alpha_l[5]); 
  Ghat_l[6] = 0.1767766952966368*(alpha_l[25]*fUpwind_l[31]+fUpwind_l[25]*alpha_l[31]+alpha_l[29]*fUpwind_l[30]+fUpwind_l[29]*alpha_l[30]+alpha_l[15]*fUpwind_l[28]+fUpwind_l[15]*alpha_l[28]+alpha_l[14]*fUpwind_l[27]+fUpwind_l[14]*alpha_l[27]+alpha_l[11]*fUpwind_l[26]+fUpwind_l[11]*alpha_l[26]+alpha_l[23]*fUpwind_l[24]+fUpwind_l[23]*alpha_l[24]+alpha_l[21]*fUpwind_l[22]+fUpwind_l[21]*alpha_l[22]+alpha_l[5]*fUpwind_l[20]+fUpwind_l[5]*alpha_l[20]+alpha_l[18]*fUpwind_l[19]+fUpwind_l[18]*alpha_l[19]+alpha_l[4]*fUpwind_l[17]+fUpwind_l[4]*alpha_l[17]+alpha_l[3]*fUpwind_l[16]+fUpwind_l[3]*alpha_l[16]+alpha_l[12]*fUpwind_l[13]+fUpwind_l[12]*alpha_l[13]+alpha_l[9]*fUpwind_l[10]+fUpwind_l[9]*alpha_l[10]+alpha_l[7]*fUpwind_l[8]+fUpwind_l[7]*alpha_l[8]+alpha_l[0]*fUpwind_l[6]+fUpwind_l[0]*alpha_l[6]+alpha_l[1]*fUpwind_l[2]+fUpwind_l[1]*alpha_l[2]); 
  Ghat_l[7] = 0.1767766952966368*(alpha_l[24]*fUpwind_l[31]+fUpwind_l[24]*alpha_l[31]+alpha_l[28]*fUpwind_l[30]+fUpwind_l[28]*alpha_l[30]+alpha_l[15]*fUpwind_l[29]+fUpwind_l[15]*alpha_l[29]+alpha_l[13]*fUpwind_l[27]+fUpwind_l[13]*alpha_l[27]+alpha_l[10]*fUpwind_l[26]+fUpwind_l[10]*alpha_l[26]+alpha_l[23]*fUpwind_l[25]+fUpwind_l[23]*alpha_l[25]+alpha_l[20]*fUpwind_l[22]+fUpwind_l[20]*alpha_l[22]+alpha_l[5]*fUpwind_l[21]+fUpwind_l[5]*alpha_l[21]+alpha_l[17]*fUpwind_l[19]+fUpwind_l[17]*alpha_l[19]+alpha_l[4]*fUpwind_l[18]+fUpwind_l[4]*alpha_l[18]+alpha_l[2]*fUpwind_l[16]+fUpwind_l[2]*alpha_l[16]+alpha_l[12]*fUpwind_l[14]+fUpwind_l[12]*alpha_l[14]+alpha_l[9]*fUpwind_l[11]+fUpwind_l[9]*alpha_l[11]+alpha_l[6]*fUpwind_l[8]+fUpwind_l[6]*alpha_l[8]+alpha_l[0]*fUpwind_l[7]+fUpwind_l[0]*alpha_l[7]+alpha_l[1]*fUpwind_l[3]+fUpwind_l[1]*alpha_l[3]); 
  Ghat_l[8] = 0.1767766952966368*(alpha_l[23]*fUpwind_l[31]+fUpwind_l[23]*alpha_l[31]+alpha_l[15]*fUpwind_l[30]+fUpwind_l[15]*alpha_l[30]+alpha_l[28]*fUpwind_l[29]+fUpwind_l[28]*alpha_l[29]+alpha_l[12]*fUpwind_l[27]+fUpwind_l[12]*alpha_l[27]+alpha_l[9]*fUpwind_l[26]+fUpwind_l[9]*alpha_l[26]+alpha_l[24]*fUpwind_l[25]+fUpwind_l[24]*alpha_l[25]+alpha_l[5]*fUpwind_l[22]+fUpwind_l[5]*alpha_l[22]+alpha_l[20]*fUpwind_l[21]+fUpwind_l[20]*alpha_l[21]+alpha_l[4]*fUpwind_l[19]+fUpwind_l[4]*alpha_l[19]+alpha_l[17]*fUpwind_l[18]+fUpwind_l[17]*alpha_l[18]+alpha_l[1]*fUpwind_l[16]+fUpwind_l[1]*alpha_l[16]+alpha_l[13]*fUpwind_l[14]+fUpwind_l[13]*alpha_l[14]+alpha_l[10]*fUpwind_l[11]+fUpwind_l[10]*alpha_l[11]+alpha_l[0]*fUpwind_l[8]+fUpwind_l[0]*alpha_l[8]+alpha_l[6]*fUpwind_l[7]+fUpwind_l[6]*alpha_l[7]+alpha_l[2]*fUpwind_l[3]+fUpwind_l[2]*alpha_l[3]); 
  Ghat_l[9] = 0.1767766952966368*(alpha_l[22]*fUpwind_l[31]+fUpwind_l[22]*alpha_l[31]+alpha_l[27]*fUpwind_l[30]+fUpwind_l[27]*alpha_l[30]+alpha_l[14]*fUpwind_l[29]+fUpwind_l[14]*alpha_l[29]+alpha_l[13]*fUpwind_l[28]+fUpwind_l[13]*alpha_l[28]+alpha_l[8]*fUpwind_l[26]+fUpwind_l[8]*alpha_l[26]+alpha_l[21]*fUpwind_l[25]+fUpwind_l[21]*alpha_l[25]+alpha_l[20]*fUpwind_l[24]+fUpwind_l[20]*alpha_l[24]+alpha_l[5]*fUpwind_l[23]+fUpwind_l[5]*alpha_l[23]+alpha_l[16]*fUpwind_l[19]+fUpwind_l[16]*alpha_l[19]+alpha_l[3]*fUpwind_l[18]+fUpwind_l[3]*alpha_l[18]+alpha_l[2]*fUpwind_l[17]+fUpwind_l[2]*alpha_l[17]+alpha_l[12]*fUpwind_l[15]+fUpwind_l[12]*alpha_l[15]+alpha_l[7]*fUpwind_l[11]+fUpwind_l[7]*alpha_l[11]+alpha_l[6]*fUpwind_l[10]+fUpwind_l[6]*alpha_l[10]+alpha_l[0]*fUpwind_l[9]+fUpwind_l[0]*alpha_l[9]+alpha_l[1]*fUpwind_l[4]+fUpwind_l[1]*alpha_l[4]); 
  Ghat_l[10] = 0.1767766952966368*(alpha_l[21]*fUpwind_l[31]+fUpwind_l[21]*alpha_l[31]+alpha_l[14]*fUpwind_l[30]+fUpwind_l[14]*alpha_l[30]+alpha_l[27]*fUpwind_l[29]+fUpwind_l[27]*alpha_l[29]+alpha_l[12]*fUpwind_l[28]+fUpwind_l[12]*alpha_l[28]+alpha_l[7]*fUpwind_l[26]+fUpwind_l[7]*alpha_l[26]+alpha_l[22]*fUpwind_l[25]+fUpwind_l[22]*alpha_l[25]+alpha_l[5]*fUpwind_l[24]+fUpwind_l[5]*alpha_l[24]+alpha_l[20]*fUpwind_l[23]+fUpwind_l[20]*alpha_l[23]+alpha_l[3]*fUpwind_l[19]+fUpwind_l[3]*alpha_l[19]+alpha_l[16]*fUpwind_l[18]+fUpwind_l[16]*alpha_l[18]+alpha_l[1]*fUpwind_l[17]+fUpwind_l[1]*alpha_l[17]+alpha_l[13]*fUpwind_l[15]+fUpwind_l[13]*alpha_l[15]+alpha_l[8]*fUpwind_l[11]+fUpwind_l[8]*alpha_l[11]+alpha_l[0]*fUpwind_l[10]+fUpwind_l[0]*alpha_l[10]+alpha_l[6]*fUpwind_l[9]+fUpwind_l[6]*alpha_l[9]+alpha_l[2]*fUpwind_l[4]+fUpwind_l[2]*alpha_l[4]); 
  Ghat_l[11] = 0.1767766952966368*(alpha_l[20]*fUpwind_l[31]+fUpwind_l[20]*alpha_l[31]+alpha_l[13]*fUpwind_l[30]+fUpwind_l[13]*alpha_l[30]+alpha_l[12]*fUpwind_l[29]+fUpwind_l[12]*alpha_l[29]+alpha_l[27]*fUpwind_l[28]+fUpwind_l[27]*alpha_l[28]+alpha_l[6]*fUpwind_l[26]+fUpwind_l[6]*alpha_l[26]+alpha_l[5]*fUpwind_l[25]+fUpwind_l[5]*alpha_l[25]+alpha_l[22]*fUpwind_l[24]+fUpwind_l[22]*alpha_l[24]+alpha_l[21]*fUpwind_l[23]+fUpwind_l[21]*alpha_l[23]+alpha_l[2]*fUpwind_l[19]+fUpwind_l[2]*alpha_l[19]+alpha_l[1]*fUpwind_l[18]+fUpwind_l[1]*alpha_l[18]+alpha_l[16]*fUpwind_l[17]+fUpwind_l[16]*alpha_l[17]+alpha_l[14]*fUpwind_l[15]+fUpwind_l[14]*alpha_l[15]+alpha_l[0]*fUpwind_l[11]+fUpwind_l[0]*alpha_l[11]+alpha_l[8]*fUpwind_l[10]+fUpwind_l[8]*alpha_l[10]+alpha_l[7]*fUpwind_l[9]+fUpwind_l[7]*alpha_l[9]+alpha_l[3]*fUpwind_l[4]+fUpwind_l[3]*alpha_l[4]); 
  Ghat_l[12] = 0.1767766952966368*(alpha_l[19]*fUpwind_l[31]+fUpwind_l[19]*alpha_l[31]+alpha_l[26]*fUpwind_l[30]+fUpwind_l[26]*alpha_l[30]+alpha_l[11]*fUpwind_l[29]+fUpwind_l[11]*alpha_l[29]+alpha_l[10]*fUpwind_l[28]+fUpwind_l[10]*alpha_l[28]+alpha_l[8]*fUpwind_l[27]+fUpwind_l[8]*alpha_l[27]+alpha_l[18]*fUpwind_l[25]+fUpwind_l[18]*alpha_l[25]+alpha_l[17]*fUpwind_l[24]+fUpwind_l[17]*alpha_l[24]+alpha_l[4]*fUpwind_l[23]+fUpwind_l[4]*alpha_l[23]+alpha_l[16]*fUpwind_l[22]+fUpwind_l[16]*alpha_l[22]+alpha_l[3]*fUpwind_l[21]+fUpwind_l[3]*alpha_l[21]+alpha_l[2]*fUpwind_l[20]+fUpwind_l[2]*alpha_l[20]+alpha_l[9]*fUpwind_l[15]+fUpwind_l[9]*alpha_l[15]+alpha_l[7]*fUpwind_l[14]+fUpwind_l[7]*alpha_l[14]+alpha_l[6]*fUpwind_l[13]+fUpwind_l[6]*alpha_l[13]+alpha_l[0]*fUpwind_l[12]+fUpwind_l[0]*alpha_l[12]+alpha_l[1]*fUpwind_l[5]+fUpwind_l[1]*alpha_l[5]); 
  Ghat_l[13] = 0.1767766952966368*(alpha_l[18]*fUpwind_l[31]+fUpwind_l[18]*alpha_l[31]+alpha_l[11]*fUpwind_l[30]+fUpwind_l[11]*alpha_l[30]+alpha_l[26]*fUpwind_l[29]+fUpwind_l[26]*alpha_l[29]+alpha_l[9]*fUpwind_l[28]+fUpwind_l[9]*alpha_l[28]+alpha_l[7]*fUpwind_l[27]+fUpwind_l[7]*alpha_l[27]+alpha_l[19]*fUpwind_l[25]+fUpwind_l[19]*alpha_l[25]+alpha_l[4]*fUpwind_l[24]+fUpwind_l[4]*alpha_l[24]+alpha_l[17]*fUpwind_l[23]+fUpwind_l[17]*alpha_l[23]+alpha_l[3]*fUpwind_l[22]+fUpwind_l[3]*alpha_l[22]+alpha_l[16]*fUpwind_l[21]+fUpwind_l[16]*alpha_l[21]+alpha_l[1]*fUpwind_l[20]+fUpwind_l[1]*alpha_l[20]+alpha_l[10]*fUpwind_l[15]+fUpwind_l[10]*alpha_l[15]+alpha_l[8]*fUpwind_l[14]+fUpwind_l[8]*alpha_l[14]+alpha_l[0]*fUpwind_l[13]+fUpwind_l[0]*alpha_l[13]+alpha_l[6]*fUpwind_l[12]+fUpwind_l[6]*alpha_l[12]+alpha_l[2]*fUpwind_l[5]+fUpwind_l[2]*alpha_l[5]); 
  Ghat_l[14] = 0.1767766952966368*(alpha_l[17]*fUpwind_l[31]+fUpwind_l[17]*alpha_l[31]+alpha_l[10]*fUpwind_l[30]+fUpwind_l[10]*alpha_l[30]+alpha_l[9]*fUpwind_l[29]+fUpwind_l[9]*alpha_l[29]+alpha_l[26]*fUpwind_l[28]+fUpwind_l[26]*alpha_l[28]+alpha_l[6]*fUpwind_l[27]+fUpwind_l[6]*alpha_l[27]+alpha_l[4]*fUpwind_l[25]+fUpwind_l[4]*alpha_l[25]+alpha_l[19]*fUpwind_l[24]+fUpwind_l[19]*alpha_l[24]+alpha_l[18]*fUpwind_l[23]+fUpwind_l[18]*alpha_l[23]+alpha_l[2]*fUpwind_l[22]+fUpwind_l[2]*alpha_l[22]+alpha_l[1]*fUpwind_l[21]+fUpwind_l[1]*alpha_l[21]+alpha_l[16]*fUpwind_l[20]+fUpwind_l[16]*alpha_l[20]+alpha_l[11]*fUpwind_l[15]+fUpwind_l[11]*alpha_l[15]+alpha_l[0]*fUpwind_l[14]+fUpwind_l[0]*alpha_l[14]+alpha_l[8]*fUpwind_l[13]+fUpwind_l[8]*alpha_l[13]+alpha_l[7]*fUpwind_l[12]+fUpwind_l[7]*alpha_l[12]+alpha_l[3]*fUpwind_l[5]+fUpwind_l[3]*alpha_l[5]); 
  Ghat_l[15] = 0.1767766952966368*(alpha_l[16]*fUpwind_l[31]+fUpwind_l[16]*alpha_l[31]+alpha_l[8]*fUpwind_l[30]+fUpwind_l[8]*alpha_l[30]+alpha_l[7]*fUpwind_l[29]+fUpwind_l[7]*alpha_l[29]+alpha_l[6]*fUpwind_l[28]+fUpwind_l[6]*alpha_l[28]+alpha_l[26]*fUpwind_l[27]+fUpwind_l[26]*alpha_l[27]+alpha_l[3]*fUpwind_l[25]+fUpwind_l[3]*alpha_l[25]+alpha_l[2]*fUpwind_l[24]+fUpwind_l[2]*alpha_l[24]+alpha_l[1]*fUpwind_l[23]+fUpwind_l[1]*alpha_l[23]+alpha_l[19]*fUpwind_l[22]+fUpwind_l[19]*alpha_l[22]+alpha_l[18]*fUpwind_l[21]+fUpwind_l[18]*alpha_l[21]+alpha_l[17]*fUpwind_l[20]+fUpwind_l[17]*alpha_l[20]+alpha_l[0]*fUpwind_l[15]+fUpwind_l[0]*alpha_l[15]+alpha_l[11]*fUpwind_l[14]+fUpwind_l[11]*alpha_l[14]+alpha_l[10]*fUpwind_l[13]+fUpwind_l[10]*alpha_l[13]+alpha_l[9]*fUpwind_l[12]+fUpwind_l[9]*alpha_l[12]+alpha_l[4]*fUpwind_l[5]+fUpwind_l[4]*alpha_l[5]); 
  Ghat_l[16] = 0.1767766952966368*(alpha_l[15]*fUpwind_l[31]+fUpwind_l[15]*alpha_l[31]+alpha_l[23]*fUpwind_l[30]+fUpwind_l[23]*alpha_l[30]+alpha_l[24]*fUpwind_l[29]+fUpwind_l[24]*alpha_l[29]+alpha_l[25]*fUpwind_l[28]+fUpwind_l[25]*alpha_l[28]+alpha_l[5]*fUpwind_l[27]+fUpwind_l[5]*alpha_l[27]+alpha_l[4]*fUpwind_l[26]+fUpwind_l[4]*alpha_l[26]+alpha_l[12]*fUpwind_l[22]+fUpwind_l[12]*alpha_l[22]+alpha_l[13]*fUpwind_l[21]+fUpwind_l[13]*alpha_l[21]+alpha_l[14]*fUpwind_l[20]+fUpwind_l[14]*alpha_l[20]+alpha_l[9]*fUpwind_l[19]+fUpwind_l[9]*alpha_l[19]+alpha_l[10]*fUpwind_l[18]+fUpwind_l[10]*alpha_l[18]+alpha_l[11]*fUpwind_l[17]+fUpwind_l[11]*alpha_l[17]+alpha_l[0]*fUpwind_l[16]+fUpwind_l[0]*alpha_l[16]+alpha_l[1]*fUpwind_l[8]+fUpwind_l[1]*alpha_l[8]+alpha_l[2]*fUpwind_l[7]+fUpwind_l[2]*alpha_l[7]+alpha_l[3]*fUpwind_l[6]+fUpwind_l[3]*alpha_l[6]); 
  Ghat_l[17] = 0.1767766952966368*(alpha_l[14]*fUpwind_l[31]+fUpwind_l[14]*alpha_l[31]+alpha_l[21]*fUpwind_l[30]+fUpwind_l[21]*alpha_l[30]+alpha_l[22]*fUpwind_l[29]+fUpwind_l[22]*alpha_l[29]+alpha_l[5]*fUpwind_l[28]+fUpwind_l[5]*alpha_l[28]+alpha_l[25]*fUpwind_l[27]+fUpwind_l[25]*alpha_l[27]+alpha_l[3]*fUpwind_l[26]+fUpwind_l[3]*alpha_l[26]+alpha_l[12]*fUpwind_l[24]+fUpwind_l[12]*alpha_l[24]+alpha_l[13]*fUpwind_l[23]+fUpwind_l[13]*alpha_l[23]+alpha_l[15]*fUpwind_l[20]+fUpwind_l[15]*alpha_l[20]+alpha_l[7]*fUpwind_l[19]+fUpwind_l[7]*alpha_l[19]+alpha_l[8]*fUpwind_l[18]+fUpwind_l[8]*alpha_l[18]+alpha_l[0]*fUpwind_l[17]+fUpwind_l[0]*alpha_l[17]+alpha_l[11]*fUpwind_l[16]+fUpwind_l[11]*alpha_l[16]+alpha_l[1]*fUpwind_l[10]+fUpwind_l[1]*alpha_l[10]+alpha_l[2]*fUpwind_l[9]+fUpwind_l[2]*alpha_l[9]+alpha_l[4]*fUpwind_l[6]+fUpwind_l[4]*alpha_l[6]); 
  Ghat_l[18] = 0.1767766952966368*(alpha_l[13]*fUpwind_l[31]+fUpwind_l[13]*alpha_l[31]+alpha_l[20]*fUpwind_l[30]+fUpwind_l[20]*alpha_l[30]+alpha_l[5]*fUpwind_l[29]+fUpwind_l[5]*alpha_l[29]+alpha_l[22]*fUpwind_l[28]+fUpwind_l[22]*alpha_l[28]+alpha_l[24]*fUpwind_l[27]+fUpwind_l[24]*alpha_l[27]+alpha_l[2]*fUpwind_l[26]+fUpwind_l[2]*alpha_l[26]+alpha_l[12]*fUpwind_l[25]+fUpwind_l[12]*alpha_l[25]+alpha_l[14]*fUpwind_l[23]+fUpwind_l[14]*alpha_l[23]+alpha_l[15]*fUpwind_l[21]+fUpwind_l[15]*alpha_l[21]+alpha_l[6]*fUpwind_l[19]+fUpwind_l[6]*alpha_l[19]+alpha_l[0]*fUpwind_l[18]+fUpwind_l[0]*alpha_l[18]+alpha_l[8]*fUpwind_l[17]+fUpwind_l[8]*alpha_l[17]+alpha_l[10]*fUpwind_l[16]+fUpwind_l[10]*alpha_l[16]+alpha_l[1]*fUpwind_l[11]+fUpwind_l[1]*alpha_l[11]+alpha_l[3]*fUpwind_l[9]+fUpwind_l[3]*alpha_l[9]+alpha_l[4]*fUpwind_l[7]+fUpwind_l[4]*alpha_l[7]); 
  Ghat_l[19] = 0.1767766952966368*(alpha_l[12]*fUpwind_l[31]+fUpwind_l[12]*alpha_l[31]+alpha_l[5]*fUpwind_l[30]+fUpwind_l[5]*alpha_l[30]+alpha_l[20]*fUpwind_l[29]+fUpwind_l[20]*alpha_l[29]+alpha_l[21]*fUpwind_l[28]+fUpwind_l[21]*alpha_l[28]+alpha_l[23]*fUpwind_l[27]+fUpwind_l[23]*alpha_l[27]+alpha_l[1]*fUpwind_l[26]+fUpwind_l[1]*alpha_l[26]+alpha_l[13]*fUpwind_l[25]+fUpwind_l[13]*alpha_l[25]+alpha_l[14]*fUpwind_l[24]+fUpwind_l[14]*alpha_l[24]+alpha_l[15]*fUpwind_l[22]+fUpwind_l[15]*alpha_l[22]+alpha_l[0]*fUpwind_l[19]+fUpwind_l[0]*alpha_l[19]+alpha_l[6]*fUpwind_l[18]+fUpwind_l[6]*alpha_l[18]+alpha_l[7]*fUpwind_l[17]+fUpwind_l[7]*alpha_l[17]+alpha_l[9]*fUpwind_l[16]+fUpwind_l[9]*alpha_l[16]+alpha_l[2]*fUpwind_l[11]+fUpwind_l[2]*alpha_l[11]+alpha_l[3]*fUpwind_l[10]+fUpwind_l[3]*alpha_l[10]+alpha_l[4]*fUpwind_l[8]+fUpwind_l[4]*alpha_l[8]); 
  Ghat_l[20] = 0.1767766952966368*(alpha_l[11]*fUpwind_l[31]+fUpwind_l[11]*alpha_l[31]+alpha_l[18]*fUpwind_l[30]+fUpwind_l[18]*alpha_l[30]+alpha_l[19]*fUpwind_l[29]+fUpwind_l[19]*alpha_l[29]+alpha_l[4]*fUpwind_l[28]+fUpwind_l[4]*alpha_l[28]+alpha_l[3]*fUpwind_l[27]+fUpwind_l[3]*alpha_l[27]+alpha_l[25]*fUpwind_l[26]+fUpwind_l[25]*alpha_l[26]+alpha_l[9]*fUpwind_l[24]+fUpwind_l[9]*alpha_l[24]+alpha_l[10]*fUpwind_l[23]+fUpwind_l[10]*alpha_l[23]+alpha_l[7]*fUpwind_l[22]+fUpwind_l[7]*alpha_l[22]+alpha_l[8]*fUpwind_l[21]+fUpwind_l[8]*alpha_l[21]+alpha_l[0]*fUpwind_l[20]+fUpwind_l[0]*alpha_l[20]+alpha_l[15]*fUpwind_l[17]+fUpwind_l[15]*alpha_l[17]+alpha_l[14]*fUpwind_l[16]+fUpwind_l[14]*alpha_l[16]+alpha_l[1]*fUpwind_l[13]+fUpwind_l[1]*alpha_l[13]+alpha_l[2]*fUpwind_l[12]+fUpwind_l[2]*alpha_l[12]+alpha_l[5]*fUpwind_l[6]+fUpwind_l[5]*alpha_l[6]); 
  Ghat_l[21] = 0.1767766952966368*(alpha_l[10]*fUpwind_l[31]+fUpwind_l[10]*alpha_l[31]+alpha_l[17]*fUpwind_l[30]+fUpwind_l[17]*alpha_l[30]+alpha_l[4]*fUpwind_l[29]+fUpwind_l[4]*alpha_l[29]+alpha_l[19]*fUpwind_l[28]+fUpwind_l[19]*alpha_l[28]+alpha_l[2]*fUpwind_l[27]+fUpwind_l[2]*alpha_l[27]+alpha_l[24]*fUpwind_l[26]+fUpwind_l[24]*alpha_l[26]+alpha_l[9]*fUpwind_l[25]+fUpwind_l[9]*alpha_l[25]+alpha_l[11]*fUpwind_l[23]+fUpwind_l[11]*alpha_l[23]+alpha_l[6]*fUpwind_l[22]+fUpwind_l[6]*alpha_l[22]+alpha_l[0]*fUpwind_l[21]+fUpwind_l[0]*alpha_l[21]+alpha_l[8]*fUpwind_l[20]+fUpwind_l[8]*alpha_l[20]+alpha_l[15]*fUpwind_l[18]+fUpwind_l[15]*alpha_l[18]+alpha_l[13]*fUpwind_l[16]+fUpwind_l[13]*alpha_l[16]+alpha_l[1]*fUpwind_l[14]+fUpwind_l[1]*alpha_l[14]+alpha_l[3]*fUpwind_l[12]+fUpwind_l[3]*alpha_l[12]+alpha_l[5]*fUpwind_l[7]+fUpwind_l[5]*alpha_l[7]); 
  Ghat_l[22] = 0.1767766952966368*(alpha_l[9]*fUpwind_l[31]+fUpwind_l[9]*alpha_l[31]+alpha_l[4]*fUpwind_l[30]+fUpwind_l[4]*alpha_l[30]+alpha_l[17]*fUpwind_l[29]+fUpwind_l[17]*alpha_l[29]+alpha_l[18]*fUpwind_l[28]+fUpwind_l[18]*alpha_l[28]+alpha_l[1]*fUpwind_l[27]+fUpwind_l[1]*alpha_l[27]+alpha_l[23]*fUpwind_l[26]+fUpwind_l[23]*alpha_l[26]+alpha_l[10]*fUpwind_l[25]+fUpwind_l[10]*alpha_l[25]+alpha_l[11]*fUpwind_l[24]+fUpwind_l[11]*alpha_l[24]+alpha_l[0]*fUpwind_l[22]+fUpwind_l[0]*alpha_l[22]+alpha_l[6]*fUpwind_l[21]+fUpwind_l[6]*alpha_l[21]+alpha_l[7]*fUpwind_l[20]+fUpwind_l[7]*alpha_l[20]+alpha_l[15]*fUpwind_l[19]+fUpwind_l[15]*alpha_l[19]+alpha_l[12]*fUpwind_l[16]+fUpwind_l[12]*alpha_l[16]+alpha_l[2]*fUpwind_l[14]+fUpwind_l[2]*alpha_l[14]+alpha_l[3]*fUpwind_l[13]+fUpwind_l[3]*alpha_l[13]+alpha_l[5]*fUpwind_l[8]+fUpwind_l[5]*alpha_l[8]); 
  Ghat_l[23] = 0.1767766952966368*(alpha_l[8]*fUpwind_l[31]+fUpwind_l[8]*alpha_l[31]+alpha_l[16]*fUpwind_l[30]+fUpwind_l[16]*alpha_l[30]+alpha_l[3]*fUpwind_l[29]+fUpwind_l[3]*alpha_l[29]+alpha_l[2]*fUpwind_l[28]+fUpwind_l[2]*alpha_l[28]+alpha_l[19]*fUpwind_l[27]+fUpwind_l[19]*alpha_l[27]+alpha_l[22]*fUpwind_l[26]+fUpwind_l[22]*alpha_l[26]+alpha_l[7]*fUpwind_l[25]+fUpwind_l[7]*alpha_l[25]+alpha_l[6]*fUpwind_l[24]+fUpwind_l[6]*alpha_l[24]+alpha_l[0]*fUpwind_l[23]+fUpwind_l[0]*alpha_l[23]+alpha_l[11]*fUpwind_l[21]+fUpwind_l[11]*alpha_l[21]+alpha_l[10]*fUpwind_l[20]+fUpwind_l[10]*alpha_l[20]+alpha_l[14]*fUpwind_l[18]+fUpwind_l[14]*alpha_l[18]+alpha_l[13]*fUpwind_l[17]+fUpwind_l[13]*alpha_l[17]+alpha_l[1]*fUpwind_l[15]+fUpwind_l[1]*alpha_l[15]+alpha_l[4]*fUpwind_l[12]+fUpwind_l[4]*alpha_l[12]+alpha_l[5]*fUpwind_l[9]+fUpwind_l[5]*alpha_l[9]); 
  Ghat_l[24] = 0.1767766952966368*(alpha_l[7]*fUpwind_l[31]+fUpwind_l[7]*alpha_l[31]+alpha_l[3]*fUpwind_l[30]+fUpwind_l[3]*alpha_l[30]+alpha_l[16]*fUpwind_l[29]+fUpwind_l[16]*alpha_l[29]+alpha_l[1]*fUpwind_l[28]+fUpwind_l[1]*alpha_l[28]+alpha_l[18]*fUpwind_l[27]+fUpwind_l[18]*alpha_l[27]+alpha_l[21]*fUpwind_l[26]+fUpwind_l[21]*alpha_l[26]+alpha_l[8]*fUpwind_l[25]+fUpwind_l[8]*alpha_l[25]+alpha_l[0]*fUpwind_l[24]+fUpwind_l[0]*alpha_l[24]+alpha_l[6]*fUpwind_l[23]+fUpwind_l[6]*alpha_l[23]+alpha_l[11]*fUpwind_l[22]+fUpwind_l[11]*alpha_l[22]+alpha_l[9]*fUpwind_l[20]+fUpwind_l[9]*alpha_l[20]+alpha_l[14]*fUpwind_l[19]+fUpwind_l[14]*alpha_l[19]+alpha_l[12]*fUpwind_l[17]+fUpwind_l[12]*alpha_l[17]+alpha_l[2]*fUpwind_l[15]+fUpwind_l[2]*alpha_l[15]+alpha_l[4]*fUpwind_l[13]+fUpwind_l[4]*alpha_l[13]+alpha_l[5]*fUpwind_l[10]+fUpwind_l[5]*alpha_l[10]); 
  Ghat_l[25] = 0.1767766952966368*(alpha_l[6]*fUpwind_l[31]+fUpwind_l[6]*alpha_l[31]+alpha_l[2]*fUpwind_l[30]+fUpwind_l[2]*alpha_l[30]+alpha_l[1]*fUpwind_l[29]+fUpwind_l[1]*alpha_l[29]+alpha_l[16]*fUpwind_l[28]+fUpwind_l[16]*alpha_l[28]+alpha_l[17]*fUpwind_l[27]+fUpwind_l[17]*alpha_l[27]+alpha_l[20]*fUpwind_l[26]+fUpwind_l[20]*alpha_l[26]+alpha_l[0]*fUpwind_l[25]+fUpwind_l[0]*alpha_l[25]+alpha_l[8]*fUpwind_l[24]+fUpwind_l[8]*alpha_l[24]+alpha_l[7]*fUpwind_l[23]+fUpwind_l[7]*alpha_l[23]+alpha_l[10]*fUpwind_l[22]+fUpwind_l[10]*alpha_l[22]+alpha_l[9]*fUpwind_l[21]+fUpwind_l[9]*alpha_l[21]+alpha_l[13]*fUpwind_l[19]+fUpwind_l[13]*alpha_l[19]+alpha_l[12]*fUpwind_l[18]+fUpwind_l[12]*alpha_l[18]+alpha_l[3]*fUpwind_l[15]+fUpwind_l[3]*alpha_l[15]+alpha_l[4]*fUpwind_l[14]+fUpwind_l[4]*alpha_l[14]+alpha_l[5]*fUpwind_l[11]+fUpwind_l[5]*alpha_l[11]); 
  Ghat_l[26] = 0.1767766952966368*(alpha_l[5]*fUpwind_l[31]+fUpwind_l[5]*alpha_l[31]+alpha_l[12]*fUpwind_l[30]+fUpwind_l[12]*alpha_l[30]+alpha_l[13]*fUpwind_l[29]+fUpwind_l[13]*alpha_l[29]+alpha_l[14]*fUpwind_l[28]+fUpwind_l[14]*alpha_l[28]+alpha_l[15]*fUpwind_l[27]+fUpwind_l[15]*alpha_l[27]+alpha_l[0]*fUpwind_l[26]+fUpwind_l[0]*alpha_l[26]+alpha_l[20]*fUpwind_l[25]+fUpwind_l[20]*alpha_l[25]+alpha_l[21]*fUpwind_l[24]+fUpwind_l[21]*alpha_l[24]+alpha_l[22]*fUpwind_l[23]+fUpwind_l[22]*alpha_l[23]+alpha_l[1]*fUpwind_l[19]+fUpwind_l[1]*alpha_l[19]+alpha_l[2]*fUpwind_l[18]+fUpwind_l[2]*alpha_l[18]+alpha_l[3]*fUpwind_l[17]+fUpwind_l[3]*alpha_l[17]+alpha_l[4]*fUpwind_l[16]+fUpwind_l[4]*alpha_l[16]+alpha_l[6]*fUpwind_l[11]+fUpwind_l[6]*alpha_l[11]+alpha_l[7]*fUpwind_l[10]+fUpwind_l[7]*alpha_l[10]+alpha_l[8]*fUpwind_l[9]+fUpwind_l[8]*alpha_l[9]); 
  Ghat_l[27] = 0.1767766952966368*(alpha_l[4]*fUpwind_l[31]+fUpwind_l[4]*alpha_l[31]+alpha_l[9]*fUpwind_l[30]+fUpwind_l[9]*alpha_l[30]+alpha_l[10]*fUpwind_l[29]+fUpwind_l[10]*alpha_l[29]+alpha_l[11]*fUpwind_l[28]+fUpwind_l[11]*alpha_l[28]+alpha_l[0]*fUpwind_l[27]+fUpwind_l[0]*alpha_l[27]+alpha_l[15]*fUpwind_l[26]+fUpwind_l[15]*alpha_l[26]+alpha_l[17]*fUpwind_l[25]+fUpwind_l[17]*alpha_l[25]+alpha_l[18]*fUpwind_l[24]+fUpwind_l[18]*alpha_l[24]+alpha_l[19]*fUpwind_l[23]+fUpwind_l[19]*alpha_l[23]+alpha_l[1]*fUpwind_l[22]+fUpwind_l[1]*alpha_l[22]+alpha_l[2]*fUpwind_l[21]+fUpwind_l[2]*alpha_l[21]+alpha_l[3]*fUpwind_l[20]+fUpwind_l[3]*alpha_l[20]+alpha_l[5]*fUpwind_l[16]+fUpwind_l[5]*alpha_l[16]+alpha_l[6]*fUpwind_l[14]+fUpwind_l[6]*alpha_l[14]+alpha_l[7]*fUpwind_l[13]+fUpwind_l[7]*alpha_l[13]+alpha_l[8]*fUpwind_l[12]+fUpwind_l[8]*alpha_l[12]); 
  Ghat_l[28] = 0.1767766952966368*(alpha_l[3]*fUpwind_l[31]+fUpwind_l[3]*alpha_l[31]+alpha_l[7]*fUpwind_l[30]+fUpwind_l[7]*alpha_l[30]+alpha_l[8]*fUpwind_l[29]+fUpwind_l[8]*alpha_l[29]+alpha_l[0]*fUpwind_l[28]+fUpwind_l[0]*alpha_l[28]+alpha_l[11]*fUpwind_l[27]+fUpwind_l[11]*alpha_l[27]+alpha_l[14]*fUpwind_l[26]+fUpwind_l[14]*alpha_l[26]+alpha_l[16]*fUpwind_l[25]+fUpwind_l[16]*alpha_l[25]+alpha_l[1]*fUpwind_l[24]+fUpwind_l[1]*alpha_l[24]+alpha_l[2]*fUpwind_l[23]+fUpwind_l[2]*alpha_l[23]+alpha_l[18]*fUpwind_l[22]+fUpwind_l[18]*alpha_l[22]+alpha_l[19]*fUpwind_l[21]+fUpwind_l[19]*alpha_l[21]+alpha_l[4]*fUpwind_l[20]+fUpwind_l[4]*alpha_l[20]+alpha_l[5]*fUpwind_l[17]+fUpwind_l[5]*alpha_l[17]+alpha_l[6]*fUpwind_l[15]+fUpwind_l[6]*alpha_l[15]+alpha_l[9]*fUpwind_l[13]+fUpwind_l[9]*alpha_l[13]+alpha_l[10]*fUpwind_l[12]+fUpwind_l[10]*alpha_l[12]); 
  Ghat_l[29] = 0.1767766952966368*(alpha_l[2]*fUpwind_l[31]+fUpwind_l[2]*alpha_l[31]+alpha_l[6]*fUpwind_l[30]+fUpwind_l[6]*alpha_l[30]+alpha_l[0]*fUpwind_l[29]+fUpwind_l[0]*alpha_l[29]+alpha_l[8]*fUpwind_l[28]+fUpwind_l[8]*alpha_l[28]+alpha_l[10]*fUpwind_l[27]+fUpwind_l[10]*alpha_l[27]+alpha_l[13]*fUpwind_l[26]+fUpwind_l[13]*alpha_l[26]+alpha_l[1]*fUpwind_l[25]+fUpwind_l[1]*alpha_l[25]+alpha_l[16]*fUpwind_l[24]+fUpwind_l[16]*alpha_l[24]+alpha_l[3]*fUpwind_l[23]+fUpwind_l[3]*alpha_l[23]+alpha_l[17]*fUpwind_l[22]+fUpwind_l[17]*alpha_l[22]+alpha_l[4]*fUpwind_l[21]+fUpwind_l[4]*alpha_l[21]+alpha_l[19]*fUpwind_l[20]+fUpwind_l[19]*alpha_l[20]+alpha_l[5]*fUpwind_l[18]+fUpwind_l[5]*alpha_l[18]+alpha_l[7]*fUpwind_l[15]+fUpwind_l[7]*alpha_l[15]+alpha_l[9]*fUpwind_l[14]+fUpwind_l[9]*alpha_l[14]+alpha_l[11]*fUpwind_l[12]+fUpwind_l[11]*alpha_l[12]); 
  Ghat_l[30] = 0.1767766952966368*(alpha_l[1]*fUpwind_l[31]+fUpwind_l[1]*alpha_l[31]+alpha_l[0]*fUpwind_l[30]+fUpwind_l[0]*alpha_l[30]+alpha_l[6]*fUpwind_l[29]+fUpwind_l[6]*alpha_l[29]+alpha_l[7]*fUpwind_l[28]+fUpwind_l[7]*alpha_l[28]+alpha_l[9]*fUpwind_l[27]+fUpwind_l[9]*alpha_l[27]+alpha_l[12]*fUpwind_l[26]+fUpwind_l[12]*alpha_l[26]+alpha_l[2]*fUpwind_l[25]+fUpwind_l[2]*alpha_l[25]+alpha_l[3]*fUpwind_l[24]+fUpwind_l[3]*alpha_l[24]+alpha_l[16]*fUpwind_l[23]+fUpwind_l[16]*alpha_l[23]+alpha_l[4]*fUpwind_l[22]+fUpwind_l[4]*alpha_l[22]+alpha_l[17]*fUpwind_l[21]+fUpwind_l[17]*alpha_l[21]+alpha_l[18]*fUpwind_l[20]+fUpwind_l[18]*alpha_l[20]+alpha_l[5]*fUpwind_l[19]+fUpwind_l[5]*alpha_l[19]+alpha_l[8]*fUpwind_l[15]+fUpwind_l[8]*alpha_l[15]+alpha_l[10]*fUpwind_l[14]+fUpwind_l[10]*alpha_l[14]+alpha_l[11]*fUpwind_l[13]+fUpwind_l[11]*alpha_l[13]); 
  Ghat_l[31] = 0.1767766952966368*(alpha_l[0]*fUpwind_l[31]+fUpwind_l[0]*alpha_l[31]+alpha_l[1]*fUpwind_l[30]+fUpwind_l[1]*alpha_l[30]+alpha_l[2]*fUpwind_l[29]+fUpwind_l[2]*alpha_l[29]+alpha_l[3]*fUpwind_l[28]+fUpwind_l[3]*alpha_l[28]+alpha_l[4]*fUpwind_l[27]+fUpwind_l[4]*alpha_l[27]+alpha_l[5]*fUpwind_l[26]+fUpwind_l[5]*alpha_l[26]+alpha_l[6]*fUpwind_l[25]+fUpwind_l[6]*alpha_l[25]+alpha_l[7]*fUpwind_l[24]+fUpwind_l[7]*alpha_l[24]+alpha_l[8]*fUpwind_l[23]+fUpwind_l[8]*alpha_l[23]+alpha_l[9]*fUpwind_l[22]+fUpwind_l[9]*alpha_l[22]+alpha_l[10]*fUpwind_l[21]+fUpwind_l[10]*alpha_l[21]+alpha_l[11]*fUpwind_l[20]+fUpwind_l[11]*alpha_l[20]+alpha_l[12]*fUpwind_l[19]+fUpwind_l[12]*alpha_l[19]+alpha_l[13]*fUpwind_l[18]+fUpwind_l[13]*alpha_l[18]+alpha_l[14]*fUpwind_l[17]+fUpwind_l[14]*alpha_l[17]+alpha_l[15]*fUpwind_l[16]+fUpwind_l[15]*alpha_l[16]); 

  Ghat_r[0] = 0.1767766952966368*(alpha_r[31]*fUpwind_r[31]+alpha_r[30]*fUpwind_r[30]+alpha_r[29]*fUpwind_r[29]+alpha_r[28]*fUpwind_r[28]+alpha_r[27]*fUpwind_r[27]+alpha_r[26]*fUpwind_r[26]+alpha_r[25]*fUpwind_r[25]+alpha_r[24]*fUpwind_r[24]+alpha_r[23]*fUpwind_r[23]+alpha_r[22]*fUpwind_r[22]+alpha_r[21]*fUpwind_r[21]+alpha_r[20]*fUpwind_r[20]+alpha_r[19]*fUpwind_r[19]+alpha_r[18]*fUpwind_r[18]+alpha_r[17]*fUpwind_r[17]+alpha_r[16]*fUpwind_r[16]+alpha_r[15]*fUpwind_r[15]+alpha_r[14]*fUpwind_r[14]+alpha_r[13]*fUpwind_r[13]+alpha_r[12]*fUpwind_r[12]+alpha_r[11]*fUpwind_r[11]+alpha_r[10]*fUpwind_r[10]+alpha_r[9]*fUpwind_r[9]+alpha_r[8]*fUpwind_r[8]+alpha_r[7]*fUpwind_r[7]+alpha_r[6]*fUpwind_r[6]+alpha_r[5]*fUpwind_r[5]+alpha_r[4]*fUpwind_r[4]+alpha_r[3]*fUpwind_r[3]+alpha_r[2]*fUpwind_r[2]+alpha_r[1]*fUpwind_r[1]+alpha_r[0]*fUpwind_r[0]); 
  Ghat_r[1] = 0.1767766952966368*(alpha_r[30]*fUpwind_r[31]+fUpwind_r[30]*alpha_r[31]+alpha_r[25]*fUpwind_r[29]+fUpwind_r[25]*alpha_r[29]+alpha_r[24]*fUpwind_r[28]+fUpwind_r[24]*alpha_r[28]+alpha_r[22]*fUpwind_r[27]+fUpwind_r[22]*alpha_r[27]+alpha_r[19]*fUpwind_r[26]+fUpwind_r[19]*alpha_r[26]+alpha_r[15]*fUpwind_r[23]+fUpwind_r[15]*alpha_r[23]+alpha_r[14]*fUpwind_r[21]+fUpwind_r[14]*alpha_r[21]+alpha_r[13]*fUpwind_r[20]+fUpwind_r[13]*alpha_r[20]+alpha_r[11]*fUpwind_r[18]+fUpwind_r[11]*alpha_r[18]+alpha_r[10]*fUpwind_r[17]+fUpwind_r[10]*alpha_r[17]+alpha_r[8]*fUpwind_r[16]+fUpwind_r[8]*alpha_r[16]+alpha_r[5]*fUpwind_r[12]+fUpwind_r[5]*alpha_r[12]+alpha_r[4]*fUpwind_r[9]+fUpwind_r[4]*alpha_r[9]+alpha_r[3]*fUpwind_r[7]+fUpwind_r[3]*alpha_r[7]+alpha_r[2]*fUpwind_r[6]+fUpwind_r[2]*alpha_r[6]+alpha_r[0]*fUpwind_r[1]+fUpwind_r[0]*alpha_r[1]); 
  Ghat_r[2] = 0.1767766952966368*(alpha_r[29]*fUpwind_r[31]+fUpwind_r[29]*alpha_r[31]+alpha_r[25]*fUpwind_r[30]+fUpwind_r[25]*alpha_r[30]+alpha_r[23]*fUpwind_r[28]+fUpwind_r[23]*alpha_r[28]+alpha_r[21]*fUpwind_r[27]+fUpwind_r[21]*alpha_r[27]+alpha_r[18]*fUpwind_r[26]+fUpwind_r[18]*alpha_r[26]+alpha_r[15]*fUpwind_r[24]+fUpwind_r[15]*alpha_r[24]+alpha_r[14]*fUpwind_r[22]+fUpwind_r[14]*alpha_r[22]+alpha_r[12]*fUpwind_r[20]+fUpwind_r[12]*alpha_r[20]+alpha_r[11]*fUpwind_r[19]+fUpwind_r[11]*alpha_r[19]+alpha_r[9]*fUpwind_r[17]+fUpwind_r[9]*alpha_r[17]+alpha_r[7]*fUpwind_r[16]+fUpwind_r[7]*alpha_r[16]+alpha_r[5]*fUpwind_r[13]+fUpwind_r[5]*alpha_r[13]+alpha_r[4]*fUpwind_r[10]+fUpwind_r[4]*alpha_r[10]+alpha_r[3]*fUpwind_r[8]+fUpwind_r[3]*alpha_r[8]+alpha_r[1]*fUpwind_r[6]+fUpwind_r[1]*alpha_r[6]+alpha_r[0]*fUpwind_r[2]+fUpwind_r[0]*alpha_r[2]); 
  Ghat_r[3] = 0.1767766952966368*(alpha_r[28]*fUpwind_r[31]+fUpwind_r[28]*alpha_r[31]+alpha_r[24]*fUpwind_r[30]+fUpwind_r[24]*alpha_r[30]+alpha_r[23]*fUpwind_r[29]+fUpwind_r[23]*alpha_r[29]+alpha_r[20]*fUpwind_r[27]+fUpwind_r[20]*alpha_r[27]+alpha_r[17]*fUpwind_r[26]+fUpwind_r[17]*alpha_r[26]+alpha_r[15]*fUpwind_r[25]+fUpwind_r[15]*alpha_r[25]+alpha_r[13]*fUpwind_r[22]+fUpwind_r[13]*alpha_r[22]+alpha_r[12]*fUpwind_r[21]+fUpwind_r[12]*alpha_r[21]+alpha_r[10]*fUpwind_r[19]+fUpwind_r[10]*alpha_r[19]+alpha_r[9]*fUpwind_r[18]+fUpwind_r[9]*alpha_r[18]+alpha_r[6]*fUpwind_r[16]+fUpwind_r[6]*alpha_r[16]+alpha_r[5]*fUpwind_r[14]+fUpwind_r[5]*alpha_r[14]+alpha_r[4]*fUpwind_r[11]+fUpwind_r[4]*alpha_r[11]+alpha_r[2]*fUpwind_r[8]+fUpwind_r[2]*alpha_r[8]+alpha_r[1]*fUpwind_r[7]+fUpwind_r[1]*alpha_r[7]+alpha_r[0]*fUpwind_r[3]+fUpwind_r[0]*alpha_r[3]); 
  Ghat_r[4] = 0.1767766952966368*(alpha_r[27]*fUpwind_r[31]+fUpwind_r[27]*alpha_r[31]+alpha_r[22]*fUpwind_r[30]+fUpwind_r[22]*alpha_r[30]+alpha_r[21]*fUpwind_r[29]+fUpwind_r[21]*alpha_r[29]+alpha_r[20]*fUpwind_r[28]+fUpwind_r[20]*alpha_r[28]+alpha_r[16]*fUpwind_r[26]+fUpwind_r[16]*alpha_r[26]+alpha_r[14]*fUpwind_r[25]+fUpwind_r[14]*alpha_r[25]+alpha_r[13]*fUpwind_r[24]+fUpwind_r[13]*alpha_r[24]+alpha_r[12]*fUpwind_r[23]+fUpwind_r[12]*alpha_r[23]+alpha_r[8]*fUpwind_r[19]+fUpwind_r[8]*alpha_r[19]+alpha_r[7]*fUpwind_r[18]+fUpwind_r[7]*alpha_r[18]+alpha_r[6]*fUpwind_r[17]+fUpwind_r[6]*alpha_r[17]+alpha_r[5]*fUpwind_r[15]+fUpwind_r[5]*alpha_r[15]+alpha_r[3]*fUpwind_r[11]+fUpwind_r[3]*alpha_r[11]+alpha_r[2]*fUpwind_r[10]+fUpwind_r[2]*alpha_r[10]+alpha_r[1]*fUpwind_r[9]+fUpwind_r[1]*alpha_r[9]+alpha_r[0]*fUpwind_r[4]+fUpwind_r[0]*alpha_r[4]); 
  Ghat_r[5] = 0.1767766952966368*(alpha_r[26]*fUpwind_r[31]+fUpwind_r[26]*alpha_r[31]+alpha_r[19]*fUpwind_r[30]+fUpwind_r[19]*alpha_r[30]+alpha_r[18]*fUpwind_r[29]+fUpwind_r[18]*alpha_r[29]+alpha_r[17]*fUpwind_r[28]+fUpwind_r[17]*alpha_r[28]+alpha_r[16]*fUpwind_r[27]+fUpwind_r[16]*alpha_r[27]+alpha_r[11]*fUpwind_r[25]+fUpwind_r[11]*alpha_r[25]+alpha_r[10]*fUpwind_r[24]+fUpwind_r[10]*alpha_r[24]+alpha_r[9]*fUpwind_r[23]+fUpwind_r[9]*alpha_r[23]+alpha_r[8]*fUpwind_r[22]+fUpwind_r[8]*alpha_r[22]+alpha_r[7]*fUpwind_r[21]+fUpwind_r[7]*alpha_r[21]+alpha_r[6]*fUpwind_r[20]+fUpwind_r[6]*alpha_r[20]+alpha_r[4]*fUpwind_r[15]+fUpwind_r[4]*alpha_r[15]+alpha_r[3]*fUpwind_r[14]+fUpwind_r[3]*alpha_r[14]+alpha_r[2]*fUpwind_r[13]+fUpwind_r[2]*alpha_r[13]+alpha_r[1]*fUpwind_r[12]+fUpwind_r[1]*alpha_r[12]+alpha_r[0]*fUpwind_r[5]+fUpwind_r[0]*alpha_r[5]); 
  Ghat_r[6] = 0.1767766952966368*(alpha_r[25]*fUpwind_r[31]+fUpwind_r[25]*alpha_r[31]+alpha_r[29]*fUpwind_r[30]+fUpwind_r[29]*alpha_r[30]+alpha_r[15]*fUpwind_r[28]+fUpwind_r[15]*alpha_r[28]+alpha_r[14]*fUpwind_r[27]+fUpwind_r[14]*alpha_r[27]+alpha_r[11]*fUpwind_r[26]+fUpwind_r[11]*alpha_r[26]+alpha_r[23]*fUpwind_r[24]+fUpwind_r[23]*alpha_r[24]+alpha_r[21]*fUpwind_r[22]+fUpwind_r[21]*alpha_r[22]+alpha_r[5]*fUpwind_r[20]+fUpwind_r[5]*alpha_r[20]+alpha_r[18]*fUpwind_r[19]+fUpwind_r[18]*alpha_r[19]+alpha_r[4]*fUpwind_r[17]+fUpwind_r[4]*alpha_r[17]+alpha_r[3]*fUpwind_r[16]+fUpwind_r[3]*alpha_r[16]+alpha_r[12]*fUpwind_r[13]+fUpwind_r[12]*alpha_r[13]+alpha_r[9]*fUpwind_r[10]+fUpwind_r[9]*alpha_r[10]+alpha_r[7]*fUpwind_r[8]+fUpwind_r[7]*alpha_r[8]+alpha_r[0]*fUpwind_r[6]+fUpwind_r[0]*alpha_r[6]+alpha_r[1]*fUpwind_r[2]+fUpwind_r[1]*alpha_r[2]); 
  Ghat_r[7] = 0.1767766952966368*(alpha_r[24]*fUpwind_r[31]+fUpwind_r[24]*alpha_r[31]+alpha_r[28]*fUpwind_r[30]+fUpwind_r[28]*alpha_r[30]+alpha_r[15]*fUpwind_r[29]+fUpwind_r[15]*alpha_r[29]+alpha_r[13]*fUpwind_r[27]+fUpwind_r[13]*alpha_r[27]+alpha_r[10]*fUpwind_r[26]+fUpwind_r[10]*alpha_r[26]+alpha_r[23]*fUpwind_r[25]+fUpwind_r[23]*alpha_r[25]+alpha_r[20]*fUpwind_r[22]+fUpwind_r[20]*alpha_r[22]+alpha_r[5]*fUpwind_r[21]+fUpwind_r[5]*alpha_r[21]+alpha_r[17]*fUpwind_r[19]+fUpwind_r[17]*alpha_r[19]+alpha_r[4]*fUpwind_r[18]+fUpwind_r[4]*alpha_r[18]+alpha_r[2]*fUpwind_r[16]+fUpwind_r[2]*alpha_r[16]+alpha_r[12]*fUpwind_r[14]+fUpwind_r[12]*alpha_r[14]+alpha_r[9]*fUpwind_r[11]+fUpwind_r[9]*alpha_r[11]+alpha_r[6]*fUpwind_r[8]+fUpwind_r[6]*alpha_r[8]+alpha_r[0]*fUpwind_r[7]+fUpwind_r[0]*alpha_r[7]+alpha_r[1]*fUpwind_r[3]+fUpwind_r[1]*alpha_r[3]); 
  Ghat_r[8] = 0.1767766952966368*(alpha_r[23]*fUpwind_r[31]+fUpwind_r[23]*alpha_r[31]+alpha_r[15]*fUpwind_r[30]+fUpwind_r[15]*alpha_r[30]+alpha_r[28]*fUpwind_r[29]+fUpwind_r[28]*alpha_r[29]+alpha_r[12]*fUpwind_r[27]+fUpwind_r[12]*alpha_r[27]+alpha_r[9]*fUpwind_r[26]+fUpwind_r[9]*alpha_r[26]+alpha_r[24]*fUpwind_r[25]+fUpwind_r[24]*alpha_r[25]+alpha_r[5]*fUpwind_r[22]+fUpwind_r[5]*alpha_r[22]+alpha_r[20]*fUpwind_r[21]+fUpwind_r[20]*alpha_r[21]+alpha_r[4]*fUpwind_r[19]+fUpwind_r[4]*alpha_r[19]+alpha_r[17]*fUpwind_r[18]+fUpwind_r[17]*alpha_r[18]+alpha_r[1]*fUpwind_r[16]+fUpwind_r[1]*alpha_r[16]+alpha_r[13]*fUpwind_r[14]+fUpwind_r[13]*alpha_r[14]+alpha_r[10]*fUpwind_r[11]+fUpwind_r[10]*alpha_r[11]+alpha_r[0]*fUpwind_r[8]+fUpwind_r[0]*alpha_r[8]+alpha_r[6]*fUpwind_r[7]+fUpwind_r[6]*alpha_r[7]+alpha_r[2]*fUpwind_r[3]+fUpwind_r[2]*alpha_r[3]); 
  Ghat_r[9] = 0.1767766952966368*(alpha_r[22]*fUpwind_r[31]+fUpwind_r[22]*alpha_r[31]+alpha_r[27]*fUpwind_r[30]+fUpwind_r[27]*alpha_r[30]+alpha_r[14]*fUpwind_r[29]+fUpwind_r[14]*alpha_r[29]+alpha_r[13]*fUpwind_r[28]+fUpwind_r[13]*alpha_r[28]+alpha_r[8]*fUpwind_r[26]+fUpwind_r[8]*alpha_r[26]+alpha_r[21]*fUpwind_r[25]+fUpwind_r[21]*alpha_r[25]+alpha_r[20]*fUpwind_r[24]+fUpwind_r[20]*alpha_r[24]+alpha_r[5]*fUpwind_r[23]+fUpwind_r[5]*alpha_r[23]+alpha_r[16]*fUpwind_r[19]+fUpwind_r[16]*alpha_r[19]+alpha_r[3]*fUpwind_r[18]+fUpwind_r[3]*alpha_r[18]+alpha_r[2]*fUpwind_r[17]+fUpwind_r[2]*alpha_r[17]+alpha_r[12]*fUpwind_r[15]+fUpwind_r[12]*alpha_r[15]+alpha_r[7]*fUpwind_r[11]+fUpwind_r[7]*alpha_r[11]+alpha_r[6]*fUpwind_r[10]+fUpwind_r[6]*alpha_r[10]+alpha_r[0]*fUpwind_r[9]+fUpwind_r[0]*alpha_r[9]+alpha_r[1]*fUpwind_r[4]+fUpwind_r[1]*alpha_r[4]); 
  Ghat_r[10] = 0.1767766952966368*(alpha_r[21]*fUpwind_r[31]+fUpwind_r[21]*alpha_r[31]+alpha_r[14]*fUpwind_r[30]+fUpwind_r[14]*alpha_r[30]+alpha_r[27]*fUpwind_r[29]+fUpwind_r[27]*alpha_r[29]+alpha_r[12]*fUpwind_r[28]+fUpwind_r[12]*alpha_r[28]+alpha_r[7]*fUpwind_r[26]+fUpwind_r[7]*alpha_r[26]+alpha_r[22]*fUpwind_r[25]+fUpwind_r[22]*alpha_r[25]+alpha_r[5]*fUpwind_r[24]+fUpwind_r[5]*alpha_r[24]+alpha_r[20]*fUpwind_r[23]+fUpwind_r[20]*alpha_r[23]+alpha_r[3]*fUpwind_r[19]+fUpwind_r[3]*alpha_r[19]+alpha_r[16]*fUpwind_r[18]+fUpwind_r[16]*alpha_r[18]+alpha_r[1]*fUpwind_r[17]+fUpwind_r[1]*alpha_r[17]+alpha_r[13]*fUpwind_r[15]+fUpwind_r[13]*alpha_r[15]+alpha_r[8]*fUpwind_r[11]+fUpwind_r[8]*alpha_r[11]+alpha_r[0]*fUpwind_r[10]+fUpwind_r[0]*alpha_r[10]+alpha_r[6]*fUpwind_r[9]+fUpwind_r[6]*alpha_r[9]+alpha_r[2]*fUpwind_r[4]+fUpwind_r[2]*alpha_r[4]); 
  Ghat_r[11] = 0.1767766952966368*(alpha_r[20]*fUpwind_r[31]+fUpwind_r[20]*alpha_r[31]+alpha_r[13]*fUpwind_r[30]+fUpwind_r[13]*alpha_r[30]+alpha_r[12]*fUpwind_r[29]+fUpwind_r[12]*alpha_r[29]+alpha_r[27]*fUpwind_r[28]+fUpwind_r[27]*alpha_r[28]+alpha_r[6]*fUpwind_r[26]+fUpwind_r[6]*alpha_r[26]+alpha_r[5]*fUpwind_r[25]+fUpwind_r[5]*alpha_r[25]+alpha_r[22]*fUpwind_r[24]+fUpwind_r[22]*alpha_r[24]+alpha_r[21]*fUpwind_r[23]+fUpwind_r[21]*alpha_r[23]+alpha_r[2]*fUpwind_r[19]+fUpwind_r[2]*alpha_r[19]+alpha_r[1]*fUpwind_r[18]+fUpwind_r[1]*alpha_r[18]+alpha_r[16]*fUpwind_r[17]+fUpwind_r[16]*alpha_r[17]+alpha_r[14]*fUpwind_r[15]+fUpwind_r[14]*alpha_r[15]+alpha_r[0]*fUpwind_r[11]+fUpwind_r[0]*alpha_r[11]+alpha_r[8]*fUpwind_r[10]+fUpwind_r[8]*alpha_r[10]+alpha_r[7]*fUpwind_r[9]+fUpwind_r[7]*alpha_r[9]+alpha_r[3]*fUpwind_r[4]+fUpwind_r[3]*alpha_r[4]); 
  Ghat_r[12] = 0.1767766952966368*(alpha_r[19]*fUpwind_r[31]+fUpwind_r[19]*alpha_r[31]+alpha_r[26]*fUpwind_r[30]+fUpwind_r[26]*alpha_r[30]+alpha_r[11]*fUpwind_r[29]+fUpwind_r[11]*alpha_r[29]+alpha_r[10]*fUpwind_r[28]+fUpwind_r[10]*alpha_r[28]+alpha_r[8]*fUpwind_r[27]+fUpwind_r[8]*alpha_r[27]+alpha_r[18]*fUpwind_r[25]+fUpwind_r[18]*alpha_r[25]+alpha_r[17]*fUpwind_r[24]+fUpwind_r[17]*alpha_r[24]+alpha_r[4]*fUpwind_r[23]+fUpwind_r[4]*alpha_r[23]+alpha_r[16]*fUpwind_r[22]+fUpwind_r[16]*alpha_r[22]+alpha_r[3]*fUpwind_r[21]+fUpwind_r[3]*alpha_r[21]+alpha_r[2]*fUpwind_r[20]+fUpwind_r[2]*alpha_r[20]+alpha_r[9]*fUpwind_r[15]+fUpwind_r[9]*alpha_r[15]+alpha_r[7]*fUpwind_r[14]+fUpwind_r[7]*alpha_r[14]+alpha_r[6]*fUpwind_r[13]+fUpwind_r[6]*alpha_r[13]+alpha_r[0]*fUpwind_r[12]+fUpwind_r[0]*alpha_r[12]+alpha_r[1]*fUpwind_r[5]+fUpwind_r[1]*alpha_r[5]); 
  Ghat_r[13] = 0.1767766952966368*(alpha_r[18]*fUpwind_r[31]+fUpwind_r[18]*alpha_r[31]+alpha_r[11]*fUpwind_r[30]+fUpwind_r[11]*alpha_r[30]+alpha_r[26]*fUpwind_r[29]+fUpwind_r[26]*alpha_r[29]+alpha_r[9]*fUpwind_r[28]+fUpwind_r[9]*alpha_r[28]+alpha_r[7]*fUpwind_r[27]+fUpwind_r[7]*alpha_r[27]+alpha_r[19]*fUpwind_r[25]+fUpwind_r[19]*alpha_r[25]+alpha_r[4]*fUpwind_r[24]+fUpwind_r[4]*alpha_r[24]+alpha_r[17]*fUpwind_r[23]+fUpwind_r[17]*alpha_r[23]+alpha_r[3]*fUpwind_r[22]+fUpwind_r[3]*alpha_r[22]+alpha_r[16]*fUpwind_r[21]+fUpwind_r[16]*alpha_r[21]+alpha_r[1]*fUpwind_r[20]+fUpwind_r[1]*alpha_r[20]+alpha_r[10]*fUpwind_r[15]+fUpwind_r[10]*alpha_r[15]+alpha_r[8]*fUpwind_r[14]+fUpwind_r[8]*alpha_r[14]+alpha_r[0]*fUpwind_r[13]+fUpwind_r[0]*alpha_r[13]+alpha_r[6]*fUpwind_r[12]+fUpwind_r[6]*alpha_r[12]+alpha_r[2]*fUpwind_r[5]+fUpwind_r[2]*alpha_r[5]); 
  Ghat_r[14] = 0.1767766952966368*(alpha_r[17]*fUpwind_r[31]+fUpwind_r[17]*alpha_r[31]+alpha_r[10]*fUpwind_r[30]+fUpwind_r[10]*alpha_r[30]+alpha_r[9]*fUpwind_r[29]+fUpwind_r[9]*alpha_r[29]+alpha_r[26]*fUpwind_r[28]+fUpwind_r[26]*alpha_r[28]+alpha_r[6]*fUpwind_r[27]+fUpwind_r[6]*alpha_r[27]+alpha_r[4]*fUpwind_r[25]+fUpwind_r[4]*alpha_r[25]+alpha_r[19]*fUpwind_r[24]+fUpwind_r[19]*alpha_r[24]+alpha_r[18]*fUpwind_r[23]+fUpwind_r[18]*alpha_r[23]+alpha_r[2]*fUpwind_r[22]+fUpwind_r[2]*alpha_r[22]+alpha_r[1]*fUpwind_r[21]+fUpwind_r[1]*alpha_r[21]+alpha_r[16]*fUpwind_r[20]+fUpwind_r[16]*alpha_r[20]+alpha_r[11]*fUpwind_r[15]+fUpwind_r[11]*alpha_r[15]+alpha_r[0]*fUpwind_r[14]+fUpwind_r[0]*alpha_r[14]+alpha_r[8]*fUpwind_r[13]+fUpwind_r[8]*alpha_r[13]+alpha_r[7]*fUpwind_r[12]+fUpwind_r[7]*alpha_r[12]+alpha_r[3]*fUpwind_r[5]+fUpwind_r[3]*alpha_r[5]); 
  Ghat_r[15] = 0.1767766952966368*(alpha_r[16]*fUpwind_r[31]+fUpwind_r[16]*alpha_r[31]+alpha_r[8]*fUpwind_r[30]+fUpwind_r[8]*alpha_r[30]+alpha_r[7]*fUpwind_r[29]+fUpwind_r[7]*alpha_r[29]+alpha_r[6]*fUpwind_r[28]+fUpwind_r[6]*alpha_r[28]+alpha_r[26]*fUpwind_r[27]+fUpwind_r[26]*alpha_r[27]+alpha_r[3]*fUpwind_r[25]+fUpwind_r[3]*alpha_r[25]+alpha_r[2]*fUpwind_r[24]+fUpwind_r[2]*alpha_r[24]+alpha_r[1]*fUpwind_r[23]+fUpwind_r[1]*alpha_r[23]+alpha_r[19]*fUpwind_r[22]+fUpwind_r[19]*alpha_r[22]+alpha_r[18]*fUpwind_r[21]+fUpwind_r[18]*alpha_r[21]+alpha_r[17]*fUpwind_r[20]+fUpwind_r[17]*alpha_r[20]+alpha_r[0]*fUpwind_r[15]+fUpwind_r[0]*alpha_r[15]+alpha_r[11]*fUpwind_r[14]+fUpwind_r[11]*alpha_r[14]+alpha_r[10]*fUpwind_r[13]+fUpwind_r[10]*alpha_r[13]+alpha_r[9]*fUpwind_r[12]+fUpwind_r[9]*alpha_r[12]+alpha_r[4]*fUpwind_r[5]+fUpwind_r[4]*alpha_r[5]); 
  Ghat_r[16] = 0.1767766952966368*(alpha_r[15]*fUpwind_r[31]+fUpwind_r[15]*alpha_r[31]+alpha_r[23]*fUpwind_r[30]+fUpwind_r[23]*alpha_r[30]+alpha_r[24]*fUpwind_r[29]+fUpwind_r[24]*alpha_r[29]+alpha_r[25]*fUpwind_r[28]+fUpwind_r[25]*alpha_r[28]+alpha_r[5]*fUpwind_r[27]+fUpwind_r[5]*alpha_r[27]+alpha_r[4]*fUpwind_r[26]+fUpwind_r[4]*alpha_r[26]+alpha_r[12]*fUpwind_r[22]+fUpwind_r[12]*alpha_r[22]+alpha_r[13]*fUpwind_r[21]+fUpwind_r[13]*alpha_r[21]+alpha_r[14]*fUpwind_r[20]+fUpwind_r[14]*alpha_r[20]+alpha_r[9]*fUpwind_r[19]+fUpwind_r[9]*alpha_r[19]+alpha_r[10]*fUpwind_r[18]+fUpwind_r[10]*alpha_r[18]+alpha_r[11]*fUpwind_r[17]+fUpwind_r[11]*alpha_r[17]+alpha_r[0]*fUpwind_r[16]+fUpwind_r[0]*alpha_r[16]+alpha_r[1]*fUpwind_r[8]+fUpwind_r[1]*alpha_r[8]+alpha_r[2]*fUpwind_r[7]+fUpwind_r[2]*alpha_r[7]+alpha_r[3]*fUpwind_r[6]+fUpwind_r[3]*alpha_r[6]); 
  Ghat_r[17] = 0.1767766952966368*(alpha_r[14]*fUpwind_r[31]+fUpwind_r[14]*alpha_r[31]+alpha_r[21]*fUpwind_r[30]+fUpwind_r[21]*alpha_r[30]+alpha_r[22]*fUpwind_r[29]+fUpwind_r[22]*alpha_r[29]+alpha_r[5]*fUpwind_r[28]+fUpwind_r[5]*alpha_r[28]+alpha_r[25]*fUpwind_r[27]+fUpwind_r[25]*alpha_r[27]+alpha_r[3]*fUpwind_r[26]+fUpwind_r[3]*alpha_r[26]+alpha_r[12]*fUpwind_r[24]+fUpwind_r[12]*alpha_r[24]+alpha_r[13]*fUpwind_r[23]+fUpwind_r[13]*alpha_r[23]+alpha_r[15]*fUpwind_r[20]+fUpwind_r[15]*alpha_r[20]+alpha_r[7]*fUpwind_r[19]+fUpwind_r[7]*alpha_r[19]+alpha_r[8]*fUpwind_r[18]+fUpwind_r[8]*alpha_r[18]+alpha_r[0]*fUpwind_r[17]+fUpwind_r[0]*alpha_r[17]+alpha_r[11]*fUpwind_r[16]+fUpwind_r[11]*alpha_r[16]+alpha_r[1]*fUpwind_r[10]+fUpwind_r[1]*alpha_r[10]+alpha_r[2]*fUpwind_r[9]+fUpwind_r[2]*alpha_r[9]+alpha_r[4]*fUpwind_r[6]+fUpwind_r[4]*alpha_r[6]); 
  Ghat_r[18] = 0.1767766952966368*(alpha_r[13]*fUpwind_r[31]+fUpwind_r[13]*alpha_r[31]+alpha_r[20]*fUpwind_r[30]+fUpwind_r[20]*alpha_r[30]+alpha_r[5]*fUpwind_r[29]+fUpwind_r[5]*alpha_r[29]+alpha_r[22]*fUpwind_r[28]+fUpwind_r[22]*alpha_r[28]+alpha_r[24]*fUpwind_r[27]+fUpwind_r[24]*alpha_r[27]+alpha_r[2]*fUpwind_r[26]+fUpwind_r[2]*alpha_r[26]+alpha_r[12]*fUpwind_r[25]+fUpwind_r[12]*alpha_r[25]+alpha_r[14]*fUpwind_r[23]+fUpwind_r[14]*alpha_r[23]+alpha_r[15]*fUpwind_r[21]+fUpwind_r[15]*alpha_r[21]+alpha_r[6]*fUpwind_r[19]+fUpwind_r[6]*alpha_r[19]+alpha_r[0]*fUpwind_r[18]+fUpwind_r[0]*alpha_r[18]+alpha_r[8]*fUpwind_r[17]+fUpwind_r[8]*alpha_r[17]+alpha_r[10]*fUpwind_r[16]+fUpwind_r[10]*alpha_r[16]+alpha_r[1]*fUpwind_r[11]+fUpwind_r[1]*alpha_r[11]+alpha_r[3]*fUpwind_r[9]+fUpwind_r[3]*alpha_r[9]+alpha_r[4]*fUpwind_r[7]+fUpwind_r[4]*alpha_r[7]); 
  Ghat_r[19] = 0.1767766952966368*(alpha_r[12]*fUpwind_r[31]+fUpwind_r[12]*alpha_r[31]+alpha_r[5]*fUpwind_r[30]+fUpwind_r[5]*alpha_r[30]+alpha_r[20]*fUpwind_r[29]+fUpwind_r[20]*alpha_r[29]+alpha_r[21]*fUpwind_r[28]+fUpwind_r[21]*alpha_r[28]+alpha_r[23]*fUpwind_r[27]+fUpwind_r[23]*alpha_r[27]+alpha_r[1]*fUpwind_r[26]+fUpwind_r[1]*alpha_r[26]+alpha_r[13]*fUpwind_r[25]+fUpwind_r[13]*alpha_r[25]+alpha_r[14]*fUpwind_r[24]+fUpwind_r[14]*alpha_r[24]+alpha_r[15]*fUpwind_r[22]+fUpwind_r[15]*alpha_r[22]+alpha_r[0]*fUpwind_r[19]+fUpwind_r[0]*alpha_r[19]+alpha_r[6]*fUpwind_r[18]+fUpwind_r[6]*alpha_r[18]+alpha_r[7]*fUpwind_r[17]+fUpwind_r[7]*alpha_r[17]+alpha_r[9]*fUpwind_r[16]+fUpwind_r[9]*alpha_r[16]+alpha_r[2]*fUpwind_r[11]+fUpwind_r[2]*alpha_r[11]+alpha_r[3]*fUpwind_r[10]+fUpwind_r[3]*alpha_r[10]+alpha_r[4]*fUpwind_r[8]+fUpwind_r[4]*alpha_r[8]); 
  Ghat_r[20] = 0.1767766952966368*(alpha_r[11]*fUpwind_r[31]+fUpwind_r[11]*alpha_r[31]+alpha_r[18]*fUpwind_r[30]+fUpwind_r[18]*alpha_r[30]+alpha_r[19]*fUpwind_r[29]+fUpwind_r[19]*alpha_r[29]+alpha_r[4]*fUpwind_r[28]+fUpwind_r[4]*alpha_r[28]+alpha_r[3]*fUpwind_r[27]+fUpwind_r[3]*alpha_r[27]+alpha_r[25]*fUpwind_r[26]+fUpwind_r[25]*alpha_r[26]+alpha_r[9]*fUpwind_r[24]+fUpwind_r[9]*alpha_r[24]+alpha_r[10]*fUpwind_r[23]+fUpwind_r[10]*alpha_r[23]+alpha_r[7]*fUpwind_r[22]+fUpwind_r[7]*alpha_r[22]+alpha_r[8]*fUpwind_r[21]+fUpwind_r[8]*alpha_r[21]+alpha_r[0]*fUpwind_r[20]+fUpwind_r[0]*alpha_r[20]+alpha_r[15]*fUpwind_r[17]+fUpwind_r[15]*alpha_r[17]+alpha_r[14]*fUpwind_r[16]+fUpwind_r[14]*alpha_r[16]+alpha_r[1]*fUpwind_r[13]+fUpwind_r[1]*alpha_r[13]+alpha_r[2]*fUpwind_r[12]+fUpwind_r[2]*alpha_r[12]+alpha_r[5]*fUpwind_r[6]+fUpwind_r[5]*alpha_r[6]); 
  Ghat_r[21] = 0.1767766952966368*(alpha_r[10]*fUpwind_r[31]+fUpwind_r[10]*alpha_r[31]+alpha_r[17]*fUpwind_r[30]+fUpwind_r[17]*alpha_r[30]+alpha_r[4]*fUpwind_r[29]+fUpwind_r[4]*alpha_r[29]+alpha_r[19]*fUpwind_r[28]+fUpwind_r[19]*alpha_r[28]+alpha_r[2]*fUpwind_r[27]+fUpwind_r[2]*alpha_r[27]+alpha_r[24]*fUpwind_r[26]+fUpwind_r[24]*alpha_r[26]+alpha_r[9]*fUpwind_r[25]+fUpwind_r[9]*alpha_r[25]+alpha_r[11]*fUpwind_r[23]+fUpwind_r[11]*alpha_r[23]+alpha_r[6]*fUpwind_r[22]+fUpwind_r[6]*alpha_r[22]+alpha_r[0]*fUpwind_r[21]+fUpwind_r[0]*alpha_r[21]+alpha_r[8]*fUpwind_r[20]+fUpwind_r[8]*alpha_r[20]+alpha_r[15]*fUpwind_r[18]+fUpwind_r[15]*alpha_r[18]+alpha_r[13]*fUpwind_r[16]+fUpwind_r[13]*alpha_r[16]+alpha_r[1]*fUpwind_r[14]+fUpwind_r[1]*alpha_r[14]+alpha_r[3]*fUpwind_r[12]+fUpwind_r[3]*alpha_r[12]+alpha_r[5]*fUpwind_r[7]+fUpwind_r[5]*alpha_r[7]); 
  Ghat_r[22] = 0.1767766952966368*(alpha_r[9]*fUpwind_r[31]+fUpwind_r[9]*alpha_r[31]+alpha_r[4]*fUpwind_r[30]+fUpwind_r[4]*alpha_r[30]+alpha_r[17]*fUpwind_r[29]+fUpwind_r[17]*alpha_r[29]+alpha_r[18]*fUpwind_r[28]+fUpwind_r[18]*alpha_r[28]+alpha_r[1]*fUpwind_r[27]+fUpwind_r[1]*alpha_r[27]+alpha_r[23]*fUpwind_r[26]+fUpwind_r[23]*alpha_r[26]+alpha_r[10]*fUpwind_r[25]+fUpwind_r[10]*alpha_r[25]+alpha_r[11]*fUpwind_r[24]+fUpwind_r[11]*alpha_r[24]+alpha_r[0]*fUpwind_r[22]+fUpwind_r[0]*alpha_r[22]+alpha_r[6]*fUpwind_r[21]+fUpwind_r[6]*alpha_r[21]+alpha_r[7]*fUpwind_r[20]+fUpwind_r[7]*alpha_r[20]+alpha_r[15]*fUpwind_r[19]+fUpwind_r[15]*alpha_r[19]+alpha_r[12]*fUpwind_r[16]+fUpwind_r[12]*alpha_r[16]+alpha_r[2]*fUpwind_r[14]+fUpwind_r[2]*alpha_r[14]+alpha_r[3]*fUpwind_r[13]+fUpwind_r[3]*alpha_r[13]+alpha_r[5]*fUpwind_r[8]+fUpwind_r[5]*alpha_r[8]); 
  Ghat_r[23] = 0.1767766952966368*(alpha_r[8]*fUpwind_r[31]+fUpwind_r[8]*alpha_r[31]+alpha_r[16]*fUpwind_r[30]+fUpwind_r[16]*alpha_r[30]+alpha_r[3]*fUpwind_r[29]+fUpwind_r[3]*alpha_r[29]+alpha_r[2]*fUpwind_r[28]+fUpwind_r[2]*alpha_r[28]+alpha_r[19]*fUpwind_r[27]+fUpwind_r[19]*alpha_r[27]+alpha_r[22]*fUpwind_r[26]+fUpwind_r[22]*alpha_r[26]+alpha_r[7]*fUpwind_r[25]+fUpwind_r[7]*alpha_r[25]+alpha_r[6]*fUpwind_r[24]+fUpwind_r[6]*alpha_r[24]+alpha_r[0]*fUpwind_r[23]+fUpwind_r[0]*alpha_r[23]+alpha_r[11]*fUpwind_r[21]+fUpwind_r[11]*alpha_r[21]+alpha_r[10]*fUpwind_r[20]+fUpwind_r[10]*alpha_r[20]+alpha_r[14]*fUpwind_r[18]+fUpwind_r[14]*alpha_r[18]+alpha_r[13]*fUpwind_r[17]+fUpwind_r[13]*alpha_r[17]+alpha_r[1]*fUpwind_r[15]+fUpwind_r[1]*alpha_r[15]+alpha_r[4]*fUpwind_r[12]+fUpwind_r[4]*alpha_r[12]+alpha_r[5]*fUpwind_r[9]+fUpwind_r[5]*alpha_r[9]); 
  Ghat_r[24] = 0.1767766952966368*(alpha_r[7]*fUpwind_r[31]+fUpwind_r[7]*alpha_r[31]+alpha_r[3]*fUpwind_r[30]+fUpwind_r[3]*alpha_r[30]+alpha_r[16]*fUpwind_r[29]+fUpwind_r[16]*alpha_r[29]+alpha_r[1]*fUpwind_r[28]+fUpwind_r[1]*alpha_r[28]+alpha_r[18]*fUpwind_r[27]+fUpwind_r[18]*alpha_r[27]+alpha_r[21]*fUpwind_r[26]+fUpwind_r[21]*alpha_r[26]+alpha_r[8]*fUpwind_r[25]+fUpwind_r[8]*alpha_r[25]+alpha_r[0]*fUpwind_r[24]+fUpwind_r[0]*alpha_r[24]+alpha_r[6]*fUpwind_r[23]+fUpwind_r[6]*alpha_r[23]+alpha_r[11]*fUpwind_r[22]+fUpwind_r[11]*alpha_r[22]+alpha_r[9]*fUpwind_r[20]+fUpwind_r[9]*alpha_r[20]+alpha_r[14]*fUpwind_r[19]+fUpwind_r[14]*alpha_r[19]+alpha_r[12]*fUpwind_r[17]+fUpwind_r[12]*alpha_r[17]+alpha_r[2]*fUpwind_r[15]+fUpwind_r[2]*alpha_r[15]+alpha_r[4]*fUpwind_r[13]+fUpwind_r[4]*alpha_r[13]+alpha_r[5]*fUpwind_r[10]+fUpwind_r[5]*alpha_r[10]); 
  Ghat_r[25] = 0.1767766952966368*(alpha_r[6]*fUpwind_r[31]+fUpwind_r[6]*alpha_r[31]+alpha_r[2]*fUpwind_r[30]+fUpwind_r[2]*alpha_r[30]+alpha_r[1]*fUpwind_r[29]+fUpwind_r[1]*alpha_r[29]+alpha_r[16]*fUpwind_r[28]+fUpwind_r[16]*alpha_r[28]+alpha_r[17]*fUpwind_r[27]+fUpwind_r[17]*alpha_r[27]+alpha_r[20]*fUpwind_r[26]+fUpwind_r[20]*alpha_r[26]+alpha_r[0]*fUpwind_r[25]+fUpwind_r[0]*alpha_r[25]+alpha_r[8]*fUpwind_r[24]+fUpwind_r[8]*alpha_r[24]+alpha_r[7]*fUpwind_r[23]+fUpwind_r[7]*alpha_r[23]+alpha_r[10]*fUpwind_r[22]+fUpwind_r[10]*alpha_r[22]+alpha_r[9]*fUpwind_r[21]+fUpwind_r[9]*alpha_r[21]+alpha_r[13]*fUpwind_r[19]+fUpwind_r[13]*alpha_r[19]+alpha_r[12]*fUpwind_r[18]+fUpwind_r[12]*alpha_r[18]+alpha_r[3]*fUpwind_r[15]+fUpwind_r[3]*alpha_r[15]+alpha_r[4]*fUpwind_r[14]+fUpwind_r[4]*alpha_r[14]+alpha_r[5]*fUpwind_r[11]+fUpwind_r[5]*alpha_r[11]); 
  Ghat_r[26] = 0.1767766952966368*(alpha_r[5]*fUpwind_r[31]+fUpwind_r[5]*alpha_r[31]+alpha_r[12]*fUpwind_r[30]+fUpwind_r[12]*alpha_r[30]+alpha_r[13]*fUpwind_r[29]+fUpwind_r[13]*alpha_r[29]+alpha_r[14]*fUpwind_r[28]+fUpwind_r[14]*alpha_r[28]+alpha_r[15]*fUpwind_r[27]+fUpwind_r[15]*alpha_r[27]+alpha_r[0]*fUpwind_r[26]+fUpwind_r[0]*alpha_r[26]+alpha_r[20]*fUpwind_r[25]+fUpwind_r[20]*alpha_r[25]+alpha_r[21]*fUpwind_r[24]+fUpwind_r[21]*alpha_r[24]+alpha_r[22]*fUpwind_r[23]+fUpwind_r[22]*alpha_r[23]+alpha_r[1]*fUpwind_r[19]+fUpwind_r[1]*alpha_r[19]+alpha_r[2]*fUpwind_r[18]+fUpwind_r[2]*alpha_r[18]+alpha_r[3]*fUpwind_r[17]+fUpwind_r[3]*alpha_r[17]+alpha_r[4]*fUpwind_r[16]+fUpwind_r[4]*alpha_r[16]+alpha_r[6]*fUpwind_r[11]+fUpwind_r[6]*alpha_r[11]+alpha_r[7]*fUpwind_r[10]+fUpwind_r[7]*alpha_r[10]+alpha_r[8]*fUpwind_r[9]+fUpwind_r[8]*alpha_r[9]); 
  Ghat_r[27] = 0.1767766952966368*(alpha_r[4]*fUpwind_r[31]+fUpwind_r[4]*alpha_r[31]+alpha_r[9]*fUpwind_r[30]+fUpwind_r[9]*alpha_r[30]+alpha_r[10]*fUpwind_r[29]+fUpwind_r[10]*alpha_r[29]+alpha_r[11]*fUpwind_r[28]+fUpwind_r[11]*alpha_r[28]+alpha_r[0]*fUpwind_r[27]+fUpwind_r[0]*alpha_r[27]+alpha_r[15]*fUpwind_r[26]+fUpwind_r[15]*alpha_r[26]+alpha_r[17]*fUpwind_r[25]+fUpwind_r[17]*alpha_r[25]+alpha_r[18]*fUpwind_r[24]+fUpwind_r[18]*alpha_r[24]+alpha_r[19]*fUpwind_r[23]+fUpwind_r[19]*alpha_r[23]+alpha_r[1]*fUpwind_r[22]+fUpwind_r[1]*alpha_r[22]+alpha_r[2]*fUpwind_r[21]+fUpwind_r[2]*alpha_r[21]+alpha_r[3]*fUpwind_r[20]+fUpwind_r[3]*alpha_r[20]+alpha_r[5]*fUpwind_r[16]+fUpwind_r[5]*alpha_r[16]+alpha_r[6]*fUpwind_r[14]+fUpwind_r[6]*alpha_r[14]+alpha_r[7]*fUpwind_r[13]+fUpwind_r[7]*alpha_r[13]+alpha_r[8]*fUpwind_r[12]+fUpwind_r[8]*alpha_r[12]); 
  Ghat_r[28] = 0.1767766952966368*(alpha_r[3]*fUpwind_r[31]+fUpwind_r[3]*alpha_r[31]+alpha_r[7]*fUpwind_r[30]+fUpwind_r[7]*alpha_r[30]+alpha_r[8]*fUpwind_r[29]+fUpwind_r[8]*alpha_r[29]+alpha_r[0]*fUpwind_r[28]+fUpwind_r[0]*alpha_r[28]+alpha_r[11]*fUpwind_r[27]+fUpwind_r[11]*alpha_r[27]+alpha_r[14]*fUpwind_r[26]+fUpwind_r[14]*alpha_r[26]+alpha_r[16]*fUpwind_r[25]+fUpwind_r[16]*alpha_r[25]+alpha_r[1]*fUpwind_r[24]+fUpwind_r[1]*alpha_r[24]+alpha_r[2]*fUpwind_r[23]+fUpwind_r[2]*alpha_r[23]+alpha_r[18]*fUpwind_r[22]+fUpwind_r[18]*alpha_r[22]+alpha_r[19]*fUpwind_r[21]+fUpwind_r[19]*alpha_r[21]+alpha_r[4]*fUpwind_r[20]+fUpwind_r[4]*alpha_r[20]+alpha_r[5]*fUpwind_r[17]+fUpwind_r[5]*alpha_r[17]+alpha_r[6]*fUpwind_r[15]+fUpwind_r[6]*alpha_r[15]+alpha_r[9]*fUpwind_r[13]+fUpwind_r[9]*alpha_r[13]+alpha_r[10]*fUpwind_r[12]+fUpwind_r[10]*alpha_r[12]); 
  Ghat_r[29] = 0.1767766952966368*(alpha_r[2]*fUpwind_r[31]+fUpwind_r[2]*alpha_r[31]+alpha_r[6]*fUpwind_r[30]+fUpwind_r[6]*alpha_r[30]+alpha_r[0]*fUpwind_r[29]+fUpwind_r[0]*alpha_r[29]+alpha_r[8]*fUpwind_r[28]+fUpwind_r[8]*alpha_r[28]+alpha_r[10]*fUpwind_r[27]+fUpwind_r[10]*alpha_r[27]+alpha_r[13]*fUpwind_r[26]+fUpwind_r[13]*alpha_r[26]+alpha_r[1]*fUpwind_r[25]+fUpwind_r[1]*alpha_r[25]+alpha_r[16]*fUpwind_r[24]+fUpwind_r[16]*alpha_r[24]+alpha_r[3]*fUpwind_r[23]+fUpwind_r[3]*alpha_r[23]+alpha_r[17]*fUpwind_r[22]+fUpwind_r[17]*alpha_r[22]+alpha_r[4]*fUpwind_r[21]+fUpwind_r[4]*alpha_r[21]+alpha_r[19]*fUpwind_r[20]+fUpwind_r[19]*alpha_r[20]+alpha_r[5]*fUpwind_r[18]+fUpwind_r[5]*alpha_r[18]+alpha_r[7]*fUpwind_r[15]+fUpwind_r[7]*alpha_r[15]+alpha_r[9]*fUpwind_r[14]+fUpwind_r[9]*alpha_r[14]+alpha_r[11]*fUpwind_r[12]+fUpwind_r[11]*alpha_r[12]); 
  Ghat_r[30] = 0.1767766952966368*(alpha_r[1]*fUpwind_r[31]+fUpwind_r[1]*alpha_r[31]+alpha_r[0]*fUpwind_r[30]+fUpwind_r[0]*alpha_r[30]+alpha_r[6]*fUpwind_r[29]+fUpwind_r[6]*alpha_r[29]+alpha_r[7]*fUpwind_r[28]+fUpwind_r[7]*alpha_r[28]+alpha_r[9]*fUpwind_r[27]+fUpwind_r[9]*alpha_r[27]+alpha_r[12]*fUpwind_r[26]+fUpwind_r[12]*alpha_r[26]+alpha_r[2]*fUpwind_r[25]+fUpwind_r[2]*alpha_r[25]+alpha_r[3]*fUpwind_r[24]+fUpwind_r[3]*alpha_r[24]+alpha_r[16]*fUpwind_r[23]+fUpwind_r[16]*alpha_r[23]+alpha_r[4]*fUpwind_r[22]+fUpwind_r[4]*alpha_r[22]+alpha_r[17]*fUpwind_r[21]+fUpwind_r[17]*alpha_r[21]+alpha_r[18]*fUpwind_r[20]+fUpwind_r[18]*alpha_r[20]+alpha_r[5]*fUpwind_r[19]+fUpwind_r[5]*alpha_r[19]+alpha_r[8]*fUpwind_r[15]+fUpwind_r[8]*alpha_r[15]+alpha_r[10]*fUpwind_r[14]+fUpwind_r[10]*alpha_r[14]+alpha_r[11]*fUpwind_r[13]+fUpwind_r[11]*alpha_r[13]); 
  Ghat_r[31] = 0.1767766952966368*(alpha_r[0]*fUpwind_r[31]+fUpwind_r[0]*alpha_r[31]+alpha_r[1]*fUpwind_r[30]+fUpwind_r[1]*alpha_r[30]+alpha_r[2]*fUpwind_r[29]+fUpwind_r[2]*alpha_r[29]+alpha_r[3]*fUpwind_r[28]+fUpwind_r[3]*alpha_r[28]+alpha_r[4]*fUpwind_r[27]+fUpwind_r[4]*alpha_r[27]+alpha_r[5]*fUpwind_r[26]+fUpwind_r[5]*alpha_r[26]+alpha_r[6]*fUpwind_r[25]+fUpwind_r[6]*alpha_r[25]+alpha_r[7]*fUpwind_r[24]+fUpwind_r[7]*alpha_r[24]+alpha_r[8]*fUpwind_r[23]+fUpwind_r[8]*alpha_r[23]+alpha_r[9]*fUpwind_r[22]+fUpwind_r[9]*alpha_r[22]+alpha_r[10]*fUpwind_r[21]+fUpwind_r[10]*alpha_r[21]+alpha_r[11]*fUpwind_r[20]+fUpwind_r[11]*alpha_r[20]+alpha_r[12]*fUpwind_r[19]+fUpwind_r[12]*alpha_r[19]+alpha_r[13]*fUpwind_r[18]+fUpwind_r[13]*alpha_r[18]+alpha_r[14]*fUpwind_r[17]+fUpwind_r[14]*alpha_r[17]+alpha_r[15]*fUpwind_r[16]+fUpwind_r[15]*alpha_r[16]); 

  out[0] += (0.7071067811865475*Ghat_l[0]-0.7071067811865475*Ghat_r[0])*dv10; 
  out[1] += (0.7071067811865475*Ghat_l[1]-0.7071067811865475*Ghat_r[1])*dv10; 
  out[2] += (0.7071067811865475*Ghat_l[2]-0.7071067811865475*Ghat_r[2])*dv10; 
  out[3] += (0.7071067811865475*Ghat_l[3]-0.7071067811865475*Ghat_r[3])*dv10; 
  out[4] += -1.224744871391589*(Ghat_r[0]+Ghat_l[0])*dv10; 
  out[5] += (0.7071067811865475*Ghat_l[4]-0.7071067811865475*Ghat_r[4])*dv10; 
  out[6] += (0.7071067811865475*Ghat_l[5]-0.7071067811865475*Ghat_r[5])*dv10; 
  out[7] += (0.7071067811865475*Ghat_l[6]-0.7071067811865475*Ghat_r[6])*dv10; 
  out[8] += (0.7071067811865475*Ghat_l[7]-0.7071067811865475*Ghat_r[7])*dv10; 
  out[9] += (0.7071067811865475*Ghat_l[8]-0.7071067811865475*Ghat_r[8])*dv10; 
  out[10] += -1.224744871391589*(Ghat_r[1]+Ghat_l[1])*dv10; 
  out[11] += -1.224744871391589*(Ghat_r[2]+Ghat_l[2])*dv10; 
  out[12] += -1.224744871391589*(Ghat_r[3]+Ghat_l[3])*dv10; 
  out[13] += (0.7071067811865475*Ghat_l[9]-0.7071067811865475*Ghat_r[9])*dv10; 
  out[14] += (0.7071067811865475*Ghat_l[10]-0.7071067811865475*Ghat_r[10])*dv10; 
  out[15] += (0.7071067811865475*Ghat_l[11]-0.7071067811865475*Ghat_r[11])*dv10; 
  out[16] += -1.224744871391589*(Ghat_r[4]+Ghat_l[4])*dv10; 
  out[17] += (0.7071067811865475*Ghat_l[12]-0.7071067811865475*Ghat_r[12])*dv10; 
  out[18] += (0.7071067811865475*Ghat_l[13]-0.7071067811865475*Ghat_r[13])*dv10; 
  out[19] += (0.7071067811865475*Ghat_l[14]-0.7071067811865475*Ghat_r[14])*dv10; 
  out[20] += -1.224744871391589*(Ghat_r[5]+Ghat_l[5])*dv10; 
  out[21] += (0.7071067811865475*Ghat_l[15]-0.7071067811865475*Ghat_r[15])*dv10; 
  out[22] += (0.7071067811865475*Ghat_l[16]-0.7071067811865475*Ghat_r[16])*dv10; 
  out[23] += -1.224744871391589*(Ghat_r[6]+Ghat_l[6])*dv10; 
  out[24] += -1.224744871391589*(Ghat_r[7]+Ghat_l[7])*dv10; 
  out[25] += -1.224744871391589*(Ghat_r[8]+Ghat_l[8])*dv10; 
  out[26] += (0.7071067811865475*Ghat_l[17]-0.7071067811865475*Ghat_r[17])*dv10; 
  out[27] += (0.7071067811865475*Ghat_l[18]-0.7071067811865475*Ghat_r[18])*dv10; 
  out[28] += (0.7071067811865475*Ghat_l[19]-0.7071067811865475*Ghat_r[19])*dv10; 
  out[29] += -1.224744871391589*(Ghat_r[9]+Ghat_l[9])*dv10; 
  out[30] += -1.224744871391589*(Ghat_r[10]+Ghat_l[10])*dv10; 
  out[31] += -1.224744871391589*(Ghat_r[11]+Ghat_l[11])*dv10; 
  out[32] += (0.7071067811865475*Ghat_l[20]-0.7071067811865475*Ghat_r[20])*dv10; 
  out[33] += (0.7071067811865475*Ghat_l[21]-0.7071067811865475*Ghat_r[21])*dv10; 
  out[34] += (0.7071067811865475*Ghat_l[22]-0.7071067811865475*Ghat_r[22])*dv10; 
  out[35] += -1.224744871391589*(Ghat_r[12]+Ghat_l[12])*dv10; 
  out[36] += -1.224744871391589*(Ghat_r[13]+Ghat_l[13])*dv10; 
  out[37] += -1.224744871391589*(Ghat_r[14]+Ghat_l[14])*dv10; 
  out[38] += (0.7071067811865475*Ghat_l[23]-0.7071067811865475*Ghat_r[23])*dv10; 
  out[39] += (0.7071067811865475*Ghat_l[24]-0.7071067811865475*Ghat_r[24])*dv10; 
  out[40] += (0.7071067811865475*Ghat_l[25]-0.7071067811865475*Ghat_r[25])*dv10; 
  out[41] += -1.224744871391589*(Ghat_r[15]+Ghat_l[15])*dv10; 
  out[42] += -1.224744871391589*(Ghat_r[16]+Ghat_l[16])*dv10; 
  out[43] += (0.7071067811865475*Ghat_l[26]-0.7071067811865475*Ghat_r[26])*dv10; 
  out[44] += -1.224744871391589*(Ghat_r[17]+Ghat_l[17])*dv10; 
  out[45] += -1.224744871391589*(Ghat_r[18]+Ghat_l[18])*dv10; 
  out[46] += -1.224744871391589*(Ghat_r[19]+Ghat_l[19])*dv10; 
  out[47] += (0.7071067811865475*Ghat_l[27]-0.7071067811865475*Ghat_r[27])*dv10; 
  out[48] += -1.224744871391589*(Ghat_r[20]+Ghat_l[20])*dv10; 
  out[49] += -1.224744871391589*(Ghat_r[21]+Ghat_l[21])*dv10; 
  out[50] += -1.224744871391589*(Ghat_r[22]+Ghat_l[22])*dv10; 
  out[51] += (0.7071067811865475*Ghat_l[28]-0.7071067811865475*Ghat_r[28])*dv10; 
  out[52] += (0.7071067811865475*Ghat_l[29]-0.7071067811865475*Ghat_r[29])*dv10; 
  out[53] += (0.7071067811865475*Ghat_l[30]-0.7071067811865475*Ghat_r[30])*dv10; 
  out[54] += -1.224744871391589*(Ghat_r[23]+Ghat_l[23])*dv10; 
  out[55] += -1.224744871391589*(Ghat_r[24]+Ghat_l[24])*dv10; 
  out[56] += -1.224744871391589*(Ghat_r[25]+Ghat_l[25])*dv10; 
  out[57] += -1.224744871391589*(Ghat_r[26]+Ghat_l[26])*dv10; 
  out[58] += -1.224744871391589*(Ghat_r[27]+Ghat_l[27])*dv10; 
  out[59] += (0.7071067811865475*Ghat_l[31]-0.7071067811865475*Ghat_r[31])*dv10; 
  out[60] += -1.224744871391589*(Ghat_r[28]+Ghat_l[28])*dv10; 
  out[61] += -1.224744871391589*(Ghat_r[29]+Ghat_l[29])*dv10; 
  out[62] += -1.224744871391589*(Ghat_r[30]+Ghat_l[30])*dv10; 
  out[63] += -1.224744871391589*(Ghat_r[31]+Ghat_l[31])*dv10; 

} 
