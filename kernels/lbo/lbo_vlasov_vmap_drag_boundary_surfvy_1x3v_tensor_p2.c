#include <gkyl_lbo_vlasov_kernels.h> 
#include <gkyl_basis_tensor_4x_p2_surfx3_eval_quad.h> 
#include <gkyl_basis_tensor_4x_p2_upwind_quad_to_modal.h> 
GKYL_CU_DH double lbo_vlasov_vmap_drag_boundary_surfvy_1x3v_tensor_p2(const double *w, const double *dxv, const double *vmap, const double *jacob_vel_inv, const double *nuSum, const double *nuPrimMomsSum, const int edge, const double *fSkin, const double *fEdge, double* GKYL_RESTRICT out) 
{ 
  // w[4]: Cell-center coordinates. 
  // dxv[4]: Cell spacing. 
  // vmap: Velocity-space nonuniform mapping in each dimension. 
  // jacob_vel_inv: Inverse of velocity space Jacobian in each dimension. 
  // nuSum: collisionalities added (self and cross species collisionalities). 
  // nuPrimMomsSum[12]: sum of bulk velocities and thermal speeds (squared) times their respective collisionalities. 
  // fSkin/Edge: Distribution function in cells 
  // out: Incremented distribution function in cell 
  double rdv2 = 2.0/dxv[2]; 

  const double *v1 = &vmap[4]; 
  const double *jacob_vel_inv1 = &jacob_vel_inv[3]; 
  const double *sumNuUy = &nuPrimMomsSum[3]; 

  double alphaDrSurf[27] = {0.0}; 
  double fUpwindQuad[27] = {0.0};
  double fUpwind[27] = {0.0};;
  double Ghat[27] = {0.0}; 

  if (edge == -1) { 

  alphaDrSurf[0] = nuSum[0]*((5.916079783099617*jacob_vel_inv1[2]+4.58257569495584*jacob_vel_inv1[1]+2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]+3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+(nuSum[0]*(3.872983346207417*v1[1]+2.23606797749979*v1[0])-3.16227766016838*sumNuUy[0])*jacob_vel_inv1[2]+nuSum[0]*(3.0*jacob_vel_inv1[1]+1.732050807568877*jacob_vel_inv1[0])*v1[1]+(1.732050807568877*nuSum[0]*v1[0]-2.449489742783178*sumNuUy[0])*jacob_vel_inv1[1]+jacob_vel_inv1[0]*(nuSum[0]*v1[0]-1.414213562373095*sumNuUy[0]); 
  alphaDrSurf[1] = nuSum[1]*((5.916079783099617*jacob_vel_inv1[2]+4.58257569495584*jacob_vel_inv1[1]+2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]+3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+(3.872983346207417*nuSum[1]*v1[1]-3.16227766016838*sumNuUy[1]+2.23606797749979*v1[0]*nuSum[1])*jacob_vel_inv1[2]+(3.0*jacob_vel_inv1[1]+1.732050807568877*jacob_vel_inv1[0])*nuSum[1]*v1[1]+((-2.449489742783178*jacob_vel_inv1[1])-1.414213562373095*jacob_vel_inv1[0])*sumNuUy[1]+v1[0]*(1.732050807568877*jacob_vel_inv1[1]+jacob_vel_inv1[0])*nuSum[1]; 
  alphaDrSurf[7] = nuSum[2]*((5.916079783099617*jacob_vel_inv1[2]+4.58257569495584*jacob_vel_inv1[1]+2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]+3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+((-3.16227766016838*jacob_vel_inv1[2])-2.449489742783178*jacob_vel_inv1[1]-1.414213562373095*jacob_vel_inv1[0])*sumNuUy[2]+((3.872983346207417*v1[1]+2.23606797749979*v1[0])*jacob_vel_inv1[2]+(3.0*jacob_vel_inv1[1]+1.732050807568877*jacob_vel_inv1[0])*v1[1]+v1[0]*(1.732050807568877*jacob_vel_inv1[1]+jacob_vel_inv1[0]))*nuSum[2]; 

  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[0] = tensor_4x_p2_surfx3_eval_quad_node_0_r(fSkin); 
    fUpwindQuad[1] = tensor_4x_p2_surfx3_eval_quad_node_1_r(fSkin); 
    fUpwindQuad[2] = tensor_4x_p2_surfx3_eval_quad_node_2_r(fSkin); 
    fUpwindQuad[3] = tensor_4x_p2_surfx3_eval_quad_node_3_r(fSkin); 
    fUpwindQuad[4] = tensor_4x_p2_surfx3_eval_quad_node_4_r(fSkin); 
    fUpwindQuad[5] = tensor_4x_p2_surfx3_eval_quad_node_5_r(fSkin); 
    fUpwindQuad[6] = tensor_4x_p2_surfx3_eval_quad_node_6_r(fSkin); 
    fUpwindQuad[7] = tensor_4x_p2_surfx3_eval_quad_node_7_r(fSkin); 
    fUpwindQuad[8] = tensor_4x_p2_surfx3_eval_quad_node_8_r(fSkin); 
  } else { 
    fUpwindQuad[0] = tensor_4x_p2_surfx3_eval_quad_node_0_l(fEdge); 
    fUpwindQuad[1] = tensor_4x_p2_surfx3_eval_quad_node_1_l(fEdge); 
    fUpwindQuad[2] = tensor_4x_p2_surfx3_eval_quad_node_2_l(fEdge); 
    fUpwindQuad[3] = tensor_4x_p2_surfx3_eval_quad_node_3_l(fEdge); 
    fUpwindQuad[4] = tensor_4x_p2_surfx3_eval_quad_node_4_l(fEdge); 
    fUpwindQuad[5] = tensor_4x_p2_surfx3_eval_quad_node_5_l(fEdge); 
    fUpwindQuad[6] = tensor_4x_p2_surfx3_eval_quad_node_6_l(fEdge); 
    fUpwindQuad[7] = tensor_4x_p2_surfx3_eval_quad_node_7_l(fEdge); 
    fUpwindQuad[8] = tensor_4x_p2_surfx3_eval_quad_node_8_l(fEdge); 
  } 
  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[9] = tensor_4x_p2_surfx3_eval_quad_node_9_r(fSkin); 
    fUpwindQuad[10] = tensor_4x_p2_surfx3_eval_quad_node_10_r(fSkin); 
    fUpwindQuad[11] = tensor_4x_p2_surfx3_eval_quad_node_11_r(fSkin); 
    fUpwindQuad[12] = tensor_4x_p2_surfx3_eval_quad_node_12_r(fSkin); 
    fUpwindQuad[13] = tensor_4x_p2_surfx3_eval_quad_node_13_r(fSkin); 
    fUpwindQuad[14] = tensor_4x_p2_surfx3_eval_quad_node_14_r(fSkin); 
    fUpwindQuad[15] = tensor_4x_p2_surfx3_eval_quad_node_15_r(fSkin); 
    fUpwindQuad[16] = tensor_4x_p2_surfx3_eval_quad_node_16_r(fSkin); 
    fUpwindQuad[17] = tensor_4x_p2_surfx3_eval_quad_node_17_r(fSkin); 
  } else { 
    fUpwindQuad[9] = tensor_4x_p2_surfx3_eval_quad_node_9_l(fEdge); 
    fUpwindQuad[10] = tensor_4x_p2_surfx3_eval_quad_node_10_l(fEdge); 
    fUpwindQuad[11] = tensor_4x_p2_surfx3_eval_quad_node_11_l(fEdge); 
    fUpwindQuad[12] = tensor_4x_p2_surfx3_eval_quad_node_12_l(fEdge); 
    fUpwindQuad[13] = tensor_4x_p2_surfx3_eval_quad_node_13_l(fEdge); 
    fUpwindQuad[14] = tensor_4x_p2_surfx3_eval_quad_node_14_l(fEdge); 
    fUpwindQuad[15] = tensor_4x_p2_surfx3_eval_quad_node_15_l(fEdge); 
    fUpwindQuad[16] = tensor_4x_p2_surfx3_eval_quad_node_16_l(fEdge); 
    fUpwindQuad[17] = tensor_4x_p2_surfx3_eval_quad_node_17_l(fEdge); 
  } 
  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[18] = tensor_4x_p2_surfx3_eval_quad_node_18_r(fSkin); 
    fUpwindQuad[19] = tensor_4x_p2_surfx3_eval_quad_node_19_r(fSkin); 
    fUpwindQuad[20] = tensor_4x_p2_surfx3_eval_quad_node_20_r(fSkin); 
    fUpwindQuad[21] = tensor_4x_p2_surfx3_eval_quad_node_21_r(fSkin); 
    fUpwindQuad[22] = tensor_4x_p2_surfx3_eval_quad_node_22_r(fSkin); 
    fUpwindQuad[23] = tensor_4x_p2_surfx3_eval_quad_node_23_r(fSkin); 
    fUpwindQuad[24] = tensor_4x_p2_surfx3_eval_quad_node_24_r(fSkin); 
    fUpwindQuad[25] = tensor_4x_p2_surfx3_eval_quad_node_25_r(fSkin); 
    fUpwindQuad[26] = tensor_4x_p2_surfx3_eval_quad_node_26_r(fSkin); 
  } else { 
    fUpwindQuad[18] = tensor_4x_p2_surfx3_eval_quad_node_18_l(fEdge); 
    fUpwindQuad[19] = tensor_4x_p2_surfx3_eval_quad_node_19_l(fEdge); 
    fUpwindQuad[20] = tensor_4x_p2_surfx3_eval_quad_node_20_l(fEdge); 
    fUpwindQuad[21] = tensor_4x_p2_surfx3_eval_quad_node_21_l(fEdge); 
    fUpwindQuad[22] = tensor_4x_p2_surfx3_eval_quad_node_22_l(fEdge); 
    fUpwindQuad[23] = tensor_4x_p2_surfx3_eval_quad_node_23_l(fEdge); 
    fUpwindQuad[24] = tensor_4x_p2_surfx3_eval_quad_node_24_l(fEdge); 
    fUpwindQuad[25] = tensor_4x_p2_surfx3_eval_quad_node_25_l(fEdge); 
    fUpwindQuad[26] = tensor_4x_p2_surfx3_eval_quad_node_26_l(fEdge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  tensor_4x_p2_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[7]+0.3535533905932737*alphaDrSurf[1]*fUpwind[1]+0.3535533905932737*alphaDrSurf[0]*fUpwind[0]; 
  Ghat[1] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[7]+0.3162277660168379*fUpwind[1]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[1]+0.3535533905932737*fUpwind[0]*alphaDrSurf[1]; 
  Ghat[2] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[11]+0.3535533905932737*alphaDrSurf[1]*fUpwind[4]+0.3535533905932737*alphaDrSurf[0]*fUpwind[2]; 
  Ghat[3] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[13]+0.3535533905932737*alphaDrSurf[1]*fUpwind[5]+0.3535533905932737*alphaDrSurf[0]*fUpwind[3]; 
  Ghat[4] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[11]+0.3162277660168379*fUpwind[4]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[4]+0.3535533905932737*alphaDrSurf[1]*fUpwind[2]; 
  Ghat[5] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[13]+0.3162277660168379*fUpwind[5]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[5]+0.3535533905932737*alphaDrSurf[1]*fUpwind[3]; 
  Ghat[6] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[17]+0.3535533905932737*alphaDrSurf[1]*fUpwind[10]+0.3535533905932737*alphaDrSurf[0]*fUpwind[6]; 
  Ghat[7] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[7]+0.3535533905932737*fUpwind[0]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[1]; 
  Ghat[8] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[20]+0.3535533905932737*alphaDrSurf[1]*fUpwind[12]+0.3535533905932737*alphaDrSurf[0]*fUpwind[8]; 
  Ghat[9] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[21]+0.3535533905932737*alphaDrSurf[1]*fUpwind[15]+0.3535533905932737*alphaDrSurf[0]*fUpwind[9]; 
  Ghat[10] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[17]+0.3162277660168379*alphaDrSurf[7]*fUpwind[10]+0.3535533905932737*alphaDrSurf[0]*fUpwind[10]+0.3535533905932737*alphaDrSurf[1]*fUpwind[6]; 
  Ghat[11] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[11]+0.3535533905932737*alphaDrSurf[0]*fUpwind[11]+0.3535533905932737*fUpwind[2]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[4]; 
  Ghat[12] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[20]+0.3162277660168379*alphaDrSurf[7]*fUpwind[12]+0.3535533905932737*alphaDrSurf[0]*fUpwind[12]+0.3535533905932737*alphaDrSurf[1]*fUpwind[8]; 
  Ghat[13] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[13]+0.3535533905932737*alphaDrSurf[0]*fUpwind[13]+0.3535533905932737*fUpwind[3]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[5]; 
  Ghat[14] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[23]+0.3535533905932737*alphaDrSurf[1]*fUpwind[18]+0.3535533905932737*alphaDrSurf[0]*fUpwind[14]; 
  Ghat[15] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[21]+0.3162277660168379*alphaDrSurf[7]*fUpwind[15]+0.3535533905932737*alphaDrSurf[0]*fUpwind[15]+0.3535533905932737*alphaDrSurf[1]*fUpwind[9]; 
  Ghat[16] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[24]+0.3535533905932737*alphaDrSurf[1]*fUpwind[19]+0.3535533905932737*alphaDrSurf[0]*fUpwind[16]; 
  Ghat[17] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[17]+0.3535533905932737*alphaDrSurf[0]*fUpwind[17]+0.3162277660168379*alphaDrSurf[1]*fUpwind[10]+0.3535533905932737*fUpwind[6]*alphaDrSurf[7]; 
  Ghat[18] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[23]+0.3162277660168379*alphaDrSurf[7]*fUpwind[18]+0.3535533905932737*alphaDrSurf[0]*fUpwind[18]+0.3535533905932737*alphaDrSurf[1]*fUpwind[14]; 
  Ghat[19] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[24]+0.3162277660168379*alphaDrSurf[7]*fUpwind[19]+0.3535533905932737*alphaDrSurf[0]*fUpwind[19]+0.3535533905932737*alphaDrSurf[1]*fUpwind[16]; 
  Ghat[20] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[20]+0.3535533905932737*alphaDrSurf[0]*fUpwind[20]+0.3162277660168379*alphaDrSurf[1]*fUpwind[12]+0.3535533905932737*alphaDrSurf[7]*fUpwind[8]; 
  Ghat[21] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[21]+0.3535533905932737*alphaDrSurf[0]*fUpwind[21]+0.3162277660168379*alphaDrSurf[1]*fUpwind[15]+0.3535533905932737*alphaDrSurf[7]*fUpwind[9]; 
  Ghat[22] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[26]+0.3535533905932737*alphaDrSurf[1]*fUpwind[25]+0.3535533905932737*alphaDrSurf[0]*fUpwind[22]; 
  Ghat[23] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[23]+0.3535533905932737*alphaDrSurf[0]*fUpwind[23]+0.3162277660168379*alphaDrSurf[1]*fUpwind[18]+0.3535533905932737*alphaDrSurf[7]*fUpwind[14]; 
  Ghat[24] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[24]+0.3535533905932737*alphaDrSurf[0]*fUpwind[24]+0.3162277660168379*alphaDrSurf[1]*fUpwind[19]+0.3535533905932737*alphaDrSurf[7]*fUpwind[16]; 
  Ghat[25] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[26]+0.3162277660168379*alphaDrSurf[7]*fUpwind[25]+0.3535533905932737*alphaDrSurf[0]*fUpwind[25]+0.3535533905932737*alphaDrSurf[1]*fUpwind[22]; 
  Ghat[26] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[26]+0.3535533905932737*alphaDrSurf[0]*fUpwind[26]+0.3162277660168379*alphaDrSurf[1]*fUpwind[25]+0.3535533905932737*alphaDrSurf[7]*fUpwind[22]; 

  out[0] += 0.7071067811865475*Ghat[0]*rdv2; 
  out[1] += 0.7071067811865475*Ghat[1]*rdv2; 
  out[2] += 0.7071067811865475*Ghat[2]*rdv2; 
  out[3] += 1.224744871391589*Ghat[0]*rdv2; 
  out[4] += 0.7071067811865475*Ghat[3]*rdv2; 
  out[5] += 0.7071067811865475*Ghat[4]*rdv2; 
  out[6] += 1.224744871391589*Ghat[1]*rdv2; 
  out[7] += 1.224744871391589*Ghat[2]*rdv2; 
  out[8] += 0.7071067811865475*Ghat[5]*rdv2; 
  out[9] += 0.7071067811865475*Ghat[6]*rdv2; 
  out[10] += 1.224744871391589*Ghat[3]*rdv2; 
  out[11] += 0.7071067811865475*Ghat[7]*rdv2; 
  out[12] += 0.7071067811865475*Ghat[8]*rdv2; 
  out[13] += 1.58113883008419*Ghat[0]*rdv2; 
  out[14] += 0.7071067811865475*Ghat[9]*rdv2; 
  out[15] += 1.224744871391589*Ghat[4]*rdv2; 
  out[16] += 0.7071067811865475*Ghat[10]*rdv2; 
  out[17] += 1.224744871391589*Ghat[5]*rdv2; 
  out[18] += 1.224744871391589*Ghat[6]*rdv2; 
  out[19] += 0.7071067811865475*Ghat[11]*rdv2; 
  out[20] += 0.7071067811865475*Ghat[12]*rdv2; 
  out[21] += 1.224744871391589*Ghat[7]*rdv2; 
  out[22] += 1.224744871391589*Ghat[8]*rdv2; 
  out[23] += 1.58113883008419*Ghat[1]*rdv2; 
  out[24] += 1.58113883008419*Ghat[2]*rdv2; 
  out[25] += 0.7071067811865475*Ghat[13]*rdv2; 
  out[26] += 0.7071067811865475*Ghat[14]*rdv2; 
  out[27] += 1.58113883008419*Ghat[3]*rdv2; 
  out[28] += 0.7071067811865475*Ghat[15]*rdv2; 
  out[29] += 0.7071067811865475*Ghat[16]*rdv2; 
  out[30] += 1.224744871391589*Ghat[9]*rdv2; 
  out[31] += 1.224744871391589*Ghat[10]*rdv2; 
  out[32] += 1.224744871391589*Ghat[11]*rdv2; 
  out[33] += 1.224744871391589*Ghat[12]*rdv2; 
  out[34] += 1.58113883008419*Ghat[4]*rdv2; 
  out[35] += 0.7071067811865475*Ghat[17]*rdv2; 
  out[36] += 0.7071067811865475*Ghat[18]*rdv2; 
  out[37] += 1.224744871391589*Ghat[13]*rdv2; 
  out[38] += 1.224744871391589*Ghat[14]*rdv2; 
  out[39] += 1.58113883008419*Ghat[5]*rdv2; 
  out[40] += 1.58113883008419*Ghat[6]*rdv2; 
  out[41] += 0.7071067811865475*Ghat[19]*rdv2; 
  out[42] += 1.224744871391589*Ghat[15]*rdv2; 
  out[43] += 1.224744871391589*Ghat[16]*rdv2; 
  out[44] += 0.7071067811865475*Ghat[20]*rdv2; 
  out[45] += 1.58113883008419*Ghat[7]*rdv2; 
  out[46] += 1.58113883008419*Ghat[8]*rdv2; 
  out[47] += 0.7071067811865475*Ghat[21]*rdv2; 
  out[48] += 0.7071067811865475*Ghat[22]*rdv2; 
  out[49] += 1.58113883008419*Ghat[9]*rdv2; 
  out[50] += 1.224744871391589*Ghat[17]*rdv2; 
  out[51] += 1.224744871391589*Ghat[18]*rdv2; 
  out[52] += 1.58113883008419*Ghat[10]*rdv2; 
  out[53] += 1.224744871391589*Ghat[19]*rdv2; 
  out[54] += 1.224744871391589*Ghat[20]*rdv2; 
  out[55] += 1.58113883008419*Ghat[11]*rdv2; 
  out[56] += 1.58113883008419*Ghat[12]*rdv2; 
  out[57] += 0.7071067811865475*Ghat[23]*rdv2; 
  out[58] += 1.58113883008419*Ghat[13]*rdv2; 
  out[59] += 1.58113883008419*Ghat[14]*rdv2; 
  out[60] += 0.7071067811865475*Ghat[24]*rdv2; 
  out[61] += 0.7071067811865475*Ghat[25]*rdv2; 
  out[62] += 1.224744871391589*Ghat[21]*rdv2; 
  out[63] += 1.224744871391589*Ghat[22]*rdv2; 
  out[64] += 1.58113883008419*Ghat[15]*rdv2; 
  out[65] += 1.58113883008419*Ghat[16]*rdv2; 
  out[66] += 1.224744871391589*Ghat[23]*rdv2; 
  out[67] += 1.58113883008419*Ghat[17]*rdv2; 
  out[68] += 1.58113883008419*Ghat[18]*rdv2; 
  out[69] += 1.224744871391589*Ghat[24]*rdv2; 
  out[70] += 1.224744871391589*Ghat[25]*rdv2; 
  out[71] += 1.58113883008419*Ghat[19]*rdv2; 
  out[72] += 1.58113883008419*Ghat[20]*rdv2; 
  out[73] += 0.7071067811865475*Ghat[26]*rdv2; 
  out[74] += 1.58113883008419*Ghat[21]*rdv2; 
  out[75] += 1.58113883008419*Ghat[22]*rdv2; 
  out[76] += 1.58113883008419*Ghat[23]*rdv2; 
  out[77] += 1.224744871391589*Ghat[26]*rdv2; 
  out[78] += 1.58113883008419*Ghat[24]*rdv2; 
  out[79] += 1.58113883008419*Ghat[25]*rdv2; 
  out[80] += 1.58113883008419*Ghat[26]*rdv2; 

  } else { 

  alphaDrSurf[0] = nuSum[0]*(((-5.916079783099617*jacob_vel_inv1[2])+4.58257569495584*jacob_vel_inv1[1]-2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]-3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+(nuSum[0]*(2.23606797749979*v1[0]-3.872983346207417*v1[1])-3.16227766016838*sumNuUy[0])*jacob_vel_inv1[2]+nuSum[0]*(3.0*jacob_vel_inv1[1]-1.732050807568877*jacob_vel_inv1[0])*v1[1]+(2.449489742783178*sumNuUy[0]-1.732050807568877*nuSum[0]*v1[0])*jacob_vel_inv1[1]+jacob_vel_inv1[0]*(nuSum[0]*v1[0]-1.414213562373095*sumNuUy[0]); 
  alphaDrSurf[1] = nuSum[1]*(((-5.916079783099617*jacob_vel_inv1[2])+4.58257569495584*jacob_vel_inv1[1]-2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]-3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+((-3.872983346207417*nuSum[1]*v1[1])-3.16227766016838*sumNuUy[1]+2.23606797749979*v1[0]*nuSum[1])*jacob_vel_inv1[2]+(3.0*jacob_vel_inv1[1]-1.732050807568877*jacob_vel_inv1[0])*nuSum[1]*v1[1]+(2.449489742783178*jacob_vel_inv1[1]-1.414213562373095*jacob_vel_inv1[0])*sumNuUy[1]+v1[0]*(jacob_vel_inv1[0]-1.732050807568877*jacob_vel_inv1[1])*nuSum[1]; 
  alphaDrSurf[7] = nuSum[2]*(((-5.916079783099617*jacob_vel_inv1[2])+4.58257569495584*jacob_vel_inv1[1]-2.645751311064591*jacob_vel_inv1[0])*v1[3]+(5.0*jacob_vel_inv1[2]-3.872983346207417*jacob_vel_inv1[1]+2.23606797749979*jacob_vel_inv1[0])*v1[2])+((-3.16227766016838*jacob_vel_inv1[2])+2.449489742783178*jacob_vel_inv1[1]-1.414213562373095*jacob_vel_inv1[0])*sumNuUy[2]+((2.23606797749979*v1[0]-3.872983346207417*v1[1])*jacob_vel_inv1[2]+(3.0*jacob_vel_inv1[1]-1.732050807568877*jacob_vel_inv1[0])*v1[1]+v1[0]*(jacob_vel_inv1[0]-1.732050807568877*jacob_vel_inv1[1]))*nuSum[2]; 

  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[0] = tensor_4x_p2_surfx3_eval_quad_node_0_r(fEdge); 
    fUpwindQuad[1] = tensor_4x_p2_surfx3_eval_quad_node_1_r(fEdge); 
    fUpwindQuad[2] = tensor_4x_p2_surfx3_eval_quad_node_2_r(fEdge); 
    fUpwindQuad[3] = tensor_4x_p2_surfx3_eval_quad_node_3_r(fEdge); 
    fUpwindQuad[4] = tensor_4x_p2_surfx3_eval_quad_node_4_r(fEdge); 
    fUpwindQuad[5] = tensor_4x_p2_surfx3_eval_quad_node_5_r(fEdge); 
    fUpwindQuad[6] = tensor_4x_p2_surfx3_eval_quad_node_6_r(fEdge); 
    fUpwindQuad[7] = tensor_4x_p2_surfx3_eval_quad_node_7_r(fEdge); 
    fUpwindQuad[8] = tensor_4x_p2_surfx3_eval_quad_node_8_r(fEdge); 
  } else { 
    fUpwindQuad[0] = tensor_4x_p2_surfx3_eval_quad_node_0_l(fSkin); 
    fUpwindQuad[1] = tensor_4x_p2_surfx3_eval_quad_node_1_l(fSkin); 
    fUpwindQuad[2] = tensor_4x_p2_surfx3_eval_quad_node_2_l(fSkin); 
    fUpwindQuad[3] = tensor_4x_p2_surfx3_eval_quad_node_3_l(fSkin); 
    fUpwindQuad[4] = tensor_4x_p2_surfx3_eval_quad_node_4_l(fSkin); 
    fUpwindQuad[5] = tensor_4x_p2_surfx3_eval_quad_node_5_l(fSkin); 
    fUpwindQuad[6] = tensor_4x_p2_surfx3_eval_quad_node_6_l(fSkin); 
    fUpwindQuad[7] = tensor_4x_p2_surfx3_eval_quad_node_7_l(fSkin); 
    fUpwindQuad[8] = tensor_4x_p2_surfx3_eval_quad_node_8_l(fSkin); 
  } 
  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[9] = tensor_4x_p2_surfx3_eval_quad_node_9_r(fEdge); 
    fUpwindQuad[10] = tensor_4x_p2_surfx3_eval_quad_node_10_r(fEdge); 
    fUpwindQuad[11] = tensor_4x_p2_surfx3_eval_quad_node_11_r(fEdge); 
    fUpwindQuad[12] = tensor_4x_p2_surfx3_eval_quad_node_12_r(fEdge); 
    fUpwindQuad[13] = tensor_4x_p2_surfx3_eval_quad_node_13_r(fEdge); 
    fUpwindQuad[14] = tensor_4x_p2_surfx3_eval_quad_node_14_r(fEdge); 
    fUpwindQuad[15] = tensor_4x_p2_surfx3_eval_quad_node_15_r(fEdge); 
    fUpwindQuad[16] = tensor_4x_p2_surfx3_eval_quad_node_16_r(fEdge); 
    fUpwindQuad[17] = tensor_4x_p2_surfx3_eval_quad_node_17_r(fEdge); 
  } else { 
    fUpwindQuad[9] = tensor_4x_p2_surfx3_eval_quad_node_9_l(fSkin); 
    fUpwindQuad[10] = tensor_4x_p2_surfx3_eval_quad_node_10_l(fSkin); 
    fUpwindQuad[11] = tensor_4x_p2_surfx3_eval_quad_node_11_l(fSkin); 
    fUpwindQuad[12] = tensor_4x_p2_surfx3_eval_quad_node_12_l(fSkin); 
    fUpwindQuad[13] = tensor_4x_p2_surfx3_eval_quad_node_13_l(fSkin); 
    fUpwindQuad[14] = tensor_4x_p2_surfx3_eval_quad_node_14_l(fSkin); 
    fUpwindQuad[15] = tensor_4x_p2_surfx3_eval_quad_node_15_l(fSkin); 
    fUpwindQuad[16] = tensor_4x_p2_surfx3_eval_quad_node_16_l(fSkin); 
    fUpwindQuad[17] = tensor_4x_p2_surfx3_eval_quad_node_17_l(fSkin); 
  } 
  if (0.3162277660168378*alphaDrSurf[7]-0.4743416490252568*alphaDrSurf[1]+0.3535533905932734*alphaDrSurf[0] < 0) { 
    fUpwindQuad[18] = tensor_4x_p2_surfx3_eval_quad_node_18_r(fEdge); 
    fUpwindQuad[19] = tensor_4x_p2_surfx3_eval_quad_node_19_r(fEdge); 
    fUpwindQuad[20] = tensor_4x_p2_surfx3_eval_quad_node_20_r(fEdge); 
    fUpwindQuad[21] = tensor_4x_p2_surfx3_eval_quad_node_21_r(fEdge); 
    fUpwindQuad[22] = tensor_4x_p2_surfx3_eval_quad_node_22_r(fEdge); 
    fUpwindQuad[23] = tensor_4x_p2_surfx3_eval_quad_node_23_r(fEdge); 
    fUpwindQuad[24] = tensor_4x_p2_surfx3_eval_quad_node_24_r(fEdge); 
    fUpwindQuad[25] = tensor_4x_p2_surfx3_eval_quad_node_25_r(fEdge); 
    fUpwindQuad[26] = tensor_4x_p2_surfx3_eval_quad_node_26_r(fEdge); 
  } else { 
    fUpwindQuad[18] = tensor_4x_p2_surfx3_eval_quad_node_18_l(fSkin); 
    fUpwindQuad[19] = tensor_4x_p2_surfx3_eval_quad_node_19_l(fSkin); 
    fUpwindQuad[20] = tensor_4x_p2_surfx3_eval_quad_node_20_l(fSkin); 
    fUpwindQuad[21] = tensor_4x_p2_surfx3_eval_quad_node_21_l(fSkin); 
    fUpwindQuad[22] = tensor_4x_p2_surfx3_eval_quad_node_22_l(fSkin); 
    fUpwindQuad[23] = tensor_4x_p2_surfx3_eval_quad_node_23_l(fSkin); 
    fUpwindQuad[24] = tensor_4x_p2_surfx3_eval_quad_node_24_l(fSkin); 
    fUpwindQuad[25] = tensor_4x_p2_surfx3_eval_quad_node_25_l(fSkin); 
    fUpwindQuad[26] = tensor_4x_p2_surfx3_eval_quad_node_26_l(fSkin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  tensor_4x_p2_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[7]+0.3535533905932737*alphaDrSurf[1]*fUpwind[1]+0.3535533905932737*alphaDrSurf[0]*fUpwind[0]; 
  Ghat[1] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[7]+0.3162277660168379*fUpwind[1]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[1]+0.3535533905932737*fUpwind[0]*alphaDrSurf[1]; 
  Ghat[2] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[11]+0.3535533905932737*alphaDrSurf[1]*fUpwind[4]+0.3535533905932737*alphaDrSurf[0]*fUpwind[2]; 
  Ghat[3] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[13]+0.3535533905932737*alphaDrSurf[1]*fUpwind[5]+0.3535533905932737*alphaDrSurf[0]*fUpwind[3]; 
  Ghat[4] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[11]+0.3162277660168379*fUpwind[4]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[4]+0.3535533905932737*alphaDrSurf[1]*fUpwind[2]; 
  Ghat[5] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[13]+0.3162277660168379*fUpwind[5]*alphaDrSurf[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[5]+0.3535533905932737*alphaDrSurf[1]*fUpwind[3]; 
  Ghat[6] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[17]+0.3535533905932737*alphaDrSurf[1]*fUpwind[10]+0.3535533905932737*alphaDrSurf[0]*fUpwind[6]; 
  Ghat[7] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[7]+0.3535533905932737*alphaDrSurf[0]*fUpwind[7]+0.3535533905932737*fUpwind[0]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[1]; 
  Ghat[8] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[20]+0.3535533905932737*alphaDrSurf[1]*fUpwind[12]+0.3535533905932737*alphaDrSurf[0]*fUpwind[8]; 
  Ghat[9] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[21]+0.3535533905932737*alphaDrSurf[1]*fUpwind[15]+0.3535533905932737*alphaDrSurf[0]*fUpwind[9]; 
  Ghat[10] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[17]+0.3162277660168379*alphaDrSurf[7]*fUpwind[10]+0.3535533905932737*alphaDrSurf[0]*fUpwind[10]+0.3535533905932737*alphaDrSurf[1]*fUpwind[6]; 
  Ghat[11] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[11]+0.3535533905932737*alphaDrSurf[0]*fUpwind[11]+0.3535533905932737*fUpwind[2]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[4]; 
  Ghat[12] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[20]+0.3162277660168379*alphaDrSurf[7]*fUpwind[12]+0.3535533905932737*alphaDrSurf[0]*fUpwind[12]+0.3535533905932737*alphaDrSurf[1]*fUpwind[8]; 
  Ghat[13] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[13]+0.3535533905932737*alphaDrSurf[0]*fUpwind[13]+0.3535533905932737*fUpwind[3]*alphaDrSurf[7]+0.3162277660168379*alphaDrSurf[1]*fUpwind[5]; 
  Ghat[14] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[23]+0.3535533905932737*alphaDrSurf[1]*fUpwind[18]+0.3535533905932737*alphaDrSurf[0]*fUpwind[14]; 
  Ghat[15] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[21]+0.3162277660168379*alphaDrSurf[7]*fUpwind[15]+0.3535533905932737*alphaDrSurf[0]*fUpwind[15]+0.3535533905932737*alphaDrSurf[1]*fUpwind[9]; 
  Ghat[16] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[24]+0.3535533905932737*alphaDrSurf[1]*fUpwind[19]+0.3535533905932737*alphaDrSurf[0]*fUpwind[16]; 
  Ghat[17] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[17]+0.3535533905932737*alphaDrSurf[0]*fUpwind[17]+0.3162277660168379*alphaDrSurf[1]*fUpwind[10]+0.3535533905932737*fUpwind[6]*alphaDrSurf[7]; 
  Ghat[18] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[23]+0.3162277660168379*alphaDrSurf[7]*fUpwind[18]+0.3535533905932737*alphaDrSurf[0]*fUpwind[18]+0.3535533905932737*alphaDrSurf[1]*fUpwind[14]; 
  Ghat[19] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[24]+0.3162277660168379*alphaDrSurf[7]*fUpwind[19]+0.3535533905932737*alphaDrSurf[0]*fUpwind[19]+0.3535533905932737*alphaDrSurf[1]*fUpwind[16]; 
  Ghat[20] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[20]+0.3535533905932737*alphaDrSurf[0]*fUpwind[20]+0.3162277660168379*alphaDrSurf[1]*fUpwind[12]+0.3535533905932737*alphaDrSurf[7]*fUpwind[8]; 
  Ghat[21] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[21]+0.3535533905932737*alphaDrSurf[0]*fUpwind[21]+0.3162277660168379*alphaDrSurf[1]*fUpwind[15]+0.3535533905932737*alphaDrSurf[7]*fUpwind[9]; 
  Ghat[22] = 0.3535533905932737*alphaDrSurf[7]*fUpwind[26]+0.3535533905932737*alphaDrSurf[1]*fUpwind[25]+0.3535533905932737*alphaDrSurf[0]*fUpwind[22]; 
  Ghat[23] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[23]+0.3535533905932737*alphaDrSurf[0]*fUpwind[23]+0.3162277660168379*alphaDrSurf[1]*fUpwind[18]+0.3535533905932737*alphaDrSurf[7]*fUpwind[14]; 
  Ghat[24] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[24]+0.3535533905932737*alphaDrSurf[0]*fUpwind[24]+0.3162277660168379*alphaDrSurf[1]*fUpwind[19]+0.3535533905932737*alphaDrSurf[7]*fUpwind[16]; 
  Ghat[25] = 0.3162277660168379*alphaDrSurf[1]*fUpwind[26]+0.3162277660168379*alphaDrSurf[7]*fUpwind[25]+0.3535533905932737*alphaDrSurf[0]*fUpwind[25]+0.3535533905932737*alphaDrSurf[1]*fUpwind[22]; 
  Ghat[26] = 0.2258769757263128*alphaDrSurf[7]*fUpwind[26]+0.3535533905932737*alphaDrSurf[0]*fUpwind[26]+0.3162277660168379*alphaDrSurf[1]*fUpwind[25]+0.3535533905932737*alphaDrSurf[7]*fUpwind[22]; 

  out[0] += -0.7071067811865475*Ghat[0]*rdv2; 
  out[1] += -0.7071067811865475*Ghat[1]*rdv2; 
  out[2] += -0.7071067811865475*Ghat[2]*rdv2; 
  out[3] += 1.224744871391589*Ghat[0]*rdv2; 
  out[4] += -0.7071067811865475*Ghat[3]*rdv2; 
  out[5] += -0.7071067811865475*Ghat[4]*rdv2; 
  out[6] += 1.224744871391589*Ghat[1]*rdv2; 
  out[7] += 1.224744871391589*Ghat[2]*rdv2; 
  out[8] += -0.7071067811865475*Ghat[5]*rdv2; 
  out[9] += -0.7071067811865475*Ghat[6]*rdv2; 
  out[10] += 1.224744871391589*Ghat[3]*rdv2; 
  out[11] += -0.7071067811865475*Ghat[7]*rdv2; 
  out[12] += -0.7071067811865475*Ghat[8]*rdv2; 
  out[13] += -1.58113883008419*Ghat[0]*rdv2; 
  out[14] += -0.7071067811865475*Ghat[9]*rdv2; 
  out[15] += 1.224744871391589*Ghat[4]*rdv2; 
  out[16] += -0.7071067811865475*Ghat[10]*rdv2; 
  out[17] += 1.224744871391589*Ghat[5]*rdv2; 
  out[18] += 1.224744871391589*Ghat[6]*rdv2; 
  out[19] += -0.7071067811865475*Ghat[11]*rdv2; 
  out[20] += -0.7071067811865475*Ghat[12]*rdv2; 
  out[21] += 1.224744871391589*Ghat[7]*rdv2; 
  out[22] += 1.224744871391589*Ghat[8]*rdv2; 
  out[23] += -1.58113883008419*Ghat[1]*rdv2; 
  out[24] += -1.58113883008419*Ghat[2]*rdv2; 
  out[25] += -0.7071067811865475*Ghat[13]*rdv2; 
  out[26] += -0.7071067811865475*Ghat[14]*rdv2; 
  out[27] += -1.58113883008419*Ghat[3]*rdv2; 
  out[28] += -0.7071067811865475*Ghat[15]*rdv2; 
  out[29] += -0.7071067811865475*Ghat[16]*rdv2; 
  out[30] += 1.224744871391589*Ghat[9]*rdv2; 
  out[31] += 1.224744871391589*Ghat[10]*rdv2; 
  out[32] += 1.224744871391589*Ghat[11]*rdv2; 
  out[33] += 1.224744871391589*Ghat[12]*rdv2; 
  out[34] += -1.58113883008419*Ghat[4]*rdv2; 
  out[35] += -0.7071067811865475*Ghat[17]*rdv2; 
  out[36] += -0.7071067811865475*Ghat[18]*rdv2; 
  out[37] += 1.224744871391589*Ghat[13]*rdv2; 
  out[38] += 1.224744871391589*Ghat[14]*rdv2; 
  out[39] += -1.58113883008419*Ghat[5]*rdv2; 
  out[40] += -1.58113883008419*Ghat[6]*rdv2; 
  out[41] += -0.7071067811865475*Ghat[19]*rdv2; 
  out[42] += 1.224744871391589*Ghat[15]*rdv2; 
  out[43] += 1.224744871391589*Ghat[16]*rdv2; 
  out[44] += -0.7071067811865475*Ghat[20]*rdv2; 
  out[45] += -1.58113883008419*Ghat[7]*rdv2; 
  out[46] += -1.58113883008419*Ghat[8]*rdv2; 
  out[47] += -0.7071067811865475*Ghat[21]*rdv2; 
  out[48] += -0.7071067811865475*Ghat[22]*rdv2; 
  out[49] += -1.58113883008419*Ghat[9]*rdv2; 
  out[50] += 1.224744871391589*Ghat[17]*rdv2; 
  out[51] += 1.224744871391589*Ghat[18]*rdv2; 
  out[52] += -1.58113883008419*Ghat[10]*rdv2; 
  out[53] += 1.224744871391589*Ghat[19]*rdv2; 
  out[54] += 1.224744871391589*Ghat[20]*rdv2; 
  out[55] += -1.58113883008419*Ghat[11]*rdv2; 
  out[56] += -1.58113883008419*Ghat[12]*rdv2; 
  out[57] += -0.7071067811865475*Ghat[23]*rdv2; 
  out[58] += -1.58113883008419*Ghat[13]*rdv2; 
  out[59] += -1.58113883008419*Ghat[14]*rdv2; 
  out[60] += -0.7071067811865475*Ghat[24]*rdv2; 
  out[61] += -0.7071067811865475*Ghat[25]*rdv2; 
  out[62] += 1.224744871391589*Ghat[21]*rdv2; 
  out[63] += 1.224744871391589*Ghat[22]*rdv2; 
  out[64] += -1.58113883008419*Ghat[15]*rdv2; 
  out[65] += -1.58113883008419*Ghat[16]*rdv2; 
  out[66] += 1.224744871391589*Ghat[23]*rdv2; 
  out[67] += -1.58113883008419*Ghat[17]*rdv2; 
  out[68] += -1.58113883008419*Ghat[18]*rdv2; 
  out[69] += 1.224744871391589*Ghat[24]*rdv2; 
  out[70] += 1.224744871391589*Ghat[25]*rdv2; 
  out[71] += -1.58113883008419*Ghat[19]*rdv2; 
  out[72] += -1.58113883008419*Ghat[20]*rdv2; 
  out[73] += -0.7071067811865475*Ghat[26]*rdv2; 
  out[74] += -1.58113883008419*Ghat[21]*rdv2; 
  out[75] += -1.58113883008419*Ghat[22]*rdv2; 
  out[76] += -1.58113883008419*Ghat[23]*rdv2; 
  out[77] += 1.224744871391589*Ghat[26]*rdv2; 
  out[78] += -1.58113883008419*Ghat[24]*rdv2; 
  out[79] += -1.58113883008419*Ghat[25]*rdv2; 
  out[80] += -1.58113883008419*Ghat[26]*rdv2; 

  } 

  return 0.;

} 
