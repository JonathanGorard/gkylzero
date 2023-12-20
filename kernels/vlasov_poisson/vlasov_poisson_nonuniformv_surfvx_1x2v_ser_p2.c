#include <gkyl_vlasov_poisson_kernels.h> 
#include <gkyl_basis_ser_3x_p2_surfx2_eval_quad.h> 
#include <gkyl_basis_ser_3x_p2_upwind_quad_to_modal.h> 
GKYL_CU_DH double vlasov_poisson_nonuniformv_surfvx_1x2v_ser_p2(const double *w, const double *dxv, const double *vcoord, const double *field, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w:         Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // vcoord:    Discrete (DG) velocity coordinate.
  // field:     potential (scaled by appropriate factors).
  // fl/fc/fr:  Input Distribution function in left/center/right cells 
  // out:       Output distribution function in center cell 
  const double rdv2 = 2/dxv[1]; 
  const double *phi = &field[0]; 
  const double rdx2 = 2/dxv[0]; 
  double alpha[8] = {0.0}; 

  alpha[0] = -2.449489742783178*phi[1]*rdx2; 
  alpha[1] = -5.477225575051662*phi[2]*rdx2; 

  double fUpwindQuad_l[9] = {0.0};
  double fUpwindQuad_r[9] = {0.0};
  double fUpwind_l[8] = {0.0};
  double fUpwind_r[8] = {0.0};
  double Ghat_l[8] = {0.0}; 
  double Ghat_r[8] = {0.0}; 

  if (0.5*alpha[0]-0.6708203932499369*alpha[1] > 0) { 
    fUpwindQuad_l[0] = ser_3x_p2_surfx2_eval_quad_node_0_r(fl); 
    fUpwindQuad_r[0] = ser_3x_p2_surfx2_eval_quad_node_0_r(fc); 
    fUpwindQuad_l[1] = ser_3x_p2_surfx2_eval_quad_node_1_r(fl); 
    fUpwindQuad_r[1] = ser_3x_p2_surfx2_eval_quad_node_1_r(fc); 
    fUpwindQuad_l[2] = ser_3x_p2_surfx2_eval_quad_node_2_r(fl); 
    fUpwindQuad_r[2] = ser_3x_p2_surfx2_eval_quad_node_2_r(fc); 
  } else { 
    fUpwindQuad_l[0] = ser_3x_p2_surfx2_eval_quad_node_0_l(fc); 
    fUpwindQuad_r[0] = ser_3x_p2_surfx2_eval_quad_node_0_l(fr); 
    fUpwindQuad_l[1] = ser_3x_p2_surfx2_eval_quad_node_1_l(fc); 
    fUpwindQuad_r[1] = ser_3x_p2_surfx2_eval_quad_node_1_l(fr); 
    fUpwindQuad_l[2] = ser_3x_p2_surfx2_eval_quad_node_2_l(fc); 
    fUpwindQuad_r[2] = ser_3x_p2_surfx2_eval_quad_node_2_l(fr); 
  } 
  if (0.5*alpha[0]-0.6708203932499369*alpha[1] > 0) { 
    fUpwindQuad_l[3] = ser_3x_p2_surfx2_eval_quad_node_3_r(fl); 
    fUpwindQuad_r[3] = ser_3x_p2_surfx2_eval_quad_node_3_r(fc); 
    fUpwindQuad_l[4] = ser_3x_p2_surfx2_eval_quad_node_4_r(fl); 
    fUpwindQuad_r[4] = ser_3x_p2_surfx2_eval_quad_node_4_r(fc); 
    fUpwindQuad_l[5] = ser_3x_p2_surfx2_eval_quad_node_5_r(fl); 
    fUpwindQuad_r[5] = ser_3x_p2_surfx2_eval_quad_node_5_r(fc); 
  } else { 
    fUpwindQuad_l[3] = ser_3x_p2_surfx2_eval_quad_node_3_l(fc); 
    fUpwindQuad_r[3] = ser_3x_p2_surfx2_eval_quad_node_3_l(fr); 
    fUpwindQuad_l[4] = ser_3x_p2_surfx2_eval_quad_node_4_l(fc); 
    fUpwindQuad_r[4] = ser_3x_p2_surfx2_eval_quad_node_4_l(fr); 
    fUpwindQuad_l[5] = ser_3x_p2_surfx2_eval_quad_node_5_l(fc); 
    fUpwindQuad_r[5] = ser_3x_p2_surfx2_eval_quad_node_5_l(fr); 
  } 
  if (0.5*alpha[0]-0.6708203932499369*alpha[1] > 0) { 
    fUpwindQuad_l[6] = ser_3x_p2_surfx2_eval_quad_node_6_r(fl); 
    fUpwindQuad_r[6] = ser_3x_p2_surfx2_eval_quad_node_6_r(fc); 
    fUpwindQuad_l[7] = ser_3x_p2_surfx2_eval_quad_node_7_r(fl); 
    fUpwindQuad_r[7] = ser_3x_p2_surfx2_eval_quad_node_7_r(fc); 
    fUpwindQuad_l[8] = ser_3x_p2_surfx2_eval_quad_node_8_r(fl); 
    fUpwindQuad_r[8] = ser_3x_p2_surfx2_eval_quad_node_8_r(fc); 
  } else { 
    fUpwindQuad_l[6] = ser_3x_p2_surfx2_eval_quad_node_6_l(fc); 
    fUpwindQuad_r[6] = ser_3x_p2_surfx2_eval_quad_node_6_l(fr); 
    fUpwindQuad_l[7] = ser_3x_p2_surfx2_eval_quad_node_7_l(fc); 
    fUpwindQuad_r[7] = ser_3x_p2_surfx2_eval_quad_node_7_l(fr); 
    fUpwindQuad_l[8] = ser_3x_p2_surfx2_eval_quad_node_8_l(fc); 
    fUpwindQuad_r[8] = ser_3x_p2_surfx2_eval_quad_node_8_l(fr); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  ser_3x_p2_upwind_quad_to_modal(fUpwindQuad_l, fUpwind_l); 
  ser_3x_p2_upwind_quad_to_modal(fUpwindQuad_r, fUpwind_r); 

  Ghat_l[0] = 0.5*alpha[1]*fUpwind_l[1]+0.5*alpha[0]*fUpwind_l[0]; 
  Ghat_l[1] = 0.4472135954999579*alpha[1]*fUpwind_l[4]+0.5*alpha[0]*fUpwind_l[1]+0.5*fUpwind_l[0]*alpha[1]; 
  Ghat_l[2] = 0.5*alpha[1]*fUpwind_l[3]+0.5*alpha[0]*fUpwind_l[2]; 
  Ghat_l[3] = 0.447213595499958*alpha[1]*fUpwind_l[6]+0.5*alpha[0]*fUpwind_l[3]+0.5*alpha[1]*fUpwind_l[2]; 
  Ghat_l[4] = 0.5*alpha[0]*fUpwind_l[4]+0.4472135954999579*alpha[1]*fUpwind_l[1]; 
  Ghat_l[5] = 0.5000000000000001*alpha[1]*fUpwind_l[7]+0.5*alpha[0]*fUpwind_l[5]; 
  Ghat_l[6] = 0.5*alpha[0]*fUpwind_l[6]+0.447213595499958*alpha[1]*fUpwind_l[3]; 
  Ghat_l[7] = 0.5*alpha[0]*fUpwind_l[7]+0.5000000000000001*alpha[1]*fUpwind_l[5]; 

  Ghat_r[0] = 0.5*alpha[1]*fUpwind_r[1]+0.5*alpha[0]*fUpwind_r[0]; 
  Ghat_r[1] = 0.4472135954999579*alpha[1]*fUpwind_r[4]+0.5*alpha[0]*fUpwind_r[1]+0.5*fUpwind_r[0]*alpha[1]; 
  Ghat_r[2] = 0.5*alpha[1]*fUpwind_r[3]+0.5*alpha[0]*fUpwind_r[2]; 
  Ghat_r[3] = 0.447213595499958*alpha[1]*fUpwind_r[6]+0.5*alpha[0]*fUpwind_r[3]+0.5*alpha[1]*fUpwind_r[2]; 
  Ghat_r[4] = 0.5*alpha[0]*fUpwind_r[4]+0.4472135954999579*alpha[1]*fUpwind_r[1]; 
  Ghat_r[5] = 0.5000000000000001*alpha[1]*fUpwind_r[7]+0.5*alpha[0]*fUpwind_r[5]; 
  Ghat_r[6] = 0.5*alpha[0]*fUpwind_r[6]+0.447213595499958*alpha[1]*fUpwind_r[3]; 
  Ghat_r[7] = 0.5*alpha[0]*fUpwind_r[7]+0.5000000000000001*alpha[1]*fUpwind_r[5]; 

  out[0] += (0.7071067811865475*Ghat_l[0]-0.7071067811865475*Ghat_r[0])*rdv2; 
  out[1] += (0.7071067811865475*Ghat_l[1]-0.7071067811865475*Ghat_r[1])*rdv2; 
  out[2] += -1.224744871391589*(Ghat_r[0]+Ghat_l[0])*rdv2; 
  out[3] += (0.7071067811865475*Ghat_l[2]-0.7071067811865475*Ghat_r[2])*rdv2; 
  out[4] += -1.224744871391589*(Ghat_r[1]+Ghat_l[1])*rdv2; 
  out[5] += (0.7071067811865475*Ghat_l[3]-0.7071067811865475*Ghat_r[3])*rdv2; 
  out[6] += -1.224744871391589*(Ghat_r[2]+Ghat_l[2])*rdv2; 
  out[7] += (0.7071067811865475*Ghat_l[4]-0.7071067811865475*Ghat_r[4])*rdv2; 
  out[8] += (1.58113883008419*Ghat_l[0]-1.58113883008419*Ghat_r[0])*rdv2; 
  out[9] += (0.7071067811865475*Ghat_l[5]-0.7071067811865475*Ghat_r[5])*rdv2; 
  out[10] += -1.224744871391589*(Ghat_r[3]+Ghat_l[3])*rdv2; 
  out[11] += -1.224744871391589*(Ghat_r[4]+Ghat_l[4])*rdv2; 
  out[12] += (1.58113883008419*Ghat_l[1]-1.58113883008419*Ghat_r[1])*rdv2; 
  out[13] += (0.7071067811865475*Ghat_l[6]-0.7071067811865475*Ghat_r[6])*rdv2; 
  out[14] += (1.58113883008419*Ghat_l[2]-1.58113883008419*Ghat_r[2])*rdv2; 
  out[15] += (0.7071067811865475*Ghat_l[7]-0.7071067811865475*Ghat_r[7])*rdv2; 
  out[16] += -1.224744871391589*(Ghat_r[5]+Ghat_l[5])*rdv2; 
  out[17] += -1.224744871391589*(Ghat_r[6]+Ghat_l[6])*rdv2; 
  out[18] += (1.58113883008419*Ghat_l[3]-1.58113883008419*Ghat_r[3])*rdv2; 
  out[19] += -1.224744871391589*(Ghat_r[7]+Ghat_l[7])*rdv2; 

  return 0.;

} 
