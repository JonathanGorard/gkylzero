#include <gkyl_vlasov_sr_kernels.h> 
#include <gkyl_basis_hyb_1x2v_p1_surfx2_eval_quad.h> 
#include <gkyl_basis_hyb_1x2v_p1_upwind_quad_to_modal.h> 
GKYL_CU_DH double vlasov_sr_vmap_surfvx_1x2v_ser_p1(const double *w, const double *dxv, const double *jacob_vel_inv, const double *gamma, const double *qmem, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:             Cell-center coordinates.
  // dxv[NDIM]:           Cell spacing.
  // jacob_vel_inv[VDIM]: Inverse velocity space Jacobian in each direction.
  // gamma:               Particle Lorentz boost factor sqrt(1 + p^2).
  // qmem:                q/m*EM fields.
  // fl/fc/fr:            Input Distribution function in left/center/right cells 
  // out:                 Output distribution function in center cell 
  const double dv10 = 2.0/dxv[1]; 
  const double dv11 = 2.0/dxv[2]; 
  const double *E0 = &qmem[0]; 
  const double *jacob_vel_inv0 = &jacob_vel_inv[0]; 
  const double *jacob_vel_inv1 = &jacob_vel_inv[3]; 
  double p1_over_gamma_l[3] = {0.0}; 
  double p1_over_gamma_r[3] = {0.0}; 
  p1_over_gamma_l[0] = (-3.354101966249684*jacob_vel_inv1[1]*gamma[7]*dv11)+1.936491673103709*jacob_vel_inv1[0]*gamma[6]*dv11+1.936491673103709*jacob_vel_inv1[1]*gamma[5]*dv11-1.5*jacob_vel_inv1[0]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[0]*gamma[2]*dv11; 
  p1_over_gamma_l[1] = (-3.0*jacob_vel_inv1[2]*gamma[7]*dv11)-3.354101966249684*jacob_vel_inv1[0]*gamma[7]*dv11+1.936491673103709*jacob_vel_inv1[1]*gamma[6]*dv11+1.732050807568877*jacob_vel_inv1[2]*gamma[5]*dv11+1.936491673103709*jacob_vel_inv1[0]*gamma[5]*dv11-1.5*jacob_vel_inv1[1]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[1]*gamma[2]*dv11; 
  p1_over_gamma_l[2] = (-3.0*jacob_vel_inv1[1]*gamma[7]*dv11)+1.936491673103709*jacob_vel_inv1[2]*gamma[6]*dv11+1.732050807568877*jacob_vel_inv1[1]*gamma[5]*dv11-1.5*jacob_vel_inv1[2]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[2]*gamma[2]*dv11; 
  p1_over_gamma_r[0] = 3.354101966249684*jacob_vel_inv1[1]*gamma[7]*dv11+1.936491673103709*jacob_vel_inv1[0]*gamma[6]*dv11+1.936491673103709*jacob_vel_inv1[1]*gamma[5]*dv11+1.5*jacob_vel_inv1[0]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[0]*gamma[2]*dv11; 
  p1_over_gamma_r[1] = 3.0*jacob_vel_inv1[2]*gamma[7]*dv11+3.354101966249684*jacob_vel_inv1[0]*gamma[7]*dv11+1.936491673103709*jacob_vel_inv1[1]*gamma[6]*dv11+1.732050807568877*jacob_vel_inv1[2]*gamma[5]*dv11+1.936491673103709*jacob_vel_inv1[0]*gamma[5]*dv11+1.5*jacob_vel_inv1[1]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[1]*gamma[2]*dv11; 
  p1_over_gamma_r[2] = 3.0*jacob_vel_inv1[1]*gamma[7]*dv11+1.936491673103709*jacob_vel_inv1[2]*gamma[6]*dv11+1.732050807568877*jacob_vel_inv1[1]*gamma[5]*dv11+1.5*jacob_vel_inv1[2]*gamma[3]*dv11+0.8660254037844386*jacob_vel_inv1[2]*gamma[2]*dv11; 

  const double *B2 = &qmem[10]; 

  double alpha_l[6] = {0.0}; 
  double alpha_r[6] = {0.0}; 

  alpha_l[0] = B2[0]*p1_over_gamma_l[0]+1.414213562373095*E0[0]; 
  alpha_l[1] = 1.414213562373095*E0[1]+p1_over_gamma_l[0]*B2[1]; 
  alpha_l[2] = B2[0]*p1_over_gamma_l[1]; 
  alpha_l[3] = B2[1]*p1_over_gamma_l[1]; 
  alpha_l[4] = B2[0]*p1_over_gamma_l[2]; 
  alpha_l[5] = 1.0*B2[1]*p1_over_gamma_l[2]; 

  alpha_r[0] = B2[0]*p1_over_gamma_r[0]+1.414213562373095*E0[0]; 
  alpha_r[1] = 1.414213562373095*E0[1]+p1_over_gamma_r[0]*B2[1]; 
  alpha_r[2] = B2[0]*p1_over_gamma_r[1]; 
  alpha_r[3] = B2[1]*p1_over_gamma_r[1]; 
  alpha_r[4] = B2[0]*p1_over_gamma_r[2]; 
  alpha_r[5] = 1.0*B2[1]*p1_over_gamma_r[2]; 

  double fUpwindQuad_l[6] = {0.0};
  double fUpwindQuad_r[6] = {0.0};
  double fUpwind_l[6] = {0.0};;
  double fUpwind_r[6] = {0.0};
  double Ghat_l[6] = {0.0}; 
  double Ghat_r[6] = {0.0}; 

  if ((-0.447213595499958*alpha_l[5])+0.4472135954999579*alpha_l[4]+0.6708203932499369*alpha_l[3]-0.6708203932499369*alpha_l[2]-0.5*alpha_l[1]+0.5*alpha_l[0] > 0) { 
    fUpwindQuad_l[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_r(fl); 
  } else { 
    fUpwindQuad_l[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_l(fc); 
  } 
  if ((-0.447213595499958*alpha_r[5])+0.4472135954999579*alpha_r[4]+0.6708203932499369*alpha_r[3]-0.6708203932499369*alpha_r[2]-0.5*alpha_r[1]+0.5*alpha_r[0] > 0) { 
    fUpwindQuad_r[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_r(fc); 
  } else { 
    fUpwindQuad_r[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_l(fr); 
  } 
  if (0.5590169943749476*alpha_l[5]-0.5590169943749475*alpha_l[4]-0.5*alpha_l[1]+0.5*alpha_l[0] > 0) { 
    fUpwindQuad_l[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_r(fl); 
  } else { 
    fUpwindQuad_l[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_l(fc); 
  } 
  if (0.5590169943749476*alpha_r[5]-0.5590169943749475*alpha_r[4]-0.5*alpha_r[1]+0.5*alpha_r[0] > 0) { 
    fUpwindQuad_r[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_r(fc); 
  } else { 
    fUpwindQuad_r[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_l(fr); 
  } 
  if ((-0.447213595499958*alpha_l[5])+0.4472135954999579*alpha_l[4]-0.6708203932499369*alpha_l[3]+0.6708203932499369*alpha_l[2]-0.5*alpha_l[1]+0.5*alpha_l[0] > 0) { 
    fUpwindQuad_l[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_r(fl); 
  } else { 
    fUpwindQuad_l[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_l(fc); 
  } 
  if ((-0.447213595499958*alpha_r[5])+0.4472135954999579*alpha_r[4]-0.6708203932499369*alpha_r[3]+0.6708203932499369*alpha_r[2]-0.5*alpha_r[1]+0.5*alpha_r[0] > 0) { 
    fUpwindQuad_r[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_r(fc); 
  } else { 
    fUpwindQuad_r[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_l(fr); 
  } 
  if (0.447213595499958*alpha_l[5]+0.4472135954999579*alpha_l[4]-0.6708203932499369*(alpha_l[3]+alpha_l[2])+0.5*(alpha_l[1]+alpha_l[0]) > 0) { 
    fUpwindQuad_l[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_r(fl); 
  } else { 
    fUpwindQuad_l[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_l(fc); 
  } 
  if (0.447213595499958*alpha_r[5]+0.4472135954999579*alpha_r[4]-0.6708203932499369*(alpha_r[3]+alpha_r[2])+0.5*(alpha_r[1]+alpha_r[0]) > 0) { 
    fUpwindQuad_r[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_r(fc); 
  } else { 
    fUpwindQuad_r[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_l(fr); 
  } 
  if ((-0.5590169943749476*alpha_l[5])-0.5590169943749475*alpha_l[4]+0.5*(alpha_l[1]+alpha_l[0]) > 0) { 
    fUpwindQuad_l[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_r(fl); 
  } else { 
    fUpwindQuad_l[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_l(fc); 
  } 
  if ((-0.5590169943749476*alpha_r[5])-0.5590169943749475*alpha_r[4]+0.5*(alpha_r[1]+alpha_r[0]) > 0) { 
    fUpwindQuad_r[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_r(fc); 
  } else { 
    fUpwindQuad_r[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_l(fr); 
  } 
  if (0.447213595499958*alpha_l[5]+0.4472135954999579*alpha_l[4]+0.6708203932499369*(alpha_l[3]+alpha_l[2])+0.5*(alpha_l[1]+alpha_l[0]) > 0) { 
    fUpwindQuad_l[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_r(fl); 
  } else { 
    fUpwindQuad_l[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_l(fc); 
  } 
  if (0.447213595499958*alpha_r[5]+0.4472135954999579*alpha_r[4]+0.6708203932499369*(alpha_r[3]+alpha_r[2])+0.5*(alpha_r[1]+alpha_r[0]) > 0) { 
    fUpwindQuad_r[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_r(fc); 
  } else { 
    fUpwindQuad_r[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_l(fr); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x2v_p1_vdir_upwind_quad_to_modal(fUpwindQuad_l, fUpwind_l); 
  hyb_1x2v_p1_vdir_upwind_quad_to_modal(fUpwindQuad_r, fUpwind_r); 
  Ghat_l[0] = 0.5*alpha_l[5]*fUpwind_l[5]+0.5*alpha_l[4]*fUpwind_l[4]+0.5*alpha_l[3]*fUpwind_l[3]+0.5*alpha_l[2]*fUpwind_l[2]+0.5*alpha_l[1]*fUpwind_l[1]+0.5*alpha_l[0]*fUpwind_l[0]; 
  Ghat_l[1] = 0.5000000000000001*alpha_l[4]*fUpwind_l[5]+0.5000000000000001*fUpwind_l[4]*alpha_l[5]+0.5*alpha_l[2]*fUpwind_l[3]+0.5*fUpwind_l[2]*alpha_l[3]+0.5*alpha_l[0]*fUpwind_l[1]+0.5*fUpwind_l[0]*alpha_l[1]; 
  Ghat_l[2] = 0.447213595499958*alpha_l[3]*fUpwind_l[5]+0.447213595499958*fUpwind_l[3]*alpha_l[5]+0.4472135954999579*alpha_l[2]*fUpwind_l[4]+0.4472135954999579*fUpwind_l[2]*alpha_l[4]+0.5*alpha_l[1]*fUpwind_l[3]+0.5*fUpwind_l[1]*alpha_l[3]+0.5*alpha_l[0]*fUpwind_l[2]+0.5*fUpwind_l[0]*alpha_l[2]; 
  Ghat_l[3] = 0.447213595499958*alpha_l[2]*fUpwind_l[5]+0.447213595499958*fUpwind_l[2]*alpha_l[5]+0.4472135954999579*alpha_l[3]*fUpwind_l[4]+0.4472135954999579*fUpwind_l[3]*alpha_l[4]+0.5*alpha_l[0]*fUpwind_l[3]+0.5*fUpwind_l[0]*alpha_l[3]+0.5*alpha_l[1]*fUpwind_l[2]+0.5*fUpwind_l[1]*alpha_l[2]; 
  Ghat_l[4] = 0.31943828249997*alpha_l[5]*fUpwind_l[5]+0.5000000000000001*alpha_l[1]*fUpwind_l[5]+0.5000000000000001*fUpwind_l[1]*alpha_l[5]+0.31943828249997*alpha_l[4]*fUpwind_l[4]+0.5*alpha_l[0]*fUpwind_l[4]+0.5*fUpwind_l[0]*alpha_l[4]+0.4472135954999579*alpha_l[3]*fUpwind_l[3]+0.4472135954999579*alpha_l[2]*fUpwind_l[2]; 
  Ghat_l[5] = 0.31943828249997*alpha_l[4]*fUpwind_l[5]+0.5*alpha_l[0]*fUpwind_l[5]+0.31943828249997*fUpwind_l[4]*alpha_l[5]+0.5*fUpwind_l[0]*alpha_l[5]+0.5000000000000001*alpha_l[1]*fUpwind_l[4]+0.5000000000000001*fUpwind_l[1]*alpha_l[4]+0.447213595499958*alpha_l[2]*fUpwind_l[3]+0.447213595499958*fUpwind_l[2]*alpha_l[3]; 

  Ghat_r[0] = 0.5*alpha_r[5]*fUpwind_r[5]+0.5*alpha_r[4]*fUpwind_r[4]+0.5*alpha_r[3]*fUpwind_r[3]+0.5*alpha_r[2]*fUpwind_r[2]+0.5*alpha_r[1]*fUpwind_r[1]+0.5*alpha_r[0]*fUpwind_r[0]; 
  Ghat_r[1] = 0.5000000000000001*alpha_r[4]*fUpwind_r[5]+0.5000000000000001*fUpwind_r[4]*alpha_r[5]+0.5*alpha_r[2]*fUpwind_r[3]+0.5*fUpwind_r[2]*alpha_r[3]+0.5*alpha_r[0]*fUpwind_r[1]+0.5*fUpwind_r[0]*alpha_r[1]; 
  Ghat_r[2] = 0.447213595499958*alpha_r[3]*fUpwind_r[5]+0.447213595499958*fUpwind_r[3]*alpha_r[5]+0.4472135954999579*alpha_r[2]*fUpwind_r[4]+0.4472135954999579*fUpwind_r[2]*alpha_r[4]+0.5*alpha_r[1]*fUpwind_r[3]+0.5*fUpwind_r[1]*alpha_r[3]+0.5*alpha_r[0]*fUpwind_r[2]+0.5*fUpwind_r[0]*alpha_r[2]; 
  Ghat_r[3] = 0.447213595499958*alpha_r[2]*fUpwind_r[5]+0.447213595499958*fUpwind_r[2]*alpha_r[5]+0.4472135954999579*alpha_r[3]*fUpwind_r[4]+0.4472135954999579*fUpwind_r[3]*alpha_r[4]+0.5*alpha_r[0]*fUpwind_r[3]+0.5*fUpwind_r[0]*alpha_r[3]+0.5*alpha_r[1]*fUpwind_r[2]+0.5*fUpwind_r[1]*alpha_r[2]; 
  Ghat_r[4] = 0.31943828249997*alpha_r[5]*fUpwind_r[5]+0.5000000000000001*alpha_r[1]*fUpwind_r[5]+0.5000000000000001*fUpwind_r[1]*alpha_r[5]+0.31943828249997*alpha_r[4]*fUpwind_r[4]+0.5*alpha_r[0]*fUpwind_r[4]+0.5*fUpwind_r[0]*alpha_r[4]+0.4472135954999579*alpha_r[3]*fUpwind_r[3]+0.4472135954999579*alpha_r[2]*fUpwind_r[2]; 
  Ghat_r[5] = 0.31943828249997*alpha_r[4]*fUpwind_r[5]+0.5*alpha_r[0]*fUpwind_r[5]+0.31943828249997*fUpwind_r[4]*alpha_r[5]+0.5*fUpwind_r[0]*alpha_r[5]+0.5000000000000001*alpha_r[1]*fUpwind_r[4]+0.5000000000000001*fUpwind_r[1]*alpha_r[4]+0.447213595499958*alpha_r[2]*fUpwind_r[3]+0.447213595499958*fUpwind_r[2]*alpha_r[3]; 

  out[0] += ((1.118033988749895*Ghat_l[0]-1.118033988749895*Ghat_r[0])*jacob_vel_inv0[2]-0.8660254037844386*(Ghat_r[0]+Ghat_l[0])*jacob_vel_inv0[1]+(0.5*Ghat_l[0]-0.5*Ghat_r[0])*jacob_vel_inv0[0])*dv10; 
  out[1] += ((1.118033988749895*Ghat_l[1]-1.118033988749895*Ghat_r[1])*jacob_vel_inv0[2]-0.8660254037844386*(Ghat_r[1]+Ghat_l[1])*jacob_vel_inv0[1]+jacob_vel_inv0[0]*(0.5*Ghat_l[1]-0.5*Ghat_r[1]))*dv10; 
  out[2] += ((-1.936491673103709*(Ghat_r[0]+Ghat_l[0])*jacob_vel_inv0[2])+(1.5*Ghat_l[0]-1.5*Ghat_r[0])*jacob_vel_inv0[1]-0.8660254037844386*(Ghat_r[0]+Ghat_l[0])*jacob_vel_inv0[0])*dv10; 
  out[3] += ((1.118033988749895*Ghat_l[2]-1.118033988749895*Ghat_r[2])*jacob_vel_inv0[2]+((-0.8660254037844386*jacob_vel_inv0[1])-0.5*jacob_vel_inv0[0])*Ghat_r[2]+(0.5*jacob_vel_inv0[0]-0.8660254037844386*jacob_vel_inv0[1])*Ghat_l[2])*dv10; 
  out[4] += ((-1.936491673103709*(Ghat_r[1]+Ghat_l[1])*jacob_vel_inv0[2])+(1.5*Ghat_l[1]-1.5*Ghat_r[1])*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0]*(Ghat_r[1]+Ghat_l[1]))*dv10; 
  out[5] += (((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat_r[3]+(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat_l[3])*dv10; 
  out[6] += ((-1.936491673103709*(Ghat_r[2]+Ghat_l[2])*jacob_vel_inv0[2])+((-1.5*jacob_vel_inv0[1])-0.8660254037844386*jacob_vel_inv0[0])*Ghat_r[2]+(1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*Ghat_l[2])*dv10; 
  out[7] += (((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*Ghat_r[3]+((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*Ghat_l[3])*dv10; 
  out[8] += ((2.5*Ghat_l[0]-2.5*Ghat_r[0])*jacob_vel_inv0[2]-1.936491673103709*(Ghat_r[0]+Ghat_l[0])*jacob_vel_inv0[1]+(1.118033988749895*Ghat_l[0]-1.118033988749895*Ghat_r[0])*jacob_vel_inv0[0])*dv10; 
  out[9] += ((2.5*Ghat_l[1]-2.5*Ghat_r[1])*jacob_vel_inv0[2]-1.936491673103709*(Ghat_r[1]+Ghat_l[1])*jacob_vel_inv0[1]+jacob_vel_inv0[0]*(1.118033988749895*Ghat_l[1]-1.118033988749895*Ghat_r[1]))*dv10; 
  out[10] += ((2.5*Ghat_l[2]-2.5*Ghat_r[2])*jacob_vel_inv0[2]+((-1.936491673103709*jacob_vel_inv0[1])-1.118033988749895*jacob_vel_inv0[0])*Ghat_r[2]+(1.118033988749895*jacob_vel_inv0[0]-1.936491673103709*jacob_vel_inv0[1])*Ghat_l[2])*dv10; 
  out[11] += (((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*Ghat_r[3]+(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*Ghat_l[3])*dv10; 
  out[12] += (((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat_r[4]+(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat_l[4])*dv10; 
  out[13] += (((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat_r[5]+(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat_l[5])*dv10; 
  out[14] += (((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat_r[4]+((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat_l[4])*dv10; 
  out[15] += (((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat_r[5]+((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat_l[5])*dv10; 

  return 0.;

} 