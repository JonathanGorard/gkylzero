#include <gkyl_vlasov_sr_kernels.h> 
#include <gkyl_basis_hyb_1x2v_p1_surfx2_eval_quad.h> 
#include <gkyl_basis_hyb_1x2v_p1_upwind_quad_to_modal.h> 
GKYL_CU_DH double vlasov_sr_vmap_boundary_surfvx_1x2v_ser_p1(const double *w, const double *dxv, const double *jacob_vel_inv, const double *gamma, const double *qmem, const int edge, const double *fEdge, const double *fSkin, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:             Cell-center coordinates.
  // dxv[NDIM]:           Cell spacing.
  // jacob_vel_inv[VDIM]: Inverse velocity space Jacobian in each direction.
  // gamma:               Particle Lorentz boost factor sqrt(1 + p^2).
  // qmem:                q/m*EM fields.
  // edge:                Determines if the update is for the left edge (-1) or right edge (+1).
  // fSkin/fEdge:         Input Distribution function in skin cell/last edge cell 
  // out:                 Output distribution function in skin cell 
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

  double alpha[6] = {0.0}; 

  double fUpwindQuad[6] = {0.0};
  double fUpwind[6] = {0.0};
  double Ghat[6] = {0.0}; 

  if (edge == -1) { 

  alpha[0] = B2[0]*p1_over_gamma_r[0]+1.414213562373095*E0[0]; 
  alpha[1] = 1.414213562373095*E0[1]+p1_over_gamma_r[0]*B2[1]; 
  alpha[2] = B2[0]*p1_over_gamma_r[1]; 
  alpha[3] = B2[1]*p1_over_gamma_r[1]; 
  alpha[4] = B2[0]*p1_over_gamma_r[2]; 
  alpha[5] = 1.0*B2[1]*p1_over_gamma_r[2]; 

  if ((-0.447213595499958*alpha[5])+0.4472135954999579*alpha[4]+0.6708203932499369*alpha[3]-0.6708203932499369*alpha[2]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_r(fSkin); 
  } else { 
    fUpwindQuad[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_l(fEdge); 
  } 
  if (0.5590169943749476*alpha[5]-0.5590169943749475*alpha[4]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_r(fSkin); 
  } else { 
    fUpwindQuad[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_l(fEdge); 
  } 
  if ((-0.447213595499958*alpha[5])+0.4472135954999579*alpha[4]-0.6708203932499369*alpha[3]+0.6708203932499369*alpha[2]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_r(fSkin); 
  } else { 
    fUpwindQuad[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_l(fEdge); 
  } 
  if (0.447213595499958*alpha[5]+0.4472135954999579*alpha[4]-0.6708203932499369*(alpha[3]+alpha[2])+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_r(fSkin); 
  } else { 
    fUpwindQuad[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_l(fEdge); 
  } 
  if ((-0.5590169943749476*alpha[5])-0.5590169943749475*alpha[4]+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_r(fSkin); 
  } else { 
    fUpwindQuad[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_l(fEdge); 
  } 
  if (0.447213595499958*alpha[5]+0.4472135954999579*alpha[4]+0.6708203932499369*(alpha[3]+alpha[2])+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_r(fSkin); 
  } else { 
    fUpwindQuad[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_l(fEdge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x2v_p1_vdir_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.5*alpha[5]*fUpwind[5]+0.5*alpha[4]*fUpwind[4]+0.5*alpha[3]*fUpwind[3]+0.5*alpha[2]*fUpwind[2]+0.5*alpha[1]*fUpwind[1]+0.5*alpha[0]*fUpwind[0]; 
  Ghat[1] = 0.5000000000000001*alpha[4]*fUpwind[5]+0.5000000000000001*fUpwind[4]*alpha[5]+0.5*alpha[2]*fUpwind[3]+0.5*fUpwind[2]*alpha[3]+0.5*alpha[0]*fUpwind[1]+0.5*fUpwind[0]*alpha[1]; 
  Ghat[2] = 0.447213595499958*alpha[3]*fUpwind[5]+0.447213595499958*fUpwind[3]*alpha[5]+0.4472135954999579*alpha[2]*fUpwind[4]+0.4472135954999579*fUpwind[2]*alpha[4]+0.5*alpha[1]*fUpwind[3]+0.5*fUpwind[1]*alpha[3]+0.5*alpha[0]*fUpwind[2]+0.5*fUpwind[0]*alpha[2]; 
  Ghat[3] = 0.447213595499958*alpha[2]*fUpwind[5]+0.447213595499958*fUpwind[2]*alpha[5]+0.4472135954999579*alpha[3]*fUpwind[4]+0.4472135954999579*fUpwind[3]*alpha[4]+0.5*alpha[0]*fUpwind[3]+0.5*fUpwind[0]*alpha[3]+0.5*alpha[1]*fUpwind[2]+0.5*fUpwind[1]*alpha[2]; 
  Ghat[4] = 0.31943828249997*alpha[5]*fUpwind[5]+0.5000000000000001*alpha[1]*fUpwind[5]+0.5000000000000001*fUpwind[1]*alpha[5]+0.31943828249997*alpha[4]*fUpwind[4]+0.5*alpha[0]*fUpwind[4]+0.5*fUpwind[0]*alpha[4]+0.4472135954999579*alpha[3]*fUpwind[3]+0.4472135954999579*alpha[2]*fUpwind[2]; 
  Ghat[5] = 0.31943828249997*alpha[4]*fUpwind[5]+0.5*alpha[0]*fUpwind[5]+0.31943828249997*fUpwind[4]*alpha[5]+0.5*fUpwind[0]*alpha[5]+0.5000000000000001*alpha[1]*fUpwind[4]+0.5000000000000001*fUpwind[1]*alpha[4]+0.447213595499958*alpha[2]*fUpwind[3]+0.447213595499958*fUpwind[2]*alpha[3]; 

  out[0] += Ghat[0]*((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*dv10; 
  out[1] += Ghat[1]*((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*dv10; 
  out[2] += Ghat[0]*((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[3] += Ghat[2]*((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*dv10; 
  out[4] += Ghat[1]*((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[5] += ((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[6] += Ghat[2]*((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[7] += ((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[8] += Ghat[0]*((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[9] += Ghat[1]*((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[10] += Ghat[2]*((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[11] += ((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[12] += ((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat[4]*dv10; 
  out[13] += ((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*Ghat[5]*dv10; 
  out[14] += ((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat[4]*dv10; 
  out[15] += ((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat[5]*dv10; 

  } else { 

  alpha[0] = B2[0]*p1_over_gamma_l[0]+1.414213562373095*E0[0]; 
  alpha[1] = 1.414213562373095*E0[1]+p1_over_gamma_l[0]*B2[1]; 
  alpha[2] = B2[0]*p1_over_gamma_l[1]; 
  alpha[3] = B2[1]*p1_over_gamma_l[1]; 
  alpha[4] = B2[0]*p1_over_gamma_l[2]; 
  alpha[5] = 1.0*B2[1]*p1_over_gamma_l[2]; 

  if ((-0.447213595499958*alpha[5])+0.4472135954999579*alpha[4]+0.6708203932499369*alpha[3]-0.6708203932499369*alpha[2]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_r(fEdge); 
  } else { 
    fUpwindQuad[0] = hyb_1x2v_p1_surfx2_eval_quad_node_0_l(fSkin); 
  } 
  if (0.5590169943749476*alpha[5]-0.5590169943749475*alpha[4]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_r(fEdge); 
  } else { 
    fUpwindQuad[1] = hyb_1x2v_p1_surfx2_eval_quad_node_1_l(fSkin); 
  } 
  if ((-0.447213595499958*alpha[5])+0.4472135954999579*alpha[4]-0.6708203932499369*alpha[3]+0.6708203932499369*alpha[2]-0.5*alpha[1]+0.5*alpha[0] > 0) { 
    fUpwindQuad[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_r(fEdge); 
  } else { 
    fUpwindQuad[2] = hyb_1x2v_p1_surfx2_eval_quad_node_2_l(fSkin); 
  } 
  if (0.447213595499958*alpha[5]+0.4472135954999579*alpha[4]-0.6708203932499369*(alpha[3]+alpha[2])+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_r(fEdge); 
  } else { 
    fUpwindQuad[3] = hyb_1x2v_p1_surfx2_eval_quad_node_3_l(fSkin); 
  } 
  if ((-0.5590169943749476*alpha[5])-0.5590169943749475*alpha[4]+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_r(fEdge); 
  } else { 
    fUpwindQuad[4] = hyb_1x2v_p1_surfx2_eval_quad_node_4_l(fSkin); 
  } 
  if (0.447213595499958*alpha[5]+0.4472135954999579*alpha[4]+0.6708203932499369*(alpha[3]+alpha[2])+0.5*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_r(fEdge); 
  } else { 
    fUpwindQuad[5] = hyb_1x2v_p1_surfx2_eval_quad_node_5_l(fSkin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x2v_p1_vdir_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.5*alpha[5]*fUpwind[5]+0.5*alpha[4]*fUpwind[4]+0.5*alpha[3]*fUpwind[3]+0.5*alpha[2]*fUpwind[2]+0.5*alpha[1]*fUpwind[1]+0.5*alpha[0]*fUpwind[0]; 
  Ghat[1] = 0.5000000000000001*alpha[4]*fUpwind[5]+0.5000000000000001*fUpwind[4]*alpha[5]+0.5*alpha[2]*fUpwind[3]+0.5*fUpwind[2]*alpha[3]+0.5*alpha[0]*fUpwind[1]+0.5*fUpwind[0]*alpha[1]; 
  Ghat[2] = 0.447213595499958*alpha[3]*fUpwind[5]+0.447213595499958*fUpwind[3]*alpha[5]+0.4472135954999579*alpha[2]*fUpwind[4]+0.4472135954999579*fUpwind[2]*alpha[4]+0.5*alpha[1]*fUpwind[3]+0.5*fUpwind[1]*alpha[3]+0.5*alpha[0]*fUpwind[2]+0.5*fUpwind[0]*alpha[2]; 
  Ghat[3] = 0.447213595499958*alpha[2]*fUpwind[5]+0.447213595499958*fUpwind[2]*alpha[5]+0.4472135954999579*alpha[3]*fUpwind[4]+0.4472135954999579*fUpwind[3]*alpha[4]+0.5*alpha[0]*fUpwind[3]+0.5*fUpwind[0]*alpha[3]+0.5*alpha[1]*fUpwind[2]+0.5*fUpwind[1]*alpha[2]; 
  Ghat[4] = 0.31943828249997*alpha[5]*fUpwind[5]+0.5000000000000001*alpha[1]*fUpwind[5]+0.5000000000000001*fUpwind[1]*alpha[5]+0.31943828249997*alpha[4]*fUpwind[4]+0.5*alpha[0]*fUpwind[4]+0.5*fUpwind[0]*alpha[4]+0.4472135954999579*alpha[3]*fUpwind[3]+0.4472135954999579*alpha[2]*fUpwind[2]; 
  Ghat[5] = 0.31943828249997*alpha[4]*fUpwind[5]+0.5*alpha[0]*fUpwind[5]+0.31943828249997*fUpwind[4]*alpha[5]+0.5*fUpwind[0]*alpha[5]+0.5000000000000001*alpha[1]*fUpwind[4]+0.5000000000000001*fUpwind[1]*alpha[4]+0.447213595499958*alpha[2]*fUpwind[3]+0.447213595499958*fUpwind[2]*alpha[3]; 

  out[0] += Ghat[0]*(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*dv10; 
  out[1] += Ghat[1]*(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*dv10; 
  out[2] += Ghat[0]*((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[3] += Ghat[2]*(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*dv10; 
  out[4] += Ghat[1]*((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[5] += (1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[6] += Ghat[2]*((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[7] += ((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[8] += Ghat[0]*(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[9] += Ghat[1]*(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[10] += Ghat[2]*(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[11] += (2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*Ghat[3]*dv10; 
  out[12] += (1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat[4]*dv10; 
  out[13] += (1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*Ghat[5]*dv10; 
  out[14] += ((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat[4]*dv10; 
  out[15] += ((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844387*jacob_vel_inv0[0])*Ghat[5]*dv10; 

  } 
  return 0.;

} 