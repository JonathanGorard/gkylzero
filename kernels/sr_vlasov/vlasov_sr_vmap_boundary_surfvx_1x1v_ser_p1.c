#include <gkyl_vlasov_sr_kernels.h> 
#include <gkyl_basis_hyb_1x1v_p1_surfx2_eval_quad.h> 
#include <gkyl_basis_hyb_1x1v_p1_upwind_quad_to_modal.h> 
GKYL_CU_DH double vlasov_sr_vmap_boundary_surfvx_1x1v_ser_p1(const double *w, const double *dxv, const double *jacob_vel_inv, const double *gamma, const double *qmem, const int edge, const double *fEdge, const double *fSkin, double* GKYL_RESTRICT out) 
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
  const double *E0 = &qmem[0]; 
  const double *jacob_vel_inv0 = &jacob_vel_inv[0]; 
  double alpha[2] = {0.0}; 

  double fUpwindQuad[2] = {0.0};
  double fUpwind[2] = {0.0};
  double Ghat[2] = {0.0}; 

  if (edge == -1) { 

  alpha[0] = E0[0]; 
  alpha[1] = E0[1]; 

  if (0.7071067811865475*alpha[0]-0.7071067811865475*alpha[1] > 0) { 
    fUpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(fSkin); 
  } else { 
    fUpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(fEdge); 
  } 
  if (0.7071067811865475*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(fSkin); 
  } else { 
    fUpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(fEdge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.7071067811865475*alpha[1]*fUpwind[1]+0.7071067811865475*alpha[0]*fUpwind[0]; 
  Ghat[1] = 0.7071067811865475*alpha[0]*fUpwind[1]+0.7071067811865475*fUpwind[0]*alpha[1]; 

  out[0] += Ghat[0]*((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*dv10; 
  out[1] += Ghat[1]*((-1.118033988749895*jacob_vel_inv0[2])-0.8660254037844386*jacob_vel_inv0[1]-0.5*jacob_vel_inv0[0])*dv10; 
  out[2] += Ghat[0]*((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[3] += Ghat[1]*((-1.936491673103709*jacob_vel_inv0[2])-1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[4] += Ghat[0]*((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[5] += Ghat[1]*((-2.5*jacob_vel_inv0[2])-1.936491673103709*jacob_vel_inv0[1]-1.118033988749895*jacob_vel_inv0[0])*dv10; 

  } else { 

  alpha[0] = E0[0]; 
  alpha[1] = E0[1]; 

  if (0.7071067811865475*alpha[0]-0.7071067811865475*alpha[1] > 0) { 
    fUpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(fEdge); 
  } else { 
    fUpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(fSkin); 
  } 
  if (0.7071067811865475*(alpha[1]+alpha[0]) > 0) { 
    fUpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(fEdge); 
  } else { 
    fUpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(fSkin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(fUpwindQuad, fUpwind); 

  Ghat[0] = 0.7071067811865475*alpha[1]*fUpwind[1]+0.7071067811865475*alpha[0]*fUpwind[0]; 
  Ghat[1] = 0.7071067811865475*alpha[0]*fUpwind[1]+0.7071067811865475*fUpwind[0]*alpha[1]; 

  out[0] += Ghat[0]*(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*dv10; 
  out[1] += Ghat[1]*(1.118033988749895*jacob_vel_inv0[2]-0.8660254037844386*jacob_vel_inv0[1]+0.5*jacob_vel_inv0[0])*dv10; 
  out[2] += Ghat[0]*((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[3] += Ghat[1]*((-1.936491673103709*jacob_vel_inv0[2])+1.5*jacob_vel_inv0[1]-0.8660254037844386*jacob_vel_inv0[0])*dv10; 
  out[4] += Ghat[0]*(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*dv10; 
  out[5] += Ghat[1]*(2.5*jacob_vel_inv0[2]-1.936491673103709*jacob_vel_inv0[1]+1.118033988749895*jacob_vel_inv0[0])*dv10; 

  } 
  return 0.;

} 