#include <gkyl_vlasov_pkpm_kernels.h> 
#include <gkyl_basis_hyb_1x1v_p1_surfx2_eval_quad.h> 
#include <gkyl_basis_hyb_1x1v_p1_upwind_quad_to_modal.h> 
GKYL_CU_DH void vlasov_pkpm_boundary_surfvpar_1x1v_ser_p1(const double *w, const double *dxv, const double *pkpm_accel_vars, 
     const double *g_dist_sourceEdge, const double *g_dist_sourceSkin, 
     const int edge, const double *fEdge, const double *fSkin, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:                Cell-center coordinates.
  // dxv[NDIM]:              Cell spacing.
  // pkpm_accel_vars:        pkpm acceleration variables
  // g_dist_sourceEdge/Skin: 2.0*T_perp/m*(2.0*T_perp/m*G_1 + T_perp/m*(F_2 - F_0)) in skin cell/last edge cell.
  // edge:                   Determines if the update is for the left edge (-1) or right edge (+1).
  // fSkin/fEdge:            Input Distribution function [F_0, T_perp G_1 = T_perp (F_1 - F_0)] in skin cell/last edge cell 
  // out:                    Incremented distribution function in center cell.
  const double dv1par = 2.0/dxv[1]; 
  const double dvpar = dxv[1], wvpar = w[1]; 
  const double *F_0Skin = &fSkin[0]; 
  const double *G_1Skin = &fSkin[6]; 
  const double *F_0Edge = &fEdge[0]; 
  const double *G_1Edge = &fEdge[6]; 
  const double *F_0_sourceSkin = &fSkin[6]; 
  const double *G_1_sourceSkin = &g_dist_sourceSkin[0]; 
  const double *F_0_sourceEdge = &fEdge[6]; 
  const double *G_1_sourceEdge = &g_dist_sourceEdge[0]; 
  const double *div_b = &pkpm_accel_vars[0]; 
  const double *bb_grad_u = &pkpm_accel_vars[2]; 
  const double *p_force = &pkpm_accel_vars[4]; 

  double *out_F_0 = &out[0]; 
  double *out_G_1 = &out[6]; 
  double div_b_Surf[2] = {0.0}; 
  div_b_Surf[0] = div_b[0]; 
  div_b_Surf[1] = div_b[1]; 

  double alphaSurf[2] = {0.0}; 
  double F_0_UpwindQuad[2] = {0.0};
  double F_0_Upwind[2] = {0.0};;
  double Ghat_F_0[2] = {0.0}; 
  double G_1_UpwindQuad[2] = {0.0};
  double G_1_Upwind[2] = {0.0};;
  double Ghat_G_1[2] = {0.0}; 
  double F_0_div_b_UpwindQuad[2] = {0.0};
  double F_0_div_b_Upwind[2] = {0.0};;
  double Ghat_F_0_div_b[2] = {0.0}; 
  double G_1_div_b_UpwindQuad[2] = {0.0};
  double G_1_div_b_Upwind[2] = {0.0};;
  double Ghat_G_1_div_b[2] = {0.0}; 

  if (edge == -1) { 

  alphaSurf[0] = (-1.0*bb_grad_u[0]*wvpar)-0.5*bb_grad_u[0]*dvpar+p_force[0]; 
  alphaSurf[1] = (-1.0*bb_grad_u[1]*wvpar)-0.5*bb_grad_u[1]*dvpar+p_force[1]; 

  if (0.7071067811865475*alphaSurf[0]-0.7071067811865475*alphaSurf[1] > 0) { 
    F_0_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(F_0Skin); 
    G_1_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(G_1Skin); 
  } else { 
    F_0_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(F_0Edge); 
    G_1_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(G_1Edge); 
  } 
  if (0.7071067811865475*(alphaSurf[1]+alphaSurf[0]) > 0) { 
    F_0_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(F_0Skin); 
    G_1_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(G_1Skin); 
  } else { 
    F_0_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(F_0Edge); 
    G_1_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(G_1Edge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(F_0_UpwindQuad, F_0_Upwind); 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(G_1_UpwindQuad, G_1_Upwind); 

  if (0.7071067811865475*div_b_Surf[0]-0.7071067811865475*div_b_Surf[1] > 0) { 
    F_0_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(F_0_sourceSkin); 
    G_1_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(G_1_sourceSkin); 
  } else { 
    F_0_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(F_0_sourceEdge); 
    G_1_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(G_1_sourceEdge); 
  } 
  if (0.7071067811865475*(div_b_Surf[1]+div_b_Surf[0]) > 0) { 
    F_0_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(F_0_sourceSkin); 
    G_1_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(G_1_sourceSkin); 
  } else { 
    F_0_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(F_0_sourceEdge); 
    G_1_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(G_1_sourceEdge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(F_0_div_b_UpwindQuad, F_0_div_b_Upwind); 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(G_1_div_b_UpwindQuad, G_1_div_b_Upwind); 

  Ghat_F_0[0] = 0.7071067811865475*F_0_Upwind[1]*alphaSurf[1]+0.7071067811865475*F_0_Upwind[0]*alphaSurf[0]; 
  Ghat_F_0[1] = 0.7071067811865475*F_0_Upwind[0]*alphaSurf[1]+0.7071067811865475*alphaSurf[0]*F_0_Upwind[1]; 
  Ghat_G_1[0] = 0.7071067811865475*G_1_Upwind[1]*alphaSurf[1]+0.7071067811865475*G_1_Upwind[0]*alphaSurf[0]; 
  Ghat_G_1[1] = 0.7071067811865475*G_1_Upwind[0]*alphaSurf[1]+0.7071067811865475*alphaSurf[0]*G_1_Upwind[1]; 
  Ghat_F_0_div_b[0] = 0.7071067811865475*F_0_div_b_Upwind[1]*div_b_Surf[1]+0.7071067811865475*F_0_div_b_Upwind[0]*div_b_Surf[0]; 
  Ghat_F_0_div_b[1] = 0.7071067811865475*F_0_div_b_Upwind[0]*div_b_Surf[1]+0.7071067811865475*div_b_Surf[0]*F_0_div_b_Upwind[1]; 
  Ghat_G_1_div_b[0] = 0.7071067811865475*G_1_div_b_Upwind[1]*div_b_Surf[1]+0.7071067811865475*G_1_div_b_Upwind[0]*div_b_Surf[0]; 
  Ghat_G_1_div_b[1] = 0.7071067811865475*G_1_div_b_Upwind[0]*div_b_Surf[1]+0.7071067811865475*div_b_Surf[0]*G_1_div_b_Upwind[1]; 

  out_F_0[0] += (-0.7071067811865475*Ghat_F_0_div_b[0]*dv1par)-0.7071067811865475*Ghat_F_0[0]*dv1par; 
  out_F_0[1] += (-0.7071067811865475*Ghat_F_0_div_b[1]*dv1par)-0.7071067811865475*Ghat_F_0[1]*dv1par; 
  out_F_0[2] += (-1.224744871391589*Ghat_F_0_div_b[0]*dv1par)-1.224744871391589*Ghat_F_0[0]*dv1par; 
  out_F_0[3] += (-1.224744871391589*Ghat_F_0_div_b[1]*dv1par)-1.224744871391589*Ghat_F_0[1]*dv1par; 
  out_F_0[4] += (-1.58113883008419*Ghat_F_0_div_b[0]*dv1par)-1.58113883008419*Ghat_F_0[0]*dv1par; 
  out_F_0[5] += (-1.58113883008419*Ghat_F_0_div_b[1]*dv1par)-1.58113883008419*Ghat_F_0[1]*dv1par; 
  out_G_1[0] += (-0.7071067811865475*Ghat_G_1_div_b[0]*dv1par)-0.7071067811865475*Ghat_G_1[0]*dv1par; 
  out_G_1[1] += (-0.7071067811865475*Ghat_G_1_div_b[1]*dv1par)-0.7071067811865475*Ghat_G_1[1]*dv1par; 
  out_G_1[2] += (-1.224744871391589*Ghat_G_1_div_b[0]*dv1par)-1.224744871391589*Ghat_G_1[0]*dv1par; 
  out_G_1[3] += (-1.224744871391589*Ghat_G_1_div_b[1]*dv1par)-1.224744871391589*Ghat_G_1[1]*dv1par; 
  out_G_1[4] += (-1.58113883008419*Ghat_G_1_div_b[0]*dv1par)-1.58113883008419*Ghat_G_1[0]*dv1par; 
  out_G_1[5] += (-1.58113883008419*Ghat_G_1_div_b[1]*dv1par)-1.58113883008419*Ghat_G_1[1]*dv1par; 

  } else { 

  alphaSurf[0] = (-1.0*bb_grad_u[0]*wvpar)+0.5*bb_grad_u[0]*dvpar+p_force[0]; 
  alphaSurf[1] = (-1.0*bb_grad_u[1]*wvpar)+0.5*bb_grad_u[1]*dvpar+p_force[1]; 

  if (0.7071067811865475*alphaSurf[0]-0.7071067811865475*alphaSurf[1] > 0) { 
    F_0_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(F_0Edge); 
    G_1_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(G_1Edge); 
  } else { 
    F_0_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(F_0Skin); 
    G_1_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(G_1Skin); 
  } 
  if (0.7071067811865475*(alphaSurf[1]+alphaSurf[0]) > 0) { 
    F_0_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(F_0Edge); 
    G_1_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(G_1Edge); 
  } else { 
    F_0_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(F_0Skin); 
    G_1_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(G_1Skin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(F_0_UpwindQuad, F_0_Upwind); 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(G_1_UpwindQuad, G_1_Upwind); 

  if (0.7071067811865475*div_b_Surf[0]-0.7071067811865475*div_b_Surf[1] > 0) { 
    F_0_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(F_0_sourceEdge); 
    G_1_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_r(G_1_sourceEdge); 
  } else { 
    F_0_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(F_0_sourceSkin); 
    G_1_div_b_UpwindQuad[0] = hyb_1x1v_p1_surfx2_eval_quad_node_0_l(G_1_sourceSkin); 
  } 
  if (0.7071067811865475*(div_b_Surf[1]+div_b_Surf[0]) > 0) { 
    F_0_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(F_0_sourceEdge); 
    G_1_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_r(G_1_sourceEdge); 
  } else { 
    F_0_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(F_0_sourceSkin); 
    G_1_div_b_UpwindQuad[1] = hyb_1x1v_p1_surfx2_eval_quad_node_1_l(G_1_sourceSkin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(F_0_div_b_UpwindQuad, F_0_div_b_Upwind); 
  hyb_1x1v_p1_vdir_upwind_quad_to_modal(G_1_div_b_UpwindQuad, G_1_div_b_Upwind); 

  Ghat_F_0[0] = 0.7071067811865475*F_0_Upwind[1]*alphaSurf[1]+0.7071067811865475*F_0_Upwind[0]*alphaSurf[0]; 
  Ghat_F_0[1] = 0.7071067811865475*F_0_Upwind[0]*alphaSurf[1]+0.7071067811865475*alphaSurf[0]*F_0_Upwind[1]; 
  Ghat_G_1[0] = 0.7071067811865475*G_1_Upwind[1]*alphaSurf[1]+0.7071067811865475*G_1_Upwind[0]*alphaSurf[0]; 
  Ghat_G_1[1] = 0.7071067811865475*G_1_Upwind[0]*alphaSurf[1]+0.7071067811865475*alphaSurf[0]*G_1_Upwind[1]; 
  Ghat_F_0_div_b[0] = 0.7071067811865475*F_0_div_b_Upwind[1]*div_b_Surf[1]+0.7071067811865475*F_0_div_b_Upwind[0]*div_b_Surf[0]; 
  Ghat_F_0_div_b[1] = 0.7071067811865475*F_0_div_b_Upwind[0]*div_b_Surf[1]+0.7071067811865475*div_b_Surf[0]*F_0_div_b_Upwind[1]; 
  Ghat_G_1_div_b[0] = 0.7071067811865475*G_1_div_b_Upwind[1]*div_b_Surf[1]+0.7071067811865475*G_1_div_b_Upwind[0]*div_b_Surf[0]; 
  Ghat_G_1_div_b[1] = 0.7071067811865475*G_1_div_b_Upwind[0]*div_b_Surf[1]+0.7071067811865475*div_b_Surf[0]*G_1_div_b_Upwind[1]; 

  out_F_0[0] += 0.7071067811865475*Ghat_F_0_div_b[0]*dv1par+0.7071067811865475*Ghat_F_0[0]*dv1par; 
  out_F_0[1] += 0.7071067811865475*Ghat_F_0_div_b[1]*dv1par+0.7071067811865475*Ghat_F_0[1]*dv1par; 
  out_F_0[2] += (-1.224744871391589*Ghat_F_0_div_b[0]*dv1par)-1.224744871391589*Ghat_F_0[0]*dv1par; 
  out_F_0[3] += (-1.224744871391589*Ghat_F_0_div_b[1]*dv1par)-1.224744871391589*Ghat_F_0[1]*dv1par; 
  out_F_0[4] += 1.58113883008419*Ghat_F_0_div_b[0]*dv1par+1.58113883008419*Ghat_F_0[0]*dv1par; 
  out_F_0[5] += 1.58113883008419*Ghat_F_0_div_b[1]*dv1par+1.58113883008419*Ghat_F_0[1]*dv1par; 
  out_G_1[0] += 0.7071067811865475*Ghat_G_1_div_b[0]*dv1par+0.7071067811865475*Ghat_G_1[0]*dv1par; 
  out_G_1[1] += 0.7071067811865475*Ghat_G_1_div_b[1]*dv1par+0.7071067811865475*Ghat_G_1[1]*dv1par; 
  out_G_1[2] += (-1.224744871391589*Ghat_G_1_div_b[0]*dv1par)-1.224744871391589*Ghat_G_1[0]*dv1par; 
  out_G_1[3] += (-1.224744871391589*Ghat_G_1_div_b[1]*dv1par)-1.224744871391589*Ghat_G_1[1]*dv1par; 
  out_G_1[4] += 1.58113883008419*Ghat_G_1_div_b[0]*dv1par+1.58113883008419*Ghat_G_1[0]*dv1par; 
  out_G_1[5] += 1.58113883008419*Ghat_G_1_div_b[1]*dv1par+1.58113883008419*Ghat_G_1[1]*dv1par; 

  } 
} 