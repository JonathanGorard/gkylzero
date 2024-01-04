#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_upvy_upvz(const double *dxv, double gamma, const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9], const double* fpo_dgdv_surf, double *diff_coeff) { 
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[9]: 9 cell stencil of Rosenbluth potential G. 
  // fpo_g_surf_stencil[9]: 9 cell stencil of surface projection of G. 
  // diff_coeff: Output array for diffusion tensor. 

  double dv1_pv1 = 2.0/dxv[2]; 
  double dv1_pv2 = 2.0/dxv[3]; 
  double dv1_sq = 4.0/dxv[2]/dxv[3]; 
 
  const double* GBL = fpo_g_stencil[0]; 
  const double* GCL = fpo_g_stencil[1]; 
  const double* GBC = fpo_g_stencil[2]; 
  const double* GCC = fpo_g_stencil[3]; 

  const double* g_surf_CL = fpo_g_surf_stencil[1]; 
  const double* g_surf_CL_pv2 = &g_surf_CL[16]; 
  const double* g_surf_CC = fpo_g_surf_stencil[3]; 
  const double* g_surf_CC_pv2 = &g_surf_CC[16]; 
  
  const double* g_surf_CC_pv1 = &g_surf_CC[8]; 
  const double* dgdpv1_surf_CC_pv2 = &fpo_dgdv_surf[56]; 
  const double* dgdpv2_surf_CC_pv1 = &fpo_dgdv_surf[40]; 
  const double* dgdpv1_surf_CC_pv1 = &fpo_dgdv_surf[32]; 
  
  double surft1_upper[8], surft1_lower[8]; 
  double surft2_upper[8], surft2_lower[8]; 
  
  double *diff_coeff_vxvy = &diff_coeff[40]; 
  double *diff_coeff_vxvz = &diff_coeff[80]; 
  double *diff_coeff_vyvx = &diff_coeff[120]; 
  double *diff_coeff_vyvz = &diff_coeff[200]; 
  double *diff_coeff_vzvx = &diff_coeff[240]; 
  double *diff_coeff_vzvy = &diff_coeff[280]; 
  
  double *out = diff_coeff_vyvz; 
  
  surft1_upper[0] = dgdpv1_surf_CC_pv2[0]/dv1_pv1; 
  surft1_upper[1] = dgdpv1_surf_CC_pv2[1]/dv1_pv1; 
  surft1_upper[2] = dgdpv1_surf_CC_pv2[2]/dv1_pv1; 
  surft1_upper[3] = dgdpv1_surf_CC_pv2[3]/dv1_pv1; 
  surft1_upper[4] = dgdpv1_surf_CC_pv2[4]/dv1_pv1; 
  surft1_upper[5] = dgdpv1_surf_CC_pv2[5]/dv1_pv1; 
  surft1_upper[6] = dgdpv1_surf_CC_pv2[6]/dv1_pv1; 
  surft1_upper[7] = dgdpv1_surf_CC_pv2[7]/dv1_pv1; 
  surft1_lower[0] = -((0.14433756729740646*dgdpv1_surf_CC_pv1[3])/dv1_pv1)+(0.08333333333333333*dgdpv1_surf_CC_pv1[0])/dv1_pv1+0.036828478186799324*GCL[10]-0.5524271728019898*GCC[10]-0.036828478186799324*GBL[10]+0.5524271728019898*GBC[10]+0.03827327723098713*GCL[4]-0.03827327723098713*GCC[4]-0.03827327723098713*GBL[4]+0.03827327723098713*GBC[4]-0.03189439769248927*GCL[3]+0.478415965387339*GCC[3]-0.03189439769248927*GBL[3]+0.478415965387339*GBC[3]-0.033145630368119385*GCL[0]+0.033145630368119385*GCC[0]-0.033145630368119385*GBL[0]+0.033145630368119385*GBC[0]; 
  surft1_lower[1] = -((0.14433756729740646*dgdpv1_surf_CC_pv1[5])/dv1_pv1)+(0.08333333333333333*dgdpv1_surf_CC_pv1[1])/dv1_pv1+0.036828478186799324*GCL[13]-0.5524271728019898*GCC[13]-0.036828478186799324*GBL[13]+0.5524271728019898*GBC[13]+0.03827327723098713*GCL[8]-0.03827327723098713*GCC[8]-0.03827327723098713*GBL[8]+0.03827327723098713*GBC[8]-0.03189439769248927*GCL[6]+0.478415965387339*GCC[6]-0.03189439769248927*GBL[6]+0.478415965387339*GBC[6]-0.033145630368119385*GCL[1]+0.033145630368119385*GCC[1]-0.033145630368119385*GBL[1]+0.033145630368119385*GBC[1]; 
  surft1_lower[2] = -((0.14433756729740646*dgdpv1_surf_CC_pv1[6])/dv1_pv1)+(0.08333333333333333*dgdpv1_surf_CC_pv1[2])/dv1_pv1+0.036828478186799324*GCL[14]-0.5524271728019898*GCC[14]-0.036828478186799324*GBL[14]+0.5524271728019898*GBC[14]+0.03827327723098713*GCL[9]-0.03827327723098713*GCC[9]-0.03827327723098713*GBL[9]+0.03827327723098713*GBC[9]-0.03189439769248927*GCL[7]+0.478415965387339*GCC[7]-0.03189439769248927*GBL[7]+0.478415965387339*GBC[7]-0.033145630368119385*GCL[2]+0.033145630368119385*GCC[2]-0.033145630368119385*GBL[2]+0.033145630368119385*GBC[2]; 
  surft1_lower[3] = -((0.25*dgdpv1_surf_CC_pv1[3])/dv1_pv1)+(0.14433756729740646*dgdpv1_surf_CC_pv1[0])/dv1_pv1-0.34445949507888407*GCL[10]-0.5485836403108156*GCC[10]+0.34445949507888407*GBL[10]+0.5485836403108156*GBC[10]-0.2872621298570347*GCL[4]+0.2872621298570347*GCC[4]+0.2872621298570347*GBL[4]-0.2872621298570347*GBC[4]+0.2983106733130745*GCL[3]+0.4750873686097112*GCC[3]+0.2983106733130745*GBL[3]+0.4750873686097112*GBC[3]+0.24877630200141632*GCL[0]-0.24877630200141632*GCC[0]+0.24877630200141632*GBL[0]-0.24877630200141632*GBC[0]; 
  surft1_lower[4] = -((0.14433756729740646*dgdpv1_surf_CC_pv1[7])/dv1_pv1)+(0.08333333333333333*dgdpv1_surf_CC_pv1[4])/dv1_pv1+0.036828478186799324*GCL[15]-0.5524271728019898*GCC[15]-0.036828478186799324*GBL[15]+0.5524271728019898*GBC[15]+0.03827327723098713*GCL[12]-0.03827327723098713*GCC[12]-0.03827327723098713*GBL[12]+0.03827327723098713*GBC[12]-0.03189439769248927*GCL[11]+0.478415965387339*GCC[11]-0.03189439769248927*GBL[11]+0.478415965387339*GBC[11]-0.033145630368119385*GCL[5]+0.033145630368119385*GCC[5]-0.033145630368119385*GBL[5]+0.033145630368119385*GBC[5]; 
  surft1_lower[5] = -((0.25*dgdpv1_surf_CC_pv1[5])/dv1_pv1)+(0.14433756729740646*dgdpv1_surf_CC_pv1[1])/dv1_pv1-0.34445949507888407*GCL[13]-0.5485836403108156*GCC[13]+0.34445949507888407*GBL[13]+0.5485836403108156*GBC[13]-0.2872621298570347*GCL[8]+0.2872621298570347*GCC[8]+0.2872621298570347*GBL[8]-0.2872621298570347*GBC[8]+0.2983106733130745*GCL[6]+0.4750873686097112*GCC[6]+0.2983106733130745*GBL[6]+0.4750873686097112*GBC[6]+0.24877630200141632*GCL[1]-0.24877630200141632*GCC[1]+0.24877630200141632*GBL[1]-0.24877630200141632*GBC[1]; 
  surft1_lower[6] = -((0.25*dgdpv1_surf_CC_pv1[6])/dv1_pv1)+(0.14433756729740646*dgdpv1_surf_CC_pv1[2])/dv1_pv1-0.34445949507888407*GCL[14]-0.5485836403108156*GCC[14]+0.34445949507888407*GBL[14]+0.5485836403108156*GBC[14]-0.2872621298570347*GCL[9]+0.2872621298570347*GCC[9]+0.2872621298570347*GBL[9]-0.2872621298570347*GBC[9]+0.2983106733130745*GCL[7]+0.4750873686097112*GCC[7]+0.2983106733130745*GBL[7]+0.4750873686097112*GBC[7]+0.24877630200141632*GCL[2]-0.24877630200141632*GCC[2]+0.24877630200141632*GBL[2]-0.24877630200141632*GBC[2]; 
  surft1_lower[7] = -((0.25*dgdpv1_surf_CC_pv1[7])/dv1_pv1)+(0.14433756729740646*dgdpv1_surf_CC_pv1[4])/dv1_pv1-0.34445949507888407*GCL[15]-0.5485836403108156*GCC[15]+0.34445949507888407*GBL[15]+0.5485836403108156*GBC[15]-0.2872621298570347*GCL[12]+0.2872621298570347*GCC[12]+0.2872621298570347*GBL[12]-0.2872621298570347*GBC[12]+0.2983106733130745*GCL[11]+0.4750873686097112*GCC[11]+0.2983106733130745*GBL[11]+0.4750873686097112*GBC[11]+0.24877630200141632*GCL[5]-0.24877630200141632*GCC[5]+0.24877630200141632*GBL[5]-0.24877630200141632*GBC[5]; 

  surft2_upper[0] = g_surf_CC_pv1[0]; 
  surft2_upper[1] = g_surf_CC_pv1[1]; 
  surft2_upper[2] = g_surf_CC_pv1[2]; 
  surft2_upper[3] = g_surf_CC_pv1[3]; 
  surft2_upper[4] = g_surf_CC_pv1[4]; 
  surft2_upper[5] = g_surf_CC_pv1[5]; 
  surft2_upper[6] = g_surf_CC_pv1[6]; 
  surft2_upper[7] = g_surf_CC_pv1[7]; 
  surft2_lower[0] = 0.408248290463863*GCL[3]-0.408248290463863*GCC[3]+0.3535533905932737*GCL[0]+0.3535533905932737*GCC[0]; 
  surft2_lower[1] = 0.408248290463863*GCL[6]-0.408248290463863*GCC[6]+0.3535533905932737*GCL[1]+0.3535533905932737*GCC[1]; 
  surft2_lower[2] = 0.408248290463863*GCL[7]-0.408248290463863*GCC[7]+0.3535533905932737*GCL[2]+0.3535533905932737*GCC[2]; 
  surft2_lower[3] = 0.408248290463863*GCL[10]-0.408248290463863*GCC[10]+0.3535533905932737*GCL[4]+0.3535533905932737*GCC[4]; 
  surft2_lower[4] = 0.408248290463863*GCL[11]-0.408248290463863*GCC[11]+0.3535533905932737*GCL[5]+0.3535533905932737*GCC[5]; 
  surft2_lower[5] = 0.408248290463863*GCL[13]-0.408248290463863*GCC[13]+0.3535533905932737*GCL[8]+0.3535533905932737*GCC[8]; 
  surft2_lower[6] = 0.408248290463863*GCL[14]-0.408248290463863*GCC[14]+0.3535533905932737*GCL[9]+0.3535533905932737*GCC[9]; 
  surft2_lower[7] = 0.408248290463863*GCL[15]-0.408248290463863*GCC[15]+0.3535533905932737*GCL[12]+0.3535533905932737*GCC[12]; 

  out[0] = 0.7071067811865475*surft1_upper[0]*dv1_sq*gamma-0.7071067811865475*surft1_lower[0]*dv1_sq*gamma; 
  out[1] = 0.7071067811865475*surft1_upper[1]*dv1_sq*gamma-0.7071067811865475*surft1_lower[1]*dv1_sq*gamma; 
  out[2] = 0.7071067811865475*surft1_upper[2]*dv1_sq*gamma-0.7071067811865475*surft1_lower[2]*dv1_sq*gamma; 
  out[3] = 0.7071067811865475*surft1_upper[3]*dv1_sq*gamma-0.7071067811865475*surft1_lower[3]*dv1_sq*gamma; 
  out[4] = -(1.224744871391589*surft2_upper[0]*dv1_sq*gamma)+1.224744871391589*surft2_lower[0]*dv1_sq*gamma+1.224744871391589*surft1_upper[0]*dv1_sq*gamma+1.224744871391589*surft1_lower[0]*dv1_sq*gamma; 
  out[5] = 0.7071067811865475*surft1_upper[4]*dv1_sq*gamma-0.7071067811865475*surft1_lower[4]*dv1_sq*gamma; 
  out[6] = 0.7071067811865475*surft1_upper[5]*dv1_sq*gamma-0.7071067811865475*surft1_lower[5]*dv1_sq*gamma; 
  out[7] = 0.7071067811865475*surft1_upper[6]*dv1_sq*gamma-0.7071067811865475*surft1_lower[6]*dv1_sq*gamma; 
  out[8] = -(1.224744871391589*surft2_upper[1]*dv1_sq*gamma)+1.224744871391589*surft2_lower[1]*dv1_sq*gamma+1.224744871391589*surft1_upper[1]*dv1_sq*gamma+1.224744871391589*surft1_lower[1]*dv1_sq*gamma; 
  out[9] = -(1.224744871391589*surft2_upper[2]*dv1_sq*gamma)+1.224744871391589*surft2_lower[2]*dv1_sq*gamma+1.224744871391589*surft1_upper[2]*dv1_sq*gamma+1.224744871391589*surft1_lower[2]*dv1_sq*gamma; 
  out[10] = 1.224744871391589*surft1_upper[3]*dv1_sq*gamma+1.224744871391589*surft1_lower[3]*dv1_sq*gamma-2.1213203435596424*surft2_upper[0]*dv1_sq*gamma-2.1213203435596424*surft2_lower[0]*dv1_sq*gamma+3.0*GCC[0]*dv1_sq*gamma; 
  out[11] = 0.7071067811865475*surft1_upper[7]*dv1_sq*gamma-0.7071067811865475*surft1_lower[7]*dv1_sq*gamma; 
  out[12] = -(1.224744871391589*surft2_upper[4]*dv1_sq*gamma)+1.224744871391589*surft2_lower[4]*dv1_sq*gamma+1.224744871391589*surft1_upper[4]*dv1_sq*gamma+1.224744871391589*surft1_lower[4]*dv1_sq*gamma; 
  out[13] = 1.224744871391589*surft1_upper[5]*dv1_sq*gamma+1.224744871391589*surft1_lower[5]*dv1_sq*gamma-2.1213203435596424*surft2_upper[1]*dv1_sq*gamma-2.1213203435596424*surft2_lower[1]*dv1_sq*gamma+3.0*GCC[1]*dv1_sq*gamma; 
  out[14] = 1.224744871391589*surft1_upper[6]*dv1_sq*gamma+1.224744871391589*surft1_lower[6]*dv1_sq*gamma-2.1213203435596424*surft2_upper[2]*dv1_sq*gamma-2.1213203435596424*surft2_lower[2]*dv1_sq*gamma+3.0*GCC[2]*dv1_sq*gamma; 
  out[15] = 1.224744871391589*surft1_upper[7]*dv1_sq*gamma+1.224744871391589*surft1_lower[7]*dv1_sq*gamma+3.0*GCC[5]*dv1_sq*gamma-2.1213203435596424*surft2_upper[4]*dv1_sq*gamma-2.1213203435596424*surft2_lower[4]*dv1_sq*gamma; 
  out[22] = 3.0*GCC[16]*dv1_sq*gamma; 
  out[23] = 3.0*GCC[17]*dv1_sq*gamma; 
  out[27] = 6.7082039324993685*GCC[3]*dv1_sq*gamma-2.7386127875258306*surft2_upper[0]*dv1_sq*gamma+2.7386127875258306*surft2_lower[0]*dv1_sq*gamma; 
  out[29] = 6.708203932499369*GCC[6]*dv1_sq*gamma-2.7386127875258306*surft2_upper[1]*dv1_sq*gamma+2.7386127875258306*surft2_lower[1]*dv1_sq*gamma; 
  out[30] = 6.708203932499369*GCC[7]*dv1_sq*gamma-2.7386127875258306*surft2_upper[2]*dv1_sq*gamma+2.7386127875258306*surft2_lower[2]*dv1_sq*gamma; 
  out[31] = 6.7082039324993685*GCC[11]*dv1_sq*gamma-2.7386127875258306*surft2_upper[4]*dv1_sq*gamma+2.7386127875258306*surft2_lower[4]*dv1_sq*gamma; 
  out[32] = -(2.7386127875258306*surft2_upper[3]*dv1_sq*gamma)+2.7386127875258306*surft2_lower[3]*dv1_sq*gamma+1.5811388300841895*surft1_upper[0]*dv1_sq*gamma-1.5811388300841895*surft1_lower[0]*dv1_sq*gamma; 
  out[33] = -(2.7386127875258306*surft2_upper[5]*dv1_sq*gamma)+2.7386127875258306*surft2_lower[5]*dv1_sq*gamma+1.5811388300841898*surft1_upper[1]*dv1_sq*gamma-1.5811388300841898*surft1_lower[1]*dv1_sq*gamma; 
  out[34] = -(2.7386127875258306*surft2_upper[6]*dv1_sq*gamma)+2.7386127875258306*surft2_lower[6]*dv1_sq*gamma+1.5811388300841898*surft1_upper[2]*dv1_sq*gamma-1.5811388300841898*surft1_lower[2]*dv1_sq*gamma; 
  out[35] = 6.7082039324993685*GCC[4]*dv1_sq*gamma-4.743416490252569*surft2_upper[3]*dv1_sq*gamma-4.743416490252569*surft2_lower[3]*dv1_sq*gamma+1.5811388300841898*surft1_upper[3]*dv1_sq*gamma-1.5811388300841898*surft1_lower[3]*dv1_sq*gamma; 
  out[36] = -(2.7386127875258306*surft2_upper[7]*dv1_sq*gamma)+2.7386127875258306*surft2_lower[7]*dv1_sq*gamma+1.5811388300841895*surft1_upper[4]*dv1_sq*gamma-1.5811388300841895*surft1_lower[4]*dv1_sq*gamma; 
  out[37] = 6.708203932499369*GCC[8]*dv1_sq*gamma-4.743416490252569*surft2_upper[5]*dv1_sq*gamma-4.743416490252569*surft2_lower[5]*dv1_sq*gamma+1.5811388300841895*surft1_upper[5]*dv1_sq*gamma-1.5811388300841895*surft1_lower[5]*dv1_sq*gamma; 
  out[38] = 6.708203932499369*GCC[9]*dv1_sq*gamma-4.743416490252569*surft2_upper[6]*dv1_sq*gamma-4.743416490252569*surft2_lower[6]*dv1_sq*gamma+1.5811388300841895*surft1_upper[6]*dv1_sq*gamma-1.5811388300841895*surft1_lower[6]*dv1_sq*gamma; 
  out[39] = 6.7082039324993685*GCC[12]*dv1_sq*gamma-4.743416490252569*surft2_upper[7]*dv1_sq*gamma-4.743416490252569*surft2_lower[7]*dv1_sq*gamma+1.5811388300841898*surft1_upper[7]*dv1_sq*gamma-1.5811388300841898*surft1_lower[7]*dv1_sq*gamma; 
} 

