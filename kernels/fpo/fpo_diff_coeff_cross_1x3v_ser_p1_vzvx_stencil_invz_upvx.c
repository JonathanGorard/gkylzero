#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_invz_upvx(const double *dxv, const double *gamma, const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9], const double* fpo_dgdv_surf, double *diff_coeff) { 
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[9]: 9 cell stencil of Rosenbluth potential G. 
  // fpo_g_surf_stencil[9]: 9 cell stencil of surface projection of G. 
  // fpo_dgdv_surf: Surface expansion of dG/dv in center cell. 
  // diff_coeff: Output array for diffusion tensor. 

  // Use cell-average value for gamma. 
 double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1_pv1 = 2.0/dxv[3]; 
  double dv1_pv2 = 2.0/dxv[1]; 
  double dv1_sq = 4.0/dxv[3]/dxv[1]; 
 
  const double* GBL = fpo_g_stencil[0]; 
  const double* GBC = fpo_g_stencil[1]; 
  const double* GBR = fpo_g_stencil[2]; 
  const double* GCL = fpo_g_stencil[3]; 
  const double* GCC = fpo_g_stencil[4]; 
  const double* GCR = fpo_g_stencil[5]; 

  const double* g_surf_CL = fpo_g_surf_stencil[3]; 
  const double* g_surf_CL_pv2 = &g_surf_CL[0]; 
  const double* g_surf_CC = fpo_g_surf_stencil[4]; 
  const double* g_surf_CC_pv2 = &g_surf_CC[0]; 
  const double* g_surf_CR = fpo_g_surf_stencil[5]; 
  const double* g_surf_CR_pv2 = &g_surf_CR[0]; 
  
  const double* g_surf_CC_pv1 = &g_surf_CC[16]; 
  const double* dgdpv1_surf_CC_pv2 = &fpo_dgdv_surf[16]; 
  const double* dgdpv2_surf_CC_pv1 = &fpo_dgdv_surf[48]; 
  const double* dgdpv1_surf_CC_pv1 = &fpo_dgdv_surf[64]; 
  
  double surft1_upper[8], surft1_lower[8]; 
  double surft2_upper[8], surft2_lower[8]; 
  
  double *diff_coeff_vxvy = &diff_coeff[40]; 
  double *diff_coeff_vxvz = &diff_coeff[80]; 
  double *diff_coeff_vyvx = &diff_coeff[120]; 
  double *diff_coeff_vyvz = &diff_coeff[200]; 
  double *diff_coeff_vzvx = &diff_coeff[240]; 
  double *diff_coeff_vzvy = &diff_coeff[280]; 
  
  double *out = diff_coeff_vzvx; 
  
  surft1_upper[0] = dgdpv1_surf_CC_pv2[0]/dv1_pv1; 
  surft1_upper[1] = dgdpv1_surf_CC_pv2[1]/dv1_pv1; 
  surft1_upper[2] = dgdpv1_surf_CC_pv2[2]/dv1_pv1; 
  surft1_upper[3] = dgdpv1_surf_CC_pv2[3]/dv1_pv1; 
  surft1_upper[4] = dgdpv1_surf_CC_pv2[4]/dv1_pv1; 
  surft1_upper[5] = dgdpv1_surf_CC_pv2[5]/dv1_pv1; 
  surft1_upper[6] = dgdpv1_surf_CC_pv2[6]/dv1_pv1; 
  surft1_upper[7] = dgdpv1_surf_CC_pv2[7]/dv1_pv1; 
  surft1_lower[0] = 0.11785113019775789*GCR[9]+0.11785113019775789*GCL[9]-0.2357022603955158*GCC[9]-0.11785113019775789*GBR[9]-0.11785113019775789*GBL[9]+0.2357022603955158*GBC[9]-0.10206207261596573*GCR[4]-0.10206207261596573*GCL[4]+0.20412414523193148*GCC[4]-0.10206207261596573*GBR[4]-0.10206207261596573*GBL[4]+0.20412414523193148*GBC[4]-0.10206207261596573*GCR[2]+0.10206207261596573*GCL[2]+0.10206207261596573*GBR[2]-0.10206207261596573*GBL[2]+0.0883883476483184*GCR[0]-0.0883883476483184*GCL[0]+0.0883883476483184*GBR[0]-0.0883883476483184*GBL[0]; 
  surft1_lower[1] = 0.11785113019775789*GCR[12]+0.11785113019775789*GCL[12]-0.2357022603955158*GCC[12]-0.11785113019775789*GBR[12]-0.11785113019775789*GBL[12]+0.2357022603955158*GBC[12]-0.10206207261596573*GCR[8]-0.10206207261596573*GCL[8]+0.20412414523193148*GCC[8]-0.10206207261596573*GBR[8]-0.10206207261596573*GBL[8]+0.20412414523193148*GBC[8]-0.10206207261596573*GCR[5]+0.10206207261596573*GCL[5]+0.10206207261596573*GBR[5]-0.10206207261596573*GBL[5]+0.0883883476483184*GCR[1]-0.0883883476483184*GCL[1]+0.0883883476483184*GBR[1]-0.0883883476483184*GBL[1]; 
  surft1_lower[2] = 0.11785113019775789*GCR[14]+0.11785113019775789*GCL[14]-0.2357022603955158*GCC[14]-0.11785113019775789*GBR[14]-0.11785113019775789*GBL[14]+0.2357022603955158*GBC[14]-0.10206207261596573*GCR[10]-0.10206207261596573*GCL[10]+0.20412414523193148*GCC[10]-0.10206207261596573*GBR[10]-0.10206207261596573*GBL[10]+0.20412414523193148*GBC[10]-0.10206207261596573*GCR[7]+0.10206207261596573*GCL[7]+0.10206207261596573*GBR[7]-0.10206207261596573*GBL[7]+0.0883883476483184*GCR[3]-0.0883883476483184*GCL[3]+0.0883883476483184*GBR[3]-0.0883883476483184*GBL[3]; 
  surft1_lower[3] = 0.20412414523193148*GCR[9]-0.20412414523193148*GCL[9]-0.20412414523193148*GBR[9]+0.20412414523193148*GBL[9]-0.1767766952966368*GCR[4]+0.1767766952966368*GCL[4]-0.1767766952966368*GBR[4]+0.1767766952966368*GBL[4]-0.1767766952966368*GCR[2]-0.1767766952966368*GCL[2]+0.3535533905932737*GCC[2]+0.1767766952966368*GBR[2]+0.1767766952966368*GBL[2]-0.3535533905932737*GBC[2]+0.15309310892394856*GCR[0]+0.15309310892394856*GCL[0]-0.3061862178478971*GCC[0]+0.15309310892394856*GBR[0]+0.15309310892394856*GBL[0]-0.3061862178478971*GBC[0]; 
  surft1_lower[4] = 0.11785113019775789*GCR[15]+0.11785113019775789*GCL[15]-0.2357022603955158*GCC[15]-0.11785113019775789*GBR[15]-0.11785113019775789*GBL[15]+0.2357022603955158*GBC[15]-0.10206207261596573*GCR[13]-0.10206207261596573*GCL[13]+0.20412414523193148*GCC[13]-0.10206207261596573*GBR[13]-0.10206207261596573*GBL[13]+0.20412414523193148*GBC[13]-0.10206207261596573*GCR[11]+0.10206207261596573*GCL[11]+0.10206207261596573*GBR[11]-0.10206207261596573*GBL[11]+0.0883883476483184*GCR[6]-0.0883883476483184*GCL[6]+0.0883883476483184*GBR[6]-0.0883883476483184*GBL[6]; 
  surft1_lower[5] = 0.20412414523193148*GCR[12]-0.20412414523193148*GCL[12]-0.20412414523193148*GBR[12]+0.20412414523193148*GBL[12]-0.1767766952966368*GCR[8]+0.1767766952966368*GCL[8]-0.1767766952966368*GBR[8]+0.1767766952966368*GBL[8]-0.1767766952966368*GCR[5]-0.1767766952966368*GCL[5]+0.3535533905932737*GCC[5]+0.1767766952966368*GBR[5]+0.1767766952966368*GBL[5]-0.3535533905932737*GBC[5]+0.15309310892394856*GCR[1]+0.15309310892394856*GCL[1]-0.3061862178478971*GCC[1]+0.15309310892394856*GBR[1]+0.15309310892394856*GBL[1]-0.3061862178478971*GBC[1]; 
  surft1_lower[6] = 0.20412414523193148*GCR[14]-0.20412414523193148*GCL[14]-0.20412414523193148*GBR[14]+0.20412414523193148*GBL[14]-0.1767766952966368*GCR[10]+0.1767766952966368*GCL[10]-0.1767766952966368*GBR[10]+0.1767766952966368*GBL[10]-0.1767766952966368*GCR[7]-0.1767766952966368*GCL[7]+0.3535533905932737*GCC[7]+0.1767766952966368*GBR[7]+0.1767766952966368*GBL[7]-0.3535533905932737*GBC[7]+0.15309310892394856*GCR[3]+0.15309310892394856*GCL[3]-0.3061862178478971*GCC[3]+0.15309310892394856*GBR[3]+0.15309310892394856*GBL[3]-0.3061862178478971*GBC[3]; 
  surft1_lower[7] = 0.20412414523193148*GCR[15]-0.20412414523193148*GCL[15]-0.20412414523193148*GBR[15]+0.20412414523193148*GBL[15]-0.1767766952966368*GCR[13]+0.1767766952966368*GCL[13]-0.1767766952966368*GBR[13]+0.1767766952966368*GBL[13]-0.1767766952966368*GCR[11]-0.1767766952966368*GCL[11]+0.3535533905932737*GCC[11]+0.1767766952966368*GBR[11]+0.1767766952966368*GBL[11]-0.3535533905932737*GBC[11]+0.15309310892394856*GCR[6]+0.15309310892394856*GCL[6]-0.3061862178478971*GCC[6]+0.15309310892394856*GBR[6]+0.15309310892394856*GBL[6]-0.3061862178478971*GBC[6]; 

  surft2_upper[0] = -(0.408248290463863*GCR[4])+0.408248290463863*GCC[4]+0.3535533905932737*GCR[0]+0.3535533905932737*GCC[0]; 
  surft2_upper[1] = -(0.408248290463863*GCR[8])+0.408248290463863*GCC[8]+0.3535533905932737*GCR[1]+0.3535533905932737*GCC[1]; 
  surft2_upper[2] = -(0.408248290463863*GCR[9])+0.408248290463863*GCC[9]+0.3535533905932737*GCR[2]+0.3535533905932737*GCC[2]; 
  surft2_upper[3] = -(0.408248290463863*GCR[10])+0.408248290463863*GCC[10]+0.3535533905932737*GCR[3]+0.3535533905932737*GCC[3]; 
  surft2_upper[4] = -(0.408248290463863*GCR[12])+0.408248290463863*GCC[12]+0.3535533905932737*GCR[5]+0.3535533905932737*GCC[5]; 
  surft2_upper[5] = -(0.408248290463863*GCR[13])+0.408248290463863*GCC[13]+0.3535533905932737*GCR[6]+0.3535533905932737*GCC[6]; 
  surft2_upper[6] = -(0.408248290463863*GCR[14])+0.408248290463863*GCC[14]+0.3535533905932737*GCR[7]+0.3535533905932737*GCC[7]; 
  surft2_upper[7] = -(0.408248290463863*GCR[15])+0.408248290463863*GCC[15]+0.3535533905932737*GCR[11]+0.3535533905932737*GCC[11]; 
  surft2_lower[0] = 0.408248290463863*GCL[4]-0.408248290463863*GCC[4]+0.3535533905932737*GCL[0]+0.3535533905932737*GCC[0]; 
  surft2_lower[1] = 0.408248290463863*GCL[8]-0.408248290463863*GCC[8]+0.3535533905932737*GCL[1]+0.3535533905932737*GCC[1]; 
  surft2_lower[2] = 0.408248290463863*GCL[9]-0.408248290463863*GCC[9]+0.3535533905932737*GCL[2]+0.3535533905932737*GCC[2]; 
  surft2_lower[3] = 0.408248290463863*GCL[10]-0.408248290463863*GCC[10]+0.3535533905932737*GCL[3]+0.3535533905932737*GCC[3]; 
  surft2_lower[4] = 0.408248290463863*GCL[12]-0.408248290463863*GCC[12]+0.3535533905932737*GCL[5]+0.3535533905932737*GCC[5]; 
  surft2_lower[5] = 0.408248290463863*GCL[13]-0.408248290463863*GCC[13]+0.3535533905932737*GCL[6]+0.3535533905932737*GCC[6]; 
  surft2_lower[6] = 0.408248290463863*GCL[14]-0.408248290463863*GCC[14]+0.3535533905932737*GCL[7]+0.3535533905932737*GCC[7]; 
  surft2_lower[7] = 0.408248290463863*GCL[15]-0.408248290463863*GCC[15]+0.3535533905932737*GCL[11]+0.3535533905932737*GCC[11]; 

  out[0] = 0.7071067811865475*surft1_upper[0]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[1] = 0.7071067811865475*surft1_upper[1]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[2] = -(1.224744871391589*surft2_upper[0]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[3] = 0.7071067811865475*surft1_upper[2]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[4] = 0.7071067811865475*surft1_upper[3]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[5] = -(1.224744871391589*surft2_upper[1]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[6] = 0.7071067811865475*surft1_upper[4]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[7] = -(1.224744871391589*surft2_upper[3]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[3]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[2]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[8] = 0.7071067811865475*surft1_upper[5]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[9] = 1.224744871391589*surft1_upper[3]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[3]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[0]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[0]*dv1_sq*gamma_avg+3.0*GCC[0]*dv1_sq*gamma_avg; 
  out[10] = 0.7071067811865475*surft1_upper[6]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[11] = -(1.224744871391589*surft2_upper[5]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[5]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[4]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[12] = 1.224744871391589*surft1_upper[5]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[5]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[1]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[1]*dv1_sq*gamma_avg+3.0*GCC[1]*dv1_sq*gamma_avg; 
  out[13] = 0.7071067811865475*surft1_upper[7]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[7]*dv1_sq*gamma_avg; 
  out[14] = 1.224744871391589*surft1_upper[6]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[6]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[3]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[3]*dv1_sq*gamma_avg+3.0*GCC[3]*dv1_sq*gamma_avg; 
  out[15] = 1.224744871391589*surft1_upper[7]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[7]*dv1_sq*gamma_avg+3.0*GCC[6]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[5]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[5]*dv1_sq*gamma_avg; 
  out[16] = -(2.7386127875258306*surft2_upper[2]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[2]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[0]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[17] = -(2.7386127875258306*surft2_upper[4]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[4]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[1]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[18] = -(2.7386127875258306*surft2_upper[6]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[2]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[19] = 1.5811388300841898*surft1_upper[3]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[3]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[2]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[2]*dv1_sq*gamma_avg+6.7082039324993685*GCC[2]*dv1_sq*gamma_avg; 
  out[20] = -(2.7386127875258306*surft2_upper[7]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[7]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[4]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[21] = 1.5811388300841895*surft1_upper[5]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[5]*dv1_sq*gamma_avg+6.708203932499369*GCC[5]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[4]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[4]*dv1_sq*gamma_avg; 
  out[22] = 6.708203932499369*GCC[7]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[6]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[6]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[23] = 6.7082039324993685*GCC[11]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[7]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[7]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[7]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[7]*dv1_sq*gamma_avg; 
  out[30] = 3.0*GCC[24]*dv1_sq*gamma_avg; 
  out[31] = 3.0*GCC[25]*dv1_sq*gamma_avg; 
  out[34] = 6.7082039324993685*GCC[4]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[0]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[0]*dv1_sq*gamma_avg; 
  out[36] = 6.708203932499369*GCC[8]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[1]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[1]*dv1_sq*gamma_avg; 
  out[38] = 6.708203932499369*GCC[10]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[3]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[3]*dv1_sq*gamma_avg; 
  out[39] = 6.7082039324993685*GCC[13]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[5]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[5]*dv1_sq*gamma_avg; 
} 

