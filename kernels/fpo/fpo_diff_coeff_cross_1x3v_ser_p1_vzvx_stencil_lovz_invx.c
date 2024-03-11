#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_diff_coeff_cross_1x3v_vzvx_ser_p1_lovz_invx(const double *dxv, const double *gamma, const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9], const double* fpo_dgdv_surf, double *diff_coeff) { 
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
 
  const double* GBC = fpo_g_stencil[0]; 
  const double* GBR = fpo_g_stencil[1]; 
  const double* GCC = fpo_g_stencil[2]; 
  const double* GCR = fpo_g_stencil[3]; 
  const double* GTC = fpo_g_stencil[4]; 
  const double* GTR = fpo_g_stencil[5]; 

  const double* g_surf_CC = fpo_g_surf_stencil[2]; 
  const double* g_surf_CC_pv2 = &g_surf_CC[0]; 
  const double* g_surf_CR = fpo_g_surf_stencil[3]; 
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
  
  surft1_upper[0] = 0.11785113019775789*GTR[9]-0.11785113019775789*GTC[9]-0.2357022603955158*GCR[9]+0.2357022603955158*GCC[9]+0.11785113019775789*GBR[9]-0.11785113019775789*GBC[9]-0.10206207261596573*GTR[4]+0.10206207261596573*GTC[4]+0.10206207261596573*GBR[4]-0.10206207261596573*GBC[4]-0.10206207261596573*GTR[2]-0.10206207261596573*GTC[2]+0.20412414523193148*GCR[2]+0.20412414523193148*GCC[2]-0.10206207261596573*GBR[2]-0.10206207261596573*GBC[2]+0.0883883476483184*GTR[0]+0.0883883476483184*GTC[0]-0.0883883476483184*GBR[0]-0.0883883476483184*GBC[0]; 
  surft1_upper[1] = 0.11785113019775789*GTR[12]-0.11785113019775789*GTC[12]-0.2357022603955158*GCR[12]+0.2357022603955158*GCC[12]+0.11785113019775789*GBR[12]-0.11785113019775789*GBC[12]-0.10206207261596573*GTR[8]+0.10206207261596573*GTC[8]+0.10206207261596573*GBR[8]-0.10206207261596573*GBC[8]-0.10206207261596573*GTR[5]-0.10206207261596573*GTC[5]+0.20412414523193148*GCR[5]+0.20412414523193148*GCC[5]-0.10206207261596573*GBR[5]-0.10206207261596573*GBC[5]+0.0883883476483184*GTR[1]+0.0883883476483184*GTC[1]-0.0883883476483184*GBR[1]-0.0883883476483184*GBC[1]; 
  surft1_upper[2] = 0.20412414523193148*GTR[9]-0.20412414523193148*GTC[9]-0.20412414523193148*GBR[9]+0.20412414523193148*GBC[9]-0.1767766952966368*GTR[4]+0.1767766952966368*GTC[4]+0.3535533905932737*GCR[4]-0.3535533905932737*GCC[4]-0.1767766952966368*GBR[4]+0.1767766952966368*GBC[4]-0.1767766952966368*GTR[2]-0.1767766952966368*GTC[2]+0.1767766952966368*GBR[2]+0.1767766952966368*GBC[2]+0.15309310892394856*GTR[0]+0.15309310892394856*GTC[0]-0.3061862178478971*GCR[0]-0.3061862178478971*GCC[0]+0.15309310892394856*GBR[0]+0.15309310892394856*GBC[0]; 
  surft1_upper[3] = 0.11785113019775789*GTR[14]-0.11785113019775789*GTC[14]-0.2357022603955158*GCR[14]+0.2357022603955158*GCC[14]+0.11785113019775789*GBR[14]-0.11785113019775789*GBC[14]-0.10206207261596573*GTR[10]+0.10206207261596573*GTC[10]+0.10206207261596573*GBR[10]-0.10206207261596573*GBC[10]-0.10206207261596573*GTR[7]-0.10206207261596573*GTC[7]+0.20412414523193148*GCR[7]+0.20412414523193148*GCC[7]-0.10206207261596573*GBR[7]-0.10206207261596573*GBC[7]+0.0883883476483184*GTR[3]+0.0883883476483184*GTC[3]-0.0883883476483184*GBR[3]-0.0883883476483184*GBC[3]; 
  surft1_upper[4] = 0.20412414523193148*GTR[12]-0.20412414523193148*GTC[12]-0.20412414523193148*GBR[12]+0.20412414523193148*GBC[12]-0.1767766952966368*GTR[8]+0.1767766952966368*GTC[8]+0.3535533905932737*GCR[8]-0.3535533905932737*GCC[8]-0.1767766952966368*GBR[8]+0.1767766952966368*GBC[8]-0.1767766952966368*GTR[5]-0.1767766952966368*GTC[5]+0.1767766952966368*GBR[5]+0.1767766952966368*GBC[5]+0.15309310892394856*GTR[1]+0.15309310892394856*GTC[1]-0.3061862178478971*GCR[1]-0.3061862178478971*GCC[1]+0.15309310892394856*GBR[1]+0.15309310892394856*GBC[1]; 
  surft1_upper[5] = 0.11785113019775789*GTR[15]-0.11785113019775789*GTC[15]-0.2357022603955158*GCR[15]+0.2357022603955158*GCC[15]+0.11785113019775789*GBR[15]-0.11785113019775789*GBC[15]-0.10206207261596573*GTR[13]+0.10206207261596573*GTC[13]+0.10206207261596573*GBR[13]-0.10206207261596573*GBC[13]-0.10206207261596573*GTR[11]-0.10206207261596573*GTC[11]+0.20412414523193148*GCR[11]+0.20412414523193148*GCC[11]-0.10206207261596573*GBR[11]-0.10206207261596573*GBC[11]+0.0883883476483184*GTR[6]+0.0883883476483184*GTC[6]-0.0883883476483184*GBR[6]-0.0883883476483184*GBC[6]; 
  surft1_upper[6] = 0.20412414523193148*GTR[14]-0.20412414523193148*GTC[14]-0.20412414523193148*GBR[14]+0.20412414523193148*GBC[14]-0.1767766952966368*GTR[10]+0.1767766952966368*GTC[10]+0.3535533905932737*GCR[10]-0.3535533905932737*GCC[10]-0.1767766952966368*GBR[10]+0.1767766952966368*GBC[10]-0.1767766952966368*GTR[7]-0.1767766952966368*GTC[7]+0.1767766952966368*GBR[7]+0.1767766952966368*GBC[7]+0.15309310892394856*GTR[3]+0.15309310892394856*GTC[3]-0.3061862178478971*GCR[3]-0.3061862178478971*GCC[3]+0.15309310892394856*GBR[3]+0.15309310892394856*GBC[3]; 
  surft1_upper[7] = 0.20412414523193148*GTR[15]-0.20412414523193148*GTC[15]-0.20412414523193148*GBR[15]+0.20412414523193148*GBC[15]-0.1767766952966368*GTR[13]+0.1767766952966368*GTC[13]+0.3535533905932737*GCR[13]-0.3535533905932737*GCC[13]-0.1767766952966368*GBR[13]+0.1767766952966368*GBC[13]-0.1767766952966368*GTR[11]-0.1767766952966368*GTC[11]+0.1767766952966368*GBR[11]+0.1767766952966368*GBC[11]+0.15309310892394856*GTR[6]+0.15309310892394856*GTC[6]-0.3061862178478971*GCR[6]-0.3061862178478971*GCC[6]+0.15309310892394856*GBR[6]+0.15309310892394856*GBC[6]; 
  surft1_lower[0] = dgdpv2_surf_CC_pv1[0]/dv1_pv2; 
  surft1_lower[1] = dgdpv2_surf_CC_pv1[1]/dv1_pv2; 
  surft1_lower[2] = dgdpv2_surf_CC_pv1[2]/dv1_pv2; 
  surft1_lower[3] = dgdpv2_surf_CC_pv1[3]/dv1_pv2; 
  surft1_lower[4] = dgdpv2_surf_CC_pv1[4]/dv1_pv2; 
  surft1_lower[5] = dgdpv2_surf_CC_pv1[5]/dv1_pv2; 
  surft1_lower[6] = dgdpv2_surf_CC_pv1[6]/dv1_pv2; 
  surft1_lower[7] = dgdpv2_surf_CC_pv1[7]/dv1_pv2; 

  surft2_upper[0] = -(0.408248290463863*GTC[2])+0.408248290463863*GCC[2]+0.3535533905932737*GTC[0]+0.3535533905932737*GCC[0]; 
  surft2_upper[1] = -(0.408248290463863*GTC[5])+0.408248290463863*GCC[5]+0.3535533905932737*GTC[1]+0.3535533905932737*GCC[1]; 
  surft2_upper[2] = -(0.408248290463863*GTC[7])+0.408248290463863*GCC[7]+0.3535533905932737*GTC[3]+0.3535533905932737*GCC[3]; 
  surft2_upper[3] = -(0.408248290463863*GTC[9])+0.408248290463863*GCC[9]+0.3535533905932737*GTC[4]+0.3535533905932737*GCC[4]; 
  surft2_upper[4] = -(0.408248290463863*GTC[11])+0.408248290463863*GCC[11]+0.3535533905932737*GTC[6]+0.3535533905932737*GCC[6]; 
  surft2_upper[5] = -(0.408248290463863*GTC[12])+0.408248290463863*GCC[12]+0.3535533905932737*GTC[8]+0.3535533905932737*GCC[8]; 
  surft2_upper[6] = -(0.408248290463863*GTC[14])+0.408248290463863*GCC[14]+0.3535533905932737*GTC[10]+0.3535533905932737*GCC[10]; 
  surft2_upper[7] = -(0.408248290463863*GTC[15])+0.408248290463863*GCC[15]+0.3535533905932737*GTC[13]+0.3535533905932737*GCC[13]; 
  surft2_lower[0] = -(0.408248290463863*GCC[2])+0.408248290463863*GBC[2]+0.3535533905932737*GCC[0]+0.3535533905932737*GBC[0]; 
  surft2_lower[1] = -(0.408248290463863*GCC[5])+0.408248290463863*GBC[5]+0.3535533905932737*GCC[1]+0.3535533905932737*GBC[1]; 
  surft2_lower[2] = -(0.408248290463863*GCC[7])+0.408248290463863*GBC[7]+0.3535533905932737*GCC[3]+0.3535533905932737*GBC[3]; 
  surft2_lower[3] = -(0.408248290463863*GCC[9])+0.408248290463863*GBC[9]+0.3535533905932737*GCC[4]+0.3535533905932737*GBC[4]; 
  surft2_lower[4] = -(0.408248290463863*GCC[11])+0.408248290463863*GBC[11]+0.3535533905932737*GCC[6]+0.3535533905932737*GBC[6]; 
  surft2_lower[5] = -(0.408248290463863*GCC[12])+0.408248290463863*GBC[12]+0.3535533905932737*GCC[8]+0.3535533905932737*GBC[8]; 
  surft2_lower[6] = -(0.408248290463863*GCC[14])+0.408248290463863*GBC[14]+0.3535533905932737*GCC[10]+0.3535533905932737*GBC[10]; 
  surft2_lower[7] = -(0.408248290463863*GCC[15])+0.408248290463863*GBC[15]+0.3535533905932737*GCC[13]+0.3535533905932737*GBC[13]; 

  out[0] = 0.7071067811865475*surft1_upper[0]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[1] = 0.7071067811865475*surft1_upper[1]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[2] = 0.7071067811865475*surft1_upper[2]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[3] = 0.7071067811865475*surft1_upper[3]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[4] = -(1.224744871391589*surft2_upper[0]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[5] = 0.7071067811865475*surft1_upper[4]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[6] = 0.7071067811865475*surft1_upper[5]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[7] = 0.7071067811865475*surft1_upper[6]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[8] = -(1.224744871391589*surft2_upper[1]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[9] = 1.224744871391589*surft1_upper[2]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[2]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[0]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[0]*dv1_sq*gamma_avg+3.0*GCC[0]*dv1_sq*gamma_avg; 
  out[10] = 1.224744871391589*surft1_upper[3]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[3]*dv1_sq*gamma_avg-1.224744871391589*surft2_upper[2]*dv1_sq*gamma_avg+1.224744871391589*surft2_lower[2]*dv1_sq*gamma_avg; 
  out[11] = 0.7071067811865475*surft1_upper[7]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[7]*dv1_sq*gamma_avg; 
  out[12] = 1.224744871391589*surft1_upper[4]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[4]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[1]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[1]*dv1_sq*gamma_avg+3.0*GCC[1]*dv1_sq*gamma_avg; 
  out[13] = 1.224744871391589*surft1_upper[5]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[5]*dv1_sq*gamma_avg-1.224744871391589*surft2_upper[4]*dv1_sq*gamma_avg+1.224744871391589*surft2_lower[4]*dv1_sq*gamma_avg; 
  out[14] = 1.224744871391589*surft1_upper[6]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[6]*dv1_sq*gamma_avg+3.0*GCC[3]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[2]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[2]*dv1_sq*gamma_avg; 
  out[15] = 1.224744871391589*surft1_upper[7]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[7]*dv1_sq*gamma_avg+3.0*GCC[6]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[4]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[4]*dv1_sq*gamma_avg; 
  out[19] = 6.7082039324993685*GCC[2]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[0]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[0]*dv1_sq*gamma_avg; 
  out[21] = 6.708203932499369*GCC[5]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[1]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[1]*dv1_sq*gamma_avg; 
  out[22] = 6.708203932499369*GCC[7]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[2]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[2]*dv1_sq*gamma_avg; 
  out[23] = 6.7082039324993685*GCC[11]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[4]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[4]*dv1_sq*gamma_avg; 
  out[30] = 3.0*GCC[24]*dv1_sq*gamma_avg; 
  out[31] = 3.0*GCC[25]*dv1_sq*gamma_avg; 
  out[32] = -(2.7386127875258306*surft2_upper[3]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[3]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[0]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[33] = -(2.7386127875258306*surft2_upper[5]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[5]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[1]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[34] = 6.7082039324993685*GCC[4]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[3]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[3]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[2]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[35] = -(2.7386127875258306*surft2_upper[6]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[3]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[36] = 6.708203932499369*GCC[8]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[5]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[5]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[4]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[37] = -(2.7386127875258306*surft2_upper[7]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[7]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[5]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[38] = 6.708203932499369*GCC[10]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[6]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[6]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[39] = 6.7082039324993685*GCC[13]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[7]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[7]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[7]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[7]*dv1_sq*gamma_avg; 
} 

