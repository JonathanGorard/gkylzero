#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_diff_coeff_cross_1x3v_vyvz_ser_p1_lovy_upvz(const double *dxv, const double *gamma, const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9], const double* fpo_dgdv_surf, double *diff_coeff) { 
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[9]: 9 cell stencil of Rosenbluth potential G. 
  // fpo_g_surf_stencil[9]: 9 cell stencil of surface projection of G. 
  // fpo_dgdv_surf: Surface expansion of dG/dv in center cell. 
  // diff_coeff: Output array for diffusion tensor. 

  // Use cell-average value for gamma. 
  double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1 = 2.0/dxv[2]; 
  double dv2 = 2.0/dxv[3]; 
  double dv1_sq = 4.0/dxv[2]/dxv[3]; 
 
  double surft1_lo[8] = {0.0}; 
  double surft1_up[8] = {0.0}; 
  double surft2_lo[8] = {0.0}; 
  double surft2_up[8] = {0.0}; 
  double vol[40] = {0.0}; 
  double *out = &diff_coeff[200]; 

  const double* GBC = fpo_g_stencil[0]; 
  const double* GCC = fpo_g_stencil[1]; 
  const double* G_surf_CC_vz = &fpo_g_surf_stencil[1][16]; 
  const double* GBR = fpo_g_stencil[2]; 
  const double* GCR = fpo_g_stencil[3]; 
  const double* dGdvz_surf_CC_vy = &fpo_dgdv_surf[40]; 

  surft1_lo[0] = dGdvz_surf_CC_vy[0]/dv2; 
  surft1_lo[1] = dGdvz_surf_CC_vy[1]/dv2; 
  surft1_lo[2] = dGdvz_surf_CC_vy[2]/dv2; 
  surft1_lo[3] = dGdvz_surf_CC_vy[3]/dv2; 
  surft1_lo[4] = dGdvz_surf_CC_vy[4]/dv2; 
  surft1_lo[5] = dGdvz_surf_CC_vy[5]/dv2; 
  surft1_lo[6] = dGdvz_surf_CC_vy[6]/dv2; 
  surft1_lo[7] = dGdvz_surf_CC_vy[7]/dv2; 
  surft1_up[0] = -(0.7071067811865475*GCR[10])+0.7071067811865475*GCC[10]+0.6123724356957944*GCR[4]+0.6123724356957944*GCC[4]; 
  surft1_up[1] = -(0.7071067811865475*GCR[13])+0.7071067811865475*GCC[13]+0.6123724356957944*GCR[8]+0.6123724356957944*GCC[8]; 
  surft1_up[2] = -(0.7071067811865475*GCR[14])+0.7071067811865475*GCC[14]+0.6123724356957944*GCR[9]+0.6123724356957944*GCC[9]; 
  surft1_up[4] = -(0.7071067811865475*GCR[15])+0.7071067811865475*GCC[15]+0.6123724356957944*GCR[12]+0.6123724356957944*GCC[12]; 

  surft2_lo[0] = -(0.408248290463863*GCC[4])+0.408248290463863*GBC[4]+0.3535533905932737*GCC[0]+0.3535533905932737*GBC[0]; 
  surft2_lo[1] = -(0.408248290463863*GCC[8])+0.408248290463863*GBC[8]+0.3535533905932737*GCC[1]+0.3535533905932737*GBC[1]; 
  surft2_lo[2] = -(0.408248290463863*GCC[9])+0.408248290463863*GBC[9]+0.3535533905932737*GCC[2]+0.3535533905932737*GBC[2]; 
  surft2_lo[3] = -(0.408248290463863*GCC[10])+0.408248290463863*GBC[10]+0.3535533905932737*GCC[3]+0.3535533905932737*GBC[3]; 
  surft2_lo[4] = -(0.408248290463863*GCC[12])+0.408248290463863*GBC[12]+0.3535533905932737*GCC[5]+0.3535533905932737*GBC[5]; 
  surft2_lo[5] = -(0.408248290463863*GCC[13])+0.408248290463863*GBC[13]+0.3535533905932737*GCC[6]+0.3535533905932737*GBC[6]; 
  surft2_lo[6] = -(0.408248290463863*GCC[14])+0.408248290463863*GBC[14]+0.3535533905932737*GCC[7]+0.3535533905932737*GBC[7]; 
  surft2_lo[7] = -(0.408248290463863*GCC[15])+0.408248290463863*GBC[15]+0.3535533905932737*GCC[11]+0.3535533905932737*GBC[11]; 
  surft2_up[0] = 1.5811388300841895*GCC[32]+1.224744871391589*GCC[4]+0.7071067811865475*GCC[0]; 
  surft2_up[1] = 1.5811388300841898*GCC[33]+1.224744871391589*GCC[8]+0.7071067811865475*GCC[1]; 
  surft2_up[2] = 1.5811388300841898*GCC[34]+1.224744871391589*GCC[9]+0.7071067811865475*GCC[2]; 
  surft2_up[3] = 1.5811388300841898*GCC[35]+1.224744871391589*GCC[10]+0.7071067811865475*GCC[3]; 
  surft2_up[4] = 1.5811388300841895*GCC[36]+1.224744871391589*GCC[12]+0.7071067811865475*GCC[5]; 
  surft2_up[5] = 1.5811388300841895*GCC[37]+1.224744871391589*GCC[13]+0.7071067811865475*GCC[6]; 
  surft2_up[6] = 1.5811388300841895*GCC[38]+1.224744871391589*GCC[14]+0.7071067811865475*GCC[7]; 
  surft2_up[7] = 1.5811388300841898*GCC[39]+1.224744871391589*GCC[15]+0.7071067811865475*GCC[11]; 

  vol[10] = 3.0*GCC[0]; 
  vol[13] = 3.0*GCC[1]; 
  vol[14] = 3.0*GCC[2]; 
  vol[15] = 3.0*GCC[5]; 
  vol[22] = 3.0*GCC[16]; 
  vol[23] = 3.0*GCC[17]; 
  vol[27] = 6.7082039324993685*GCC[3]; 
  vol[29] = 6.708203932499369*GCC[6]; 
  vol[30] = 6.708203932499369*GCC[7]; 
  vol[31] = 6.7082039324993685*GCC[11]; 
  vol[35] = 6.7082039324993685*GCC[4]; 
  vol[37] = 6.708203932499369*GCC[8]; 
  vol[38] = 6.708203932499369*GCC[9]; 
  vol[39] = 6.7082039324993685*GCC[12]; 

  out[0] = (vol[0]+0.7071067811865475*surft1_up[0]-0.7071067811865475*surft1_lo[0])*dv1_sq*gamma_avg; 
  out[1] = (vol[1]+0.7071067811865475*surft1_up[1]-0.7071067811865475*surft1_lo[1])*dv1_sq*gamma_avg; 
  out[2] = (vol[2]+0.7071067811865475*surft1_up[2]-0.7071067811865475*surft1_lo[2])*dv1_sq*gamma_avg; 
  out[3] = (vol[3]-1.224744871391589*surft2_up[0]+1.224744871391589*surft2_lo[0]+1.224744871391589*surft1_up[0]+1.224744871391589*surft1_lo[0])*dv1_sq*gamma_avg; 
  out[4] = (vol[4]+0.7071067811865475*surft1_up[3]-0.7071067811865475*surft1_lo[3])*dv1_sq*gamma_avg; 
  out[5] = (vol[5]+0.7071067811865475*surft1_up[4]-0.7071067811865475*surft1_lo[4])*dv1_sq*gamma_avg; 
  out[6] = (vol[6]-1.224744871391589*surft2_up[1]+1.224744871391589*surft2_lo[1]+1.224744871391589*surft1_up[1]+1.224744871391589*surft1_lo[1])*dv1_sq*gamma_avg; 
  out[7] = (vol[7]-1.224744871391589*surft2_up[2]+1.224744871391589*surft2_lo[2]+1.224744871391589*surft1_up[2]+1.224744871391589*surft1_lo[2])*dv1_sq*gamma_avg; 
  out[8] = (vol[8]+0.7071067811865475*surft1_up[5]-0.7071067811865475*surft1_lo[5])*dv1_sq*gamma_avg; 
  out[9] = (vol[9]+0.7071067811865475*surft1_up[6]-0.7071067811865475*surft1_lo[6])*dv1_sq*gamma_avg; 
  out[10] = (vol[10]+1.224744871391589*surft1_up[3]+1.224744871391589*surft1_lo[3]-2.1213203435596424*surft2_up[0]-2.1213203435596424*surft2_lo[0])*dv1_sq*gamma_avg; 
  out[11] = (vol[11]-1.224744871391589*surft2_up[4]+1.224744871391589*surft2_lo[4]+1.224744871391589*surft1_up[4]+1.224744871391589*surft1_lo[4])*dv1_sq*gamma_avg; 
  out[12] = (vol[12]+0.7071067811865475*surft1_up[7]-0.7071067811865475*surft1_lo[7])*dv1_sq*gamma_avg; 
  out[13] = (vol[13]+1.224744871391589*surft1_up[5]+1.224744871391589*surft1_lo[5]-2.1213203435596424*surft2_up[1]-2.1213203435596424*surft2_lo[1])*dv1_sq*gamma_avg; 
  out[14] = (vol[14]+1.224744871391589*surft1_up[6]+1.224744871391589*surft1_lo[6]-2.1213203435596424*surft2_up[2]-2.1213203435596424*surft2_lo[2])*dv1_sq*gamma_avg; 
  out[15] = (vol[15]+1.224744871391589*surft1_up[7]+1.224744871391589*surft1_lo[7]-2.1213203435596424*surft2_up[4]-2.1213203435596424*surft2_lo[4])*dv1_sq*gamma_avg; 
  out[16] = vol[16]*dv1_sq*gamma_avg; 
  out[17] = vol[17]*dv1_sq*gamma_avg; 
  out[18] = vol[18]*dv1_sq*gamma_avg; 
  out[19] = vol[19]*dv1_sq*gamma_avg; 
  out[20] = vol[20]*dv1_sq*gamma_avg; 
  out[21] = vol[21]*dv1_sq*gamma_avg; 
  out[22] = vol[22]*dv1_sq*gamma_avg; 
  out[23] = vol[23]*dv1_sq*gamma_avg; 
  out[24] = (vol[24]-2.7386127875258306*surft2_up[3]+2.7386127875258306*surft2_lo[3]+1.5811388300841895*surft1_up[0]-1.5811388300841895*surft1_lo[0])*dv1_sq*gamma_avg; 
  out[25] = (vol[25]-2.7386127875258306*surft2_up[5]+2.7386127875258306*surft2_lo[5]+1.5811388300841898*surft1_up[1]-1.5811388300841898*surft1_lo[1])*dv1_sq*gamma_avg; 
  out[26] = (vol[26]-2.7386127875258306*surft2_up[6]+2.7386127875258306*surft2_lo[6]+1.5811388300841898*surft1_up[2]-1.5811388300841898*surft1_lo[2])*dv1_sq*gamma_avg; 
  out[27] = (vol[27]-4.743416490252569*surft2_up[3]-4.743416490252569*surft2_lo[3]+1.5811388300841898*surft1_up[3]-1.5811388300841898*surft1_lo[3])*dv1_sq*gamma_avg; 
  out[28] = (vol[28]-2.7386127875258306*surft2_up[7]+2.7386127875258306*surft2_lo[7]+1.5811388300841895*surft1_up[4]-1.5811388300841895*surft1_lo[4])*dv1_sq*gamma_avg; 
  out[29] = (vol[29]-4.743416490252569*surft2_up[5]-4.743416490252569*surft2_lo[5]+1.5811388300841895*surft1_up[5]-1.5811388300841895*surft1_lo[5])*dv1_sq*gamma_avg; 
  out[30] = (vol[30]-4.743416490252569*surft2_up[6]-4.743416490252569*surft2_lo[6]+1.5811388300841895*surft1_up[6]-1.5811388300841895*surft1_lo[6])*dv1_sq*gamma_avg; 
  out[31] = (vol[31]-4.743416490252569*surft2_up[7]-4.743416490252569*surft2_lo[7]+1.5811388300841898*surft1_up[7]-1.5811388300841898*surft1_lo[7])*dv1_sq*gamma_avg; 
  out[32] = vol[32]*dv1_sq*gamma_avg; 
  out[33] = vol[33]*dv1_sq*gamma_avg; 
  out[34] = vol[34]*dv1_sq*gamma_avg; 
  out[35] = (vol[35]-2.7386127875258306*surft2_up[0]+2.7386127875258306*surft2_lo[0])*dv1_sq*gamma_avg; 
  out[36] = vol[36]*dv1_sq*gamma_avg; 
  out[37] = (vol[37]-2.7386127875258306*surft2_up[1]+2.7386127875258306*surft2_lo[1])*dv1_sq*gamma_avg; 
  out[38] = (vol[38]-2.7386127875258306*surft2_up[2]+2.7386127875258306*surft2_lo[2])*dv1_sq*gamma_avg; 
  out[39] = (vol[39]-2.7386127875258306*surft2_up[4]+2.7386127875258306*surft2_lo[4])*dv1_sq*gamma_avg; 
} 
