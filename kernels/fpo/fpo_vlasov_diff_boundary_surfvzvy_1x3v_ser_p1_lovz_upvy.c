#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_vlasov_diff_boundary_surfvzvy_1x3v_ser_p1_lovz_upvy(const double *dxv, const double* diff_coeff_stencil[9], const double* f_stencil[9], double* out) { 
  // dxv[NDIM]: Cell spacing in each direction. 
  // diff_coeff_stencil[3]: 3-cell stencil of diffusion tensor. 
  // f_stencil[9]: 9-cell stencil of distribution function. 
  // out: Incremented output. 


  double dv1_sq = 4.0/dxv[3]/dxv[2]; 
 
  double D_rec_lo[8] = {0.0}; 
  double D_rec_up[8] = {0.0}; 
  double f_rec_lo[8] = {0.0}; 
  double f_rec_up[8] = {0.0}; 
  double df_rec_lo[8] = {0.0}; 
  double df_rec_up[8] = {0.0}; 
  double surft1_lo[8] = {0.0}; 
  double surft1_up[8] = {0.0}; 
  double surft2_lo[8] = {0.0}; 
  double surft2_up[8] = {0.0}; 

  const double* DBC = &diff_coeff_stencil[0][280]; 
  const double* fBC = f_stencil[0]; 
  const double* DBR = &diff_coeff_stencil[1][280]; 
  const double* fBR = f_stencil[1]; 
  const double* DCC = &diff_coeff_stencil[2][280]; 
  const double* fCC = f_stencil[2]; 
  const double* DCR = &diff_coeff_stencil[3][280]; 
  const double* fCR = f_stencil[3]; 

  D_rec_lo[0] = 1.5811388300841895*DCC[32]-1.224744871391589*DCC[4]+0.7071067811865475*DCC[0]; 
  D_rec_lo[1] = 1.5811388300841898*DCC[33]-1.224744871391589*DCC[8]+0.7071067811865475*DCC[1]; 
  D_rec_lo[2] = 1.5811388300841898*DCC[34]-1.224744871391589*DCC[9]+0.7071067811865475*DCC[2]; 
  D_rec_lo[3] = 1.5811388300841898*DCC[35]-1.224744871391589*DCC[10]+0.7071067811865475*DCC[3]; 
  D_rec_lo[4] = 1.5811388300841895*DCC[36]-1.224744871391589*DCC[12]+0.7071067811865475*DCC[5]; 
  D_rec_lo[5] = 1.5811388300841895*DCC[37]-1.224744871391589*DCC[13]+0.7071067811865475*DCC[6]; 
  D_rec_lo[6] = 1.5811388300841895*DCC[38]-1.224744871391589*DCC[14]+0.7071067811865475*DCC[7]; 
  D_rec_lo[7] = 1.5811388300841898*DCC[39]-1.224744871391589*DCC[15]+0.7071067811865475*DCC[11]; 
  D_rec_up[0] = -(0.408248290463863*DCR[4])+0.408248290463863*DCC[4]+0.3535533905932737*DCR[0]+0.3535533905932737*DCC[0]; 
  D_rec_up[1] = -(0.408248290463863*DCR[8])+0.408248290463863*DCC[8]+0.3535533905932737*DCR[1]+0.3535533905932737*DCC[1]; 
  D_rec_up[2] = -(0.408248290463863*DCR[9])+0.408248290463863*DCC[9]+0.3535533905932737*DCR[2]+0.3535533905932737*DCC[2]; 
  D_rec_up[3] = -(0.408248290463863*DCR[10])+0.408248290463863*DCC[10]+0.3535533905932737*DCR[3]+0.3535533905932737*DCC[3]; 
  D_rec_up[4] = -(0.408248290463863*DCR[12])+0.408248290463863*DCC[12]+0.3535533905932737*DCR[5]+0.3535533905932737*DCC[5]; 
  D_rec_up[5] = -(0.408248290463863*DCR[13])+0.408248290463863*DCC[13]+0.3535533905932737*DCR[6]+0.3535533905932737*DCC[6]; 
  D_rec_up[6] = -(0.408248290463863*DCR[14])+0.408248290463863*DCC[14]+0.3535533905932737*DCR[7]+0.3535533905932737*DCC[7]; 
  D_rec_up[7] = -(0.408248290463863*DCR[15])+0.408248290463863*DCC[15]+0.3535533905932737*DCR[11]+0.3535533905932737*DCC[11]; 

  f_rec_lo[0] = 1.5811388300841895*fCC[32]-1.224744871391589*fCC[4]+0.7071067811865475*fCC[0]; 
  f_rec_lo[1] = 1.5811388300841898*fCC[33]-1.224744871391589*fCC[8]+0.7071067811865475*fCC[1]; 
  f_rec_lo[2] = 1.5811388300841898*fCC[34]-1.224744871391589*fCC[9]+0.7071067811865475*fCC[2]; 
  f_rec_lo[3] = 1.5811388300841898*fCC[35]-1.224744871391589*fCC[10]+0.7071067811865475*fCC[3]; 
  f_rec_lo[4] = 1.5811388300841895*fCC[36]-1.224744871391589*fCC[12]+0.7071067811865475*fCC[5]; 
  f_rec_lo[5] = 1.5811388300841895*fCC[37]-1.224744871391589*fCC[13]+0.7071067811865475*fCC[6]; 
  f_rec_lo[6] = 1.5811388300841895*fCC[38]-1.224744871391589*fCC[14]+0.7071067811865475*fCC[7]; 
  f_rec_lo[7] = 1.5811388300841898*fCC[39]-1.224744871391589*fCC[15]+0.7071067811865475*fCC[11]; 
  f_rec_up[0] = -(0.408248290463863*fCR[4])+0.408248290463863*fCC[4]+0.3535533905932737*fCR[0]+0.3535533905932737*fCC[0]; 
  f_rec_up[1] = -(0.408248290463863*fCR[8])+0.408248290463863*fCC[8]+0.3535533905932737*fCR[1]+0.3535533905932737*fCC[1]; 
  f_rec_up[2] = -(0.408248290463863*fCR[9])+0.408248290463863*fCC[9]+0.3535533905932737*fCR[2]+0.3535533905932737*fCC[2]; 
  f_rec_up[3] = -(0.408248290463863*fCR[10])+0.408248290463863*fCC[10]+0.3535533905932737*fCR[3]+0.3535533905932737*fCC[3]; 
  f_rec_up[4] = -(0.408248290463863*fCR[12])+0.408248290463863*fCC[12]+0.3535533905932737*fCR[5]+0.3535533905932737*fCC[5]; 
  f_rec_up[5] = -(0.408248290463863*fCR[13])+0.408248290463863*fCC[13]+0.3535533905932737*fCR[6]+0.3535533905932737*fCC[6]; 
  f_rec_up[6] = -(0.408248290463863*fCR[14])+0.408248290463863*fCC[14]+0.3535533905932737*fCR[7]+0.3535533905932737*fCC[7]; 
  f_rec_up[7] = -(0.408248290463863*fCR[15])+0.408248290463863*fCC[15]+0.3535533905932737*fCR[11]+0.3535533905932737*fCC[11]; 

  df_rec_up[0] = -(0.11785113019775789*fCR[10])+0.11785113019775789*fCC[10]+0.11785113019775789*fBR[10]-0.11785113019775789*fBC[10]+0.10206207261596573*fCR[4]-0.10206207261596573*fCC[4]+0.10206207261596573*fBR[4]-0.10206207261596573*fBC[4]+0.8660254037844386*f_rec_up[3]+0.10206207261596573*fCR[3]+0.10206207261596573*fCC[3]-0.10206207261596573*fBR[3]-0.10206207261596573*fBC[3]+0.5*f_rec_up[0]-0.0883883476483184*fCR[0]-0.0883883476483184*fCC[0]-0.0883883476483184*fBR[0]-0.0883883476483184*fBC[0]; 
  df_rec_up[1] = -(0.11785113019775789*fCR[13])+0.11785113019775789*fCC[13]+0.11785113019775789*fBR[13]-0.11785113019775789*fBC[13]+0.10206207261596573*fCR[8]-0.10206207261596573*fCC[8]+0.10206207261596573*fBR[8]-0.10206207261596573*fBC[8]+0.10206207261596573*fCR[6]+0.10206207261596573*fCC[6]-0.10206207261596573*fBR[6]-0.10206207261596573*fBC[6]+0.8660254037844386*f_rec_up[5]+0.5*f_rec_up[1]-0.0883883476483184*fCR[1]-0.0883883476483184*fCC[1]-0.0883883476483184*fBR[1]-0.0883883476483184*fBC[1]; 
  df_rec_up[2] = -(0.11785113019775789*fCR[14])+0.11785113019775789*fCC[14]+0.11785113019775789*fBR[14]-0.11785113019775789*fBC[14]+0.10206207261596573*fCR[9]-0.10206207261596573*fCC[9]+0.10206207261596573*fBR[9]-0.10206207261596573*fBC[9]+0.10206207261596573*fCR[7]+0.10206207261596573*fCC[7]-0.10206207261596573*fBR[7]-0.10206207261596573*fBC[7]+0.8660254037844386*f_rec_up[6]+0.5*f_rec_up[2]-0.0883883476483184*fCR[2]-0.0883883476483184*fCC[2]-0.0883883476483184*fBR[2]-0.0883883476483184*fBC[2]; 
  df_rec_up[3] = 0.20412414523193148*fCR[10]-0.20412414523193148*fCC[10]-0.20412414523193148*fBR[10]+0.20412414523193148*fBC[10]+0.5303300858899105*fCR[4]-0.5303300858899105*fCC[4]-0.1767766952966368*fBR[4]+0.1767766952966368*fBC[4]+1.5*f_rec_up[3]-0.1767766952966368*fCR[3]-0.1767766952966368*fCC[3]+0.1767766952966368*fBR[3]+0.1767766952966368*fBC[3]+0.8660254037844386*f_rec_up[0]-0.45927932677184563*fCR[0]-0.45927932677184563*fCC[0]+0.15309310892394856*fBR[0]+0.15309310892394856*fBC[0]; 
  df_rec_up[4] = -(0.11785113019775789*fCR[15])+0.11785113019775789*fCC[15]+0.11785113019775789*fBR[15]-0.11785113019775789*fBC[15]+0.10206207261596573*fCR[12]-0.10206207261596573*fCC[12]+0.10206207261596573*fBR[12]-0.10206207261596573*fBC[12]+0.10206207261596573*fCR[11]+0.10206207261596573*fCC[11]-0.10206207261596573*fBR[11]-0.10206207261596573*fBC[11]+0.8660254037844386*f_rec_up[7]-0.0883883476483184*fCR[5]-0.0883883476483184*fCC[5]-0.0883883476483184*fBR[5]-0.0883883476483184*fBC[5]+0.5*f_rec_up[4]; 
  df_rec_up[5] = 0.20412414523193148*fCR[13]-0.20412414523193148*fCC[13]-0.20412414523193148*fBR[13]+0.20412414523193148*fBC[13]+0.5303300858899105*fCR[8]-0.5303300858899105*fCC[8]-0.1767766952966368*fBR[8]+0.1767766952966368*fBC[8]-0.1767766952966368*fCR[6]-0.1767766952966368*fCC[6]+0.1767766952966368*fBR[6]+0.1767766952966368*fBC[6]+1.5*f_rec_up[5]+0.8660254037844386*f_rec_up[1]-0.45927932677184563*fCR[1]-0.45927932677184563*fCC[1]+0.15309310892394856*fBR[1]+0.15309310892394856*fBC[1]; 
  df_rec_up[6] = 0.20412414523193148*fCR[14]-0.20412414523193148*fCC[14]-0.20412414523193148*fBR[14]+0.20412414523193148*fBC[14]+0.5303300858899105*fCR[9]-0.5303300858899105*fCC[9]-0.1767766952966368*fBR[9]+0.1767766952966368*fBC[9]-0.1767766952966368*fCR[7]-0.1767766952966368*fCC[7]+0.1767766952966368*fBR[7]+0.1767766952966368*fBC[7]+1.5*f_rec_up[6]+0.8660254037844386*f_rec_up[2]-0.45927932677184563*fCR[2]-0.45927932677184563*fCC[2]+0.15309310892394856*fBR[2]+0.15309310892394856*fBC[2]; 
  df_rec_up[7] = 0.20412414523193148*fCR[15]-0.20412414523193148*fCC[15]-0.20412414523193148*fBR[15]+0.20412414523193148*fBC[15]+0.5303300858899105*fCR[12]-0.5303300858899105*fCC[12]-0.1767766952966368*fBR[12]+0.1767766952966368*fBC[12]-0.1767766952966368*fCR[11]-0.1767766952966368*fCC[11]+0.1767766952966368*fBR[11]+0.1767766952966368*fBC[11]+1.5*f_rec_up[7]-0.45927932677184563*fCR[5]-0.45927932677184563*fCC[5]+0.15309310892394856*fBR[5]+0.15309310892394856*fBC[5]+0.8660254037844386*f_rec_up[4]; 

  surft1_up[0] = 0.3535533905932737*D_rec_up[7]*df_rec_up[7]+0.3535533905932737*D_rec_up[6]*df_rec_up[6]+0.3535533905932737*D_rec_up[5]*df_rec_up[5]+0.3535533905932737*D_rec_up[4]*df_rec_up[4]+0.3535533905932737*D_rec_up[3]*df_rec_up[3]+0.3535533905932737*D_rec_up[2]*df_rec_up[2]+0.3535533905932737*D_rec_up[1]*df_rec_up[1]+0.3535533905932737*D_rec_up[0]*df_rec_up[0]; 
  surft1_up[1] = 0.3535533905932737*D_rec_up[6]*df_rec_up[7]+0.3535533905932737*df_rec_up[6]*D_rec_up[7]+0.3535533905932737*D_rec_up[3]*df_rec_up[5]+0.3535533905932737*df_rec_up[3]*D_rec_up[5]+0.3535533905932737*D_rec_up[2]*df_rec_up[4]+0.3535533905932737*df_rec_up[2]*D_rec_up[4]+0.3535533905932737*D_rec_up[0]*df_rec_up[1]+0.3535533905932737*df_rec_up[0]*D_rec_up[1]; 
  surft1_up[2] = 0.3535533905932737*D_rec_up[5]*df_rec_up[7]+0.3535533905932737*df_rec_up[5]*D_rec_up[7]+0.3535533905932737*D_rec_up[3]*df_rec_up[6]+0.3535533905932737*df_rec_up[3]*D_rec_up[6]+0.3535533905932737*D_rec_up[1]*df_rec_up[4]+0.3535533905932737*df_rec_up[1]*D_rec_up[4]+0.3535533905932737*D_rec_up[0]*df_rec_up[2]+0.3535533905932737*df_rec_up[0]*D_rec_up[2]; 
  surft1_up[3] = 0.3535533905932737*D_rec_up[4]*df_rec_up[7]+0.3535533905932737*df_rec_up[4]*D_rec_up[7]+0.3535533905932737*D_rec_up[2]*df_rec_up[6]+0.3535533905932737*df_rec_up[2]*D_rec_up[6]+0.3535533905932737*D_rec_up[1]*df_rec_up[5]+0.3535533905932737*df_rec_up[1]*D_rec_up[5]+0.3535533905932737*D_rec_up[0]*df_rec_up[3]+0.3535533905932737*df_rec_up[0]*D_rec_up[3]; 
  surft1_up[4] = 0.3535533905932737*D_rec_up[3]*df_rec_up[7]+0.3535533905932737*df_rec_up[3]*D_rec_up[7]+0.3535533905932737*D_rec_up[5]*df_rec_up[6]+0.3535533905932737*df_rec_up[5]*D_rec_up[6]+0.3535533905932737*D_rec_up[0]*df_rec_up[4]+0.3535533905932737*df_rec_up[0]*D_rec_up[4]+0.3535533905932737*D_rec_up[1]*df_rec_up[2]+0.3535533905932737*df_rec_up[1]*D_rec_up[2]; 
  surft1_up[5] = 0.3535533905932737*D_rec_up[2]*df_rec_up[7]+0.3535533905932737*df_rec_up[2]*D_rec_up[7]+0.3535533905932737*D_rec_up[4]*df_rec_up[6]+0.3535533905932737*df_rec_up[4]*D_rec_up[6]+0.3535533905932737*D_rec_up[0]*df_rec_up[5]+0.3535533905932737*df_rec_up[0]*D_rec_up[5]+0.3535533905932737*D_rec_up[1]*df_rec_up[3]+0.3535533905932737*df_rec_up[1]*D_rec_up[3]; 
  surft1_up[6] = 0.3535533905932737*D_rec_up[1]*df_rec_up[7]+0.3535533905932737*df_rec_up[1]*D_rec_up[7]+0.3535533905932737*D_rec_up[0]*df_rec_up[6]+0.3535533905932737*df_rec_up[0]*D_rec_up[6]+0.3535533905932737*D_rec_up[4]*df_rec_up[5]+0.3535533905932737*df_rec_up[4]*D_rec_up[5]+0.3535533905932737*D_rec_up[2]*df_rec_up[3]+0.3535533905932737*df_rec_up[2]*D_rec_up[3]; 
  surft1_up[7] = 0.3535533905932737*D_rec_up[0]*df_rec_up[7]+0.3535533905932737*df_rec_up[0]*D_rec_up[7]+0.3535533905932737*D_rec_up[1]*df_rec_up[6]+0.3535533905932737*df_rec_up[1]*D_rec_up[6]+0.3535533905932737*D_rec_up[2]*df_rec_up[5]+0.3535533905932737*df_rec_up[2]*D_rec_up[5]+0.3535533905932737*D_rec_up[3]*df_rec_up[4]+0.3535533905932737*df_rec_up[3]*D_rec_up[4]; 

  surft2_lo[0] = 0.3535533905932737*D_rec_lo[7]*f_rec_lo[7]+0.3535533905932737*D_rec_lo[6]*f_rec_lo[6]+0.3535533905932737*D_rec_lo[5]*f_rec_lo[5]+0.3535533905932737*D_rec_lo[4]*f_rec_lo[4]+0.3535533905932737*D_rec_lo[3]*f_rec_lo[3]+0.3535533905932737*D_rec_lo[2]*f_rec_lo[2]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[1]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[0]; 
  surft2_lo[1] = 0.3535533905932737*D_rec_lo[6]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[6]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[3]*f_rec_lo[5]+0.3535533905932737*f_rec_lo[3]*D_rec_lo[5]+0.3535533905932737*D_rec_lo[2]*f_rec_lo[4]+0.3535533905932737*f_rec_lo[2]*D_rec_lo[4]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[1]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[1]; 
  surft2_lo[2] = 0.3535533905932737*D_rec_lo[5]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[5]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[3]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[3]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[4]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[4]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[2]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[2]; 
  surft2_lo[3] = 0.3535533905932737*D_rec_lo[4]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[4]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[2]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[2]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[5]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[5]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[3]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[3]; 
  surft2_lo[4] = 0.3535533905932737*D_rec_lo[3]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[3]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[5]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[5]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[4]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[4]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[2]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[2]; 
  surft2_lo[5] = 0.3535533905932737*D_rec_lo[2]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[2]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[4]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[4]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[5]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[5]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[3]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[3]; 
  surft2_lo[6] = 0.3535533905932737*D_rec_lo[1]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[0]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[4]*f_rec_lo[5]+0.3535533905932737*f_rec_lo[4]*D_rec_lo[5]+0.3535533905932737*D_rec_lo[2]*f_rec_lo[3]+0.3535533905932737*f_rec_lo[2]*D_rec_lo[3]; 
  surft2_lo[7] = 0.3535533905932737*D_rec_lo[0]*f_rec_lo[7]+0.3535533905932737*f_rec_lo[0]*D_rec_lo[7]+0.3535533905932737*D_rec_lo[1]*f_rec_lo[6]+0.3535533905932737*f_rec_lo[1]*D_rec_lo[6]+0.3535533905932737*D_rec_lo[2]*f_rec_lo[5]+0.3535533905932737*f_rec_lo[2]*D_rec_lo[5]+0.3535533905932737*D_rec_lo[3]*f_rec_lo[4]+0.3535533905932737*f_rec_lo[3]*D_rec_lo[4]; 
  surft2_up[0] = 0.3535533905932737*D_rec_up[7]*f_rec_up[7]+0.3535533905932737*D_rec_up[6]*f_rec_up[6]+0.3535533905932737*D_rec_up[5]*f_rec_up[5]+0.3535533905932737*D_rec_up[4]*f_rec_up[4]+0.3535533905932737*D_rec_up[3]*f_rec_up[3]+0.3535533905932737*D_rec_up[2]*f_rec_up[2]+0.3535533905932737*D_rec_up[1]*f_rec_up[1]+0.3535533905932737*D_rec_up[0]*f_rec_up[0]; 
  surft2_up[1] = 0.3535533905932737*D_rec_up[6]*f_rec_up[7]+0.3535533905932737*f_rec_up[6]*D_rec_up[7]+0.3535533905932737*D_rec_up[3]*f_rec_up[5]+0.3535533905932737*f_rec_up[3]*D_rec_up[5]+0.3535533905932737*D_rec_up[2]*f_rec_up[4]+0.3535533905932737*f_rec_up[2]*D_rec_up[4]+0.3535533905932737*D_rec_up[0]*f_rec_up[1]+0.3535533905932737*f_rec_up[0]*D_rec_up[1]; 
  surft2_up[2] = 0.3535533905932737*D_rec_up[5]*f_rec_up[7]+0.3535533905932737*f_rec_up[5]*D_rec_up[7]+0.3535533905932737*D_rec_up[3]*f_rec_up[6]+0.3535533905932737*f_rec_up[3]*D_rec_up[6]+0.3535533905932737*D_rec_up[1]*f_rec_up[4]+0.3535533905932737*f_rec_up[1]*D_rec_up[4]+0.3535533905932737*D_rec_up[0]*f_rec_up[2]+0.3535533905932737*f_rec_up[0]*D_rec_up[2]; 
  surft2_up[3] = 0.3535533905932737*D_rec_up[4]*f_rec_up[7]+0.3535533905932737*f_rec_up[4]*D_rec_up[7]+0.3535533905932737*D_rec_up[2]*f_rec_up[6]+0.3535533905932737*f_rec_up[2]*D_rec_up[6]+0.3535533905932737*D_rec_up[1]*f_rec_up[5]+0.3535533905932737*f_rec_up[1]*D_rec_up[5]+0.3535533905932737*D_rec_up[0]*f_rec_up[3]+0.3535533905932737*f_rec_up[0]*D_rec_up[3]; 
  surft2_up[4] = 0.3535533905932737*D_rec_up[3]*f_rec_up[7]+0.3535533905932737*f_rec_up[3]*D_rec_up[7]+0.3535533905932737*D_rec_up[5]*f_rec_up[6]+0.3535533905932737*f_rec_up[5]*D_rec_up[6]+0.3535533905932737*D_rec_up[0]*f_rec_up[4]+0.3535533905932737*f_rec_up[0]*D_rec_up[4]+0.3535533905932737*D_rec_up[1]*f_rec_up[2]+0.3535533905932737*f_rec_up[1]*D_rec_up[2]; 
  surft2_up[5] = 0.3535533905932737*D_rec_up[2]*f_rec_up[7]+0.3535533905932737*f_rec_up[2]*D_rec_up[7]+0.3535533905932737*D_rec_up[4]*f_rec_up[6]+0.3535533905932737*f_rec_up[4]*D_rec_up[6]+0.3535533905932737*D_rec_up[0]*f_rec_up[5]+0.3535533905932737*f_rec_up[0]*D_rec_up[5]+0.3535533905932737*D_rec_up[1]*f_rec_up[3]+0.3535533905932737*f_rec_up[1]*D_rec_up[3]; 
  surft2_up[6] = 0.3535533905932737*D_rec_up[1]*f_rec_up[7]+0.3535533905932737*f_rec_up[1]*D_rec_up[7]+0.3535533905932737*D_rec_up[0]*f_rec_up[6]+0.3535533905932737*f_rec_up[0]*D_rec_up[6]+0.3535533905932737*D_rec_up[4]*f_rec_up[5]+0.3535533905932737*f_rec_up[4]*D_rec_up[5]+0.3535533905932737*D_rec_up[2]*f_rec_up[3]+0.3535533905932737*f_rec_up[2]*D_rec_up[3]; 
  surft2_up[7] = 0.3535533905932737*D_rec_up[0]*f_rec_up[7]+0.3535533905932737*f_rec_up[0]*D_rec_up[7]+0.3535533905932737*D_rec_up[1]*f_rec_up[6]+0.3535533905932737*f_rec_up[1]*D_rec_up[6]+0.3535533905932737*D_rec_up[2]*f_rec_up[5]+0.3535533905932737*f_rec_up[2]*D_rec_up[5]+0.3535533905932737*D_rec_up[3]*f_rec_up[4]+0.3535533905932737*f_rec_up[3]*D_rec_up[4]; 

  out[0] += 0.35355339059327373*surft1_up[0]*dv1_sq-0.35355339059327373*surft1_lo[0]*dv1_sq; 
  out[1] += 0.35355339059327373*surft1_up[1]*dv1_sq-0.35355339059327373*surft1_lo[1]*dv1_sq; 
  out[2] += 0.35355339059327373*surft1_up[2]*dv1_sq-0.35355339059327373*surft1_lo[2]*dv1_sq; 
  out[3] += 0.35355339059327373*surft1_up[3]*dv1_sq-0.35355339059327373*surft1_lo[3]*dv1_sq-0.6123724356957945*surft2_up[0]*dv1_sq+0.6123724356957945*surft2_lo[0]*dv1_sq; 
  out[4] += 0.6123724356957945*surft1_up[0]*dv1_sq+0.6123724356957945*surft1_lo[0]*dv1_sq; 
  out[5] += 0.35355339059327373*surft1_up[4]*dv1_sq-0.35355339059327373*surft1_lo[4]*dv1_sq; 
  out[6] += 0.35355339059327373*surft1_up[5]*dv1_sq-0.35355339059327373*surft1_lo[5]*dv1_sq-0.6123724356957945*surft2_up[1]*dv1_sq+0.6123724356957945*surft2_lo[1]*dv1_sq; 
  out[7] += 0.35355339059327373*surft1_up[6]*dv1_sq-0.35355339059327373*surft1_lo[6]*dv1_sq-0.6123724356957945*surft2_up[2]*dv1_sq+0.6123724356957945*surft2_lo[2]*dv1_sq; 
  out[8] += 0.6123724356957945*surft1_up[1]*dv1_sq+0.6123724356957945*surft1_lo[1]*dv1_sq; 
  out[9] += 0.6123724356957945*surft1_up[2]*dv1_sq+0.6123724356957945*surft1_lo[2]*dv1_sq; 
  out[10] += 0.6123724356957945*surft1_up[3]*dv1_sq+0.6123724356957945*surft1_lo[3]*dv1_sq-1.0606601717798212*surft2_up[0]*dv1_sq-1.0606601717798212*surft2_lo[0]*dv1_sq; 
  out[11] += 0.35355339059327373*surft1_up[7]*dv1_sq-0.35355339059327373*surft1_lo[7]*dv1_sq-0.6123724356957945*surft2_up[4]*dv1_sq+0.6123724356957945*surft2_lo[4]*dv1_sq; 
  out[12] += 0.6123724356957945*surft1_up[4]*dv1_sq+0.6123724356957945*surft1_lo[4]*dv1_sq; 
  out[13] += 0.6123724356957945*surft1_up[5]*dv1_sq+0.6123724356957945*surft1_lo[5]*dv1_sq-1.0606601717798212*surft2_up[1]*dv1_sq-1.0606601717798212*surft2_lo[1]*dv1_sq; 
  out[14] += 0.6123724356957945*surft1_up[6]*dv1_sq+0.6123724356957945*surft1_lo[6]*dv1_sq-1.0606601717798212*surft2_up[2]*dv1_sq-1.0606601717798212*surft2_lo[2]*dv1_sq; 
  out[15] += 0.6123724356957945*surft1_up[7]*dv1_sq+0.6123724356957945*surft1_lo[7]*dv1_sq-1.0606601717798212*surft2_up[4]*dv1_sq-1.0606601717798212*surft2_lo[4]*dv1_sq; 
  out[24] += 1.3693063937629153*surft2_lo[3]*dv1_sq-1.3693063937629153*surft2_up[3]*dv1_sq; 
  out[25] += 1.3693063937629153*surft2_lo[5]*dv1_sq-1.3693063937629153*surft2_up[5]*dv1_sq; 
  out[26] += 1.3693063937629153*surft2_lo[6]*dv1_sq-1.3693063937629153*surft2_up[6]*dv1_sq; 
  out[27] += -(2.3717082451262845*surft2_up[3]*dv1_sq)-2.3717082451262845*surft2_lo[3]*dv1_sq; 
  out[28] += 1.3693063937629153*surft2_lo[7]*dv1_sq-1.3693063937629153*surft2_up[7]*dv1_sq; 
  out[29] += -(2.3717082451262845*surft2_up[5]*dv1_sq)-2.3717082451262845*surft2_lo[5]*dv1_sq; 
  out[30] += -(2.3717082451262845*surft2_up[6]*dv1_sq)-2.3717082451262845*surft2_lo[6]*dv1_sq; 
  out[31] += -(2.3717082451262845*surft2_up[7]*dv1_sq)-2.3717082451262845*surft2_lo[7]*dv1_sq; 
  out[32] += 0.7905694150420948*surft1_up[0]*dv1_sq-0.7905694150420948*surft1_lo[0]*dv1_sq; 
  out[33] += 0.7905694150420949*surft1_up[1]*dv1_sq-0.7905694150420949*surft1_lo[1]*dv1_sq; 
  out[34] += 0.7905694150420949*surft1_up[2]*dv1_sq-0.7905694150420949*surft1_lo[2]*dv1_sq; 
  out[35] += 0.7905694150420949*surft1_up[3]*dv1_sq-0.7905694150420949*surft1_lo[3]*dv1_sq-1.3693063937629153*surft2_up[0]*dv1_sq+1.3693063937629153*surft2_lo[0]*dv1_sq; 
  out[36] += 0.7905694150420948*surft1_up[4]*dv1_sq-0.7905694150420948*surft1_lo[4]*dv1_sq; 
  out[37] += 0.7905694150420948*surft1_up[5]*dv1_sq-0.7905694150420948*surft1_lo[5]*dv1_sq-1.3693063937629153*surft2_up[1]*dv1_sq+1.3693063937629153*surft2_lo[1]*dv1_sq; 
  out[38] += 0.7905694150420948*surft1_up[6]*dv1_sq-0.7905694150420948*surft1_lo[6]*dv1_sq-1.3693063937629153*surft2_up[2]*dv1_sq+1.3693063937629153*surft2_lo[2]*dv1_sq; 
  out[39] += 0.7905694150420949*surft1_up[7]*dv1_sq-0.7905694150420949*surft1_lo[7]*dv1_sq-1.3693063937629153*surft2_up[4]*dv1_sq+1.3693063937629153*surft2_lo[4]*dv1_sq; 
} 