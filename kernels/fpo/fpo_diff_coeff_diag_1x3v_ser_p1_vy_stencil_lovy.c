#include <gkyl_fpo_vlasov_kernels.h> 

GKYL_CU_DH void fpo_diff_coeff_diag_1x3v_vy_ser_p1_lovy(const double *dxv, const double *gamma, const double* fpo_g_stencil[3], const double* fpo_d2gdv2_surf, double *diff_coeff) {
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[3]: 3 cell stencil of Rosenbluth potential G. 
  // fpo_d2gdv2_surf: Surface projection of d2G/dv2 in center cell. 
  // diff_coeff: Output array for diffusion tensor. 

  // Use cell-average value for gamma. 
 double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1_sq = 4.0/dxv[2]/dxv[2]; 

  const double* G_C = fpo_g_stencil[0]; 
  const double* G_R = fpo_g_stencil[1]; 
  
  const double *fpo_d2g_surf_C_vxvx = &fpo_d2gdv2_surf[0]; 
  const double *fpo_d2g_surf_C_vyvy = &fpo_d2gdv2_surf[32]; 
  const double *fpo_d2g_surf_C_vzvz = &fpo_d2gdv2_surf[64]; 
  
  const double* d2G_surf_C = fpo_d2g_surf_C_vyvy; 
  
  double *diff_coeff_vxvx = &diff_coeff[0]; 
  double *diff_coeff_vyvy = &diff_coeff[160]; 
  double *diff_coeff_vzvz = &diff_coeff[320]; 
  
  double *out = diff_coeff_vyvy; 
  
  out[0] = -(1.4433756729740645*G_R[3]*dv1_sq*gamma_avg)-2.886751345948129*G_C[3]*dv1_sq*gamma_avg+1.25*G_R[0]*dv1_sq*gamma_avg-1.25*G_C[0]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[0]*gamma_avg; 
  out[1] = -(1.4433756729740645*G_R[6]*dv1_sq*gamma_avg)-2.886751345948129*G_C[6]*dv1_sq*gamma_avg+1.25*G_R[1]*dv1_sq*gamma_avg-1.25*G_C[1]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[1]*gamma_avg; 
  out[2] = -(1.4433756729740645*G_R[7]*dv1_sq*gamma_avg)-2.886751345948129*G_C[7]*dv1_sq*gamma_avg+1.25*G_R[2]*dv1_sq*gamma_avg-1.25*G_C[2]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[2]*gamma_avg; 
  out[3] = 0.5208333333333334*G_R[3]*dv1_sq*gamma_avg+0.10416666666666667*G_C[3]*dv1_sq*gamma_avg-0.18042195912175807*G_R[0]*dv1_sq*gamma_avg+0.18042195912175807*G_C[0]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[0]*gamma_avg; 
  out[4] = -(1.4433756729740645*G_R[10]*dv1_sq*gamma_avg)-2.886751345948129*G_C[10]*dv1_sq*gamma_avg+1.25*G_R[4]*dv1_sq*gamma_avg-1.25*G_C[4]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[3]*gamma_avg; 
  out[5] = -(1.4433756729740645*G_R[11]*dv1_sq*gamma_avg)-2.886751345948129*G_C[11]*dv1_sq*gamma_avg+1.25*G_R[5]*dv1_sq*gamma_avg-1.25*G_C[5]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[4]*gamma_avg; 
  out[6] = 0.5208333333333334*G_R[6]*dv1_sq*gamma_avg+0.10416666666666667*G_C[6]*dv1_sq*gamma_avg-0.18042195912175807*G_R[1]*dv1_sq*gamma_avg+0.18042195912175807*G_C[1]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[1]*gamma_avg; 
  out[7] = 0.5208333333333334*G_R[7]*dv1_sq*gamma_avg+0.10416666666666667*G_C[7]*dv1_sq*gamma_avg-0.18042195912175807*G_R[2]*dv1_sq*gamma_avg+0.18042195912175807*G_C[2]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[2]*gamma_avg; 
  out[8] = -(1.4433756729740645*G_R[13]*dv1_sq*gamma_avg)-2.886751345948129*G_C[13]*dv1_sq*gamma_avg+1.25*G_R[8]*dv1_sq*gamma_avg-1.25*G_C[8]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[5]*gamma_avg; 
  out[9] = -(1.4433756729740645*G_R[14]*dv1_sq*gamma_avg)-2.886751345948129*G_C[14]*dv1_sq*gamma_avg+1.25*G_R[9]*dv1_sq*gamma_avg-1.25*G_C[9]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[6]*gamma_avg; 
  out[10] = 0.5208333333333334*G_R[10]*dv1_sq*gamma_avg+0.10416666666666667*G_C[10]*dv1_sq*gamma_avg-0.18042195912175807*G_R[4]*dv1_sq*gamma_avg+0.18042195912175807*G_C[4]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[3]*gamma_avg; 
  out[11] = 0.5208333333333334*G_R[11]*dv1_sq*gamma_avg+0.10416666666666667*G_C[11]*dv1_sq*gamma_avg-0.18042195912175807*G_R[5]*dv1_sq*gamma_avg+0.18042195912175807*G_C[5]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[4]*gamma_avg; 
  out[12] = -(1.4433756729740645*G_R[15]*dv1_sq*gamma_avg)-2.886751345948129*G_C[15]*dv1_sq*gamma_avg+1.25*G_R[12]*dv1_sq*gamma_avg-1.25*G_C[12]*dv1_sq*gamma_avg+0.2357022603955158*d2G_surf_C[7]*gamma_avg; 
  out[13] = 0.5208333333333334*G_R[13]*dv1_sq*gamma_avg+0.10416666666666667*G_C[13]*dv1_sq*gamma_avg-0.18042195912175807*G_R[8]*dv1_sq*gamma_avg+0.18042195912175807*G_C[8]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[5]*gamma_avg; 
  out[14] = 0.5208333333333334*G_R[14]*dv1_sq*gamma_avg+0.10416666666666667*G_C[14]*dv1_sq*gamma_avg-0.18042195912175807*G_R[9]*dv1_sq*gamma_avg+0.18042195912175807*G_C[9]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[6]*gamma_avg; 
  out[15] = 0.5208333333333334*G_R[15]*dv1_sq*gamma_avg+0.10416666666666667*G_C[15]*dv1_sq*gamma_avg-0.18042195912175807*G_R[12]*dv1_sq*gamma_avg+0.18042195912175807*G_C[12]*dv1_sq*gamma_avg-0.34020690871988585*d2G_surf_C[7]*gamma_avg; 
  out[24] = 1.0489329895978423*G_R[3]*dv1_sq*gamma_avg+1.3716816017817939*G_C[3]*dv1_sq*gamma_avg-0.6987712429686844*G_R[0]*dv1_sq*gamma_avg+0.6987712429686844*G_C[0]*dv1_sq*gamma_avg+0.2635231383473649*d2G_surf_C[0]*gamma_avg; 
  out[25] = 1.048932989597842*G_R[6]*dv1_sq*gamma_avg+1.3716816017817937*G_C[6]*dv1_sq*gamma_avg-0.6987712429686844*G_R[1]*dv1_sq*gamma_avg+0.6987712429686844*G_C[1]*dv1_sq*gamma_avg+0.26352313834736496*d2G_surf_C[1]*gamma_avg; 
  out[26] = 1.048932989597842*G_R[7]*dv1_sq*gamma_avg+1.3716816017817937*G_C[7]*dv1_sq*gamma_avg-0.6987712429686844*G_R[2]*dv1_sq*gamma_avg+0.6987712429686844*G_C[2]*dv1_sq*gamma_avg+0.26352313834736496*d2G_surf_C[2]*gamma_avg; 
  out[27] = 1.048932989597842*G_R[10]*dv1_sq*gamma_avg+1.3716816017817937*G_C[10]*dv1_sq*gamma_avg-0.6987712429686844*G_R[4]*dv1_sq*gamma_avg+0.6987712429686844*G_C[4]*dv1_sq*gamma_avg+0.26352313834736496*d2G_surf_C[3]*gamma_avg; 
  out[28] = 1.0489329895978423*G_R[11]*dv1_sq*gamma_avg+1.3716816017817939*G_C[11]*dv1_sq*gamma_avg-0.6987712429686844*G_R[5]*dv1_sq*gamma_avg+0.6987712429686844*G_C[5]*dv1_sq*gamma_avg+0.2635231383473649*d2G_surf_C[4]*gamma_avg; 
  out[29] = 1.0489329895978423*G_R[13]*dv1_sq*gamma_avg+1.3716816017817939*G_C[13]*dv1_sq*gamma_avg-0.6987712429686844*G_R[8]*dv1_sq*gamma_avg+0.6987712429686844*G_C[8]*dv1_sq*gamma_avg+0.2635231383473649*d2G_surf_C[5]*gamma_avg; 
  out[30] = 1.0489329895978423*G_R[14]*dv1_sq*gamma_avg+1.3716816017817939*G_C[14]*dv1_sq*gamma_avg-0.6987712429686844*G_R[9]*dv1_sq*gamma_avg+0.6987712429686844*G_C[9]*dv1_sq*gamma_avg+0.2635231383473649*d2G_surf_C[6]*gamma_avg; 
  out[31] = 1.048932989597842*G_R[15]*dv1_sq*gamma_avg+1.3716816017817937*G_C[15]*dv1_sq*gamma_avg-0.6987712429686844*G_R[12]*dv1_sq*gamma_avg+0.6987712429686844*G_C[12]*dv1_sq*gamma_avg+0.26352313834736496*d2G_surf_C[7]*gamma_avg; 
} 

