#include <gkyl_vlasov_sr_kernels.h> 
GKYL_CU_DH double vlasov_sr_vmap_stream_vol_1x2v_ser_p2(const double *w, const double *dxv, const double *jacob_vel_inv, const double *gamma, const double *qmem, const double *f, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:             Cell-center coordinates.
  // dxv[NDIM]:           Cell spacing.
  // jacob_vel_inv[VDIM]: Inverse velocity space Jacobian in each direction.
  // gamma:               Particle Lorentz boost factor sqrt(1 + p^2).
  // qmem:                q/m*EM fields (unused in streaming-only update).
  // f:                   Input distribution function.
  // out:                 Incremented output.
  const double dx10 = 2.0/dxv[0]; 
  const double dv10 = 2.0/dxv[1]; 
  const double *jacob_vel_inv0 = &jacob_vel_inv[0]; 
  double p0_over_gamma[8] = {0.0}; 
  p0_over_gamma[0] = 2.738612787525831*jacob_vel_inv0[1]*gamma[4]*dv10+1.224744871391589*jacob_vel_inv0[0]*gamma[1]*dv10; 
  p0_over_gamma[1] = 2.449489742783178*jacob_vel_inv0[2]*gamma[4]*dv10+2.738612787525831*jacob_vel_inv0[0]*gamma[4]*dv10+1.224744871391589*jacob_vel_inv0[1]*gamma[1]*dv10; 
  p0_over_gamma[2] = 2.738612787525831*jacob_vel_inv0[1]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[0]*gamma[3]*dv10; 
  p0_over_gamma[3] = 2.449489742783178*jacob_vel_inv0[2]*gamma[6]*dv10+2.738612787525831*jacob_vel_inv0[0]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[1]*gamma[3]*dv10; 
  p0_over_gamma[4] = 2.449489742783178*jacob_vel_inv0[1]*gamma[4]*dv10+1.224744871391589*gamma[1]*jacob_vel_inv0[2]*dv10; 
  p0_over_gamma[5] = 1.224744871391589*jacob_vel_inv0[0]*gamma[7]*dv10; 
  p0_over_gamma[6] = 2.449489742783178*jacob_vel_inv0[1]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[2]*gamma[3]*dv10; 
  p0_over_gamma[7] = 1.224744871391589*jacob_vel_inv0[1]*gamma[7]*dv10; 

  double cflFreq_mid = 0.0; 
  double alpha_cdim[20] = {0.0}; 
  alpha_cdim[0] = 1.414213562373095*p0_over_gamma[0]*dx10; 
  alpha_cdim[2] = 1.414213562373095*p0_over_gamma[1]*dx10; 
  alpha_cdim[3] = 1.414213562373095*p0_over_gamma[2]*dx10; 
  alpha_cdim[6] = 1.414213562373095*p0_over_gamma[3]*dx10; 
  alpha_cdim[8] = 1.414213562373095*p0_over_gamma[4]*dx10; 
  alpha_cdim[9] = 1.414213562373095*p0_over_gamma[5]*dx10; 
  alpha_cdim[14] = 1.414213562373095*p0_over_gamma[6]*dx10; 
  alpha_cdim[16] = 1.414213562373095*p0_over_gamma[7]*dx10; 
  cflFreq_mid += fabs(0.1767766952966368*alpha_cdim[0]-0.1976423537605236*(alpha_cdim[9]+alpha_cdim[8])); 

  out[1] += 0.6123724356957944*(alpha_cdim[16]*f[16]+alpha_cdim[14]*f[14]+alpha_cdim[9]*f[9]+alpha_cdim[8]*f[8]+alpha_cdim[6]*f[6]+alpha_cdim[3]*f[3]+alpha_cdim[2]*f[2]+alpha_cdim[0]*f[0]); 
  out[4] += 0.6123724356957944*(alpha_cdim[9]*f[16]+f[9]*alpha_cdim[16])+0.5477225575051661*(alpha_cdim[6]*f[14]+f[6]*alpha_cdim[14]+alpha_cdim[2]*f[8]+f[2]*alpha_cdim[8])+0.6123724356957944*(alpha_cdim[3]*f[6]+f[3]*alpha_cdim[6]+alpha_cdim[0]*f[2]+f[0]*alpha_cdim[2]); 
  out[5] += 0.5477225575051661*(alpha_cdim[6]*f[16]+f[6]*alpha_cdim[16])+0.6123724356957944*(alpha_cdim[8]*f[14]+f[8]*alpha_cdim[14])+0.5477225575051661*(alpha_cdim[3]*f[9]+f[3]*alpha_cdim[9])+0.6123724356957944*(alpha_cdim[2]*f[6]+f[2]*alpha_cdim[6]+alpha_cdim[0]*f[3]+f[0]*alpha_cdim[3]); 
  out[7] += 1.369306393762915*(alpha_cdim[16]*f[19]+alpha_cdim[14]*f[18]+alpha_cdim[9]*f[15]+alpha_cdim[8]*f[12]+alpha_cdim[6]*f[10]+alpha_cdim[3]*f[5]+alpha_cdim[2]*f[4]+alpha_cdim[0]*f[1]); 
  out[10] += (0.4898979485566357*alpha_cdim[14]+0.5477225575051661*alpha_cdim[3])*f[16]+0.4898979485566357*f[14]*alpha_cdim[16]+0.5477225575051661*(f[3]*alpha_cdim[16]+alpha_cdim[2]*f[14]+f[2]*alpha_cdim[14]+alpha_cdim[6]*f[9]+f[6]*alpha_cdim[9]+alpha_cdim[6]*f[8]+f[6]*alpha_cdim[8])+0.6123724356957944*(alpha_cdim[0]*f[6]+f[0]*alpha_cdim[6]+alpha_cdim[2]*f[3]+f[2]*alpha_cdim[3]); 
  out[11] += 1.369306393762915*alpha_cdim[9]*f[19]+1.224744871391589*alpha_cdim[6]*f[18]+1.369306393762915*f[15]*alpha_cdim[16]+1.224744871391589*(f[10]*alpha_cdim[14]+alpha_cdim[2]*f[12])+1.369306393762915*alpha_cdim[3]*f[10]+1.224744871391589*f[4]*alpha_cdim[8]+1.369306393762915*(f[5]*alpha_cdim[6]+alpha_cdim[0]*f[4]+f[1]*alpha_cdim[2]); 
  out[12] += 0.5477225575051661*alpha_cdim[16]*f[16]+0.3912303982179757*alpha_cdim[14]*f[14]+0.6123724356957944*(alpha_cdim[3]*f[14]+f[3]*alpha_cdim[14])+0.3912303982179757*alpha_cdim[8]*f[8]+0.6123724356957944*(alpha_cdim[0]*f[8]+f[0]*alpha_cdim[8])+0.5477225575051661*(alpha_cdim[6]*f[6]+alpha_cdim[2]*f[2]); 
  out[13] += 1.224744871391589*alpha_cdim[6]*f[19]+1.369306393762915*alpha_cdim[8]*f[18]+1.224744871391589*(f[10]*alpha_cdim[16]+alpha_cdim[3]*f[15])+1.369306393762915*(f[12]*alpha_cdim[14]+alpha_cdim[2]*f[10])+1.224744871391589*f[5]*alpha_cdim[9]+1.369306393762915*(f[4]*alpha_cdim[6]+alpha_cdim[0]*f[5]+f[1]*alpha_cdim[3]); 
  out[15] += 0.3912303982179757*alpha_cdim[16]*f[16]+0.6123724356957944*(alpha_cdim[2]*f[16]+f[2]*alpha_cdim[16])+0.5477225575051661*alpha_cdim[14]*f[14]+0.3912303982179757*alpha_cdim[9]*f[9]+0.6123724356957944*(alpha_cdim[0]*f[9]+f[0]*alpha_cdim[9])+0.5477225575051661*(alpha_cdim[6]*f[6]+alpha_cdim[3]*f[3]); 
  out[17] += (1.095445115010332*alpha_cdim[14]+1.224744871391589*alpha_cdim[3])*f[19]+1.095445115010332*alpha_cdim[16]*f[18]+1.224744871391589*(alpha_cdim[2]*f[18]+f[5]*alpha_cdim[16]+alpha_cdim[6]*f[15]+f[4]*alpha_cdim[14]+alpha_cdim[6]*f[12]+(alpha_cdim[9]+alpha_cdim[8])*f[10])+1.369306393762915*(alpha_cdim[0]*f[10]+f[1]*alpha_cdim[6]+alpha_cdim[2]*f[5]+alpha_cdim[3]*f[4]); 
  out[18] += 0.4898979485566357*(alpha_cdim[6]*f[16]+f[6]*alpha_cdim[16])+(0.5477225575051661*alpha_cdim[9]+0.3912303982179757*alpha_cdim[8]+0.6123724356957944*alpha_cdim[0])*f[14]+(0.5477225575051661*f[9]+0.3912303982179757*f[8])*alpha_cdim[14]+0.6123724356957944*(f[0]*alpha_cdim[14]+alpha_cdim[3]*f[8]+f[3]*alpha_cdim[8])+0.5477225575051661*(alpha_cdim[2]*f[6]+f[2]*alpha_cdim[6]); 
  out[19] += (0.3912303982179757*alpha_cdim[9]+0.5477225575051661*alpha_cdim[8]+0.6123724356957944*alpha_cdim[0])*f[16]+(0.3912303982179757*f[9]+0.5477225575051661*f[8]+0.6123724356957944*f[0])*alpha_cdim[16]+0.4898979485566357*(alpha_cdim[6]*f[14]+f[6]*alpha_cdim[14])+0.6123724356957944*(alpha_cdim[2]*f[9]+f[2]*alpha_cdim[9])+0.5477225575051661*(alpha_cdim[3]*f[6]+f[3]*alpha_cdim[6]); 

  return 5.0*cflFreq_mid; 
} 