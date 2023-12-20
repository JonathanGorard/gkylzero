#include <gkyl_vlasov_poisson_kernels.h> 

GKYL_CU_DH double vlasov_poisson_nonuniformv_vol_1x3v_ser_p1(const double *w, const double *dxv, const double *vcoord, const double *field, const double *f, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:   Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // vcoord     Discrete (DG) velocity coordinate.
  // field:     potential (scaled by appropriate factors).
  // f:         Input distribution function.
  // out:       Incremented output.
  double rdx = 1./dxv[0];

  const double rdx2 = 2.*rdx; 
  const double *phi = &field[0]; 
  const double rdvx2 = 2./dxv[1]; 
  const double rdvy2 = 2./dxv[2]; 
  const double rdvz2 = 2./dxv[3]; 

  double cflFreq = 0.0; 
  double alpha_cdim[40]; 
  double alpha_vdim[120]; 

  alpha_cdim[0] = 2.828427124746191*vcoord[0]*rdx; 
  alpha_cdim[2] = 2.828427124746191*vcoord[1]*rdx; 
  alpha_cdim[16] = 2.828427124746191*vcoord[7]*rdx; 
  cflFreq += 3.0*rdx*fmax(fabs(0.7905694150420947*vcoord[7]-0.6123724356957944*vcoord[1]+0.3535533905932737*vcoord[0]),fabs(0.7905694150420947*vcoord[7]+0.6123724356957944*vcoord[1]+0.3535533905932737*vcoord[0])); 

  alpha_vdim[0] = -4.898979485566357*phi[1]*rdvx2*rdx2; 
  cflFreq += 5.0*fabs(0.125*alpha_vdim[0]); 

  cflFreq += 5.0*fabs(0.0); 

  cflFreq += 5.0*fabs(0.0); 

  out[1] += 0.4330127018922193*(alpha_cdim[16]*f[16]+alpha_cdim[2]*f[2]+alpha_cdim[0]*f[0]); 
  out[2] += 0.4330127018922193*alpha_vdim[0]*f[0]; 
  out[5] += 0.3872983346207416*(alpha_cdim[2]*f[16]+f[2]*alpha_cdim[16])+0.4330127018922193*(alpha_cdim[0]*f[2]+f[0]*alpha_cdim[2]+alpha_vdim[0]*f[1]); 
  out[6] += 0.4330127018922193*(alpha_cdim[16]*f[18]+alpha_cdim[2]*f[7]+alpha_cdim[0]*f[3]); 
  out[7] += 0.4330127018922193*alpha_vdim[0]*f[3]; 
  out[8] += 0.4330127018922193*(alpha_cdim[16]*f[19]+alpha_cdim[2]*f[9]+alpha_cdim[0]*f[4]); 
  out[9] += 0.4330127018922193*alpha_vdim[0]*f[4]; 
  out[11] += 0.3872983346207416*(alpha_cdim[2]*f[18]+f[7]*alpha_cdim[16])+0.4330127018922193*(alpha_cdim[0]*f[7]+alpha_vdim[0]*f[6]+alpha_cdim[2]*f[3]); 
  out[12] += 0.3872983346207416*(alpha_cdim[2]*f[19]+f[9]*alpha_cdim[16])+0.4330127018922193*(alpha_cdim[0]*f[9]+alpha_vdim[0]*f[8]+alpha_cdim[2]*f[4]); 
  out[13] += 0.4330127018922193*(alpha_cdim[16]*f[22]+alpha_cdim[2]*f[14]+alpha_cdim[0]*f[10]); 
  out[14] += 0.4330127018922193*alpha_vdim[0]*f[10]; 
  out[15] += 0.3872983346207416*(alpha_cdim[2]*f[22]+f[14]*alpha_cdim[16])+0.4330127018922193*(alpha_cdim[0]*f[14]+alpha_vdim[0]*f[13]+alpha_cdim[2]*f[10]); 
  out[16] += 0.9682458365518543*alpha_vdim[0]*f[2]; 
  out[17] += 0.276641667586244*alpha_cdim[16]*f[16]+0.4330127018922193*(alpha_cdim[0]*f[16]+f[0]*alpha_cdim[16])+0.9682458365518543*alpha_vdim[0]*f[5]+0.3872983346207416*alpha_cdim[2]*f[2]; 
  out[18] += 0.9682458365518543*alpha_vdim[0]*f[7]; 
  out[19] += 0.9682458365518543*alpha_vdim[0]*f[9]; 
  out[20] += 0.276641667586244*alpha_cdim[16]*f[18]+0.4330127018922193*(alpha_cdim[0]*f[18]+f[3]*alpha_cdim[16])+0.9682458365518543*alpha_vdim[0]*f[11]+0.3872983346207416*alpha_cdim[2]*f[7]; 
  out[21] += 0.276641667586244*alpha_cdim[16]*f[19]+0.4330127018922193*(alpha_cdim[0]*f[19]+f[4]*alpha_cdim[16])+0.9682458365518543*alpha_vdim[0]*f[12]+0.3872983346207416*alpha_cdim[2]*f[9]; 
  out[22] += 0.9682458365518543*alpha_vdim[0]*f[14]; 
  out[23] += 0.276641667586244*alpha_cdim[16]*f[22]+0.4330127018922193*(alpha_cdim[0]*f[22]+f[10]*alpha_cdim[16])+0.9682458365518543*alpha_vdim[0]*f[15]+0.3872983346207416*alpha_cdim[2]*f[14]; 
  out[25] += 0.4330127018922193*(alpha_cdim[2]*f[26]+alpha_cdim[0]*f[24]); 
  out[26] += 0.4330127018922193*alpha_vdim[0]*f[24]; 
  out[28] += 0.3872983346207416*alpha_cdim[16]*f[26]+0.4330127018922193*(alpha_cdim[0]*f[26]+alpha_vdim[0]*f[25]+alpha_cdim[2]*f[24]); 
  out[29] += 0.4330127018922193*(alpha_cdim[2]*f[30]+alpha_cdim[0]*f[27]); 
  out[30] += 0.4330127018922193*alpha_vdim[0]*f[27]; 
  out[31] += 0.3872983346207416*alpha_cdim[16]*f[30]+0.4330127018922193*(alpha_cdim[0]*f[30]+alpha_vdim[0]*f[29]+alpha_cdim[2]*f[27]); 
  out[33] += 0.4330127018922193*(alpha_cdim[2]*f[34]+alpha_cdim[0]*f[32]); 
  out[34] += 0.4330127018922193*alpha_vdim[0]*f[32]; 
  out[36] += 0.3872983346207416*alpha_cdim[16]*f[34]+0.4330127018922193*(alpha_cdim[0]*f[34]+alpha_vdim[0]*f[33]+alpha_cdim[2]*f[32]); 
  out[37] += 0.4330127018922193*(alpha_cdim[2]*f[38]+alpha_cdim[0]*f[35]); 
  out[38] += 0.4330127018922193*alpha_vdim[0]*f[35]; 
  out[39] += 0.3872983346207416*alpha_cdim[16]*f[38]+0.4330127018922193*(alpha_cdim[0]*f[38]+alpha_vdim[0]*f[37]+alpha_cdim[2]*f[35]); 

  return cflFreq; 
} 


GKYL_CU_DH double vlasov_poisson_extem_nonuniformv_vol_1x3v_ser_p1(const double *w, const double *dxv, const double *vcoord, const double *field, const double *f, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:   Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // vcoord     Discrete (DG) velocity coordinate.
  // field:     potentials, including external (scaled by appropriate factors).
  // f:         Input distribution function.
  // out:       Incremented output.
  double rdx = 1.0/dxv[0];

  double dv0dx0 = dxv[1]/dxv[0]; 
  double w0dx0 = w[1]/dxv[0]; 
  const double rdx2 = 2.*rdx; 
  const double *phi = &field[0]; 
  const double rdvx2 = 2./dxv[1]; 
  const double rdvy2 = 2./dxv[2]; 
  const double rdvz2 = 2./dxv[3]; 

  const double *A0 = &field[2]; 
  const double *A1 = &field[4]; 
  const double *A2 = &field[6]; 
  double cflFreq = 0.0; 
  double alpha_cdim[40]; 
  double alpha_vdim[120]; 

  alpha_cdim[0] = 2.828427124746191*vcoord[0]*rdx; 
  alpha_cdim[2] = 2.828427124746191*vcoord[1]*rdx; 
  alpha_cdim[16] = 2.828427124746191*vcoord[7]*rdx; 
  cflFreq += 3.0*rdx*fmax(fabs(0.7905694150420947*vcoord[7]-0.6123724356957944*vcoord[1]+0.3535533905932737*vcoord[0]),fabs(0.7905694150420947*vcoord[7]+0.6123724356957944*vcoord[1]+0.3535533905932737*vcoord[0])); 

  alpha_vdim[0] = (1.732050807568877*(A2[1]*vcoord[40]+A1[1]*vcoord[20])-4.898979485566357*phi[1])*rdvx2*rdx2; 
  alpha_vdim[3] = 1.732050807568877*A1[1]*vcoord[22]*rdvx2*rdx2; 
  alpha_vdim[4] = 1.732050807568877*A2[1]*vcoord[43]*rdvx2*rdx2; 
  alpha_vdim[24] = 1.732050807568877*A1[1]*vcoord[28]*rdvx2*rdx2; 
  alpha_vdim[32] = 1.732050807568877*A2[1]*vcoord[49]*rdvx2*rdx2; 
  cflFreq += 5.0*fabs(0.125*alpha_vdim[0]-0.1397542485937369*(alpha_vdim[32]+alpha_vdim[24])); 

  alpha_vdim[40] = -1.732050807568877*vcoord[0]*A1[1]*rdvy2*rdx2; 
  alpha_vdim[42] = -1.732050807568877*A1[1]*vcoord[1]*rdvy2*rdx2; 
  alpha_vdim[56] = -1.732050807568877*A1[1]*vcoord[7]*rdvy2*rdx2; 
  cflFreq += 5.0*fabs(0.125*alpha_vdim[40]-0.1397542485937369*alpha_vdim[56]); 

  alpha_vdim[80] = -1.732050807568877*vcoord[0]*A2[1]*rdvz2*rdx2; 
  alpha_vdim[82] = -1.732050807568877*A2[1]*vcoord[1]*rdvz2*rdx2; 
  alpha_vdim[96] = -1.732050807568877*A2[1]*vcoord[7]*rdvz2*rdx2; 
  cflFreq += 5.0*fabs(0.125*alpha_vdim[80]-0.1397542485937369*alpha_vdim[96]); 

  out[1] += 0.4330127018922193*(alpha_cdim[16]*f[16]+alpha_cdim[2]*f[2]+alpha_cdim[0]*f[0]); 
  out[2] += 0.4330127018922193*(alpha_vdim[32]*f[32]+alpha_vdim[24]*f[24]+alpha_vdim[4]*f[4]+alpha_vdim[3]*f[3]+alpha_vdim[0]*f[0]); 
  out[3] += 0.4330127018922193*(f[16]*alpha_vdim[56]+f[2]*alpha_vdim[42]+f[0]*alpha_vdim[40]); 
  out[4] += 0.4330127018922193*(f[16]*alpha_vdim[96]+f[2]*alpha_vdim[82]+f[0]*alpha_vdim[80]); 
  out[5] += 0.4330127018922193*(alpha_vdim[32]*f[33]+alpha_vdim[24]*f[25])+0.3872983346207416*(alpha_cdim[2]*f[16]+f[2]*alpha_cdim[16])+0.4330127018922193*(alpha_vdim[4]*f[8]+alpha_vdim[3]*f[6]+alpha_cdim[0]*f[2]+f[0]*alpha_cdim[2]+alpha_vdim[0]*f[1]); 
  out[6] += 0.4330127018922193*(f[17]*alpha_vdim[56]+f[5]*alpha_vdim[42]+f[1]*alpha_vdim[40]+alpha_cdim[16]*f[18]+alpha_cdim[2]*f[7]+alpha_cdim[0]*f[3]); 
  out[7] += 0.3872983346207416*(f[2]*alpha_vdim[56]+f[16]*alpha_vdim[42])+0.4330127018922193*(f[0]*alpha_vdim[42]+f[2]*alpha_vdim[40]+alpha_vdim[32]*f[35])+0.3872983346207416*(alpha_vdim[3]*f[24]+f[3]*alpha_vdim[24])+0.4330127018922193*(alpha_vdim[4]*f[10]+alpha_vdim[0]*f[3]+f[0]*alpha_vdim[3]); 
  out[8] += 0.4330127018922193*(f[17]*alpha_vdim[96]+f[5]*alpha_vdim[82]+f[1]*alpha_vdim[80]+alpha_cdim[16]*f[19]+alpha_cdim[2]*f[9]+alpha_cdim[0]*f[4]); 
  out[9] += 0.3872983346207416*(f[2]*alpha_vdim[96]+f[16]*alpha_vdim[82])+0.4330127018922193*(f[0]*alpha_vdim[82]+f[2]*alpha_vdim[80])+0.3872983346207416*(alpha_vdim[4]*f[32]+f[4]*alpha_vdim[32])+0.4330127018922193*(alpha_vdim[24]*f[27]+alpha_vdim[3]*f[10]+alpha_vdim[0]*f[4]+f[0]*alpha_vdim[4]); 
  out[10] += 0.4330127018922193*(f[18]*alpha_vdim[96]+f[7]*alpha_vdim[82]+f[3]*alpha_vdim[80]+f[19]*alpha_vdim[56]+f[9]*alpha_vdim[42]+f[4]*alpha_vdim[40]); 
  out[11] += 0.3872983346207416*(f[5]*alpha_vdim[56]+f[17]*alpha_vdim[42])+0.4330127018922193*(f[1]*alpha_vdim[42]+f[5]*alpha_vdim[40]+alpha_vdim[32]*f[37])+0.3872983346207416*(alpha_vdim[3]*f[25]+f[6]*alpha_vdim[24]+alpha_cdim[2]*f[18]+f[7]*alpha_cdim[16])+0.4330127018922193*(alpha_vdim[4]*f[13]+alpha_cdim[0]*f[7]+alpha_vdim[0]*f[6]+alpha_cdim[2]*f[3]+f[1]*alpha_vdim[3]); 
  out[12] += 0.3872983346207416*(f[5]*alpha_vdim[96]+f[17]*alpha_vdim[82])+0.4330127018922193*(f[1]*alpha_vdim[82]+f[5]*alpha_vdim[80])+0.3872983346207416*(alpha_vdim[4]*f[33]+f[8]*alpha_vdim[32])+0.4330127018922193*alpha_vdim[24]*f[29]+0.3872983346207416*(alpha_cdim[2]*f[19]+f[9]*alpha_cdim[16])+0.4330127018922193*(alpha_vdim[3]*f[13]+alpha_cdim[0]*f[9]+alpha_vdim[0]*f[8]+alpha_cdim[2]*f[4]+f[1]*alpha_vdim[4]); 
  out[13] += 0.4330127018922193*(f[20]*alpha_vdim[96]+f[11]*alpha_vdim[82]+f[6]*alpha_vdim[80]+f[21]*alpha_vdim[56]+f[12]*alpha_vdim[42]+f[8]*alpha_vdim[40]+alpha_cdim[16]*f[22]+alpha_cdim[2]*f[14]+alpha_cdim[0]*f[10]); 
  out[14] += 0.3872983346207416*(f[7]*alpha_vdim[96]+f[18]*alpha_vdim[82])+0.4330127018922193*(f[3]*alpha_vdim[82]+f[7]*alpha_vdim[80])+0.3872983346207416*(f[9]*alpha_vdim[56]+f[19]*alpha_vdim[42])+0.4330127018922193*(f[4]*alpha_vdim[42]+f[9]*alpha_vdim[40])+0.3872983346207416*(alpha_vdim[4]*f[35]+f[10]*alpha_vdim[32]+alpha_vdim[3]*f[27]+f[10]*alpha_vdim[24])+0.4330127018922193*(alpha_vdim[0]*f[10]+alpha_vdim[3]*f[4]+f[3]*alpha_vdim[4]); 
  out[15] += 0.3872983346207416*(f[11]*alpha_vdim[96]+f[20]*alpha_vdim[82])+0.4330127018922193*(f[6]*alpha_vdim[82]+f[11]*alpha_vdim[80])+0.3872983346207416*(f[12]*alpha_vdim[56]+f[21]*alpha_vdim[42])+0.4330127018922193*(f[8]*alpha_vdim[42]+f[12]*alpha_vdim[40])+0.3872983346207416*(alpha_vdim[4]*f[37]+f[13]*alpha_vdim[32]+alpha_vdim[3]*f[29]+f[13]*alpha_vdim[24]+alpha_cdim[2]*f[22]+f[14]*alpha_cdim[16])+0.4330127018922193*(alpha_cdim[0]*f[14]+alpha_vdim[0]*f[13]+alpha_cdim[2]*f[10]+alpha_vdim[3]*f[8]+alpha_vdim[4]*f[6]); 
  out[16] += 0.9682458365518543*(alpha_vdim[32]*f[34]+alpha_vdim[24]*f[26]+alpha_vdim[4]*f[9]+alpha_vdim[3]*f[7]+alpha_vdim[0]*f[2]); 
  out[17] += 0.9682458365518543*(alpha_vdim[32]*f[36]+alpha_vdim[24]*f[28])+0.276641667586244*alpha_cdim[16]*f[16]+0.4330127018922193*(alpha_cdim[0]*f[16]+f[0]*alpha_cdim[16])+0.9682458365518543*(alpha_vdim[4]*f[12]+alpha_vdim[3]*f[11]+alpha_vdim[0]*f[5])+0.3872983346207416*alpha_cdim[2]*f[2]; 
  out[18] += (0.276641667586244*f[16]+0.4330127018922193*f[0])*alpha_vdim[56]+0.3872983346207416*f[2]*alpha_vdim[42]+0.4330127018922193*f[16]*alpha_vdim[40]+0.9682458365518543*alpha_vdim[32]*f[38]+0.8660254037844386*(alpha_vdim[3]*f[26]+f[7]*alpha_vdim[24])+0.9682458365518543*(alpha_vdim[4]*f[14]+alpha_vdim[0]*f[7]+f[2]*alpha_vdim[3]); 
  out[19] += (0.276641667586244*f[16]+0.4330127018922193*f[0])*alpha_vdim[96]+0.3872983346207416*f[2]*alpha_vdim[82]+0.4330127018922193*f[16]*alpha_vdim[80]+0.8660254037844386*(alpha_vdim[4]*f[34]+f[9]*alpha_vdim[32])+0.9682458365518543*(alpha_vdim[24]*f[30]+alpha_vdim[3]*f[14]+alpha_vdim[0]*f[9]+f[2]*alpha_vdim[4]); 
  out[20] += (0.276641667586244*f[17]+0.4330127018922193*f[1])*alpha_vdim[56]+0.3872983346207416*f[5]*alpha_vdim[42]+0.4330127018922193*f[17]*alpha_vdim[40]+0.9682458365518543*alpha_vdim[32]*f[39]+0.8660254037844386*(alpha_vdim[3]*f[28]+f[11]*alpha_vdim[24])+0.276641667586244*alpha_cdim[16]*f[18]+0.4330127018922193*(alpha_cdim[0]*f[18]+f[3]*alpha_cdim[16])+0.9682458365518543*(alpha_vdim[4]*f[15]+alpha_vdim[0]*f[11])+0.3872983346207416*alpha_cdim[2]*f[7]+0.9682458365518543*alpha_vdim[3]*f[5]; 
  out[21] += (0.276641667586244*f[17]+0.4330127018922193*f[1])*alpha_vdim[96]+0.3872983346207416*f[5]*alpha_vdim[82]+0.4330127018922193*f[17]*alpha_vdim[80]+0.8660254037844386*(alpha_vdim[4]*f[36]+f[12]*alpha_vdim[32])+0.9682458365518543*alpha_vdim[24]*f[31]+0.276641667586244*alpha_cdim[16]*f[19]+0.4330127018922193*(alpha_cdim[0]*f[19]+f[4]*alpha_cdim[16])+0.9682458365518543*(alpha_vdim[3]*f[15]+alpha_vdim[0]*f[12])+0.3872983346207416*alpha_cdim[2]*f[9]+0.9682458365518543*alpha_vdim[4]*f[5]; 
  out[22] += (0.276641667586244*f[18]+0.4330127018922193*f[3])*alpha_vdim[96]+0.3872983346207416*f[7]*alpha_vdim[82]+0.4330127018922193*f[18]*alpha_vdim[80]+(0.276641667586244*f[19]+0.4330127018922193*f[4])*alpha_vdim[56]+0.3872983346207416*f[9]*alpha_vdim[42]+0.4330127018922193*f[19]*alpha_vdim[40]+0.8660254037844386*(alpha_vdim[4]*f[38]+f[14]*alpha_vdim[32]+alpha_vdim[3]*f[30]+f[14]*alpha_vdim[24])+0.9682458365518543*(alpha_vdim[0]*f[14]+alpha_vdim[3]*f[9]+alpha_vdim[4]*f[7]); 
  out[23] += (0.276641667586244*f[20]+0.4330127018922193*f[6])*alpha_vdim[96]+0.3872983346207416*f[11]*alpha_vdim[82]+0.4330127018922193*f[20]*alpha_vdim[80]+(0.276641667586244*f[21]+0.4330127018922193*f[8])*alpha_vdim[56]+0.3872983346207416*f[12]*alpha_vdim[42]+0.4330127018922193*f[21]*alpha_vdim[40]+0.8660254037844386*(alpha_vdim[4]*f[39]+f[15]*alpha_vdim[32]+alpha_vdim[3]*f[31]+f[15]*alpha_vdim[24])+0.276641667586244*alpha_cdim[16]*f[22]+0.4330127018922193*(alpha_cdim[0]*f[22]+f[10]*alpha_cdim[16])+0.9682458365518543*alpha_vdim[0]*f[15]+0.3872983346207416*alpha_cdim[2]*f[14]+0.9682458365518543*(alpha_vdim[3]*f[12]+alpha_vdim[4]*f[11]); 
  out[24] += 0.9682458365518543*(f[18]*alpha_vdim[56]+f[7]*alpha_vdim[42]+f[3]*alpha_vdim[40]); 
  out[25] += 0.9682458365518543*(f[20]*alpha_vdim[56]+f[11]*alpha_vdim[42]+f[6]*alpha_vdim[40])+0.4330127018922193*(alpha_cdim[2]*f[26]+alpha_cdim[0]*f[24]); 
  out[26] += 0.8660254037844386*(f[7]*alpha_vdim[56]+f[18]*alpha_vdim[42])+0.9682458365518543*(f[3]*alpha_vdim[42]+f[7]*alpha_vdim[40])+0.4330127018922193*alpha_vdim[4]*f[27]+0.276641667586244*alpha_vdim[24]*f[24]+0.4330127018922193*(alpha_vdim[0]*f[24]+f[0]*alpha_vdim[24])+0.3872983346207416*alpha_vdim[3]*f[3]; 
  out[27] += 0.4330127018922193*(f[26]*alpha_vdim[82]+f[24]*alpha_vdim[80])+0.9682458365518543*(f[22]*alpha_vdim[56]+f[14]*alpha_vdim[42]+f[10]*alpha_vdim[40]); 
  out[28] += 0.8660254037844386*(f[11]*alpha_vdim[56]+f[20]*alpha_vdim[42])+0.9682458365518543*(f[6]*alpha_vdim[42]+f[11]*alpha_vdim[40])+0.4330127018922193*alpha_vdim[4]*f[29]+(0.3872983346207416*alpha_cdim[16]+0.4330127018922193*alpha_cdim[0])*f[26]+0.276641667586244*alpha_vdim[24]*f[25]+0.4330127018922193*(alpha_vdim[0]*f[25]+alpha_cdim[2]*f[24]+f[1]*alpha_vdim[24])+0.3872983346207416*alpha_vdim[3]*f[6]; 
  out[29] += 0.4330127018922193*(f[28]*alpha_vdim[82]+f[25]*alpha_vdim[80])+0.9682458365518543*(f[23]*alpha_vdim[56]+f[15]*alpha_vdim[42]+f[13]*alpha_vdim[40])+0.4330127018922193*(alpha_cdim[2]*f[30]+alpha_cdim[0]*f[27]); 
  out[30] += 0.3872983346207416*f[26]*alpha_vdim[96]+0.4330127018922193*(f[24]*alpha_vdim[82]+f[26]*alpha_vdim[80])+0.8660254037844386*(f[14]*alpha_vdim[56]+f[22]*alpha_vdim[42])+0.9682458365518543*(f[10]*alpha_vdim[42]+f[14]*alpha_vdim[40])+f[27]*(0.3872983346207416*alpha_vdim[32]+0.276641667586244*alpha_vdim[24])+0.4330127018922193*(alpha_vdim[0]*f[27]+alpha_vdim[4]*f[24]+f[4]*alpha_vdim[24])+0.3872983346207416*alpha_vdim[3]*f[10]; 
  out[31] += 0.3872983346207416*f[28]*alpha_vdim[96]+0.4330127018922193*(f[25]*alpha_vdim[82]+f[28]*alpha_vdim[80])+0.8660254037844386*(f[15]*alpha_vdim[56]+f[23]*alpha_vdim[42])+0.9682458365518543*(f[13]*alpha_vdim[42]+f[15]*alpha_vdim[40])+0.3872983346207416*f[29]*alpha_vdim[32]+(0.3872983346207416*alpha_cdim[16]+0.4330127018922193*alpha_cdim[0])*f[30]+0.276641667586244*alpha_vdim[24]*f[29]+0.4330127018922193*(alpha_vdim[0]*f[29]+alpha_cdim[2]*f[27]+alpha_vdim[4]*f[25]+f[8]*alpha_vdim[24])+0.3872983346207416*alpha_vdim[3]*f[13]; 
  out[32] += 0.9682458365518543*(f[19]*alpha_vdim[96]+f[9]*alpha_vdim[82]+f[4]*alpha_vdim[80]); 
  out[33] += 0.9682458365518543*(f[21]*alpha_vdim[96]+f[12]*alpha_vdim[82]+f[8]*alpha_vdim[80])+0.4330127018922193*(alpha_cdim[2]*f[34]+alpha_cdim[0]*f[32]); 
  out[34] += 0.8660254037844386*(f[9]*alpha_vdim[96]+f[19]*alpha_vdim[82])+0.9682458365518543*(f[4]*alpha_vdim[82]+f[9]*alpha_vdim[80])+0.4330127018922193*alpha_vdim[3]*f[35]+0.276641667586244*alpha_vdim[32]*f[32]+0.4330127018922193*(alpha_vdim[0]*f[32]+f[0]*alpha_vdim[32])+0.3872983346207416*alpha_vdim[4]*f[4]; 
  out[35] += 0.9682458365518543*(f[22]*alpha_vdim[96]+f[14]*alpha_vdim[82]+f[10]*alpha_vdim[80])+0.4330127018922193*(f[34]*alpha_vdim[42]+f[32]*alpha_vdim[40]); 
  out[36] += 0.8660254037844386*(f[12]*alpha_vdim[96]+f[21]*alpha_vdim[82])+0.9682458365518543*(f[8]*alpha_vdim[82]+f[12]*alpha_vdim[80])+0.4330127018922193*alpha_vdim[3]*f[37]+(0.3872983346207416*alpha_cdim[16]+0.4330127018922193*alpha_cdim[0])*f[34]+0.276641667586244*alpha_vdim[32]*f[33]+0.4330127018922193*(alpha_vdim[0]*f[33]+alpha_cdim[2]*f[32]+f[1]*alpha_vdim[32])+0.3872983346207416*alpha_vdim[4]*f[8]; 
  out[37] += 0.9682458365518543*(f[23]*alpha_vdim[96]+f[15]*alpha_vdim[82]+f[13]*alpha_vdim[80])+0.4330127018922193*(f[36]*alpha_vdim[42]+f[33]*alpha_vdim[40]+alpha_cdim[2]*f[38]+alpha_cdim[0]*f[35]); 
  out[38] += 0.8660254037844386*(f[14]*alpha_vdim[96]+f[22]*alpha_vdim[82])+0.9682458365518543*(f[10]*alpha_vdim[82]+f[14]*alpha_vdim[80])+0.3872983346207416*f[34]*alpha_vdim[56]+0.4330127018922193*(f[32]*alpha_vdim[42]+f[34]*alpha_vdim[40])+(0.276641667586244*alpha_vdim[32]+0.3872983346207416*alpha_vdim[24])*f[35]+0.4330127018922193*(alpha_vdim[0]*f[35]+alpha_vdim[3]*f[32]+f[3]*alpha_vdim[32])+0.3872983346207416*alpha_vdim[4]*f[10]; 
  out[39] += 0.8660254037844386*(f[15]*alpha_vdim[96]+f[23]*alpha_vdim[82])+0.9682458365518543*(f[13]*alpha_vdim[82]+f[15]*alpha_vdim[80])+0.3872983346207416*f[36]*alpha_vdim[56]+0.4330127018922193*(f[33]*alpha_vdim[42]+f[36]*alpha_vdim[40])+(0.3872983346207416*alpha_cdim[16]+0.4330127018922193*alpha_cdim[0])*f[38]+(0.276641667586244*alpha_vdim[32]+0.3872983346207416*alpha_vdim[24])*f[37]+0.4330127018922193*(alpha_vdim[0]*f[37]+alpha_cdim[2]*f[35]+alpha_vdim[3]*f[33]+f[6]*alpha_vdim[32])+0.3872983346207416*alpha_vdim[4]*f[13]; 

  return cflFreq; 
} 

