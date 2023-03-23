#include <gkyl_mom_vlasov_kernels.h> 
#include <gkyl_basis_ser_1x_p1_exp_sq.h> 
GKYL_CU_DH void vlasov_sr_M0_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  out[0] += 2.0*f[0]*volFact; 
  out[1] += 2.0*f[1]*volFact; 
} 
GKYL_CU_DH void vlasov_sr_M1i_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *p_over_gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  const double *p0_over_gamma = &p_over_gamma[0]; 
  const double *p1_over_gamma = &p_over_gamma[8]; 
  out[0] += (p0_over_gamma[7]*f[14]+p0_over_gamma[5]*f[12]+p0_over_gamma[6]*f[10]+p0_over_gamma[4]*f[8]+p0_over_gamma[3]*f[6]+p0_over_gamma[2]*f[3]+p0_over_gamma[1]*f[2]+f[0]*p0_over_gamma[0])*volFact; 
  out[1] += (1.0*p0_over_gamma[7]*f[15]+1.0*p0_over_gamma[5]*f[13]+1.0*p0_over_gamma[6]*f[11]+1.0*p0_over_gamma[4]*f[9]+p0_over_gamma[3]*f[7]+p0_over_gamma[2]*f[5]+p0_over_gamma[1]*f[4]+p0_over_gamma[0]*f[1])*volFact; 
  out[2] += (p1_over_gamma[7]*f[14]+p1_over_gamma[5]*f[12]+p1_over_gamma[6]*f[10]+p1_over_gamma[4]*f[8]+p1_over_gamma[3]*f[6]+p1_over_gamma[2]*f[3]+p1_over_gamma[1]*f[2]+f[0]*p1_over_gamma[0])*volFact; 
  out[3] += (1.0*p1_over_gamma[7]*f[15]+1.0*p1_over_gamma[5]*f[13]+1.0*p1_over_gamma[6]*f[11]+1.0*p1_over_gamma[4]*f[9]+p1_over_gamma[3]*f[7]+p1_over_gamma[2]*f[5]+p1_over_gamma[1]*f[4]+p1_over_gamma[0]*f[1])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_Ni_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *p_over_gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  const double *p0_over_gamma = &p_over_gamma[0]; 
  const double *p1_over_gamma = &p_over_gamma[8]; 
  out[0] += 2.0*f[0]*volFact; 
  out[1] += 2.0*f[1]*volFact; 
  out[2] += (p0_over_gamma[7]*f[14]+p0_over_gamma[5]*f[12]+p0_over_gamma[6]*f[10]+p0_over_gamma[4]*f[8]+p0_over_gamma[3]*f[6]+p0_over_gamma[2]*f[3]+p0_over_gamma[1]*f[2]+f[0]*p0_over_gamma[0])*volFact; 
  out[3] += (1.0*p0_over_gamma[7]*f[15]+1.0*p0_over_gamma[5]*f[13]+1.0*p0_over_gamma[6]*f[11]+1.0*p0_over_gamma[4]*f[9]+p0_over_gamma[3]*f[7]+p0_over_gamma[2]*f[5]+p0_over_gamma[1]*f[4]+p0_over_gamma[0]*f[1])*volFact; 
  out[4] += (p1_over_gamma[7]*f[14]+p1_over_gamma[5]*f[12]+p1_over_gamma[6]*f[10]+p1_over_gamma[4]*f[8]+p1_over_gamma[3]*f[6]+p1_over_gamma[2]*f[3]+p1_over_gamma[1]*f[2]+f[0]*p1_over_gamma[0])*volFact; 
  out[5] += (1.0*p1_over_gamma[7]*f[15]+1.0*p1_over_gamma[5]*f[13]+1.0*p1_over_gamma[6]*f[11]+1.0*p1_over_gamma[4]*f[9]+p1_over_gamma[3]*f[7]+p1_over_gamma[2]*f[5]+p1_over_gamma[1]*f[4]+p1_over_gamma[0]*f[1])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_Energy_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  out[0] += (gamma[7]*f[14]+gamma[5]*f[12]+gamma[6]*f[10]+gamma[4]*f[8]+gamma[3]*f[6]+gamma[2]*f[3]+gamma[1]*f[2]+f[0]*gamma[0])*volFact; 
  out[1] += (1.0*gamma[7]*f[15]+1.0*gamma[5]*f[13]+1.0*gamma[6]*f[11]+1.0*gamma[4]*f[9]+gamma[3]*f[7]+gamma[2]*f[5]+gamma[1]*f[4]+gamma[0]*f[1])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_Pressure_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *gamma_inv, const double *gamma, const double *GammaV2, const double *V_drift, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  const double wx1 = w[1], dv1 = dxv[1]; 
  const double wx1_sq = wx1*wx1, dv1_sq = dv1*dv1; 
  const double *V_drift_0 = &V_drift[0]; 
  double V_drift_0_sq[2] = {0.0}; 
  ser_1x_p1_exp_sq(V_drift_0, V_drift_0_sq); 
 
  const double wx2 = w[2], dv2 = dxv[2]; 
  const double wx2_sq = wx2*wx2, dv2_sq = dv2*dv2; 
  const double *V_drift_1 = &V_drift[2]; 
  double V_drift_1_sq[2] = {0.0}; 
  ser_1x_p1_exp_sq(V_drift_1, V_drift_1_sq); 
 
  double temp[16] = {0.0}; 
  double temp_sq[16] = {0.0}; 
  double p_fac[16] = {0.0}; 
  temp[0] = 2.0*V_drift_1[0]*wx2+2.0*V_drift_0[0]*wx1; 
  temp[1] = 2.0*V_drift_1[1]*wx2+2.0*V_drift_0[1]*wx1; 
  temp[2] = 0.5773502691896258*V_drift_0[0]*dv1; 
  temp[3] = 0.5773502691896258*V_drift_1[0]*dv2; 
  temp[4] = 0.5773502691896258*V_drift_0[1]*dv1; 
  temp[5] = 0.5773502691896258*V_drift_1[1]*dv2; 

  temp_sq[0] = 1.414213562373095*GammaV2[1]*V_drift_1_sq[1]*wx2_sq+1.414213562373095*GammaV2[0]*V_drift_1_sq[0]*wx2_sq+2.0*GammaV2[0]*V_drift_0[1]*V_drift_1[1]*wx1*wx2+2.0*V_drift_0[0]*GammaV2[1]*V_drift_1[1]*wx1*wx2+2.0*V_drift_1[0]*GammaV2[1]*V_drift_0[1]*wx1*wx2+2.0*GammaV2[0]*V_drift_0[0]*V_drift_1[0]*wx1*wx2+1.414213562373095*GammaV2[1]*V_drift_0_sq[1]*wx1_sq+1.414213562373095*GammaV2[0]*V_drift_0_sq[0]*wx1_sq+0.1178511301977579*GammaV2[1]*V_drift_1_sq[1]*dv2_sq+0.1178511301977579*GammaV2[0]*V_drift_1_sq[0]*dv2_sq+0.1178511301977579*GammaV2[1]*V_drift_0_sq[1]*dv1_sq+0.1178511301977579*GammaV2[0]*V_drift_0_sq[0]*dv1_sq; 
  temp_sq[1] = 1.414213562373095*GammaV2[0]*V_drift_1_sq[1]*wx2_sq+1.414213562373095*V_drift_1_sq[0]*GammaV2[1]*wx2_sq+3.6*GammaV2[1]*V_drift_0[1]*V_drift_1[1]*wx1*wx2+2.0*GammaV2[0]*V_drift_0[0]*V_drift_1[1]*wx1*wx2+2.0*GammaV2[0]*V_drift_1[0]*V_drift_0[1]*wx1*wx2+2.0*V_drift_0[0]*V_drift_1[0]*GammaV2[1]*wx1*wx2+1.414213562373095*GammaV2[0]*V_drift_0_sq[1]*wx1_sq+1.414213562373095*V_drift_0_sq[0]*GammaV2[1]*wx1_sq+0.1178511301977579*GammaV2[0]*V_drift_1_sq[1]*dv2_sq+0.1178511301977579*V_drift_1_sq[0]*GammaV2[1]*dv2_sq+0.1178511301977579*GammaV2[0]*V_drift_0_sq[1]*dv1_sq+0.1178511301977579*V_drift_0_sq[0]*GammaV2[1]*dv1_sq; 
  temp_sq[2] = 0.5773502691896258*GammaV2[0]*V_drift_0[1]*V_drift_1[1]*dv1*wx2+0.5773502691896258*V_drift_0[0]*GammaV2[1]*V_drift_1[1]*dv1*wx2+0.5773502691896258*V_drift_1[0]*GammaV2[1]*V_drift_0[1]*dv1*wx2+0.5773502691896258*GammaV2[0]*V_drift_0[0]*V_drift_1[0]*dv1*wx2+0.8164965809277261*GammaV2[1]*V_drift_0_sq[1]*dv1*wx1+0.8164965809277261*GammaV2[0]*V_drift_0_sq[0]*dv1*wx1; 
  temp_sq[3] = 0.8164965809277261*GammaV2[1]*V_drift_1_sq[1]*dv2*wx2+0.8164965809277261*GammaV2[0]*V_drift_1_sq[0]*dv2*wx2+0.5773502691896258*GammaV2[0]*V_drift_0[1]*V_drift_1[1]*dv2*wx1+0.5773502691896258*V_drift_0[0]*GammaV2[1]*V_drift_1[1]*dv2*wx1+0.5773502691896258*V_drift_1[0]*GammaV2[1]*V_drift_0[1]*dv2*wx1+0.5773502691896258*GammaV2[0]*V_drift_0[0]*V_drift_1[0]*dv2*wx1; 
  temp_sq[4] = 1.039230484541326*GammaV2[1]*V_drift_0[1]*V_drift_1[1]*dv1*wx2+0.5773502691896258*GammaV2[0]*V_drift_0[0]*V_drift_1[1]*dv1*wx2+0.5773502691896258*GammaV2[0]*V_drift_1[0]*V_drift_0[1]*dv1*wx2+0.5773502691896258*V_drift_0[0]*V_drift_1[0]*GammaV2[1]*dv1*wx2+0.8164965809277261*GammaV2[0]*V_drift_0_sq[1]*dv1*wx1+0.8164965809277261*V_drift_0_sq[0]*GammaV2[1]*dv1*wx1; 
  temp_sq[5] = 0.8164965809277261*GammaV2[0]*V_drift_1_sq[1]*dv2*wx2+0.8164965809277261*V_drift_1_sq[0]*GammaV2[1]*dv2*wx2+1.039230484541326*GammaV2[1]*V_drift_0[1]*V_drift_1[1]*dv2*wx1+0.5773502691896258*GammaV2[0]*V_drift_0[0]*V_drift_1[1]*dv2*wx1+0.5773502691896258*GammaV2[0]*V_drift_1[0]*V_drift_0[1]*dv2*wx1+0.5773502691896258*V_drift_0[0]*V_drift_1[0]*GammaV2[1]*dv2*wx1; 
  temp_sq[6] = 0.1666666666666667*GammaV2[0]*V_drift_0[1]*V_drift_1[1]*dv1*dv2+0.1666666666666667*V_drift_0[0]*GammaV2[1]*V_drift_1[1]*dv1*dv2+0.1666666666666667*V_drift_1[0]*GammaV2[1]*V_drift_0[1]*dv1*dv2+0.1666666666666667*GammaV2[0]*V_drift_0[0]*V_drift_1[0]*dv1*dv2; 
  temp_sq[7] = 0.3*GammaV2[1]*V_drift_0[1]*V_drift_1[1]*dv1*dv2+0.1666666666666667*GammaV2[0]*V_drift_0[0]*V_drift_1[1]*dv1*dv2+0.1666666666666667*GammaV2[0]*V_drift_1[0]*V_drift_0[1]*dv1*dv2+0.1666666666666667*V_drift_0[0]*V_drift_1[0]*GammaV2[1]*dv1*dv2; 
  temp_sq[8] = 0.105409255338946*GammaV2[1]*V_drift_0_sq[1]*dv1_sq+0.105409255338946*GammaV2[0]*V_drift_0_sq[0]*dv1_sq; 
  temp_sq[9] = 0.105409255338946*GammaV2[0]*V_drift_0_sq[1]*dv1_sq+0.105409255338946*V_drift_0_sq[0]*GammaV2[1]*dv1_sq; 
  temp_sq[12] = 0.105409255338946*GammaV2[1]*V_drift_1_sq[1]*dv2_sq+0.105409255338946*GammaV2[0]*V_drift_1_sq[0]*dv2_sq; 
  temp_sq[13] = 0.105409255338946*GammaV2[0]*V_drift_1_sq[1]*dv2_sq+0.105409255338946*V_drift_1_sq[0]*GammaV2[1]*dv2_sq; 

  p_fac[0] = 0.5*gamma_inv[5]*temp_sq[12]+0.5*gamma_inv[4]*temp_sq[8]+0.5*gamma_inv[3]*temp_sq[6]+0.5*gamma_inv[2]*temp_sq[3]+0.5*gamma_inv[1]*temp_sq[2]-1.414213562373095*GammaV2[1]*temp[1]+GammaV2[0]*gamma[0]+0.5*gamma_inv[0]*temp_sq[0]-1.414213562373095*GammaV2[0]*temp[0]-1.414213562373095*gamma_inv[0]; 
  p_fac[1] = 0.5000000000000001*gamma_inv[5]*temp_sq[13]+0.5000000000000001*gamma_inv[4]*temp_sq[9]+0.5*gamma_inv[3]*temp_sq[7]+0.5*gamma_inv[2]*temp_sq[5]+0.5*gamma_inv[1]*temp_sq[4]+0.5*gamma_inv[0]*temp_sq[1]-1.414213562373095*GammaV2[0]*temp[1]+gamma[0]*GammaV2[1]-1.414213562373095*temp[0]*GammaV2[1]; 
  p_fac[2] = 0.5000000000000001*gamma_inv[7]*temp_sq[12]+0.4472135954999579*gamma_inv[1]*temp_sq[8]+0.447213595499958*gamma_inv[6]*temp_sq[6]+0.5*gamma_inv[2]*temp_sq[6]-1.414213562373095*GammaV2[1]*temp[4]+0.4472135954999579*temp_sq[2]*gamma_inv[4]+0.5*gamma_inv[3]*temp_sq[3]+0.5*gamma_inv[0]*temp_sq[2]-1.414213562373095*GammaV2[0]*temp[2]+GammaV2[0]*gamma[1]+0.5*temp_sq[0]*gamma_inv[1]-1.414213562373095*gamma_inv[1]; 
  p_fac[3] = 0.4472135954999579*gamma_inv[2]*temp_sq[12]+0.5000000000000001*gamma_inv[6]*temp_sq[8]+0.447213595499958*temp_sq[6]*gamma_inv[7]+0.5*gamma_inv[1]*temp_sq[6]-1.414213562373095*GammaV2[1]*temp[5]+0.4472135954999579*temp_sq[3]*gamma_inv[5]+0.5*gamma_inv[0]*temp_sq[3]-1.414213562373095*GammaV2[0]*temp[3]+0.5*temp_sq[2]*gamma_inv[3]+GammaV2[0]*gamma[2]+0.5*temp_sq[0]*gamma_inv[2]-1.414213562373095*gamma_inv[2]; 
  p_fac[4] = 0.5*gamma_inv[7]*temp_sq[13]+0.447213595499958*gamma_inv[1]*temp_sq[9]+0.447213595499958*gamma_inv[6]*temp_sq[7]+0.5*gamma_inv[2]*temp_sq[7]+0.5*gamma_inv[3]*temp_sq[5]+0.4472135954999579*gamma_inv[4]*temp_sq[4]+0.5*gamma_inv[0]*temp_sq[4]-1.414213562373095*GammaV2[0]*temp[4]-1.414213562373095*GammaV2[1]*temp[2]+GammaV2[1]*gamma[1]+0.5*gamma_inv[1]*temp_sq[1]; 
  p_fac[5] = 0.447213595499958*gamma_inv[2]*temp_sq[13]+0.5*gamma_inv[6]*temp_sq[9]+0.447213595499958*gamma_inv[7]*temp_sq[7]+0.5*gamma_inv[1]*temp_sq[7]+0.4472135954999579*gamma_inv[5]*temp_sq[5]+0.5*gamma_inv[0]*temp_sq[5]-1.414213562373095*GammaV2[0]*temp[5]+0.5*gamma_inv[3]*temp_sq[4]-1.414213562373095*GammaV2[1]*temp[3]+GammaV2[1]*gamma[2]+0.5*temp_sq[1]*gamma_inv[2]; 
  p_fac[6] = 0.4472135954999579*gamma_inv[3]*temp_sq[12]+0.4472135954999579*gamma_inv[3]*temp_sq[8]+0.447213595499958*temp_sq[3]*gamma_inv[7]+0.4472135954999579*gamma_inv[5]*temp_sq[6]+0.4472135954999579*gamma_inv[4]*temp_sq[6]+0.5*gamma_inv[0]*temp_sq[6]+0.447213595499958*temp_sq[2]*gamma_inv[6]+GammaV2[0]*gamma[3]+0.5*gamma_inv[1]*temp_sq[3]+0.5*temp_sq[0]*gamma_inv[3]-1.414213562373095*gamma_inv[3]+0.5*gamma_inv[2]*temp_sq[2]; 
  p_fac[7] = 0.447213595499958*gamma_inv[3]*temp_sq[13]+0.447213595499958*gamma_inv[3]*temp_sq[9]+0.4472135954999579*gamma_inv[5]*temp_sq[7]+0.4472135954999579*gamma_inv[4]*temp_sq[7]+0.5*gamma_inv[0]*temp_sq[7]+0.447213595499958*temp_sq[5]*gamma_inv[7]+0.447213595499958*temp_sq[4]*gamma_inv[6]+0.5*gamma_inv[1]*temp_sq[5]+0.5*gamma_inv[2]*temp_sq[4]+GammaV2[1]*gamma[3]+0.5*temp_sq[1]*gamma_inv[3]; 
  p_fac[8] = 0.31943828249997*gamma_inv[4]*temp_sq[8]+0.5*gamma_inv[0]*temp_sq[8]+0.4472135954999579*gamma_inv[3]*temp_sq[6]+0.5000000000000001*temp_sq[3]*gamma_inv[6]+GammaV2[0]*gamma[4]+0.5*temp_sq[0]*gamma_inv[4]-1.414213562373095*gamma_inv[4]+0.4472135954999579*gamma_inv[1]*temp_sq[2]; 
  p_fac[9] = 0.31943828249997*gamma_inv[4]*temp_sq[9]+0.5*gamma_inv[0]*temp_sq[9]+0.447213595499958*gamma_inv[3]*temp_sq[7]+0.5*temp_sq[5]*gamma_inv[6]+1.0*GammaV2[1]*gamma[4]+0.447213595499958*gamma_inv[1]*temp_sq[4]+0.5000000000000001*temp_sq[1]*gamma_inv[4]; 
  p_fac[10] = 0.4472135954999579*gamma_inv[6]*temp_sq[12]+0.31943828249997*gamma_inv[6]*temp_sq[8]+0.5000000000000001*gamma_inv[2]*temp_sq[8]+0.4*temp_sq[6]*gamma_inv[7]+GammaV2[0]*gamma[6]+0.447213595499958*gamma_inv[1]*temp_sq[6]+0.5*temp_sq[0]*gamma_inv[6]-1.414213562373095*gamma_inv[6]+0.5000000000000001*temp_sq[3]*gamma_inv[4]+0.447213595499958*temp_sq[2]*gamma_inv[3]; 
  p_fac[11] = 0.4472135954999579*gamma_inv[6]*temp_sq[13]+0.31943828249997*gamma_inv[6]*temp_sq[9]+0.5000000000000001*gamma_inv[2]*temp_sq[9]+0.4*gamma_inv[7]*temp_sq[7]+0.4472135954999579*gamma_inv[1]*temp_sq[7]+1.0*GammaV2[1]*gamma[6]+0.5000000000000001*temp_sq[1]*gamma_inv[6]+0.5*gamma_inv[4]*temp_sq[5]+0.4472135954999579*gamma_inv[3]*temp_sq[4]; 
  p_fac[12] = 0.31943828249997*gamma_inv[5]*temp_sq[12]+0.5*gamma_inv[0]*temp_sq[12]+0.5000000000000001*temp_sq[2]*gamma_inv[7]+0.4472135954999579*gamma_inv[3]*temp_sq[6]+GammaV2[0]*gamma[5]+0.5*temp_sq[0]*gamma_inv[5]-1.414213562373095*gamma_inv[5]+0.4472135954999579*gamma_inv[2]*temp_sq[3]; 
  p_fac[13] = 0.31943828249997*gamma_inv[5]*temp_sq[13]+0.5*gamma_inv[0]*temp_sq[13]+0.447213595499958*gamma_inv[3]*temp_sq[7]+0.5*temp_sq[4]*gamma_inv[7]+1.0*GammaV2[1]*gamma[5]+0.447213595499958*gamma_inv[2]*temp_sq[5]+0.5000000000000001*temp_sq[1]*gamma_inv[5]; 
  p_fac[14] = 0.31943828249997*gamma_inv[7]*temp_sq[12]+0.5000000000000001*gamma_inv[1]*temp_sq[12]+0.4472135954999579*gamma_inv[7]*temp_sq[8]+GammaV2[0]*gamma[7]+0.5*temp_sq[0]*gamma_inv[7]-1.414213562373095*gamma_inv[7]+0.4*gamma_inv[6]*temp_sq[6]+0.447213595499958*gamma_inv[2]*temp_sq[6]+0.5000000000000001*temp_sq[2]*gamma_inv[5]+0.447213595499958*gamma_inv[3]*temp_sq[3]; 
  p_fac[15] = 0.31943828249997*gamma_inv[7]*temp_sq[13]+0.5000000000000001*gamma_inv[1]*temp_sq[13]+0.4472135954999579*gamma_inv[7]*temp_sq[9]+1.0*GammaV2[1]*gamma[7]+0.4*gamma_inv[6]*temp_sq[7]+0.4472135954999579*gamma_inv[2]*temp_sq[7]+0.5000000000000001*temp_sq[1]*gamma_inv[7]+0.4472135954999579*gamma_inv[3]*temp_sq[5]+0.5*temp_sq[4]*gamma_inv[5]; 

  out[0] += (0.3535533905932737*f[15]*p_fac[15]+0.3535533905932737*f[14]*p_fac[14]+0.3535533905932737*f[13]*p_fac[13]+0.3535533905932737*f[12]*p_fac[12]+0.3535533905932737*f[11]*p_fac[11]+0.3535533905932737*f[10]*p_fac[10]+0.3535533905932737*f[9]*p_fac[9]+0.3535533905932737*f[8]*p_fac[8]+0.3535533905932737*f[7]*p_fac[7]+0.3535533905932737*f[6]*p_fac[6]+0.3535533905932737*f[5]*p_fac[5]+0.3535533905932737*f[4]*p_fac[4]+0.3535533905932737*f[3]*p_fac[3]+0.3535533905932737*f[2]*p_fac[2]+0.3535533905932737*f[1]*p_fac[1]+0.3535533905932737*f[0]*p_fac[0])*volFact; 
  out[1] += (0.3535533905932737*f[14]*p_fac[15]+0.3535533905932737*p_fac[14]*f[15]+0.3535533905932737*f[12]*p_fac[13]+0.3535533905932737*p_fac[12]*f[13]+0.3535533905932737*f[10]*p_fac[11]+0.3535533905932737*p_fac[10]*f[11]+0.3535533905932737*f[8]*p_fac[9]+0.3535533905932737*p_fac[8]*f[9]+0.3535533905932737*f[6]*p_fac[7]+0.3535533905932737*p_fac[6]*f[7]+0.3535533905932737*f[3]*p_fac[5]+0.3535533905932737*p_fac[3]*f[5]+0.3535533905932737*f[2]*p_fac[4]+0.3535533905932737*p_fac[2]*f[4]+0.3535533905932737*f[0]*p_fac[1]+0.3535533905932737*p_fac[0]*f[1])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_Tij_1x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *p_over_gamma, const double *gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[1]*dxv[2]/4; 
  const double wx1 = w[1], dv1 = dxv[1]; 
  const double *p0_over_gamma = &p_over_gamma[0]; 
  const double wx2 = w[2], dv2 = dxv[2]; 
  const double *p1_over_gamma = &p_over_gamma[8]; 
  out[0] += (gamma[7]*f[14]+gamma[5]*f[12]+gamma[6]*f[10]+gamma[4]*f[8]+gamma[3]*f[6]+gamma[2]*f[3]+gamma[1]*f[2]+f[0]*gamma[0])*volFact; 
  out[1] += (1.0*gamma[7]*f[15]+1.0*gamma[5]*f[13]+1.0*gamma[6]*f[11]+1.0*gamma[4]*f[9]+gamma[3]*f[7]+gamma[2]*f[5]+gamma[1]*f[4]+gamma[0]*f[1])*volFact; 
  out[2] += volFact*(2.0*f[0]*wx1+0.5773502691896258*f[2]*dv1); 
  out[3] += volFact*(2.0*f[1]*wx1+0.5773502691896258*f[4]*dv1); 
  out[4] += volFact*(2.0*f[0]*wx2+0.5773502691896258*f[3]*dv2); 
  out[5] += volFact*(2.0*f[1]*wx2+0.5773502691896258*f[5]*dv2); 
  out[6] += volFact*(p0_over_gamma[7]*f[14]*wx1+p0_over_gamma[5]*f[12]*wx1+p0_over_gamma[6]*f[10]*wx1+p0_over_gamma[4]*f[8]*wx1+p0_over_gamma[3]*f[6]*wx1+p0_over_gamma[2]*f[3]*wx1+p0_over_gamma[1]*f[2]*wx1+f[0]*p0_over_gamma[0]*wx1+0.2886751345948129*p0_over_gamma[5]*f[14]*dv1+0.2886751345948129*p0_over_gamma[7]*f[12]*dv1+0.2581988897471611*p0_over_gamma[3]*f[10]*dv1+0.2581988897471612*p0_over_gamma[1]*f[8]*dv1+0.2581988897471611*f[6]*p0_over_gamma[6]*dv1+0.2886751345948129*p0_over_gamma[2]*f[6]*dv1+0.2581988897471612*f[2]*p0_over_gamma[4]*dv1+0.2886751345948129*f[3]*p0_over_gamma[3]*dv1+0.2886751345948129*p0_over_gamma[0]*f[2]*dv1+0.2886751345948129*f[0]*p0_over_gamma[1]*dv1); 
  out[7] += volFact*(1.0*p0_over_gamma[7]*f[15]*wx1+1.0*p0_over_gamma[5]*f[13]*wx1+1.0*p0_over_gamma[6]*f[11]*wx1+1.0*p0_over_gamma[4]*f[9]*wx1+p0_over_gamma[3]*f[7]*wx1+p0_over_gamma[2]*f[5]*wx1+p0_over_gamma[1]*f[4]*wx1+p0_over_gamma[0]*f[1]*wx1+0.2886751345948129*p0_over_gamma[5]*f[15]*dv1+0.2886751345948129*p0_over_gamma[7]*f[13]*dv1+0.2581988897471612*p0_over_gamma[3]*f[11]*dv1+0.2581988897471611*p0_over_gamma[1]*f[9]*dv1+0.2581988897471611*p0_over_gamma[6]*f[7]*dv1+0.2886751345948129*p0_over_gamma[2]*f[7]*dv1+0.2886751345948129*p0_over_gamma[3]*f[5]*dv1+0.2581988897471612*f[4]*p0_over_gamma[4]*dv1+0.2886751345948129*p0_over_gamma[0]*f[4]*dv1+0.2886751345948129*f[1]*p0_over_gamma[1]*dv1); 
  out[8] += volFact*(p0_over_gamma[7]*f[14]*wx2+p0_over_gamma[5]*f[12]*wx2+p0_over_gamma[6]*f[10]*wx2+p0_over_gamma[4]*f[8]*wx2+p0_over_gamma[3]*f[6]*wx2+p0_over_gamma[2]*f[3]*wx2+p0_over_gamma[1]*f[2]*wx2+f[0]*p0_over_gamma[0]*wx2+0.2581988897471611*p0_over_gamma[3]*f[14]*dv2+0.2581988897471612*p0_over_gamma[2]*f[12]*dv2+0.2886751345948129*p0_over_gamma[4]*f[10]*dv2+0.2886751345948129*p0_over_gamma[6]*f[8]*dv2+0.2581988897471611*f[6]*p0_over_gamma[7]*dv2+0.2886751345948129*p0_over_gamma[1]*f[6]*dv2+0.2581988897471612*f[3]*p0_over_gamma[5]*dv2+0.2886751345948129*f[2]*p0_over_gamma[3]*dv2+0.2886751345948129*p0_over_gamma[0]*f[3]*dv2+0.2886751345948129*f[0]*p0_over_gamma[2]*dv2); 
  out[9] += volFact*(1.0*p0_over_gamma[7]*f[15]*wx2+1.0*p0_over_gamma[5]*f[13]*wx2+1.0*p0_over_gamma[6]*f[11]*wx2+1.0*p0_over_gamma[4]*f[9]*wx2+p0_over_gamma[3]*f[7]*wx2+p0_over_gamma[2]*f[5]*wx2+p0_over_gamma[1]*f[4]*wx2+p0_over_gamma[0]*f[1]*wx2+0.2581988897471612*p0_over_gamma[3]*f[15]*dv2+0.2581988897471611*p0_over_gamma[2]*f[13]*dv2+0.2886751345948129*p0_over_gamma[4]*f[11]*dv2+0.2886751345948129*p0_over_gamma[6]*f[9]*dv2+0.2581988897471611*f[7]*p0_over_gamma[7]*dv2+0.2886751345948129*p0_over_gamma[1]*f[7]*dv2+0.2581988897471612*f[5]*p0_over_gamma[5]*dv2+0.2886751345948129*p0_over_gamma[0]*f[5]*dv2+0.2886751345948129*p0_over_gamma[3]*f[4]*dv2+0.2886751345948129*f[1]*p0_over_gamma[2]*dv2); 
  out[10] += volFact*(p1_over_gamma[7]*f[14]*wx2+p1_over_gamma[5]*f[12]*wx2+p1_over_gamma[6]*f[10]*wx2+p1_over_gamma[4]*f[8]*wx2+p1_over_gamma[3]*f[6]*wx2+p1_over_gamma[2]*f[3]*wx2+p1_over_gamma[1]*f[2]*wx2+f[0]*p1_over_gamma[0]*wx2+0.2581988897471611*p1_over_gamma[3]*f[14]*dv2+0.2581988897471612*p1_over_gamma[2]*f[12]*dv2+0.2886751345948129*p1_over_gamma[4]*f[10]*dv2+0.2886751345948129*p1_over_gamma[6]*f[8]*dv2+0.2581988897471611*f[6]*p1_over_gamma[7]*dv2+0.2886751345948129*p1_over_gamma[1]*f[6]*dv2+0.2581988897471612*f[3]*p1_over_gamma[5]*dv2+0.2886751345948129*f[2]*p1_over_gamma[3]*dv2+0.2886751345948129*p1_over_gamma[0]*f[3]*dv2+0.2886751345948129*f[0]*p1_over_gamma[2]*dv2); 
  out[11] += volFact*(1.0*p1_over_gamma[7]*f[15]*wx2+1.0*p1_over_gamma[5]*f[13]*wx2+1.0*p1_over_gamma[6]*f[11]*wx2+1.0*p1_over_gamma[4]*f[9]*wx2+p1_over_gamma[3]*f[7]*wx2+p1_over_gamma[2]*f[5]*wx2+p1_over_gamma[1]*f[4]*wx2+p1_over_gamma[0]*f[1]*wx2+0.2581988897471612*p1_over_gamma[3]*f[15]*dv2+0.2581988897471611*p1_over_gamma[2]*f[13]*dv2+0.2886751345948129*p1_over_gamma[4]*f[11]*dv2+0.2886751345948129*p1_over_gamma[6]*f[9]*dv2+0.2581988897471611*f[7]*p1_over_gamma[7]*dv2+0.2886751345948129*p1_over_gamma[1]*f[7]*dv2+0.2581988897471612*f[5]*p1_over_gamma[5]*dv2+0.2886751345948129*p1_over_gamma[0]*f[5]*dv2+0.2886751345948129*p1_over_gamma[3]*f[4]*dv2+0.2886751345948129*f[1]*p1_over_gamma[2]*dv2); 
} 
