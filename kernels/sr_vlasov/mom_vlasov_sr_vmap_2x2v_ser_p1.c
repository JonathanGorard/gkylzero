#include <gkyl_mom_vlasov_sr_kernels.h> 
GKYL_CU_DH void vlasov_sr_vmap_M1i_2x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *jacob_vel_inv, const double *gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[2]*dxv[3]/4; 
  const double dv10 = 2.0/dxv[2]; 
  const double *jacob_vel_inv0 = &jacob_vel_inv[0]; 
  const double dv11 = 2.0/dxv[3]; 
  const double *jacob_vel_inv1 = &jacob_vel_inv[3]; 
  double p0_over_gamma[8] = {0.0}; 
  p0_over_gamma[0] = 2.738612787525831*jacob_vel_inv0[1]*gamma[4]*dv10+1.224744871391589*jacob_vel_inv0[0]*gamma[1]*dv10; 
  p0_over_gamma[1] = 2.449489742783178*jacob_vel_inv0[2]*gamma[4]*dv10+2.738612787525831*jacob_vel_inv0[0]*gamma[4]*dv10+1.224744871391589*jacob_vel_inv0[1]*gamma[1]*dv10; 
  p0_over_gamma[2] = 2.738612787525831*jacob_vel_inv0[1]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[0]*gamma[3]*dv10; 
  p0_over_gamma[3] = 2.449489742783178*jacob_vel_inv0[2]*gamma[6]*dv10+2.738612787525831*jacob_vel_inv0[0]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[1]*gamma[3]*dv10; 
  p0_over_gamma[4] = 2.449489742783178*jacob_vel_inv0[1]*gamma[4]*dv10+1.224744871391589*gamma[1]*jacob_vel_inv0[2]*dv10; 
  p0_over_gamma[5] = 1.224744871391589*jacob_vel_inv0[0]*gamma[7]*dv10; 
  p0_over_gamma[6] = 2.449489742783178*jacob_vel_inv0[1]*gamma[6]*dv10+1.224744871391589*jacob_vel_inv0[2]*gamma[3]*dv10; 
  p0_over_gamma[7] = 1.224744871391589*jacob_vel_inv0[1]*gamma[7]*dv10; 

  double p1_over_gamma[8] = {0.0}; 
  p1_over_gamma[0] = 2.738612787525831*jacob_vel_inv1[1]*gamma[5]*dv11+1.224744871391589*jacob_vel_inv1[0]*gamma[2]*dv11; 
  p1_over_gamma[1] = 2.738612787525831*jacob_vel_inv1[1]*gamma[7]*dv11+1.224744871391589*jacob_vel_inv1[0]*gamma[3]*dv11; 
  p1_over_gamma[2] = 2.449489742783178*jacob_vel_inv1[2]*gamma[5]*dv11+2.738612787525831*jacob_vel_inv1[0]*gamma[5]*dv11+1.224744871391589*jacob_vel_inv1[1]*gamma[2]*dv11; 
  p1_over_gamma[3] = 2.449489742783178*jacob_vel_inv1[2]*gamma[7]*dv11+2.738612787525831*jacob_vel_inv1[0]*gamma[7]*dv11+1.224744871391589*jacob_vel_inv1[1]*gamma[3]*dv11; 
  p1_over_gamma[4] = 1.224744871391589*jacob_vel_inv1[0]*gamma[6]*dv11; 
  p1_over_gamma[5] = 2.449489742783178*jacob_vel_inv1[1]*gamma[5]*dv11+1.224744871391589*jacob_vel_inv1[2]*gamma[2]*dv11; 
  p1_over_gamma[6] = 1.224744871391589*jacob_vel_inv1[1]*gamma[6]*dv11; 
  p1_over_gamma[7] = 2.449489742783178*jacob_vel_inv1[1]*gamma[7]*dv11+1.224744871391589*jacob_vel_inv1[2]*gamma[3]*dv11; 

  out[0] += (p0_over_gamma[7]*f[27]+p0_over_gamma[5]*f[24]+p0_over_gamma[6]*f[19]+p0_over_gamma[4]*f[16]+p0_over_gamma[3]*f[10]+p0_over_gamma[2]*f[4]+p0_over_gamma[1]*f[3]+f[0]*p0_over_gamma[0])*volFact; 
  out[1] += (1.0*p0_over_gamma[7]*f[29]+1.0*p0_over_gamma[5]*f[25]+1.0*p0_over_gamma[6]*f[21]+1.0*p0_over_gamma[4]*f[17]+p0_over_gamma[3]*f[13]+p0_over_gamma[2]*f[8]+p0_over_gamma[1]*f[6]+p0_over_gamma[0]*f[1])*volFact; 
  out[2] += (1.0*p0_over_gamma[7]*f[30]+1.0*p0_over_gamma[5]*f[26]+1.0*p0_over_gamma[6]*f[22]+1.0*p0_over_gamma[4]*f[18]+p0_over_gamma[3]*f[14]+p0_over_gamma[2]*f[9]+p0_over_gamma[1]*f[7]+p0_over_gamma[0]*f[2])*volFact; 
  out[3] += (p0_over_gamma[7]*f[31]+p0_over_gamma[5]*f[28]+p0_over_gamma[6]*f[23]+p0_over_gamma[4]*f[20]+p0_over_gamma[3]*f[15]+p0_over_gamma[2]*f[12]+p0_over_gamma[1]*f[11]+p0_over_gamma[0]*f[5])*volFact; 
  out[4] += (p1_over_gamma[7]*f[27]+p1_over_gamma[5]*f[24]+p1_over_gamma[6]*f[19]+p1_over_gamma[4]*f[16]+p1_over_gamma[3]*f[10]+p1_over_gamma[2]*f[4]+p1_over_gamma[1]*f[3]+f[0]*p1_over_gamma[0])*volFact; 
  out[5] += (1.0*p1_over_gamma[7]*f[29]+1.0*p1_over_gamma[5]*f[25]+1.0*p1_over_gamma[6]*f[21]+1.0*p1_over_gamma[4]*f[17]+p1_over_gamma[3]*f[13]+p1_over_gamma[2]*f[8]+p1_over_gamma[1]*f[6]+p1_over_gamma[0]*f[1])*volFact; 
  out[6] += (1.0*p1_over_gamma[7]*f[30]+1.0*p1_over_gamma[5]*f[26]+1.0*p1_over_gamma[6]*f[22]+1.0*p1_over_gamma[4]*f[18]+p1_over_gamma[3]*f[14]+p1_over_gamma[2]*f[9]+p1_over_gamma[1]*f[7]+p1_over_gamma[0]*f[2])*volFact; 
  out[7] += (p1_over_gamma[7]*f[31]+p1_over_gamma[5]*f[28]+p1_over_gamma[6]*f[23]+p1_over_gamma[4]*f[20]+p1_over_gamma[3]*f[15]+p1_over_gamma[2]*f[12]+p1_over_gamma[1]*f[11]+p1_over_gamma[0]*f[5])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_vmap_M3i_2x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *vmap, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[2]*dxv[3]/4; 
  const double *p0 = &vmap[0]; 
  const double *p1 = &vmap[4]; 
  out[0] += (1.414213562373095*p0[2]*f[16]+1.414213562373095*p0[1]*f[3]+1.414213562373095*f[0]*p0[0])*volFact; 
  out[1] += (1.414213562373095*p0[2]*f[17]+1.414213562373095*p0[1]*f[6]+1.414213562373095*p0[0]*f[1])*volFact; 
  out[2] += (1.414213562373095*p0[2]*f[18]+1.414213562373095*p0[1]*f[7]+1.414213562373095*p0[0]*f[2])*volFact; 
  out[3] += (1.414213562373095*p0[2]*f[20]+1.414213562373095*p0[1]*f[11]+1.414213562373095*p0[0]*f[5])*volFact; 
  out[4] += (1.414213562373095*p1[2]*f[24]+1.414213562373095*p1[1]*f[4]+1.414213562373095*f[0]*p1[0])*volFact; 
  out[5] += (1.414213562373095*p1[2]*f[25]+1.414213562373095*p1[1]*f[8]+1.414213562373095*p1[0]*f[1])*volFact; 
  out[6] += (1.414213562373095*p1[2]*f[26]+1.414213562373095*p1[1]*f[9]+1.414213562373095*p1[0]*f[2])*volFact; 
  out[7] += (1.414213562373095*p1[2]*f[28]+1.414213562373095*p1[1]*f[12]+1.414213562373095*p1[0]*f[5])*volFact; 
} 
GKYL_CU_DH void vlasov_sr_vmap_int_mom_2x2v_ser_p1(const double *w, const double *dxv, const int *idx, const double *vmap, const double *gamma, const double *f, double* GKYL_RESTRICT out) 
{ 
  const double volFact = dxv[0]*dxv[1]*dxv[2]*dxv[3]*0.0625; 
  const double *p0 = &vmap[0]; 
  const double *p1 = &vmap[4]; 
  out[0] += 4.0*f[0]*volFact; 
  out[1] += (2.0*gamma[7]*f[27]+2.0*gamma[5]*f[24]+2.0*gamma[6]*f[19]+2.0*gamma[4]*f[16]+2.0*gamma[3]*f[10]+2.0*gamma[2]*f[4]+2.0*gamma[1]*f[3]+2.0*f[0]*gamma[0])*volFact; 
  out[2] += (2.828427124746191*p0[2]*f[16]+2.828427124746191*p0[1]*f[3]+2.828427124746191*f[0]*p0[0])*volFact; 
  out[3] += (2.828427124746191*p1[2]*f[24]+2.828427124746191*p1[1]*f[4]+2.828427124746191*f[0]*p1[0])*volFact; 
} 