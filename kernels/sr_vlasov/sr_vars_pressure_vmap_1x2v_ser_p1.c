#include <gkyl_sr_Gamma_kernels.h> 
#include <gkyl_basis_ser_1x_p3_exp_sq.h> 
GKYL_CU_DH void sr_vars_pressure_vmap_1x2v_ser_p1(const double *w, const double *dxv, const double *vmap, const double *gamma, const double *gamma_inv, const double *u_i, const double *u_i_sq, const double *GammaV, const double *GammaV_sq, const double *f, double* GKYL_RESTRICT sr_pressure) 
{ 
  // vmap:        Momentum-space nonuniform mapping.
  // gamma:       Particle Lorentz boost factor sqrt(1 + p^2).
  // gamma_inv:   Inverse particle Lorentz boost factor 1/sqrt(1 + p^2).
  // u_i:         Spatial components of bulk four-velocity = GammaV*V_drift. 
  // u_i_sq:      Squared spatial components of bulk four-velocity = u_i^2. 
  // GammaV:      Bulk four-velocity Lorentz factor = sqrt(1 + |u_i|^2). 
  // GammaV_sq:   Squared bulk four-velocity Lorentz factor = 1 + |u_i|^2. 
  // f:           Input distribution function.
  // sr_pressure: Output relativistic pressure.
  const double volFact = dxv[1]*dxv[2]/4; 
 
  const double *p0 = &vmap[0]; 
  const double *V_0 = &u_i[0]; 
  const double *V_0_sq = &u_i_sq[0]; 
 
  const double *p1 = &vmap[4]; 
  const double *V_1 = &u_i[2]; 
  const double *V_1_sq = &u_i_sq[2]; 
 
  double temp[16] = {0.0}; 
  double temp_sq[16] = {0.0}; 
  double p_fac[16] = {0.0}; 
  double p0_sq[4] = {0.0}; 
  ser_1x_p3_exp_sq(p0, p0_sq); 
  double p1_sq[4] = {0.0}; 
  ser_1x_p3_exp_sq(p1, p1_sq); 
  temp[0] = 1.414213562373095*V_1[0]*p1[0]+1.414213562373095*V_0[0]*p0[0]; 
  temp[1] = 1.414213562373095*p1[0]*V_1[1]+1.414213562373095*p0[0]*V_0[1]; 
  temp[2] = 1.414213562373095*V_0[0]*p0[1]; 
  temp[3] = 1.414213562373095*V_1[0]*p1[1]; 
  temp[4] = 1.414213562373095*V_0[1]*p0[1]; 
  temp[5] = 1.414213562373095*V_1[1]*p1[1]; 
  temp[8] = 1.414213562373095*V_0[0]*p0[2]; 
  temp[9] = 1.414213562373095*V_0[1]*p0[2]; 
  temp[12] = 1.414213562373095*V_1[0]*p1[2]; 
  temp[13] = 1.414213562373095*V_1[1]*p1[2]; 

  temp_sq[0] = 1.414213562373095*p0[0]*p1[0]*V_0[1]*V_1[1]+1.414213562373095*V_1_sq[0]*p1_sq[0]+1.414213562373095*V_0[0]*V_1[0]*p0[0]*p1[0]+1.414213562373095*V_0_sq[0]*p0_sq[0]; 
  temp_sq[1] = 1.414213562373095*p1_sq[0]*V_1_sq[1]+1.414213562373095*V_0[0]*p0[0]*p1[0]*V_1[1]+1.414213562373095*p0_sq[0]*V_0_sq[1]+1.414213562373095*V_1[0]*p0[0]*p1[0]*V_0[1]; 
  temp_sq[2] = 1.414213562373095*V_0_sq[0]*p0_sq[1]+1.414213562373095*p1[0]*V_0[1]*V_1[1]*p0[1]+1.414213562373095*V_0[0]*V_1[0]*p1[0]*p0[1]; 
  temp_sq[3] = 1.414213562373095*V_1_sq[0]*p1_sq[1]+1.414213562373095*p0[0]*V_0[1]*V_1[1]*p1[1]+1.414213562373095*V_0[0]*V_1[0]*p0[0]*p1[1]; 
  temp_sq[4] = 1.414213562373095*V_0_sq[1]*p0_sq[1]+1.414213562373095*V_0[0]*p1[0]*V_1[1]*p0[1]+1.414213562373095*V_1[0]*p1[0]*V_0[1]*p0[1]; 
  temp_sq[5] = 1.414213562373095*V_1_sq[1]*p1_sq[1]+1.414213562373095*V_0[0]*p0[0]*V_1[1]*p1[1]+1.414213562373095*V_1[0]*p0[0]*V_0[1]*p1[1]; 
  temp_sq[6] = 1.414213562373095*V_0[1]*V_1[1]*p0[1]*p1[1]+1.414213562373095*V_0[0]*V_1[0]*p0[1]*p1[1]; 
  temp_sq[7] = 1.414213562373095*V_0[0]*V_1[1]*p0[1]*p1[1]+1.414213562373095*V_1[0]*V_0[1]*p0[1]*p1[1]; 
  temp_sq[8] = 1.414213562373095*V_0_sq[0]*p0_sq[2]+1.414213562373095*p1[0]*V_0[1]*V_1[1]*p0[2]+1.414213562373095*V_0[0]*V_1[0]*p1[0]*p0[2]; 
  temp_sq[9] = 1.414213562373095*V_0_sq[1]*p0_sq[2]+1.414213562373095*V_0[0]*p1[0]*V_1[1]*p0[2]+1.414213562373095*V_1[0]*p1[0]*V_0[1]*p0[2]; 
  temp_sq[10] = 1.414213562373095*V_0[1]*V_1[1]*p1[1]*p0[2]+1.414213562373095*V_0[0]*V_1[0]*p1[1]*p0[2]; 
  temp_sq[11] = 1.414213562373095*V_0[0]*V_1[1]*p1[1]*p0[2]+1.414213562373095*V_1[0]*V_0[1]*p1[1]*p0[2]; 
  temp_sq[12] = 1.414213562373095*V_1_sq[0]*p1_sq[2]+1.414213562373095*p0[0]*V_0[1]*V_1[1]*p1[2]+1.414213562373095*V_0[0]*V_1[0]*p0[0]*p1[2]; 
  temp_sq[13] = 1.414213562373095*V_1_sq[1]*p1_sq[2]+1.414213562373095*V_0[0]*p0[0]*V_1[1]*p1[2]+1.414213562373095*V_1[0]*p0[0]*V_0[1]*p1[2]; 
  temp_sq[14] = 1.414213562373095*V_0[1]*V_1[1]*p0[1]*p1[2]+1.414213562373095*V_0[0]*V_1[0]*p0[1]*p1[2]; 
  temp_sq[15] = 1.414213562373095*V_0[0]*V_1[1]*p0[1]*p1[2]+1.414213562373095*V_1[0]*V_0[1]*p0[1]*p1[2]; 

  p_fac[0] = 0.5*gamma_inv[7]*temp_sq[14]+0.5*gamma_inv[5]*temp_sq[12]+0.5*gamma_inv[6]*temp_sq[10]+0.5*gamma_inv[4]*temp_sq[8]+0.5*gamma_inv[3]*temp_sq[6]+0.5*gamma_inv[2]*temp_sq[3]+0.5*gamma_inv[1]*temp_sq[2]-1.414213562373095*GammaV[1]*temp[1]+GammaV_sq[0]*gamma[0]+0.5*gamma_inv[0]*temp_sq[0]-1.414213562373095*GammaV[0]*temp[0]-1.414213562373095*gamma_inv[0]; 
  p_fac[1] = 0.5000000000000001*gamma_inv[7]*temp_sq[15]+0.5000000000000001*gamma_inv[5]*temp_sq[13]+0.5000000000000001*gamma_inv[6]*temp_sq[11]+0.5000000000000001*gamma_inv[4]*temp_sq[9]+0.5*gamma_inv[3]*temp_sq[7]+0.5*gamma_inv[2]*temp_sq[5]+0.5*gamma_inv[1]*temp_sq[4]+0.5*gamma_inv[0]*temp_sq[1]-1.414213562373095*GammaV[0]*temp[1]+gamma[0]*GammaV_sq[1]-1.414213562373095*temp[0]*GammaV[1]; 
  p_fac[2] = 0.5000000000000001*gamma_inv[5]*temp_sq[14]+0.5000000000000001*gamma_inv[7]*temp_sq[12]+0.447213595499958*gamma_inv[3]*temp_sq[10]+0.4472135954999579*gamma_inv[1]*temp_sq[8]+0.447213595499958*gamma_inv[6]*temp_sq[6]+0.5*gamma_inv[2]*temp_sq[6]-1.414213562373095*GammaV[1]*temp[4]+0.4472135954999579*temp_sq[2]*gamma_inv[4]+0.5*gamma_inv[3]*temp_sq[3]+0.5*gamma_inv[0]*temp_sq[2]-1.414213562373095*GammaV[0]*temp[2]+GammaV_sq[0]*gamma[1]+0.5*temp_sq[0]*gamma_inv[1]-1.414213562373095*gamma_inv[1]; 
  p_fac[3] = 0.447213595499958*gamma_inv[3]*temp_sq[14]+0.4472135954999579*gamma_inv[2]*temp_sq[12]+0.5000000000000001*gamma_inv[4]*temp_sq[10]+0.5000000000000001*gamma_inv[6]*temp_sq[8]+0.447213595499958*temp_sq[6]*gamma_inv[7]+0.5*gamma_inv[1]*temp_sq[6]-1.414213562373095*GammaV[1]*temp[5]+0.4472135954999579*temp_sq[3]*gamma_inv[5]+0.5*gamma_inv[0]*temp_sq[3]-1.414213562373095*GammaV[0]*temp[3]+0.5*temp_sq[2]*gamma_inv[3]+GammaV_sq[0]*gamma[2]+0.5*temp_sq[0]*gamma_inv[2]-1.414213562373095*gamma_inv[2]; 
  p_fac[4] = 0.5*gamma_inv[5]*temp_sq[15]+0.5*gamma_inv[7]*temp_sq[13]+0.4472135954999579*gamma_inv[3]*temp_sq[11]+0.447213595499958*gamma_inv[1]*temp_sq[9]+0.447213595499958*gamma_inv[6]*temp_sq[7]+0.5*gamma_inv[2]*temp_sq[7]+0.5*gamma_inv[3]*temp_sq[5]+0.4472135954999579*gamma_inv[4]*temp_sq[4]+0.5*gamma_inv[0]*temp_sq[4]-1.414213562373095*GammaV[0]*temp[4]-1.414213562373095*GammaV[1]*temp[2]+GammaV_sq[1]*gamma[1]+0.5*gamma_inv[1]*temp_sq[1]; 
  p_fac[5] = 0.4472135954999579*gamma_inv[3]*temp_sq[15]+0.447213595499958*gamma_inv[2]*temp_sq[13]+0.5*gamma_inv[4]*temp_sq[11]+0.5*gamma_inv[6]*temp_sq[9]+0.447213595499958*gamma_inv[7]*temp_sq[7]+0.5*gamma_inv[1]*temp_sq[7]+0.4472135954999579*gamma_inv[5]*temp_sq[5]+0.5*gamma_inv[0]*temp_sq[5]-1.414213562373095*GammaV[0]*temp[5]+0.5*gamma_inv[3]*temp_sq[4]-1.414213562373095*GammaV[1]*temp[3]+GammaV_sq[1]*gamma[2]+0.5*temp_sq[1]*gamma_inv[2]; 
  p_fac[6] = 0.4*gamma_inv[6]*temp_sq[14]+0.447213595499958*gamma_inv[2]*temp_sq[14]+0.4472135954999579*gamma_inv[3]*temp_sq[12]+0.4*gamma_inv[7]*temp_sq[10]+0.447213595499958*gamma_inv[1]*temp_sq[10]+0.4472135954999579*gamma_inv[3]*temp_sq[8]+0.447213595499958*temp_sq[3]*gamma_inv[7]+0.4472135954999579*gamma_inv[5]*temp_sq[6]+0.4472135954999579*gamma_inv[4]*temp_sq[6]+0.5*gamma_inv[0]*temp_sq[6]+0.447213595499958*temp_sq[2]*gamma_inv[6]+GammaV_sq[0]*gamma[3]+0.5*gamma_inv[1]*temp_sq[3]+0.5*temp_sq[0]*gamma_inv[3]-1.414213562373095*gamma_inv[3]+0.5*gamma_inv[2]*temp_sq[2]; 
  p_fac[7] = 0.4*gamma_inv[6]*temp_sq[15]+0.4472135954999579*gamma_inv[2]*temp_sq[15]+0.447213595499958*gamma_inv[3]*temp_sq[13]+0.4*gamma_inv[7]*temp_sq[11]+0.4472135954999579*gamma_inv[1]*temp_sq[11]+0.447213595499958*gamma_inv[3]*temp_sq[9]+0.4472135954999579*gamma_inv[5]*temp_sq[7]+0.4472135954999579*gamma_inv[4]*temp_sq[7]+0.5*gamma_inv[0]*temp_sq[7]+0.447213595499958*temp_sq[5]*gamma_inv[7]+0.447213595499958*temp_sq[4]*gamma_inv[6]+0.5*gamma_inv[1]*temp_sq[5]+0.5*gamma_inv[2]*temp_sq[4]+GammaV_sq[1]*gamma[3]+0.5*temp_sq[1]*gamma_inv[3]; 
  p_fac[8] = 0.4472135954999579*gamma_inv[7]*temp_sq[14]+0.31943828249997*gamma_inv[6]*temp_sq[10]+0.5000000000000001*gamma_inv[2]*temp_sq[10]-1.414213562373095*GammaV[1]*temp[9]+0.31943828249997*gamma_inv[4]*temp_sq[8]+0.5*gamma_inv[0]*temp_sq[8]-1.414213562373095*GammaV[0]*temp[8]+0.4472135954999579*gamma_inv[3]*temp_sq[6]+0.5000000000000001*temp_sq[3]*gamma_inv[6]+GammaV_sq[0]*gamma[4]+0.5*temp_sq[0]*gamma_inv[4]-1.414213562373095*gamma_inv[4]+0.4472135954999579*gamma_inv[1]*temp_sq[2]; 
  p_fac[9] = 0.4472135954999579*gamma_inv[7]*temp_sq[15]+0.31943828249997*gamma_inv[6]*temp_sq[11]+0.5000000000000001*gamma_inv[2]*temp_sq[11]+0.31943828249997*gamma_inv[4]*temp_sq[9]+0.5*gamma_inv[0]*temp_sq[9]-1.414213562373095*GammaV[0]*temp[9]-1.414213562373095*GammaV[1]*temp[8]+0.447213595499958*gamma_inv[3]*temp_sq[7]+0.5*temp_sq[5]*gamma_inv[6]+1.0*GammaV_sq[1]*gamma[4]+0.447213595499958*gamma_inv[1]*temp_sq[4]+0.5000000000000001*temp_sq[1]*gamma_inv[4]; 
  p_fac[10] = 0.4*gamma_inv[3]*temp_sq[14]+0.4472135954999579*gamma_inv[6]*temp_sq[12]+0.4472135954999579*gamma_inv[5]*temp_sq[10]+0.31943828249997*gamma_inv[4]*temp_sq[10]+0.5*gamma_inv[0]*temp_sq[10]+0.31943828249997*gamma_inv[6]*temp_sq[8]+0.5000000000000001*gamma_inv[2]*temp_sq[8]+0.4*temp_sq[6]*gamma_inv[7]+GammaV_sq[0]*gamma[6]+0.447213595499958*gamma_inv[1]*temp_sq[6]+0.5*temp_sq[0]*gamma_inv[6]-1.414213562373095*gamma_inv[6]+0.5000000000000001*temp_sq[3]*gamma_inv[4]+0.447213595499958*temp_sq[2]*gamma_inv[3]; 
  p_fac[11] = 0.4*gamma_inv[3]*temp_sq[15]+0.4472135954999579*gamma_inv[6]*temp_sq[13]+0.4472135954999579*gamma_inv[5]*temp_sq[11]+0.31943828249997*gamma_inv[4]*temp_sq[11]+0.5*gamma_inv[0]*temp_sq[11]+0.31943828249997*gamma_inv[6]*temp_sq[9]+0.5000000000000001*gamma_inv[2]*temp_sq[9]+0.4*gamma_inv[7]*temp_sq[7]+0.4472135954999579*gamma_inv[1]*temp_sq[7]+1.0*GammaV_sq[1]*gamma[6]+0.5000000000000001*temp_sq[1]*gamma_inv[6]+0.5*gamma_inv[4]*temp_sq[5]+0.4472135954999579*gamma_inv[3]*temp_sq[4]; 
  p_fac[12] = 0.31943828249997*gamma_inv[7]*temp_sq[14]+0.5000000000000001*gamma_inv[1]*temp_sq[14]-1.414213562373095*GammaV[1]*temp[13]+0.31943828249997*gamma_inv[5]*temp_sq[12]+0.5*gamma_inv[0]*temp_sq[12]-1.414213562373095*GammaV[0]*temp[12]+0.4472135954999579*gamma_inv[6]*temp_sq[10]+0.5000000000000001*temp_sq[2]*gamma_inv[7]+0.4472135954999579*gamma_inv[3]*temp_sq[6]+GammaV_sq[0]*gamma[5]+0.5*temp_sq[0]*gamma_inv[5]-1.414213562373095*gamma_inv[5]+0.4472135954999579*gamma_inv[2]*temp_sq[3]; 
  p_fac[13] = 0.31943828249997*gamma_inv[7]*temp_sq[15]+0.5000000000000001*gamma_inv[1]*temp_sq[15]+0.31943828249997*gamma_inv[5]*temp_sq[13]+0.5*gamma_inv[0]*temp_sq[13]-1.414213562373095*GammaV[0]*temp[13]-1.414213562373095*GammaV[1]*temp[12]+0.4472135954999579*gamma_inv[6]*temp_sq[11]+0.447213595499958*gamma_inv[3]*temp_sq[7]+0.5*temp_sq[4]*gamma_inv[7]+1.0*GammaV_sq[1]*gamma[5]+0.447213595499958*gamma_inv[2]*temp_sq[5]+0.5000000000000001*temp_sq[1]*gamma_inv[5]; 
  p_fac[14] = 0.31943828249997*gamma_inv[5]*temp_sq[14]+0.4472135954999579*gamma_inv[4]*temp_sq[14]+0.5*gamma_inv[0]*temp_sq[14]+0.31943828249997*gamma_inv[7]*temp_sq[12]+0.5000000000000001*gamma_inv[1]*temp_sq[12]+0.4*gamma_inv[3]*temp_sq[10]+0.4472135954999579*gamma_inv[7]*temp_sq[8]+GammaV_sq[0]*gamma[7]+0.5*temp_sq[0]*gamma_inv[7]-1.414213562373095*gamma_inv[7]+0.4*gamma_inv[6]*temp_sq[6]+0.447213595499958*gamma_inv[2]*temp_sq[6]+0.5000000000000001*temp_sq[2]*gamma_inv[5]+0.447213595499958*gamma_inv[3]*temp_sq[3]; 
  p_fac[15] = 0.31943828249997*gamma_inv[5]*temp_sq[15]+0.4472135954999579*gamma_inv[4]*temp_sq[15]+0.5*gamma_inv[0]*temp_sq[15]+0.31943828249997*gamma_inv[7]*temp_sq[13]+0.5000000000000001*gamma_inv[1]*temp_sq[13]+0.4*gamma_inv[3]*temp_sq[11]+0.4472135954999579*gamma_inv[7]*temp_sq[9]+1.0*GammaV_sq[1]*gamma[7]+0.4*gamma_inv[6]*temp_sq[7]+0.4472135954999579*gamma_inv[2]*temp_sq[7]+0.5000000000000001*temp_sq[1]*gamma_inv[7]+0.4472135954999579*gamma_inv[3]*temp_sq[5]+0.5*temp_sq[4]*gamma_inv[5]; 

  sr_pressure[0] += (0.3535533905932737*f[15]*p_fac[15]+0.3535533905932737*f[14]*p_fac[14]+0.3535533905932737*f[13]*p_fac[13]+0.3535533905932737*f[12]*p_fac[12]+0.3535533905932737*f[11]*p_fac[11]+0.3535533905932737*f[10]*p_fac[10]+0.3535533905932737*f[9]*p_fac[9]+0.3535533905932737*f[8]*p_fac[8]+0.3535533905932737*f[7]*p_fac[7]+0.3535533905932737*f[6]*p_fac[6]+0.3535533905932737*f[5]*p_fac[5]+0.3535533905932737*f[4]*p_fac[4]+0.3535533905932737*f[3]*p_fac[3]+0.3535533905932737*f[2]*p_fac[2]+0.3535533905932737*f[1]*p_fac[1]+0.3535533905932737*f[0]*p_fac[0])*volFact; 
  sr_pressure[1] += (0.3535533905932737*f[14]*p_fac[15]+0.3535533905932737*p_fac[14]*f[15]+0.3535533905932737*f[12]*p_fac[13]+0.3535533905932737*p_fac[12]*f[13]+0.3535533905932737*f[10]*p_fac[11]+0.3535533905932737*p_fac[10]*f[11]+0.3535533905932737*f[8]*p_fac[9]+0.3535533905932737*p_fac[8]*f[9]+0.3535533905932737*f[6]*p_fac[7]+0.3535533905932737*p_fac[6]*f[7]+0.3535533905932737*f[3]*p_fac[5]+0.3535533905932737*p_fac[3]*f[5]+0.3535533905932737*f[2]*p_fac[4]+0.3535533905932737*p_fac[2]*f[4]+0.3535533905932737*f[0]*p_fac[1]+0.3535533905932737*p_fac[0]*f[1])*volFact; 
} 
