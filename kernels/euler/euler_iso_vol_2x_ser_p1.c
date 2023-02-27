#include <gkyl_euler_kernels.h> 
GKYL_CU_DH double euler_iso_vol_2x_ser_p1(const double *w, const double *dxv, const double vth, const double *uvar, const double *statevec, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // vth: Thermal velocity.
  // uvar: [ux, uy, uz], Fluid flow.
  // statevec: [rho, rho ux, rho uy, rho uz].
  // out: Incremented output.

  const double *rho = &statevec[0]; 
  const double *rhou0 = &statevec[4]; 
  const double *rhou1 = &statevec[8]; 
  const double *rhou2 = &statevec[12]; 
  const double *uvar0 = &uvar[0]; 
  const double *uvar1 = &uvar[4]; 
  const double *uvar2 = &uvar[8]; 
  double *outrho = &out[0]; 
  double *outrhou0 = &out[4]; 
  double *outrhou1 = &out[8]; 
  double *outrhou2 = &out[12]; 
  double dx10 = 2./dxv[0]; 
  double dx11 = 2./dxv[1]; 

  double cflFreq_mid = 0.0; 
  double vthsq = vth*vth; 
  cflFreq_mid += 0.5*3.0*dx10*(fabs(0.5*uvar0[0])+vth); 
  cflFreq_mid += 0.5*3.0*dx11*(fabs(0.5*uvar1[0])+vth); 

  outrho[1] += 0.8660254037844386*rho[3]*rhou0[3]*dx10+0.8660254037844386*rho[2]*rhou0[2]*dx10+0.8660254037844386*rho[1]*rhou0[1]*dx10+0.8660254037844386*rho[0]*rhou0[0]*dx10; 
  outrho[2] += 0.8660254037844386*rho[3]*rhou1[3]*dx11+0.8660254037844386*rho[2]*rhou1[2]*dx11+0.8660254037844386*rho[1]*rhou1[1]*dx11+0.8660254037844386*rho[0]*rhou1[0]*dx11; 
  outrho[3] += 0.8660254037844386*rho[2]*rhou1[3]*dx11+0.8660254037844386*rhou1[2]*rho[3]*dx11+0.8660254037844386*rho[0]*rhou1[1]*dx11+0.8660254037844386*rhou1[0]*rho[1]*dx11+0.8660254037844386*rho[1]*rhou0[3]*dx10+0.8660254037844386*rhou0[1]*rho[3]*dx10+0.8660254037844386*rho[0]*rhou0[2]*dx10+0.8660254037844386*rhou0[0]*rho[2]*dx10; 

  outrhou0[1] += 1.732050807568877*rho[0]*dx10*vthsq+0.8660254037844386*rhou0[3]*uvar0[3]*dx10+0.8660254037844386*rhou0[2]*uvar0[2]*dx10+0.8660254037844386*rhou0[1]*uvar0[1]*dx10+0.8660254037844386*rhou0[0]*uvar0[0]*dx10; 
  outrhou0[2] += 0.8660254037844386*rhou0[3]*uvar1[3]*dx11+0.8660254037844386*rhou0[2]*uvar1[2]*dx11+0.8660254037844386*rhou0[1]*uvar1[1]*dx11+0.8660254037844386*rhou0[0]*uvar1[0]*dx11; 
  outrhou0[3] += 1.732050807568877*rho[2]*dx10*vthsq+0.8660254037844386*rhou0[2]*uvar1[3]*dx11+0.8660254037844386*uvar1[2]*rhou0[3]*dx11+0.8660254037844386*rhou0[0]*uvar1[1]*dx11+0.8660254037844386*uvar1[0]*rhou0[1]*dx11+0.8660254037844386*rhou0[1]*uvar0[3]*dx10+0.8660254037844386*uvar0[1]*rhou0[3]*dx10+0.8660254037844386*rhou0[0]*uvar0[2]*dx10+0.8660254037844386*uvar0[0]*rhou0[2]*dx10; 

  outrhou1[1] += 0.8660254037844386*rhou1[3]*uvar0[3]*dx10+0.8660254037844386*rhou1[2]*uvar0[2]*dx10+0.8660254037844386*rhou1[1]*uvar0[1]*dx10+0.8660254037844386*rhou1[0]*uvar0[0]*dx10; 
  outrhou1[2] += 1.732050807568877*rho[0]*dx11*vthsq+0.8660254037844386*rhou1[3]*uvar1[3]*dx11+0.8660254037844386*rhou1[2]*uvar1[2]*dx11+0.8660254037844386*rhou1[1]*uvar1[1]*dx11+0.8660254037844386*rhou1[0]*uvar1[0]*dx11; 
  outrhou1[3] += 1.732050807568877*rho[1]*dx11*vthsq+0.8660254037844386*rhou1[2]*uvar1[3]*dx11+0.8660254037844386*uvar1[2]*rhou1[3]*dx11+0.8660254037844386*rhou1[0]*uvar1[1]*dx11+0.8660254037844386*uvar1[0]*rhou1[1]*dx11+0.8660254037844386*rhou1[1]*uvar0[3]*dx10+0.8660254037844386*uvar0[1]*rhou1[3]*dx10+0.8660254037844386*rhou1[0]*uvar0[2]*dx10+0.8660254037844386*uvar0[0]*rhou1[2]*dx10; 

  outrhou2[1] += 0.8660254037844386*rhou2[3]*uvar0[3]*dx10+0.8660254037844386*rhou2[2]*uvar0[2]*dx10+0.8660254037844386*rhou2[1]*uvar0[1]*dx10+0.8660254037844386*rhou2[0]*uvar0[0]*dx10; 
  outrhou2[2] += 0.8660254037844386*rhou2[3]*uvar1[3]*dx11+0.8660254037844386*rhou2[2]*uvar1[2]*dx11+0.8660254037844386*rhou2[1]*uvar1[1]*dx11+0.8660254037844386*rhou2[0]*uvar1[0]*dx11; 
  outrhou2[3] += 0.8660254037844386*rhou2[2]*uvar1[3]*dx11+0.8660254037844386*uvar1[2]*rhou2[3]*dx11+0.8660254037844386*rhou2[0]*uvar1[1]*dx11+0.8660254037844386*uvar1[0]*rhou2[1]*dx11+0.8660254037844386*rhou2[1]*uvar0[3]*dx10+0.8660254037844386*uvar0[1]*rhou2[3]*dx10+0.8660254037844386*rhou2[0]*uvar0[2]*dx10+0.8660254037844386*uvar0[0]*rhou2[2]*dx10; 

  return cflFreq_mid; 
} 