#include <gkyl_dg_diffusion_kernels.h> 
GKYL_CU_DH double dg_diffusion4_pkpm_surfy_2x_ser_p2(const double* w, const double* dx, double D, 
  const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: Cell-center coordinates
  // dxv[NDIM]: Cell spacing
  // D: Diffusion coefficient in the center cell
  // ql: Input field in the left cell
  // qc: Input field in the center cell
  // qr: Input field in the right cell
  // out: Incremented output

  const double dx1 = 2.0/dx[1]; 
  const double J = -1.0*pow(dx1, 4.0);

  const double *q0l = &ql[0]; 
  const double *q0c = &qc[0]; 
  const double *q0r = &qr[0]; 
  double *out0 = &out[0]; 

  out0[0] += J*D*((-6.708203932499369*q0r[5])-6.708203932499369*q0l[5]+13.41640786499874*q0c[5]+8.11898816047911*q0r[2]-8.11898816047911*q0l[2]-4.6875*q0r[0]-4.6875*q0l[0]+9.375*q0c[0]); 
  out0[1] += J*D*((-6.708203932499369*q0r[7])-6.708203932499369*q0l[7]+13.41640786499874*q0c[7]+8.11898816047911*q0r[3]-8.11898816047911*q0l[3]-4.6875*q0r[1]-4.6875*q0l[1]+9.375*q0c[1]); 
  out0[2] += J*D*((-9.077304717673634*q0r[5])+9.077304717673634*q0l[5]+12.65625*q0r[2]+12.65625*q0l[2]+30.9375*q0c[2]-8.11898816047911*q0r[0]+8.11898816047911*q0l[0]); 
  out0[3] += J*D*((-9.077304717673634*q0r[7])+9.077304717673634*q0l[7]+12.65625*q0r[3]+12.65625*q0l[3]+30.9375*q0c[3]-8.11898816047911*q0r[1]+8.11898816047911*q0l[1]); 
  out0[4] += J*D*(8.118988160479114*q0r[6]-8.118988160479114*q0l[6]-4.6875*q0r[4]-4.6875*q0l[4]+9.375*q0c[4]); 
  out0[5] += J*D*((-0.65625*q0r[5])-0.65625*q0l[5]+40.6875*q0c[5]+4.720198453190289*q0r[2]-4.720198453190289*q0l[2]-4.192627457812106*q0r[0]-4.192627457812106*q0l[0]+8.385254915624213*q0c[0]); 
  out0[6] += J*D*(12.65625*q0r[6]+12.65625*q0l[6]+30.9375*q0c[6]-8.118988160479114*q0r[4]+8.118988160479114*q0l[4]); 
  out0[7] += J*D*((-0.65625*q0r[7])-0.65625*q0l[7]+40.6875*q0c[7]+4.72019845319029*q0r[3]-4.72019845319029*q0l[3]-4.192627457812105*q0r[1]-4.192627457812105*q0l[1]+8.38525491562421*q0c[1]); 

  const double *q1l = &ql[8]; 
  const double *q1c = &qc[8]; 
  const double *q1r = &qr[8]; 
  double *out1 = &out[8]; 

  out1[0] += J*D*((-6.708203932499369*q1r[5])-6.708203932499369*q1l[5]+13.41640786499874*q1c[5]+8.11898816047911*q1r[2]-8.11898816047911*q1l[2]-4.6875*q1r[0]-4.6875*q1l[0]+9.375*q1c[0]); 
  out1[1] += J*D*((-6.708203932499369*q1r[7])-6.708203932499369*q1l[7]+13.41640786499874*q1c[7]+8.11898816047911*q1r[3]-8.11898816047911*q1l[3]-4.6875*q1r[1]-4.6875*q1l[1]+9.375*q1c[1]); 
  out1[2] += J*D*((-9.077304717673634*q1r[5])+9.077304717673634*q1l[5]+12.65625*q1r[2]+12.65625*q1l[2]+30.9375*q1c[2]-8.11898816047911*q1r[0]+8.11898816047911*q1l[0]); 
  out1[3] += J*D*((-9.077304717673634*q1r[7])+9.077304717673634*q1l[7]+12.65625*q1r[3]+12.65625*q1l[3]+30.9375*q1c[3]-8.11898816047911*q1r[1]+8.11898816047911*q1l[1]); 
  out1[4] += J*D*(8.118988160479114*q1r[6]-8.118988160479114*q1l[6]-4.6875*q1r[4]-4.6875*q1l[4]+9.375*q1c[4]); 
  out1[5] += J*D*((-0.65625*q1r[5])-0.65625*q1l[5]+40.6875*q1c[5]+4.720198453190289*q1r[2]-4.720198453190289*q1l[2]-4.192627457812106*q1r[0]-4.192627457812106*q1l[0]+8.385254915624213*q1c[0]); 
  out1[6] += J*D*(12.65625*q1r[6]+12.65625*q1l[6]+30.9375*q1c[6]-8.118988160479114*q1r[4]+8.118988160479114*q1l[4]); 
  out1[7] += J*D*((-0.65625*q1r[7])-0.65625*q1l[7]+40.6875*q1c[7]+4.72019845319029*q1r[3]-4.72019845319029*q1l[3]-4.192627457812105*q1r[1]-4.192627457812105*q1l[1]+8.38525491562421*q1c[1]); 

  const double *q2l = &ql[16]; 
  const double *q2c = &qc[16]; 
  const double *q2r = &qr[16]; 
  double *out2 = &out[16]; 

  out2[0] += J*D*((-6.708203932499369*q2r[5])-6.708203932499369*q2l[5]+13.41640786499874*q2c[5]+8.11898816047911*q2r[2]-8.11898816047911*q2l[2]-4.6875*q2r[0]-4.6875*q2l[0]+9.375*q2c[0]); 
  out2[1] += J*D*((-6.708203932499369*q2r[7])-6.708203932499369*q2l[7]+13.41640786499874*q2c[7]+8.11898816047911*q2r[3]-8.11898816047911*q2l[3]-4.6875*q2r[1]-4.6875*q2l[1]+9.375*q2c[1]); 
  out2[2] += J*D*((-9.077304717673634*q2r[5])+9.077304717673634*q2l[5]+12.65625*q2r[2]+12.65625*q2l[2]+30.9375*q2c[2]-8.11898816047911*q2r[0]+8.11898816047911*q2l[0]); 
  out2[3] += J*D*((-9.077304717673634*q2r[7])+9.077304717673634*q2l[7]+12.65625*q2r[3]+12.65625*q2l[3]+30.9375*q2c[3]-8.11898816047911*q2r[1]+8.11898816047911*q2l[1]); 
  out2[4] += J*D*(8.118988160479114*q2r[6]-8.118988160479114*q2l[6]-4.6875*q2r[4]-4.6875*q2l[4]+9.375*q2c[4]); 
  out2[5] += J*D*((-0.65625*q2r[5])-0.65625*q2l[5]+40.6875*q2c[5]+4.720198453190289*q2r[2]-4.720198453190289*q2l[2]-4.192627457812106*q2r[0]-4.192627457812106*q2l[0]+8.385254915624213*q2c[0]); 
  out2[6] += J*D*(12.65625*q2r[6]+12.65625*q2l[6]+30.9375*q2c[6]-8.118988160479114*q2r[4]+8.118988160479114*q2l[4]); 
  out2[7] += J*D*((-0.65625*q2r[7])-0.65625*q2l[7]+40.6875*q2c[7]+4.72019845319029*q2r[3]-4.72019845319029*q2l[3]-4.192627457812105*q2r[1]-4.192627457812105*q2l[1]+8.38525491562421*q2c[1]); 

  return 0.;

} 