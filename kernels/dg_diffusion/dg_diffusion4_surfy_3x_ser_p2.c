#include <gkyl_dg_diffusion_kernels.h> 
GKYL_CU_DH double dg_diffusion4_surfy_3x_ser_p2(const double* w, const double* dx, double D, 
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

  out0[0] += J*D*((-6.708203932499369*q0r[8])-6.708203932499369*q0l[8]+13.41640786499874*q0c[8]+8.11898816047911*q0r[2]-8.11898816047911*q0l[2]-4.6875*q0r[0]-4.6875*q0l[0]+9.375*q0c[0]); 
  out0[1] += J*D*((-6.708203932499369*q0r[12])-6.708203932499369*q0l[12]+13.41640786499874*q0c[12]+8.11898816047911*q0r[4]-8.11898816047911*q0l[4]-4.6875*q0r[1]-4.6875*q0l[1]+9.375*q0c[1]); 
  out0[2] += J*D*((-9.077304717673634*q0r[8])+9.077304717673634*q0l[8]+12.65625*q0r[2]+12.65625*q0l[2]+30.9375*q0c[2]-8.11898816047911*q0r[0]+8.11898816047911*q0l[0]); 
  out0[3] += J*D*((-6.708203932499369*q0r[14])-6.708203932499369*q0l[14]+13.41640786499874*q0c[14]+8.11898816047911*q0r[6]-8.11898816047911*q0l[6]-4.6875*q0r[3]-4.6875*q0l[3]+9.375*q0c[3]); 
  out0[4] += J*D*((-9.077304717673634*q0r[12])+9.077304717673634*q0l[12]+12.65625*q0r[4]+12.65625*q0l[4]+30.9375*q0c[4]-8.11898816047911*q0r[1]+8.11898816047911*q0l[1]); 
  out0[5] += J*D*((-6.708203932499369*q0r[18])-6.708203932499369*q0l[18]+13.41640786499874*q0c[18]+8.11898816047911*q0r[10]-8.11898816047911*q0l[10]-4.6875*q0r[5]-4.6875*q0l[5]+9.375*q0c[5]); 
  out0[6] += J*D*((-9.077304717673634*q0r[14])+9.077304717673634*q0l[14]+12.65625*q0r[6]+12.65625*q0l[6]+30.9375*q0c[6]-8.11898816047911*q0r[3]+8.11898816047911*q0l[3]); 
  out0[7] += J*D*(8.118988160479114*q0r[11]-8.118988160479114*q0l[11]-4.6875*q0r[7]-4.6875*q0l[7]+9.375*q0c[7]); 
  out0[8] += J*D*((-0.65625*q0r[8])-0.65625*q0l[8]+40.6875*q0c[8]+4.720198453190289*q0r[2]-4.720198453190289*q0l[2]-4.192627457812106*q0r[0]-4.192627457812106*q0l[0]+8.385254915624213*q0c[0]); 
  out0[9] += J*D*(8.118988160479114*q0r[16]-8.118988160479114*q0l[16]-4.6875*q0r[9]-4.6875*q0l[9]+9.375*q0c[9]); 
  out0[10] += J*D*((-9.077304717673634*q0r[18])+9.077304717673634*q0l[18]+12.65625*q0r[10]+12.65625*q0l[10]+30.9375*q0c[10]-8.11898816047911*q0r[5]+8.11898816047911*q0l[5]); 
  out0[11] += J*D*(12.65625*q0r[11]+12.65625*q0l[11]+30.9375*q0c[11]-8.118988160479114*q0r[7]+8.118988160479114*q0l[7]); 
  out0[12] += J*D*((-0.65625*q0r[12])-0.65625*q0l[12]+40.6875*q0c[12]+4.72019845319029*q0r[4]-4.72019845319029*q0l[4]-4.192627457812105*q0r[1]-4.192627457812105*q0l[1]+8.38525491562421*q0c[1]); 
  out0[13] += J*D*(8.118988160479114*q0r[17]-8.118988160479114*q0l[17]-4.6875*q0r[13]-4.6875*q0l[13]+9.375*q0c[13]); 
  out0[14] += J*D*((-0.65625*q0r[14])-0.65625*q0l[14]+40.6875*q0c[14]+4.72019845319029*q0r[6]-4.72019845319029*q0l[6]-4.192627457812105*q0r[3]-4.192627457812105*q0l[3]+8.38525491562421*q0c[3]); 
  out0[15] += J*D*(8.118988160479114*q0r[19]-8.118988160479114*q0l[19]-4.6875*q0r[15]-4.6875*q0l[15]+9.375*q0c[15]); 
  out0[16] += J*D*(12.65625*q0r[16]+12.65625*q0l[16]+30.9375*q0c[16]-8.118988160479114*q0r[9]+8.118988160479114*q0l[9]); 
  out0[17] += J*D*(12.65625*q0r[17]+12.65625*q0l[17]+30.9375*q0c[17]-8.118988160479114*q0r[13]+8.118988160479114*q0l[13]); 
  out0[18] += J*D*((-0.65625*q0r[18])-0.65625*q0l[18]+40.6875*q0c[18]+4.720198453190289*q0r[10]-4.720198453190289*q0l[10]-4.192627457812106*q0r[5]-4.192627457812106*q0l[5]+8.385254915624213*q0c[5]); 
  out0[19] += J*D*(12.65625*q0r[19]+12.65625*q0l[19]+30.9375*q0c[19]-8.118988160479114*q0r[15]+8.118988160479114*q0l[15]); 

  return 0.;

} 