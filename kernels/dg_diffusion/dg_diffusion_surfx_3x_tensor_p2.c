#include <gkyl_dg_diffusion_kernels.h> 
GKYL_CU_DH double dg_diffusion_surfx_3x_tensor_p2(const double* w, const double* dx, double D, 
  const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: Cell-center coordinates
  // dxv[NDIM]: Cell spacing
  // D: Diffusion coefficient in the center cell
  // ql: Input field in the left cell
  // qc: Input field in the center cell
  // qr: Input field in the right cell
  // out: Incremented output

  const double dx1 = 2.0/dx[0]; 
  const double J = pow(dx1, 2.0);

  const double *q0l = &ql[0]; 
  const double *q0c = &qc[0]; 
  const double *q0r = &qr[0]; 
  double *out0 = &out[0]; 

  out0[0] += J*D*(0.6708203932499369*q0r[7]+0.6708203932499369*q0l[7]-1.341640786499874*q0c[7]-1.190784930203603*q0r[1]+1.190784930203603*q0l[1]+0.9375*q0r[0]+0.9375*q0l[0]-1.875*q0c[0]); 
  out0[1] += J*D*(0.7382874503707888*q0r[7]-0.7382874503707888*q0l[7]-1.453125*q0r[1]-1.453125*q0l[1]-5.34375*q0c[1]+1.190784930203603*q0r[0]-1.190784930203603*q0l[0]); 
  out0[2] += J*D*(0.6708203932499369*q0r[11]+0.6708203932499369*q0l[11]-1.341640786499874*q0c[11]-1.190784930203603*q0r[4]+1.190784930203603*q0l[4]+0.9375*q0r[2]+0.9375*q0l[2]-1.875*q0c[2]); 
  out0[3] += J*D*(0.6708203932499369*q0r[13]+0.6708203932499369*q0l[13]-1.341640786499874*q0c[13]-1.190784930203603*q0r[5]+1.190784930203603*q0l[5]+0.9375*q0r[3]+0.9375*q0l[3]-1.875*q0c[3]); 
  out0[4] += J*D*(0.7382874503707888*q0r[11]-0.7382874503707888*q0l[11]-1.453125*q0r[4]-1.453125*q0l[4]-5.34375*q0c[4]+1.190784930203603*q0r[2]-1.190784930203603*q0l[2]); 
  out0[5] += J*D*(0.7382874503707888*q0r[13]-0.7382874503707888*q0l[13]-1.453125*q0r[5]-1.453125*q0l[5]-5.34375*q0c[5]+1.190784930203603*q0r[3]-1.190784930203603*q0l[3]); 
  out0[6] += J*D*(0.6708203932499369*q0r[17]+0.6708203932499369*q0l[17]-1.341640786499874*q0c[17]-1.190784930203603*q0r[10]+1.190784930203603*q0l[10]+0.9375*q0r[6]+0.9375*q0l[6]-1.875*q0c[6]); 
  out0[7] += J*D*((-0.140625*q0r[7])-0.140625*q0l[7]-6.28125*q0c[7]-0.3025768239224545*q0r[1]+0.3025768239224545*q0l[1]+0.4192627457812106*q0r[0]+0.4192627457812106*q0l[0]-0.8385254915624212*q0c[0]); 
  out0[8] += J*D*(0.6708203932499369*q0r[20]+0.6708203932499369*q0l[20]-1.341640786499874*q0c[20]-1.190784930203603*q0r[12]+1.190784930203603*q0l[12]+0.9375*q0r[8]+0.9375*q0l[8]-1.875*q0c[8]); 
  out0[9] += J*D*(0.6708203932499369*q0r[21]+0.6708203932499369*q0l[21]-1.341640786499874*q0c[21]-1.190784930203603*q0r[15]+1.190784930203603*q0l[15]+0.9375*q0r[9]+0.9375*q0l[9]-1.875*q0c[9]); 
  out0[10] += J*D*(0.7382874503707888*q0r[17]-0.7382874503707888*q0l[17]-1.453125*q0r[10]-1.453125*q0l[10]-5.34375*q0c[10]+1.190784930203603*q0r[6]-1.190784930203603*q0l[6]); 
  out0[11] += J*D*((-0.140625*q0r[11])-0.140625*q0l[11]-6.28125*q0c[11]-0.3025768239224544*q0r[4]+0.3025768239224544*q0l[4]+0.4192627457812105*q0r[2]+0.4192627457812105*q0l[2]-0.8385254915624211*q0c[2]); 
  out0[12] += J*D*(0.7382874503707888*q0r[20]-0.7382874503707888*q0l[20]-1.453125*q0r[12]-1.453125*q0l[12]-5.34375*q0c[12]+1.190784930203603*q0r[8]-1.190784930203603*q0l[8]); 
  out0[13] += J*D*((-0.140625*q0r[13])-0.140625*q0l[13]-6.28125*q0c[13]-0.3025768239224544*q0r[5]+0.3025768239224544*q0l[5]+0.4192627457812105*q0r[3]+0.4192627457812105*q0l[3]-0.8385254915624211*q0c[3]); 
  out0[14] += J*D*(0.6708203932499369*q0r[23]+0.6708203932499369*q0l[23]-1.341640786499874*q0c[23]-1.190784930203603*q0r[18]+1.190784930203603*q0l[18]+0.9375*q0r[14]+0.9375*q0l[14]-1.875*q0c[14]); 
  out0[15] += J*D*(0.7382874503707888*q0r[21]-0.7382874503707888*q0l[21]-1.453125*q0r[15]-1.453125*q0l[15]-5.34375*q0c[15]+1.190784930203603*q0r[9]-1.190784930203603*q0l[9]); 
  out0[16] += J*D*(0.6708203932499369*q0r[24]+0.6708203932499369*q0l[24]-1.341640786499874*q0c[24]-1.190784930203603*q0r[19]+1.190784930203603*q0l[19]+0.9375*q0r[16]+0.9375*q0l[16]-1.875*q0c[16]); 
  out0[17] += J*D*((-0.140625*q0r[17])-0.140625*q0l[17]-6.28125*q0c[17]-0.3025768239224545*q0r[10]+0.3025768239224545*q0l[10]+0.4192627457812106*q0r[6]+0.4192627457812106*q0l[6]-0.8385254915624212*q0c[6]); 
  out0[18] += J*D*(0.7382874503707888*q0r[23]-0.7382874503707888*q0l[23]-1.453125*q0r[18]-1.453125*q0l[18]-5.34375*q0c[18]+1.190784930203603*q0r[14]-1.190784930203603*q0l[14]); 
  out0[19] += J*D*(0.7382874503707888*q0r[24]-0.7382874503707888*q0l[24]-1.453125*q0r[19]-1.453125*q0l[19]-5.34375*q0c[19]+1.190784930203603*q0r[16]-1.190784930203603*q0l[16]); 
  out0[20] += J*D*((-0.140625*q0r[20])-0.140625*q0l[20]-6.28125*q0c[20]-0.3025768239224544*q0r[12]+0.3025768239224544*q0l[12]+0.4192627457812106*q0r[8]+0.4192627457812106*q0l[8]-0.8385254915624212*q0c[8]); 
  out0[21] += J*D*((-0.140625*q0r[21])-0.140625*q0l[21]-6.28125*q0c[21]-0.3025768239224544*q0r[15]+0.3025768239224544*q0l[15]+0.4192627457812106*q0r[9]+0.4192627457812106*q0l[9]-0.8385254915624212*q0c[9]); 
  out0[22] += J*D*(0.6708203932499369*q0r[26]+0.6708203932499369*q0l[26]-1.341640786499874*q0c[26]-1.190784930203603*q0r[25]+1.190784930203603*q0l[25]+0.9375*q0r[22]+0.9375*q0l[22]-1.875*q0c[22]); 
  out0[23] += J*D*((-0.140625*q0r[23])-0.140625*q0l[23]-6.28125*q0c[23]-0.3025768239224545*q0r[18]+0.3025768239224545*q0l[18]+0.4192627457812105*q0r[14]+0.4192627457812105*q0l[14]-0.8385254915624211*q0c[14]); 
  out0[24] += J*D*((-0.140625*q0r[24])-0.140625*q0l[24]-6.28125*q0c[24]-0.3025768239224545*q0r[19]+0.3025768239224545*q0l[19]+0.4192627457812105*q0r[16]+0.4192627457812105*q0l[16]-0.8385254915624211*q0c[16]); 
  out0[25] += J*D*(0.7382874503707888*q0r[26]-0.7382874503707888*q0l[26]-1.453125*q0r[25]-1.453125*q0l[25]-5.34375*q0c[25]+1.190784930203603*q0r[22]-1.190784930203603*q0l[22]); 
  out0[26] += J*D*((-0.140625*q0r[26])-0.140625*q0l[26]-6.28125*q0c[26]-0.3025768239224545*q0r[25]+0.3025768239224545*q0l[25]+0.4192627457812106*q0r[22]+0.4192627457812106*q0l[22]-0.8385254915624212*q0c[22]); 

  return 0.;

} 