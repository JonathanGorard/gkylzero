#include <gkyl_dg_diffusion_kernels.h> 
GKYL_CU_DH double dg_diffusion_iso_euler_surfz_3x_tensor_p2(const double* w, const double* dx, double D, 
  const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: Cell-center coordinates
  // dxv[NDIM]: Cell spacing
  // D: Diffusion coefficient in the center cell
  // ql: Input field in the left cell
  // qc: Input field in the center cell
  // qr: Input field in the right cell
  // out: Incremented output

  const double dx1 = 2.0/dx[2]; 
  const double J = pow(dx1, 2.0);

  const double *q0l = &ql[0]; 
  const double *q0c = &qc[0]; 
  const double *q0r = &qr[0]; 
  double *out0 = &out[0]; 

  out0[0] += J*D*(0.6708203932499369*q0r[9]+0.6708203932499369*q0l[9]-1.341640786499874*q0c[9]-1.190784930203603*q0r[3]+1.190784930203603*q0l[3]+0.9375*q0r[0]+0.9375*q0l[0]-1.875*q0c[0]); 
  out0[1] += J*D*(0.6708203932499369*q0r[15]+0.6708203932499369*q0l[15]-1.341640786499874*q0c[15]-1.190784930203603*q0r[5]+1.190784930203603*q0l[5]+0.9375*q0r[1]+0.9375*q0l[1]-1.875*q0c[1]); 
  out0[2] += J*D*(0.6708203932499369*q0r[16]+0.6708203932499369*q0l[16]-1.341640786499874*q0c[16]-1.190784930203603*q0r[6]+1.190784930203603*q0l[6]+0.9375*q0r[2]+0.9375*q0l[2]-1.875*q0c[2]); 
  out0[3] += J*D*(0.7382874503707888*q0r[9]-0.7382874503707888*q0l[9]-1.453125*q0r[3]-1.453125*q0l[3]-5.34375*q0c[3]+1.190784930203603*q0r[0]-1.190784930203603*q0l[0]); 
  out0[4] += J*D*(0.6708203932499369*q0r[19]+0.6708203932499369*q0l[19]-1.341640786499874*q0c[19]-1.190784930203603*q0r[10]+1.190784930203603*q0l[10]+0.9375*q0r[4]+0.9375*q0l[4]-1.875*q0c[4]); 
  out0[5] += J*D*(0.7382874503707888*q0r[15]-0.7382874503707888*q0l[15]-1.453125*q0r[5]-1.453125*q0l[5]-5.34375*q0c[5]+1.190784930203603*q0r[1]-1.190784930203603*q0l[1]); 
  out0[6] += J*D*(0.7382874503707888*q0r[16]-0.7382874503707888*q0l[16]-1.453125*q0r[6]-1.453125*q0l[6]-5.34375*q0c[6]+1.190784930203603*q0r[2]-1.190784930203603*q0l[2]); 
  out0[7] += J*D*(0.6708203932499369*q0r[21]+0.6708203932499369*q0l[21]-1.341640786499874*q0c[21]-1.190784930203603*q0r[13]+1.190784930203603*q0l[13]+0.9375*q0r[7]+0.9375*q0l[7]-1.875*q0c[7]); 
  out0[8] += J*D*(0.6708203932499369*q0r[22]+0.6708203932499369*q0l[22]-1.341640786499874*q0c[22]-1.190784930203603*q0r[14]+1.190784930203603*q0l[14]+0.9375*q0r[8]+0.9375*q0l[8]-1.875*q0c[8]); 
  out0[9] += J*D*((-0.140625*q0r[9])-0.140625*q0l[9]-6.28125*q0c[9]-0.3025768239224545*q0r[3]+0.3025768239224545*q0l[3]+0.4192627457812106*q0r[0]+0.4192627457812106*q0l[0]-0.8385254915624212*q0c[0]); 
  out0[10] += J*D*(0.7382874503707888*q0r[19]-0.7382874503707888*q0l[19]-1.453125*q0r[10]-1.453125*q0l[10]-5.34375*q0c[10]+1.190784930203603*q0r[4]-1.190784930203603*q0l[4]); 
  out0[11] += J*D*(0.6708203932499369*q0r[24]+0.6708203932499369*q0l[24]-1.341640786499874*q0c[24]-1.190784930203603*q0r[17]+1.190784930203603*q0l[17]+0.9375*q0r[11]+0.9375*q0l[11]-1.875*q0c[11]); 
  out0[12] += J*D*(0.6708203932499369*q0r[25]+0.6708203932499369*q0l[25]-1.341640786499874*q0c[25]-1.190784930203603*q0r[18]+1.190784930203603*q0l[18]+0.9375*q0r[12]+0.9375*q0l[12]-1.875*q0c[12]); 
  out0[13] += J*D*(0.7382874503707888*q0r[21]-0.7382874503707888*q0l[21]-1.453125*q0r[13]-1.453125*q0l[13]-5.34375*q0c[13]+1.190784930203603*q0r[7]-1.190784930203603*q0l[7]); 
  out0[14] += J*D*(0.7382874503707888*q0r[22]-0.7382874503707888*q0l[22]-1.453125*q0r[14]-1.453125*q0l[14]-5.34375*q0c[14]+1.190784930203603*q0r[8]-1.190784930203603*q0l[8]); 
  out0[15] += J*D*((-0.140625*q0r[15])-0.140625*q0l[15]-6.28125*q0c[15]-0.3025768239224544*q0r[5]+0.3025768239224544*q0l[5]+0.4192627457812105*q0r[1]+0.4192627457812105*q0l[1]-0.8385254915624211*q0c[1]); 
  out0[16] += J*D*((-0.140625*q0r[16])-0.140625*q0l[16]-6.28125*q0c[16]-0.3025768239224544*q0r[6]+0.3025768239224544*q0l[6]+0.4192627457812105*q0r[2]+0.4192627457812105*q0l[2]-0.8385254915624211*q0c[2]); 
  out0[17] += J*D*(0.7382874503707888*q0r[24]-0.7382874503707888*q0l[24]-1.453125*q0r[17]-1.453125*q0l[17]-5.34375*q0c[17]+1.190784930203603*q0r[11]-1.190784930203603*q0l[11]); 
  out0[18] += J*D*(0.7382874503707888*q0r[25]-0.7382874503707888*q0l[25]-1.453125*q0r[18]-1.453125*q0l[18]-5.34375*q0c[18]+1.190784930203603*q0r[12]-1.190784930203603*q0l[12]); 
  out0[19] += J*D*((-0.140625*q0r[19])-0.140625*q0l[19]-6.28125*q0c[19]-0.3025768239224545*q0r[10]+0.3025768239224545*q0l[10]+0.4192627457812106*q0r[4]+0.4192627457812106*q0l[4]-0.8385254915624212*q0c[4]); 
  out0[20] += J*D*(0.6708203932499369*q0r[26]+0.6708203932499369*q0l[26]-1.341640786499874*q0c[26]-1.190784930203603*q0r[23]+1.190784930203603*q0l[23]+0.9375*q0r[20]+0.9375*q0l[20]-1.875*q0c[20]); 
  out0[21] += J*D*((-0.140625*q0r[21])-0.140625*q0l[21]-6.28125*q0c[21]-0.3025768239224544*q0r[13]+0.3025768239224544*q0l[13]+0.4192627457812106*q0r[7]+0.4192627457812106*q0l[7]-0.8385254915624212*q0c[7]); 
  out0[22] += J*D*((-0.140625*q0r[22])-0.140625*q0l[22]-6.28125*q0c[22]-0.3025768239224544*q0r[14]+0.3025768239224544*q0l[14]+0.4192627457812106*q0r[8]+0.4192627457812106*q0l[8]-0.8385254915624212*q0c[8]); 
  out0[23] += J*D*(0.7382874503707888*q0r[26]-0.7382874503707888*q0l[26]-1.453125*q0r[23]-1.453125*q0l[23]-5.34375*q0c[23]+1.190784930203603*q0r[20]-1.190784930203603*q0l[20]); 
  out0[24] += J*D*((-0.140625*q0r[24])-0.140625*q0l[24]-6.28125*q0c[24]-0.3025768239224545*q0r[17]+0.3025768239224545*q0l[17]+0.4192627457812105*q0r[11]+0.4192627457812105*q0l[11]-0.8385254915624211*q0c[11]); 
  out0[25] += J*D*((-0.140625*q0r[25])-0.140625*q0l[25]-6.28125*q0c[25]-0.3025768239224545*q0r[18]+0.3025768239224545*q0l[18]+0.4192627457812105*q0r[12]+0.4192627457812105*q0l[12]-0.8385254915624211*q0c[12]); 
  out0[26] += J*D*((-0.140625*q0r[26])-0.140625*q0l[26]-6.28125*q0c[26]-0.3025768239224545*q0r[23]+0.3025768239224545*q0l[23]+0.4192627457812106*q0r[20]+0.4192627457812106*q0l[20]-0.8385254915624212*q0c[20]); 

  const double *q1l = &ql[27]; 
  const double *q1c = &qc[27]; 
  const double *q1r = &qr[27]; 
  double *out1 = &out[27]; 

  out1[0] += J*D*(0.6708203932499369*q1r[9]+0.6708203932499369*q1l[9]-1.341640786499874*q1c[9]-1.190784930203603*q1r[3]+1.190784930203603*q1l[3]+0.9375*q1r[0]+0.9375*q1l[0]-1.875*q1c[0]); 
  out1[1] += J*D*(0.6708203932499369*q1r[15]+0.6708203932499369*q1l[15]-1.341640786499874*q1c[15]-1.190784930203603*q1r[5]+1.190784930203603*q1l[5]+0.9375*q1r[1]+0.9375*q1l[1]-1.875*q1c[1]); 
  out1[2] += J*D*(0.6708203932499369*q1r[16]+0.6708203932499369*q1l[16]-1.341640786499874*q1c[16]-1.190784930203603*q1r[6]+1.190784930203603*q1l[6]+0.9375*q1r[2]+0.9375*q1l[2]-1.875*q1c[2]); 
  out1[3] += J*D*(0.7382874503707888*q1r[9]-0.7382874503707888*q1l[9]-1.453125*q1r[3]-1.453125*q1l[3]-5.34375*q1c[3]+1.190784930203603*q1r[0]-1.190784930203603*q1l[0]); 
  out1[4] += J*D*(0.6708203932499369*q1r[19]+0.6708203932499369*q1l[19]-1.341640786499874*q1c[19]-1.190784930203603*q1r[10]+1.190784930203603*q1l[10]+0.9375*q1r[4]+0.9375*q1l[4]-1.875*q1c[4]); 
  out1[5] += J*D*(0.7382874503707888*q1r[15]-0.7382874503707888*q1l[15]-1.453125*q1r[5]-1.453125*q1l[5]-5.34375*q1c[5]+1.190784930203603*q1r[1]-1.190784930203603*q1l[1]); 
  out1[6] += J*D*(0.7382874503707888*q1r[16]-0.7382874503707888*q1l[16]-1.453125*q1r[6]-1.453125*q1l[6]-5.34375*q1c[6]+1.190784930203603*q1r[2]-1.190784930203603*q1l[2]); 
  out1[7] += J*D*(0.6708203932499369*q1r[21]+0.6708203932499369*q1l[21]-1.341640786499874*q1c[21]-1.190784930203603*q1r[13]+1.190784930203603*q1l[13]+0.9375*q1r[7]+0.9375*q1l[7]-1.875*q1c[7]); 
  out1[8] += J*D*(0.6708203932499369*q1r[22]+0.6708203932499369*q1l[22]-1.341640786499874*q1c[22]-1.190784930203603*q1r[14]+1.190784930203603*q1l[14]+0.9375*q1r[8]+0.9375*q1l[8]-1.875*q1c[8]); 
  out1[9] += J*D*((-0.140625*q1r[9])-0.140625*q1l[9]-6.28125*q1c[9]-0.3025768239224545*q1r[3]+0.3025768239224545*q1l[3]+0.4192627457812106*q1r[0]+0.4192627457812106*q1l[0]-0.8385254915624212*q1c[0]); 
  out1[10] += J*D*(0.7382874503707888*q1r[19]-0.7382874503707888*q1l[19]-1.453125*q1r[10]-1.453125*q1l[10]-5.34375*q1c[10]+1.190784930203603*q1r[4]-1.190784930203603*q1l[4]); 
  out1[11] += J*D*(0.6708203932499369*q1r[24]+0.6708203932499369*q1l[24]-1.341640786499874*q1c[24]-1.190784930203603*q1r[17]+1.190784930203603*q1l[17]+0.9375*q1r[11]+0.9375*q1l[11]-1.875*q1c[11]); 
  out1[12] += J*D*(0.6708203932499369*q1r[25]+0.6708203932499369*q1l[25]-1.341640786499874*q1c[25]-1.190784930203603*q1r[18]+1.190784930203603*q1l[18]+0.9375*q1r[12]+0.9375*q1l[12]-1.875*q1c[12]); 
  out1[13] += J*D*(0.7382874503707888*q1r[21]-0.7382874503707888*q1l[21]-1.453125*q1r[13]-1.453125*q1l[13]-5.34375*q1c[13]+1.190784930203603*q1r[7]-1.190784930203603*q1l[7]); 
  out1[14] += J*D*(0.7382874503707888*q1r[22]-0.7382874503707888*q1l[22]-1.453125*q1r[14]-1.453125*q1l[14]-5.34375*q1c[14]+1.190784930203603*q1r[8]-1.190784930203603*q1l[8]); 
  out1[15] += J*D*((-0.140625*q1r[15])-0.140625*q1l[15]-6.28125*q1c[15]-0.3025768239224544*q1r[5]+0.3025768239224544*q1l[5]+0.4192627457812105*q1r[1]+0.4192627457812105*q1l[1]-0.8385254915624211*q1c[1]); 
  out1[16] += J*D*((-0.140625*q1r[16])-0.140625*q1l[16]-6.28125*q1c[16]-0.3025768239224544*q1r[6]+0.3025768239224544*q1l[6]+0.4192627457812105*q1r[2]+0.4192627457812105*q1l[2]-0.8385254915624211*q1c[2]); 
  out1[17] += J*D*(0.7382874503707888*q1r[24]-0.7382874503707888*q1l[24]-1.453125*q1r[17]-1.453125*q1l[17]-5.34375*q1c[17]+1.190784930203603*q1r[11]-1.190784930203603*q1l[11]); 
  out1[18] += J*D*(0.7382874503707888*q1r[25]-0.7382874503707888*q1l[25]-1.453125*q1r[18]-1.453125*q1l[18]-5.34375*q1c[18]+1.190784930203603*q1r[12]-1.190784930203603*q1l[12]); 
  out1[19] += J*D*((-0.140625*q1r[19])-0.140625*q1l[19]-6.28125*q1c[19]-0.3025768239224545*q1r[10]+0.3025768239224545*q1l[10]+0.4192627457812106*q1r[4]+0.4192627457812106*q1l[4]-0.8385254915624212*q1c[4]); 
  out1[20] += J*D*(0.6708203932499369*q1r[26]+0.6708203932499369*q1l[26]-1.341640786499874*q1c[26]-1.190784930203603*q1r[23]+1.190784930203603*q1l[23]+0.9375*q1r[20]+0.9375*q1l[20]-1.875*q1c[20]); 
  out1[21] += J*D*((-0.140625*q1r[21])-0.140625*q1l[21]-6.28125*q1c[21]-0.3025768239224544*q1r[13]+0.3025768239224544*q1l[13]+0.4192627457812106*q1r[7]+0.4192627457812106*q1l[7]-0.8385254915624212*q1c[7]); 
  out1[22] += J*D*((-0.140625*q1r[22])-0.140625*q1l[22]-6.28125*q1c[22]-0.3025768239224544*q1r[14]+0.3025768239224544*q1l[14]+0.4192627457812106*q1r[8]+0.4192627457812106*q1l[8]-0.8385254915624212*q1c[8]); 
  out1[23] += J*D*(0.7382874503707888*q1r[26]-0.7382874503707888*q1l[26]-1.453125*q1r[23]-1.453125*q1l[23]-5.34375*q1c[23]+1.190784930203603*q1r[20]-1.190784930203603*q1l[20]); 
  out1[24] += J*D*((-0.140625*q1r[24])-0.140625*q1l[24]-6.28125*q1c[24]-0.3025768239224545*q1r[17]+0.3025768239224545*q1l[17]+0.4192627457812105*q1r[11]+0.4192627457812105*q1l[11]-0.8385254915624211*q1c[11]); 
  out1[25] += J*D*((-0.140625*q1r[25])-0.140625*q1l[25]-6.28125*q1c[25]-0.3025768239224545*q1r[18]+0.3025768239224545*q1l[18]+0.4192627457812105*q1r[12]+0.4192627457812105*q1l[12]-0.8385254915624211*q1c[12]); 
  out1[26] += J*D*((-0.140625*q1r[26])-0.140625*q1l[26]-6.28125*q1c[26]-0.3025768239224545*q1r[23]+0.3025768239224545*q1l[23]+0.4192627457812106*q1r[20]+0.4192627457812106*q1l[20]-0.8385254915624212*q1c[20]); 

  const double *q2l = &ql[54]; 
  const double *q2c = &qc[54]; 
  const double *q2r = &qr[54]; 
  double *out2 = &out[54]; 

  out2[0] += J*D*(0.6708203932499369*q2r[9]+0.6708203932499369*q2l[9]-1.341640786499874*q2c[9]-1.190784930203603*q2r[3]+1.190784930203603*q2l[3]+0.9375*q2r[0]+0.9375*q2l[0]-1.875*q2c[0]); 
  out2[1] += J*D*(0.6708203932499369*q2r[15]+0.6708203932499369*q2l[15]-1.341640786499874*q2c[15]-1.190784930203603*q2r[5]+1.190784930203603*q2l[5]+0.9375*q2r[1]+0.9375*q2l[1]-1.875*q2c[1]); 
  out2[2] += J*D*(0.6708203932499369*q2r[16]+0.6708203932499369*q2l[16]-1.341640786499874*q2c[16]-1.190784930203603*q2r[6]+1.190784930203603*q2l[6]+0.9375*q2r[2]+0.9375*q2l[2]-1.875*q2c[2]); 
  out2[3] += J*D*(0.7382874503707888*q2r[9]-0.7382874503707888*q2l[9]-1.453125*q2r[3]-1.453125*q2l[3]-5.34375*q2c[3]+1.190784930203603*q2r[0]-1.190784930203603*q2l[0]); 
  out2[4] += J*D*(0.6708203932499369*q2r[19]+0.6708203932499369*q2l[19]-1.341640786499874*q2c[19]-1.190784930203603*q2r[10]+1.190784930203603*q2l[10]+0.9375*q2r[4]+0.9375*q2l[4]-1.875*q2c[4]); 
  out2[5] += J*D*(0.7382874503707888*q2r[15]-0.7382874503707888*q2l[15]-1.453125*q2r[5]-1.453125*q2l[5]-5.34375*q2c[5]+1.190784930203603*q2r[1]-1.190784930203603*q2l[1]); 
  out2[6] += J*D*(0.7382874503707888*q2r[16]-0.7382874503707888*q2l[16]-1.453125*q2r[6]-1.453125*q2l[6]-5.34375*q2c[6]+1.190784930203603*q2r[2]-1.190784930203603*q2l[2]); 
  out2[7] += J*D*(0.6708203932499369*q2r[21]+0.6708203932499369*q2l[21]-1.341640786499874*q2c[21]-1.190784930203603*q2r[13]+1.190784930203603*q2l[13]+0.9375*q2r[7]+0.9375*q2l[7]-1.875*q2c[7]); 
  out2[8] += J*D*(0.6708203932499369*q2r[22]+0.6708203932499369*q2l[22]-1.341640786499874*q2c[22]-1.190784930203603*q2r[14]+1.190784930203603*q2l[14]+0.9375*q2r[8]+0.9375*q2l[8]-1.875*q2c[8]); 
  out2[9] += J*D*((-0.140625*q2r[9])-0.140625*q2l[9]-6.28125*q2c[9]-0.3025768239224545*q2r[3]+0.3025768239224545*q2l[3]+0.4192627457812106*q2r[0]+0.4192627457812106*q2l[0]-0.8385254915624212*q2c[0]); 
  out2[10] += J*D*(0.7382874503707888*q2r[19]-0.7382874503707888*q2l[19]-1.453125*q2r[10]-1.453125*q2l[10]-5.34375*q2c[10]+1.190784930203603*q2r[4]-1.190784930203603*q2l[4]); 
  out2[11] += J*D*(0.6708203932499369*q2r[24]+0.6708203932499369*q2l[24]-1.341640786499874*q2c[24]-1.190784930203603*q2r[17]+1.190784930203603*q2l[17]+0.9375*q2r[11]+0.9375*q2l[11]-1.875*q2c[11]); 
  out2[12] += J*D*(0.6708203932499369*q2r[25]+0.6708203932499369*q2l[25]-1.341640786499874*q2c[25]-1.190784930203603*q2r[18]+1.190784930203603*q2l[18]+0.9375*q2r[12]+0.9375*q2l[12]-1.875*q2c[12]); 
  out2[13] += J*D*(0.7382874503707888*q2r[21]-0.7382874503707888*q2l[21]-1.453125*q2r[13]-1.453125*q2l[13]-5.34375*q2c[13]+1.190784930203603*q2r[7]-1.190784930203603*q2l[7]); 
  out2[14] += J*D*(0.7382874503707888*q2r[22]-0.7382874503707888*q2l[22]-1.453125*q2r[14]-1.453125*q2l[14]-5.34375*q2c[14]+1.190784930203603*q2r[8]-1.190784930203603*q2l[8]); 
  out2[15] += J*D*((-0.140625*q2r[15])-0.140625*q2l[15]-6.28125*q2c[15]-0.3025768239224544*q2r[5]+0.3025768239224544*q2l[5]+0.4192627457812105*q2r[1]+0.4192627457812105*q2l[1]-0.8385254915624211*q2c[1]); 
  out2[16] += J*D*((-0.140625*q2r[16])-0.140625*q2l[16]-6.28125*q2c[16]-0.3025768239224544*q2r[6]+0.3025768239224544*q2l[6]+0.4192627457812105*q2r[2]+0.4192627457812105*q2l[2]-0.8385254915624211*q2c[2]); 
  out2[17] += J*D*(0.7382874503707888*q2r[24]-0.7382874503707888*q2l[24]-1.453125*q2r[17]-1.453125*q2l[17]-5.34375*q2c[17]+1.190784930203603*q2r[11]-1.190784930203603*q2l[11]); 
  out2[18] += J*D*(0.7382874503707888*q2r[25]-0.7382874503707888*q2l[25]-1.453125*q2r[18]-1.453125*q2l[18]-5.34375*q2c[18]+1.190784930203603*q2r[12]-1.190784930203603*q2l[12]); 
  out2[19] += J*D*((-0.140625*q2r[19])-0.140625*q2l[19]-6.28125*q2c[19]-0.3025768239224545*q2r[10]+0.3025768239224545*q2l[10]+0.4192627457812106*q2r[4]+0.4192627457812106*q2l[4]-0.8385254915624212*q2c[4]); 
  out2[20] += J*D*(0.6708203932499369*q2r[26]+0.6708203932499369*q2l[26]-1.341640786499874*q2c[26]-1.190784930203603*q2r[23]+1.190784930203603*q2l[23]+0.9375*q2r[20]+0.9375*q2l[20]-1.875*q2c[20]); 
  out2[21] += J*D*((-0.140625*q2r[21])-0.140625*q2l[21]-6.28125*q2c[21]-0.3025768239224544*q2r[13]+0.3025768239224544*q2l[13]+0.4192627457812106*q2r[7]+0.4192627457812106*q2l[7]-0.8385254915624212*q2c[7]); 
  out2[22] += J*D*((-0.140625*q2r[22])-0.140625*q2l[22]-6.28125*q2c[22]-0.3025768239224544*q2r[14]+0.3025768239224544*q2l[14]+0.4192627457812106*q2r[8]+0.4192627457812106*q2l[8]-0.8385254915624212*q2c[8]); 
  out2[23] += J*D*(0.7382874503707888*q2r[26]-0.7382874503707888*q2l[26]-1.453125*q2r[23]-1.453125*q2l[23]-5.34375*q2c[23]+1.190784930203603*q2r[20]-1.190784930203603*q2l[20]); 
  out2[24] += J*D*((-0.140625*q2r[24])-0.140625*q2l[24]-6.28125*q2c[24]-0.3025768239224545*q2r[17]+0.3025768239224545*q2l[17]+0.4192627457812105*q2r[11]+0.4192627457812105*q2l[11]-0.8385254915624211*q2c[11]); 
  out2[25] += J*D*((-0.140625*q2r[25])-0.140625*q2l[25]-6.28125*q2c[25]-0.3025768239224545*q2r[18]+0.3025768239224545*q2l[18]+0.4192627457812105*q2r[12]+0.4192627457812105*q2l[12]-0.8385254915624211*q2c[12]); 
  out2[26] += J*D*((-0.140625*q2r[26])-0.140625*q2l[26]-6.28125*q2c[26]-0.3025768239224545*q2r[23]+0.3025768239224545*q2l[23]+0.4192627457812106*q2r[20]+0.4192627457812106*q2l[20]-0.8385254915624212*q2c[20]); 

  const double *q3l = &ql[81]; 
  const double *q3c = &qc[81]; 
  const double *q3r = &qr[81]; 
  double *out3 = &out[81]; 

  out3[0] += J*D*(0.6708203932499369*q3r[9]+0.6708203932499369*q3l[9]-1.341640786499874*q3c[9]-1.190784930203603*q3r[3]+1.190784930203603*q3l[3]+0.9375*q3r[0]+0.9375*q3l[0]-1.875*q3c[0]); 
  out3[1] += J*D*(0.6708203932499369*q3r[15]+0.6708203932499369*q3l[15]-1.341640786499874*q3c[15]-1.190784930203603*q3r[5]+1.190784930203603*q3l[5]+0.9375*q3r[1]+0.9375*q3l[1]-1.875*q3c[1]); 
  out3[2] += J*D*(0.6708203932499369*q3r[16]+0.6708203932499369*q3l[16]-1.341640786499874*q3c[16]-1.190784930203603*q3r[6]+1.190784930203603*q3l[6]+0.9375*q3r[2]+0.9375*q3l[2]-1.875*q3c[2]); 
  out3[3] += J*D*(0.7382874503707888*q3r[9]-0.7382874503707888*q3l[9]-1.453125*q3r[3]-1.453125*q3l[3]-5.34375*q3c[3]+1.190784930203603*q3r[0]-1.190784930203603*q3l[0]); 
  out3[4] += J*D*(0.6708203932499369*q3r[19]+0.6708203932499369*q3l[19]-1.341640786499874*q3c[19]-1.190784930203603*q3r[10]+1.190784930203603*q3l[10]+0.9375*q3r[4]+0.9375*q3l[4]-1.875*q3c[4]); 
  out3[5] += J*D*(0.7382874503707888*q3r[15]-0.7382874503707888*q3l[15]-1.453125*q3r[5]-1.453125*q3l[5]-5.34375*q3c[5]+1.190784930203603*q3r[1]-1.190784930203603*q3l[1]); 
  out3[6] += J*D*(0.7382874503707888*q3r[16]-0.7382874503707888*q3l[16]-1.453125*q3r[6]-1.453125*q3l[6]-5.34375*q3c[6]+1.190784930203603*q3r[2]-1.190784930203603*q3l[2]); 
  out3[7] += J*D*(0.6708203932499369*q3r[21]+0.6708203932499369*q3l[21]-1.341640786499874*q3c[21]-1.190784930203603*q3r[13]+1.190784930203603*q3l[13]+0.9375*q3r[7]+0.9375*q3l[7]-1.875*q3c[7]); 
  out3[8] += J*D*(0.6708203932499369*q3r[22]+0.6708203932499369*q3l[22]-1.341640786499874*q3c[22]-1.190784930203603*q3r[14]+1.190784930203603*q3l[14]+0.9375*q3r[8]+0.9375*q3l[8]-1.875*q3c[8]); 
  out3[9] += J*D*((-0.140625*q3r[9])-0.140625*q3l[9]-6.28125*q3c[9]-0.3025768239224545*q3r[3]+0.3025768239224545*q3l[3]+0.4192627457812106*q3r[0]+0.4192627457812106*q3l[0]-0.8385254915624212*q3c[0]); 
  out3[10] += J*D*(0.7382874503707888*q3r[19]-0.7382874503707888*q3l[19]-1.453125*q3r[10]-1.453125*q3l[10]-5.34375*q3c[10]+1.190784930203603*q3r[4]-1.190784930203603*q3l[4]); 
  out3[11] += J*D*(0.6708203932499369*q3r[24]+0.6708203932499369*q3l[24]-1.341640786499874*q3c[24]-1.190784930203603*q3r[17]+1.190784930203603*q3l[17]+0.9375*q3r[11]+0.9375*q3l[11]-1.875*q3c[11]); 
  out3[12] += J*D*(0.6708203932499369*q3r[25]+0.6708203932499369*q3l[25]-1.341640786499874*q3c[25]-1.190784930203603*q3r[18]+1.190784930203603*q3l[18]+0.9375*q3r[12]+0.9375*q3l[12]-1.875*q3c[12]); 
  out3[13] += J*D*(0.7382874503707888*q3r[21]-0.7382874503707888*q3l[21]-1.453125*q3r[13]-1.453125*q3l[13]-5.34375*q3c[13]+1.190784930203603*q3r[7]-1.190784930203603*q3l[7]); 
  out3[14] += J*D*(0.7382874503707888*q3r[22]-0.7382874503707888*q3l[22]-1.453125*q3r[14]-1.453125*q3l[14]-5.34375*q3c[14]+1.190784930203603*q3r[8]-1.190784930203603*q3l[8]); 
  out3[15] += J*D*((-0.140625*q3r[15])-0.140625*q3l[15]-6.28125*q3c[15]-0.3025768239224544*q3r[5]+0.3025768239224544*q3l[5]+0.4192627457812105*q3r[1]+0.4192627457812105*q3l[1]-0.8385254915624211*q3c[1]); 
  out3[16] += J*D*((-0.140625*q3r[16])-0.140625*q3l[16]-6.28125*q3c[16]-0.3025768239224544*q3r[6]+0.3025768239224544*q3l[6]+0.4192627457812105*q3r[2]+0.4192627457812105*q3l[2]-0.8385254915624211*q3c[2]); 
  out3[17] += J*D*(0.7382874503707888*q3r[24]-0.7382874503707888*q3l[24]-1.453125*q3r[17]-1.453125*q3l[17]-5.34375*q3c[17]+1.190784930203603*q3r[11]-1.190784930203603*q3l[11]); 
  out3[18] += J*D*(0.7382874503707888*q3r[25]-0.7382874503707888*q3l[25]-1.453125*q3r[18]-1.453125*q3l[18]-5.34375*q3c[18]+1.190784930203603*q3r[12]-1.190784930203603*q3l[12]); 
  out3[19] += J*D*((-0.140625*q3r[19])-0.140625*q3l[19]-6.28125*q3c[19]-0.3025768239224545*q3r[10]+0.3025768239224545*q3l[10]+0.4192627457812106*q3r[4]+0.4192627457812106*q3l[4]-0.8385254915624212*q3c[4]); 
  out3[20] += J*D*(0.6708203932499369*q3r[26]+0.6708203932499369*q3l[26]-1.341640786499874*q3c[26]-1.190784930203603*q3r[23]+1.190784930203603*q3l[23]+0.9375*q3r[20]+0.9375*q3l[20]-1.875*q3c[20]); 
  out3[21] += J*D*((-0.140625*q3r[21])-0.140625*q3l[21]-6.28125*q3c[21]-0.3025768239224544*q3r[13]+0.3025768239224544*q3l[13]+0.4192627457812106*q3r[7]+0.4192627457812106*q3l[7]-0.8385254915624212*q3c[7]); 
  out3[22] += J*D*((-0.140625*q3r[22])-0.140625*q3l[22]-6.28125*q3c[22]-0.3025768239224544*q3r[14]+0.3025768239224544*q3l[14]+0.4192627457812106*q3r[8]+0.4192627457812106*q3l[8]-0.8385254915624212*q3c[8]); 
  out3[23] += J*D*(0.7382874503707888*q3r[26]-0.7382874503707888*q3l[26]-1.453125*q3r[23]-1.453125*q3l[23]-5.34375*q3c[23]+1.190784930203603*q3r[20]-1.190784930203603*q3l[20]); 
  out3[24] += J*D*((-0.140625*q3r[24])-0.140625*q3l[24]-6.28125*q3c[24]-0.3025768239224545*q3r[17]+0.3025768239224545*q3l[17]+0.4192627457812105*q3r[11]+0.4192627457812105*q3l[11]-0.8385254915624211*q3c[11]); 
  out3[25] += J*D*((-0.140625*q3r[25])-0.140625*q3l[25]-6.28125*q3c[25]-0.3025768239224545*q3r[18]+0.3025768239224545*q3l[18]+0.4192627457812105*q3r[12]+0.4192627457812105*q3l[12]-0.8385254915624211*q3c[12]); 
  out3[26] += J*D*((-0.140625*q3r[26])-0.140625*q3l[26]-6.28125*q3c[26]-0.3025768239224545*q3r[23]+0.3025768239224545*q3l[23]+0.4192627457812106*q3r[20]+0.4192627457812106*q3l[20]-0.8385254915624212*q3c[20]); 

  return 0.;

} 