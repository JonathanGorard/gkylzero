#include <gkyl_vlasov_kernels.h> 
GKYL_CU_DH double vlasov_surfx_3x3v_tensor_p1(const double *w, const double *dxv, const double *alpha_geo, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:   Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // alpha_geo: Fields used only for general geometry.
  // fl/fc/fr:  Input Distribution function in left/center/right cells.
  // out:       Incremented distribution function in center cell.
  const double dx10 = 2/dxv[0]; 
  const double dv = dxv[3], wv = w[3]; 
  double Ghat_r[80]; 
  double Ghat_l[80]; 
  if (wv>0) { 

  Ghat_r[0] = (1.224744871391589*fc[1]+0.7071067811865475*fc[0])*wv+(0.3535533905932737*fc[10]+0.2041241452319315*fc[4])*dv; 
  Ghat_r[1] = (1.224744871391589*fc[7]+0.7071067811865475*fc[2])*wv+(0.3535533905932737*fc[23]+0.2041241452319315*fc[11])*dv; 
  Ghat_r[2] = (1.224744871391589*fc[8]+0.7071067811865475*fc[3])*wv+(0.3535533905932737*fc[24]+0.2041241452319315*fc[12])*dv; 
  Ghat_r[3] = (1.224744871391589*fc[10]+0.7071067811865475*fc[4])*wv+(0.3535533905932737*fc[1]+0.2041241452319315*fc[0])*dv; 
  Ghat_r[4] = (1.224744871391589*fc[13]+0.7071067811865475*fc[5])*wv+(0.3535533905932737*fc[29]+0.2041241452319315*fc[16])*dv; 
  Ghat_r[5] = (1.224744871391589*fc[17]+0.7071067811865475*fc[6])*wv+(0.3535533905932737*fc[35]+0.2041241452319315*fc[20])*dv; 
  Ghat_r[6] = (1.224744871391589*fc[22]+0.7071067811865475*fc[9])*wv+(0.3535533905932737*fc[42]+0.2041241452319315*fc[25])*dv; 
  Ghat_r[7] = (1.224744871391589*fc[23]+0.7071067811865475*fc[11])*wv+(0.3535533905932737*fc[7]+0.2041241452319315*fc[2])*dv; 
  Ghat_r[8] = (1.224744871391589*fc[24]+0.7071067811865475*fc[12])*wv+(0.3535533905932737*fc[8]+0.2041241452319315*fc[3])*dv; 
  Ghat_r[9] = (1.224744871391589*fc[26]+0.7071067811865475*fc[14])*wv+(0.3535533905932737*fc[44]+0.2041241452319315*fc[30])*dv; 
  Ghat_r[10] = (1.224744871391589*fc[27]+0.7071067811865475*fc[15])*wv+(0.3535533905932737*fc[45]+0.2041241452319315*fc[31])*dv; 
  Ghat_r[11] = (1.224744871391589*fc[29]+0.7071067811865475*fc[16])*wv+(0.3535533905932737*fc[13]+0.2041241452319315*fc[5])*dv; 
  Ghat_r[12] = (1.224744871391589*fc[32]+0.7071067811865475*fc[18])*wv+(0.3535533905932737*fc[48]+0.2041241452319315*fc[36])*dv; 
  Ghat_r[13] = (1.224744871391589*fc[33]+0.7071067811865475*fc[19])*wv+(0.3535533905932737*fc[49]+0.2041241452319315*fc[37])*dv; 
  Ghat_r[14] = (1.224744871391589*fc[35]+0.7071067811865475*fc[20])*wv+(0.3535533905932737*fc[17]+0.2041241452319315*fc[6])*dv; 
  Ghat_r[15] = (1.224744871391589*fc[38]+0.7071067811865475*fc[21])*wv+(0.3535533905932737*fc[54]+0.2041241452319315*fc[41])*dv; 
  Ghat_r[16] = (1.224744871391589*fc[42]+0.7071067811865475*fc[25])*wv+(0.3535533905932737*fc[22]+0.2041241452319315*fc[9])*dv; 
  Ghat_r[17] = (1.224744871391589*fc[43]+0.7071067811865475*fc[28])*wv+(0.3535533905932737*fc[57]+0.2041241452319315*fc[46])*dv; 
  Ghat_r[18] = (1.224744871391589*fc[44]+0.7071067811865475*fc[30])*wv+(0.3535533905932737*fc[26]+0.2041241452319315*fc[14])*dv; 
  Ghat_r[19] = (1.224744871391589*fc[45]+0.7071067811865475*fc[31])*wv+(0.3535533905932737*fc[27]+0.2041241452319315*fc[15])*dv; 
  Ghat_r[20] = (1.224744871391589*fc[47]+0.7071067811865475*fc[34])*wv+(0.3535533905932737*fc[58]+0.2041241452319315*fc[50])*dv; 
  Ghat_r[21] = (1.224744871391589*fc[48]+0.7071067811865475*fc[36])*wv+(0.3535533905932737*fc[32]+0.2041241452319315*fc[18])*dv; 
  Ghat_r[22] = (1.224744871391589*fc[49]+0.7071067811865475*fc[37])*wv+(0.3535533905932737*fc[33]+0.2041241452319315*fc[19])*dv; 
  Ghat_r[23] = (1.224744871391589*fc[51]+0.7071067811865475*fc[39])*wv+(0.3535533905932737*fc[60]+0.2041241452319315*fc[55])*dv; 
  Ghat_r[24] = (1.224744871391589*fc[52]+0.7071067811865475*fc[40])*wv+(0.3535533905932737*fc[61]+0.2041241452319315*fc[56])*dv; 
  Ghat_r[25] = (1.224744871391589*fc[54]+0.7071067811865475*fc[41])*wv+(0.3535533905932737*fc[38]+0.2041241452319315*fc[21])*dv; 
  Ghat_r[26] = (1.224744871391589*fc[57]+0.7071067811865475*fc[46])*wv+(0.3535533905932737*fc[43]+0.2041241452319315*fc[28])*dv; 
  Ghat_r[27] = (1.224744871391589*fc[58]+0.7071067811865475*fc[50])*wv+(0.3535533905932737*fc[47]+0.2041241452319315*fc[34])*dv; 
  Ghat_r[28] = (1.224744871391589*fc[59]+0.7071067811865475*fc[53])*wv+(0.3535533905932737*fc[63]+0.2041241452319315*fc[62])*dv; 
  Ghat_r[29] = (1.224744871391589*fc[60]+0.7071067811865475*fc[55])*wv+(0.3535533905932737*fc[51]+0.2041241452319315*fc[39])*dv; 
  Ghat_r[30] = (1.224744871391589*fc[61]+0.7071067811865475*fc[56])*wv+(0.3535533905932737*fc[52]+0.2041241452319315*fc[40])*dv; 
  Ghat_r[31] = (1.224744871391589*fc[63]+0.7071067811865475*fc[62])*wv+(0.3535533905932737*fc[59]+0.2041241452319315*fc[53])*dv; 
  Ghat_r[32] = (0.3162277660168379*fc[10]+0.1825741858350554*fc[4])*dv; 
  Ghat_r[33] = (0.3162277660168379*fc[23]+0.1825741858350554*fc[11])*dv; 
  Ghat_r[34] = (0.3162277660168379*fc[24]+0.1825741858350554*fc[12])*dv; 
  Ghat_r[35] = (0.3162277660168379*fc[29]+0.1825741858350554*fc[16])*dv; 
  Ghat_r[36] = (0.3162277660168379*fc[35]+0.1825741858350554*fc[20])*dv; 
  Ghat_r[37] = (0.3162277660168379*fc[42]+0.1825741858350554*fc[25])*dv; 
  Ghat_r[38] = (0.3162277660168379*fc[44]+0.1825741858350554*fc[30])*dv; 
  Ghat_r[39] = (0.3162277660168379*fc[45]+0.1825741858350554*fc[31])*dv; 
  Ghat_r[40] = (0.3162277660168379*fc[48]+0.1825741858350554*fc[36])*dv; 
  Ghat_r[41] = (0.3162277660168379*fc[49]+0.1825741858350554*fc[37])*dv; 
  Ghat_r[42] = (0.3162277660168379*fc[54]+0.1825741858350554*fc[41])*dv; 
  Ghat_r[43] = (0.3162277660168379*fc[57]+0.1825741858350554*fc[46])*dv; 
  Ghat_r[44] = (0.3162277660168379*fc[58]+0.1825741858350554*fc[50])*dv; 
  Ghat_r[45] = (0.3162277660168379*fc[60]+0.1825741858350554*fc[55])*dv; 
  Ghat_r[46] = (0.3162277660168379*fc[61]+0.1825741858350554*fc[56])*dv; 
  Ghat_r[47] = (0.3162277660168379*fc[63]+0.1825741858350554*fc[62])*dv; 

  Ghat_l[0] = (1.224744871391589*fl[1]+0.7071067811865475*fl[0])*wv+(0.3535533905932737*fl[10]+0.2041241452319315*fl[4])*dv; 
  Ghat_l[1] = (1.224744871391589*fl[7]+0.7071067811865475*fl[2])*wv+(0.3535533905932737*fl[23]+0.2041241452319315*fl[11])*dv; 
  Ghat_l[2] = (1.224744871391589*fl[8]+0.7071067811865475*fl[3])*wv+(0.3535533905932737*fl[24]+0.2041241452319315*fl[12])*dv; 
  Ghat_l[3] = (1.224744871391589*fl[10]+0.7071067811865475*fl[4])*wv+(0.3535533905932737*fl[1]+0.2041241452319315*fl[0])*dv; 
  Ghat_l[4] = (1.224744871391589*fl[13]+0.7071067811865475*fl[5])*wv+(0.3535533905932737*fl[29]+0.2041241452319315*fl[16])*dv; 
  Ghat_l[5] = (1.224744871391589*fl[17]+0.7071067811865475*fl[6])*wv+(0.3535533905932737*fl[35]+0.2041241452319315*fl[20])*dv; 
  Ghat_l[6] = (1.224744871391589*fl[22]+0.7071067811865475*fl[9])*wv+(0.3535533905932737*fl[42]+0.2041241452319315*fl[25])*dv; 
  Ghat_l[7] = (1.224744871391589*fl[23]+0.7071067811865475*fl[11])*wv+(0.3535533905932737*fl[7]+0.2041241452319315*fl[2])*dv; 
  Ghat_l[8] = (1.224744871391589*fl[24]+0.7071067811865475*fl[12])*wv+(0.3535533905932737*fl[8]+0.2041241452319315*fl[3])*dv; 
  Ghat_l[9] = (1.224744871391589*fl[26]+0.7071067811865475*fl[14])*wv+(0.3535533905932737*fl[44]+0.2041241452319315*fl[30])*dv; 
  Ghat_l[10] = (1.224744871391589*fl[27]+0.7071067811865475*fl[15])*wv+(0.3535533905932737*fl[45]+0.2041241452319315*fl[31])*dv; 
  Ghat_l[11] = (1.224744871391589*fl[29]+0.7071067811865475*fl[16])*wv+(0.3535533905932737*fl[13]+0.2041241452319315*fl[5])*dv; 
  Ghat_l[12] = (1.224744871391589*fl[32]+0.7071067811865475*fl[18])*wv+(0.3535533905932737*fl[48]+0.2041241452319315*fl[36])*dv; 
  Ghat_l[13] = (1.224744871391589*fl[33]+0.7071067811865475*fl[19])*wv+(0.3535533905932737*fl[49]+0.2041241452319315*fl[37])*dv; 
  Ghat_l[14] = (1.224744871391589*fl[35]+0.7071067811865475*fl[20])*wv+(0.3535533905932737*fl[17]+0.2041241452319315*fl[6])*dv; 
  Ghat_l[15] = (1.224744871391589*fl[38]+0.7071067811865475*fl[21])*wv+(0.3535533905932737*fl[54]+0.2041241452319315*fl[41])*dv; 
  Ghat_l[16] = (1.224744871391589*fl[42]+0.7071067811865475*fl[25])*wv+(0.3535533905932737*fl[22]+0.2041241452319315*fl[9])*dv; 
  Ghat_l[17] = (1.224744871391589*fl[43]+0.7071067811865475*fl[28])*wv+(0.3535533905932737*fl[57]+0.2041241452319315*fl[46])*dv; 
  Ghat_l[18] = (1.224744871391589*fl[44]+0.7071067811865475*fl[30])*wv+(0.3535533905932737*fl[26]+0.2041241452319315*fl[14])*dv; 
  Ghat_l[19] = (1.224744871391589*fl[45]+0.7071067811865475*fl[31])*wv+(0.3535533905932737*fl[27]+0.2041241452319315*fl[15])*dv; 
  Ghat_l[20] = (1.224744871391589*fl[47]+0.7071067811865475*fl[34])*wv+(0.3535533905932737*fl[58]+0.2041241452319315*fl[50])*dv; 
  Ghat_l[21] = (1.224744871391589*fl[48]+0.7071067811865475*fl[36])*wv+(0.3535533905932737*fl[32]+0.2041241452319315*fl[18])*dv; 
  Ghat_l[22] = (1.224744871391589*fl[49]+0.7071067811865475*fl[37])*wv+(0.3535533905932737*fl[33]+0.2041241452319315*fl[19])*dv; 
  Ghat_l[23] = (1.224744871391589*fl[51]+0.7071067811865475*fl[39])*wv+(0.3535533905932737*fl[60]+0.2041241452319315*fl[55])*dv; 
  Ghat_l[24] = (1.224744871391589*fl[52]+0.7071067811865475*fl[40])*wv+(0.3535533905932737*fl[61]+0.2041241452319315*fl[56])*dv; 
  Ghat_l[25] = (1.224744871391589*fl[54]+0.7071067811865475*fl[41])*wv+(0.3535533905932737*fl[38]+0.2041241452319315*fl[21])*dv; 
  Ghat_l[26] = (1.224744871391589*fl[57]+0.7071067811865475*fl[46])*wv+(0.3535533905932737*fl[43]+0.2041241452319315*fl[28])*dv; 
  Ghat_l[27] = (1.224744871391589*fl[58]+0.7071067811865475*fl[50])*wv+(0.3535533905932737*fl[47]+0.2041241452319315*fl[34])*dv; 
  Ghat_l[28] = (1.224744871391589*fl[59]+0.7071067811865475*fl[53])*wv+(0.3535533905932737*fl[63]+0.2041241452319315*fl[62])*dv; 
  Ghat_l[29] = (1.224744871391589*fl[60]+0.7071067811865475*fl[55])*wv+(0.3535533905932737*fl[51]+0.2041241452319315*fl[39])*dv; 
  Ghat_l[30] = (1.224744871391589*fl[61]+0.7071067811865475*fl[56])*wv+(0.3535533905932737*fl[52]+0.2041241452319315*fl[40])*dv; 
  Ghat_l[31] = (1.224744871391589*fl[63]+0.7071067811865475*fl[62])*wv+(0.3535533905932737*fl[59]+0.2041241452319315*fl[53])*dv; 
  Ghat_l[32] = (0.3162277660168379*fl[10]+0.1825741858350554*fl[4])*dv; 
  Ghat_l[33] = (0.3162277660168379*fl[23]+0.1825741858350554*fl[11])*dv; 
  Ghat_l[34] = (0.3162277660168379*fl[24]+0.1825741858350554*fl[12])*dv; 
  Ghat_l[35] = (0.3162277660168379*fl[29]+0.1825741858350554*fl[16])*dv; 
  Ghat_l[36] = (0.3162277660168379*fl[35]+0.1825741858350554*fl[20])*dv; 
  Ghat_l[37] = (0.3162277660168379*fl[42]+0.1825741858350554*fl[25])*dv; 
  Ghat_l[38] = (0.3162277660168379*fl[44]+0.1825741858350554*fl[30])*dv; 
  Ghat_l[39] = (0.3162277660168379*fl[45]+0.1825741858350554*fl[31])*dv; 
  Ghat_l[40] = (0.3162277660168379*fl[48]+0.1825741858350554*fl[36])*dv; 
  Ghat_l[41] = (0.3162277660168379*fl[49]+0.1825741858350554*fl[37])*dv; 
  Ghat_l[42] = (0.3162277660168379*fl[54]+0.1825741858350554*fl[41])*dv; 
  Ghat_l[43] = (0.3162277660168379*fl[57]+0.1825741858350554*fl[46])*dv; 
  Ghat_l[44] = (0.3162277660168379*fl[58]+0.1825741858350554*fl[50])*dv; 
  Ghat_l[45] = (0.3162277660168379*fl[60]+0.1825741858350554*fl[55])*dv; 
  Ghat_l[46] = (0.3162277660168379*fl[61]+0.1825741858350554*fl[56])*dv; 
  Ghat_l[47] = (0.3162277660168379*fl[63]+0.1825741858350554*fl[62])*dv; 

  } else { 

  Ghat_r[0] = -0.1178511301977579*((10.39230484541326*fr[1]-6.0*fr[0])*wv+(3.0*fr[10]-1.732050807568877*fr[4])*dv); 
  Ghat_r[1] = -0.1178511301977579*((10.39230484541326*fr[7]-6.0*fr[2])*wv+(3.0*fr[23]-1.732050807568877*fr[11])*dv); 
  Ghat_r[2] = -0.1178511301977579*((10.39230484541326*fr[8]-6.0*fr[3])*wv+(3.0*fr[24]-1.732050807568877*fr[12])*dv); 
  Ghat_r[3] = -0.1178511301977579*((10.39230484541326*fr[10]-6.0*fr[4])*wv+(3.0*fr[1]-1.732050807568877*fr[0])*dv); 
  Ghat_r[4] = -0.1178511301977579*((10.39230484541326*fr[13]-6.0*fr[5])*wv+(3.0*fr[29]-1.732050807568877*fr[16])*dv); 
  Ghat_r[5] = -0.1178511301977579*((10.39230484541326*fr[17]-6.0*fr[6])*wv+(3.0*fr[35]-1.732050807568877*fr[20])*dv); 
  Ghat_r[6] = -0.1178511301977579*((10.39230484541326*fr[22]-6.0*fr[9])*wv+(3.0*fr[42]-1.732050807568877*fr[25])*dv); 
  Ghat_r[7] = -0.1178511301977579*((10.39230484541326*fr[23]-6.0*fr[11])*wv+(3.0*fr[7]-1.732050807568877*fr[2])*dv); 
  Ghat_r[8] = -0.1178511301977579*((10.39230484541326*fr[24]-6.0*fr[12])*wv+(3.0*fr[8]-1.732050807568877*fr[3])*dv); 
  Ghat_r[9] = -0.1178511301977579*((10.39230484541326*fr[26]-6.0*fr[14])*wv+(3.0*fr[44]-1.732050807568877*fr[30])*dv); 
  Ghat_r[10] = -0.1178511301977579*((10.39230484541326*fr[27]-6.0*fr[15])*wv+(3.0*fr[45]-1.732050807568877*fr[31])*dv); 
  Ghat_r[11] = -0.1178511301977579*((10.39230484541326*fr[29]-6.0*fr[16])*wv+(3.0*fr[13]-1.732050807568877*fr[5])*dv); 
  Ghat_r[12] = -0.1178511301977579*((10.39230484541326*fr[32]-6.0*fr[18])*wv+(3.0*fr[48]-1.732050807568877*fr[36])*dv); 
  Ghat_r[13] = -0.1178511301977579*((10.39230484541326*fr[33]-6.0*fr[19])*wv+(3.0*fr[49]-1.732050807568877*fr[37])*dv); 
  Ghat_r[14] = -0.1178511301977579*((10.39230484541326*fr[35]-6.0*fr[20])*wv+(3.0*fr[17]-1.732050807568877*fr[6])*dv); 
  Ghat_r[15] = -0.1178511301977579*((10.39230484541326*fr[38]-6.0*fr[21])*wv+(3.0*fr[54]-1.732050807568877*fr[41])*dv); 
  Ghat_r[16] = -0.1178511301977579*((10.39230484541326*fr[42]-6.0*fr[25])*wv+(3.0*fr[22]-1.732050807568877*fr[9])*dv); 
  Ghat_r[17] = -0.1178511301977579*((10.39230484541326*fr[43]-6.0*fr[28])*wv+(3.0*fr[57]-1.732050807568877*fr[46])*dv); 
  Ghat_r[18] = -0.1178511301977579*((10.39230484541326*fr[44]-6.0*fr[30])*wv+(3.0*fr[26]-1.732050807568877*fr[14])*dv); 
  Ghat_r[19] = -0.1178511301977579*((10.39230484541326*fr[45]-6.0*fr[31])*wv+(3.0*fr[27]-1.732050807568877*fr[15])*dv); 
  Ghat_r[20] = -0.1178511301977579*((10.39230484541326*fr[47]-6.0*fr[34])*wv+(3.0*fr[58]-1.732050807568877*fr[50])*dv); 
  Ghat_r[21] = -0.1178511301977579*((10.39230484541326*fr[48]-6.0*fr[36])*wv+(3.0*fr[32]-1.732050807568877*fr[18])*dv); 
  Ghat_r[22] = -0.1178511301977579*((10.39230484541326*fr[49]-6.0*fr[37])*wv+(3.0*fr[33]-1.732050807568877*fr[19])*dv); 
  Ghat_r[23] = -0.1178511301977579*((10.39230484541326*fr[51]-6.0*fr[39])*wv+(3.0*fr[60]-1.732050807568877*fr[55])*dv); 
  Ghat_r[24] = -0.1178511301977579*((10.39230484541326*fr[52]-6.0*fr[40])*wv+(3.0*fr[61]-1.732050807568877*fr[56])*dv); 
  Ghat_r[25] = -0.1178511301977579*((10.39230484541326*fr[54]-6.0*fr[41])*wv+(3.0*fr[38]-1.732050807568877*fr[21])*dv); 
  Ghat_r[26] = -0.1178511301977579*((10.39230484541326*fr[57]-6.0*fr[46])*wv+(3.0*fr[43]-1.732050807568877*fr[28])*dv); 
  Ghat_r[27] = -0.1178511301977579*((10.39230484541326*fr[58]-6.0*fr[50])*wv+(3.0*fr[47]-1.732050807568877*fr[34])*dv); 
  Ghat_r[28] = -0.1178511301977579*((10.39230484541326*fr[59]-6.0*fr[53])*wv+(3.0*fr[63]-1.732050807568877*fr[62])*dv); 
  Ghat_r[29] = -0.1178511301977579*((10.39230484541326*fr[60]-6.0*fr[55])*wv+(3.0*fr[51]-1.732050807568877*fr[39])*dv); 
  Ghat_r[30] = -0.1178511301977579*((10.39230484541326*fr[61]-6.0*fr[56])*wv+(3.0*fr[52]-1.732050807568877*fr[40])*dv); 
  Ghat_r[31] = -0.1178511301977579*((10.39230484541326*fr[63]-6.0*fr[62])*wv+(3.0*fr[59]-1.732050807568877*fr[53])*dv); 
  Ghat_r[32] = -0.04714045207910316*(6.708203932499369*fr[10]-3.872983346207417*fr[4])*dv; 
  Ghat_r[33] = -0.04714045207910316*(6.708203932499369*fr[23]-3.872983346207417*fr[11])*dv; 
  Ghat_r[34] = -0.04714045207910316*(6.708203932499369*fr[24]-3.872983346207417*fr[12])*dv; 
  Ghat_r[35] = -0.04714045207910316*(6.708203932499369*fr[29]-3.872983346207417*fr[16])*dv; 
  Ghat_r[36] = -0.04714045207910316*(6.708203932499369*fr[35]-3.872983346207417*fr[20])*dv; 
  Ghat_r[37] = -0.04714045207910316*(6.708203932499369*fr[42]-3.872983346207417*fr[25])*dv; 
  Ghat_r[38] = -0.04714045207910316*(6.708203932499369*fr[44]-3.872983346207417*fr[30])*dv; 
  Ghat_r[39] = -0.04714045207910316*(6.708203932499369*fr[45]-3.872983346207417*fr[31])*dv; 
  Ghat_r[40] = -0.04714045207910316*(6.708203932499369*fr[48]-3.872983346207417*fr[36])*dv; 
  Ghat_r[41] = -0.04714045207910316*(6.708203932499369*fr[49]-3.872983346207417*fr[37])*dv; 
  Ghat_r[42] = -0.04714045207910316*(6.708203932499369*fr[54]-3.872983346207417*fr[41])*dv; 
  Ghat_r[43] = -0.04714045207910316*(6.708203932499369*fr[57]-3.872983346207417*fr[46])*dv; 
  Ghat_r[44] = -0.04714045207910316*(6.708203932499369*fr[58]-3.872983346207417*fr[50])*dv; 
  Ghat_r[45] = -0.04714045207910316*(6.708203932499369*fr[60]-3.872983346207417*fr[55])*dv; 
  Ghat_r[46] = -0.04714045207910316*(6.708203932499369*fr[61]-3.872983346207417*fr[56])*dv; 
  Ghat_r[47] = -0.04714045207910316*(6.708203932499369*fr[63]-3.872983346207417*fr[62])*dv; 

  Ghat_l[0] = -0.1178511301977579*((10.39230484541326*fc[1]-6.0*fc[0])*wv+(3.0*fc[10]-1.732050807568877*fc[4])*dv); 
  Ghat_l[1] = -0.1178511301977579*((10.39230484541326*fc[7]-6.0*fc[2])*wv+(3.0*fc[23]-1.732050807568877*fc[11])*dv); 
  Ghat_l[2] = -0.1178511301977579*((10.39230484541326*fc[8]-6.0*fc[3])*wv+(3.0*fc[24]-1.732050807568877*fc[12])*dv); 
  Ghat_l[3] = -0.1178511301977579*((10.39230484541326*fc[10]-6.0*fc[4])*wv+(3.0*fc[1]-1.732050807568877*fc[0])*dv); 
  Ghat_l[4] = -0.1178511301977579*((10.39230484541326*fc[13]-6.0*fc[5])*wv+(3.0*fc[29]-1.732050807568877*fc[16])*dv); 
  Ghat_l[5] = -0.1178511301977579*((10.39230484541326*fc[17]-6.0*fc[6])*wv+(3.0*fc[35]-1.732050807568877*fc[20])*dv); 
  Ghat_l[6] = -0.1178511301977579*((10.39230484541326*fc[22]-6.0*fc[9])*wv+(3.0*fc[42]-1.732050807568877*fc[25])*dv); 
  Ghat_l[7] = -0.1178511301977579*((10.39230484541326*fc[23]-6.0*fc[11])*wv+(3.0*fc[7]-1.732050807568877*fc[2])*dv); 
  Ghat_l[8] = -0.1178511301977579*((10.39230484541326*fc[24]-6.0*fc[12])*wv+(3.0*fc[8]-1.732050807568877*fc[3])*dv); 
  Ghat_l[9] = -0.1178511301977579*((10.39230484541326*fc[26]-6.0*fc[14])*wv+(3.0*fc[44]-1.732050807568877*fc[30])*dv); 
  Ghat_l[10] = -0.1178511301977579*((10.39230484541326*fc[27]-6.0*fc[15])*wv+(3.0*fc[45]-1.732050807568877*fc[31])*dv); 
  Ghat_l[11] = -0.1178511301977579*((10.39230484541326*fc[29]-6.0*fc[16])*wv+(3.0*fc[13]-1.732050807568877*fc[5])*dv); 
  Ghat_l[12] = -0.1178511301977579*((10.39230484541326*fc[32]-6.0*fc[18])*wv+(3.0*fc[48]-1.732050807568877*fc[36])*dv); 
  Ghat_l[13] = -0.1178511301977579*((10.39230484541326*fc[33]-6.0*fc[19])*wv+(3.0*fc[49]-1.732050807568877*fc[37])*dv); 
  Ghat_l[14] = -0.1178511301977579*((10.39230484541326*fc[35]-6.0*fc[20])*wv+(3.0*fc[17]-1.732050807568877*fc[6])*dv); 
  Ghat_l[15] = -0.1178511301977579*((10.39230484541326*fc[38]-6.0*fc[21])*wv+(3.0*fc[54]-1.732050807568877*fc[41])*dv); 
  Ghat_l[16] = -0.1178511301977579*((10.39230484541326*fc[42]-6.0*fc[25])*wv+(3.0*fc[22]-1.732050807568877*fc[9])*dv); 
  Ghat_l[17] = -0.1178511301977579*((10.39230484541326*fc[43]-6.0*fc[28])*wv+(3.0*fc[57]-1.732050807568877*fc[46])*dv); 
  Ghat_l[18] = -0.1178511301977579*((10.39230484541326*fc[44]-6.0*fc[30])*wv+(3.0*fc[26]-1.732050807568877*fc[14])*dv); 
  Ghat_l[19] = -0.1178511301977579*((10.39230484541326*fc[45]-6.0*fc[31])*wv+(3.0*fc[27]-1.732050807568877*fc[15])*dv); 
  Ghat_l[20] = -0.1178511301977579*((10.39230484541326*fc[47]-6.0*fc[34])*wv+(3.0*fc[58]-1.732050807568877*fc[50])*dv); 
  Ghat_l[21] = -0.1178511301977579*((10.39230484541326*fc[48]-6.0*fc[36])*wv+(3.0*fc[32]-1.732050807568877*fc[18])*dv); 
  Ghat_l[22] = -0.1178511301977579*((10.39230484541326*fc[49]-6.0*fc[37])*wv+(3.0*fc[33]-1.732050807568877*fc[19])*dv); 
  Ghat_l[23] = -0.1178511301977579*((10.39230484541326*fc[51]-6.0*fc[39])*wv+(3.0*fc[60]-1.732050807568877*fc[55])*dv); 
  Ghat_l[24] = -0.1178511301977579*((10.39230484541326*fc[52]-6.0*fc[40])*wv+(3.0*fc[61]-1.732050807568877*fc[56])*dv); 
  Ghat_l[25] = -0.1178511301977579*((10.39230484541326*fc[54]-6.0*fc[41])*wv+(3.0*fc[38]-1.732050807568877*fc[21])*dv); 
  Ghat_l[26] = -0.1178511301977579*((10.39230484541326*fc[57]-6.0*fc[46])*wv+(3.0*fc[43]-1.732050807568877*fc[28])*dv); 
  Ghat_l[27] = -0.1178511301977579*((10.39230484541326*fc[58]-6.0*fc[50])*wv+(3.0*fc[47]-1.732050807568877*fc[34])*dv); 
  Ghat_l[28] = -0.1178511301977579*((10.39230484541326*fc[59]-6.0*fc[53])*wv+(3.0*fc[63]-1.732050807568877*fc[62])*dv); 
  Ghat_l[29] = -0.1178511301977579*((10.39230484541326*fc[60]-6.0*fc[55])*wv+(3.0*fc[51]-1.732050807568877*fc[39])*dv); 
  Ghat_l[30] = -0.1178511301977579*((10.39230484541326*fc[61]-6.0*fc[56])*wv+(3.0*fc[52]-1.732050807568877*fc[40])*dv); 
  Ghat_l[31] = -0.1178511301977579*((10.39230484541326*fc[63]-6.0*fc[62])*wv+(3.0*fc[59]-1.732050807568877*fc[53])*dv); 
  Ghat_l[32] = -0.04714045207910316*(6.708203932499369*fc[10]-3.872983346207417*fc[4])*dv; 
  Ghat_l[33] = -0.04714045207910316*(6.708203932499369*fc[23]-3.872983346207417*fc[11])*dv; 
  Ghat_l[34] = -0.04714045207910316*(6.708203932499369*fc[24]-3.872983346207417*fc[12])*dv; 
  Ghat_l[35] = -0.04714045207910316*(6.708203932499369*fc[29]-3.872983346207417*fc[16])*dv; 
  Ghat_l[36] = -0.04714045207910316*(6.708203932499369*fc[35]-3.872983346207417*fc[20])*dv; 
  Ghat_l[37] = -0.04714045207910316*(6.708203932499369*fc[42]-3.872983346207417*fc[25])*dv; 
  Ghat_l[38] = -0.04714045207910316*(6.708203932499369*fc[44]-3.872983346207417*fc[30])*dv; 
  Ghat_l[39] = -0.04714045207910316*(6.708203932499369*fc[45]-3.872983346207417*fc[31])*dv; 
  Ghat_l[40] = -0.04714045207910316*(6.708203932499369*fc[48]-3.872983346207417*fc[36])*dv; 
  Ghat_l[41] = -0.04714045207910316*(6.708203932499369*fc[49]-3.872983346207417*fc[37])*dv; 
  Ghat_l[42] = -0.04714045207910316*(6.708203932499369*fc[54]-3.872983346207417*fc[41])*dv; 
  Ghat_l[43] = -0.04714045207910316*(6.708203932499369*fc[57]-3.872983346207417*fc[46])*dv; 
  Ghat_l[44] = -0.04714045207910316*(6.708203932499369*fc[58]-3.872983346207417*fc[50])*dv; 
  Ghat_l[45] = -0.04714045207910316*(6.708203932499369*fc[60]-3.872983346207417*fc[55])*dv; 
  Ghat_l[46] = -0.04714045207910316*(6.708203932499369*fc[61]-3.872983346207417*fc[56])*dv; 
  Ghat_l[47] = -0.04714045207910316*(6.708203932499369*fc[63]-3.872983346207417*fc[62])*dv; 

  } 
  out[0] += (0.7071067811865475*Ghat_l[0]-0.7071067811865475*Ghat_r[0])*dx10; 
  out[1] += -1.224744871391589*(Ghat_r[0]+Ghat_l[0])*dx10; 
  out[2] += (0.7071067811865475*Ghat_l[1]-0.7071067811865475*Ghat_r[1])*dx10; 
  out[3] += (0.7071067811865475*Ghat_l[2]-0.7071067811865475*Ghat_r[2])*dx10; 
  out[4] += (0.7071067811865475*Ghat_l[3]-0.7071067811865475*Ghat_r[3])*dx10; 
  out[5] += (0.7071067811865475*Ghat_l[4]-0.7071067811865475*Ghat_r[4])*dx10; 
  out[6] += (0.7071067811865475*Ghat_l[5]-0.7071067811865475*Ghat_r[5])*dx10; 
  out[7] += -1.224744871391589*(Ghat_r[1]+Ghat_l[1])*dx10; 
  out[8] += -1.224744871391589*(Ghat_r[2]+Ghat_l[2])*dx10; 
  out[9] += (0.7071067811865475*Ghat_l[6]-0.7071067811865475*Ghat_r[6])*dx10; 
  out[10] += -1.224744871391589*(Ghat_r[3]+Ghat_l[3])*dx10; 
  out[11] += (0.7071067811865475*Ghat_l[7]-0.7071067811865475*Ghat_r[7])*dx10; 
  out[12] += (0.7071067811865475*Ghat_l[8]-0.7071067811865475*Ghat_r[8])*dx10; 
  out[13] += -1.224744871391589*(Ghat_r[4]+Ghat_l[4])*dx10; 
  out[14] += (0.7071067811865475*Ghat_l[9]-0.7071067811865475*Ghat_r[9])*dx10; 
  out[15] += (0.7071067811865475*Ghat_l[10]-0.7071067811865475*Ghat_r[10])*dx10; 
  out[16] += (0.7071067811865475*Ghat_l[11]-0.7071067811865475*Ghat_r[11])*dx10; 
  out[17] += -1.224744871391589*(Ghat_r[5]+Ghat_l[5])*dx10; 
  out[18] += (0.7071067811865475*Ghat_l[12]-0.7071067811865475*Ghat_r[12])*dx10; 
  out[19] += (0.7071067811865475*Ghat_l[13]-0.7071067811865475*Ghat_r[13])*dx10; 
  out[20] += (0.7071067811865475*Ghat_l[14]-0.7071067811865475*Ghat_r[14])*dx10; 
  out[21] += (0.7071067811865475*Ghat_l[15]-0.7071067811865475*Ghat_r[15])*dx10; 
  out[22] += -1.224744871391589*(Ghat_r[6]+Ghat_l[6])*dx10; 
  out[23] += -1.224744871391589*(Ghat_r[7]+Ghat_l[7])*dx10; 
  out[24] += -1.224744871391589*(Ghat_r[8]+Ghat_l[8])*dx10; 
  out[25] += (0.7071067811865475*Ghat_l[16]-0.7071067811865475*Ghat_r[16])*dx10; 
  out[26] += -1.224744871391589*(Ghat_r[9]+Ghat_l[9])*dx10; 
  out[27] += -1.224744871391589*(Ghat_r[10]+Ghat_l[10])*dx10; 
  out[28] += (0.7071067811865475*Ghat_l[17]-0.7071067811865475*Ghat_r[17])*dx10; 
  out[29] += -1.224744871391589*(Ghat_r[11]+Ghat_l[11])*dx10; 
  out[30] += (0.7071067811865475*Ghat_l[18]-0.7071067811865475*Ghat_r[18])*dx10; 
  out[31] += (0.7071067811865475*Ghat_l[19]-0.7071067811865475*Ghat_r[19])*dx10; 
  out[32] += -1.224744871391589*(Ghat_r[12]+Ghat_l[12])*dx10; 
  out[33] += -1.224744871391589*(Ghat_r[13]+Ghat_l[13])*dx10; 
  out[34] += (0.7071067811865475*Ghat_l[20]-0.7071067811865475*Ghat_r[20])*dx10; 
  out[35] += -1.224744871391589*(Ghat_r[14]+Ghat_l[14])*dx10; 
  out[36] += (0.7071067811865475*Ghat_l[21]-0.7071067811865475*Ghat_r[21])*dx10; 
  out[37] += (0.7071067811865475*Ghat_l[22]-0.7071067811865475*Ghat_r[22])*dx10; 
  out[38] += -1.224744871391589*(Ghat_r[15]+Ghat_l[15])*dx10; 
  out[39] += (0.7071067811865475*Ghat_l[23]-0.7071067811865475*Ghat_r[23])*dx10; 
  out[40] += (0.7071067811865475*Ghat_l[24]-0.7071067811865475*Ghat_r[24])*dx10; 
  out[41] += (0.7071067811865475*Ghat_l[25]-0.7071067811865475*Ghat_r[25])*dx10; 
  out[42] += -1.224744871391589*(Ghat_r[16]+Ghat_l[16])*dx10; 
  out[43] += -1.224744871391589*(Ghat_r[17]+Ghat_l[17])*dx10; 
  out[44] += -1.224744871391589*(Ghat_r[18]+Ghat_l[18])*dx10; 
  out[45] += -1.224744871391589*(Ghat_r[19]+Ghat_l[19])*dx10; 
  out[46] += (0.7071067811865475*Ghat_l[26]-0.7071067811865475*Ghat_r[26])*dx10; 
  out[47] += -1.224744871391589*(Ghat_r[20]+Ghat_l[20])*dx10; 
  out[48] += -1.224744871391589*(Ghat_r[21]+Ghat_l[21])*dx10; 
  out[49] += -1.224744871391589*(Ghat_r[22]+Ghat_l[22])*dx10; 
  out[50] += (0.7071067811865475*Ghat_l[27]-0.7071067811865475*Ghat_r[27])*dx10; 
  out[51] += -1.224744871391589*(Ghat_r[23]+Ghat_l[23])*dx10; 
  out[52] += -1.224744871391589*(Ghat_r[24]+Ghat_l[24])*dx10; 
  out[53] += (0.7071067811865475*Ghat_l[28]-0.7071067811865475*Ghat_r[28])*dx10; 
  out[54] += -1.224744871391589*(Ghat_r[25]+Ghat_l[25])*dx10; 
  out[55] += (0.7071067811865475*Ghat_l[29]-0.7071067811865475*Ghat_r[29])*dx10; 
  out[56] += (0.7071067811865475*Ghat_l[30]-0.7071067811865475*Ghat_r[30])*dx10; 
  out[57] += -1.224744871391589*(Ghat_r[26]+Ghat_l[26])*dx10; 
  out[58] += -1.224744871391589*(Ghat_r[27]+Ghat_l[27])*dx10; 
  out[59] += -1.224744871391589*(Ghat_r[28]+Ghat_l[28])*dx10; 
  out[60] += -1.224744871391589*(Ghat_r[29]+Ghat_l[29])*dx10; 
  out[61] += -1.224744871391589*(Ghat_r[30]+Ghat_l[30])*dx10; 
  out[62] += (0.7071067811865475*Ghat_l[31]-0.7071067811865475*Ghat_r[31])*dx10; 
  out[63] += -1.224744871391589*(Ghat_r[31]+Ghat_l[31])*dx10; 

  return 0.;

} 