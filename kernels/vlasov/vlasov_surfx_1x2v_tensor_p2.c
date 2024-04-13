#include <gkyl_vlasov_kernels.h> 
GKYL_CU_DH double vlasov_surfx_1x2v_tensor_p2(const double *w, const double *dxv, const double *alpha_geo, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:   Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // alpha_geo: Fields used only for general geometry.
  // fl/fc/fr:  Input Distribution function in left/center/right cells.
  // out:       Incremented distribution function in center cell.
  const double dx10 = 2/dxv[0]; 
  const double dv = dxv[1], wv = w[1]; 
  double Ghat_r[9]; 
  double Ghat_l[9]; 
  if (wv>0) { 

  Ghat_r[0] = (1.58113883008419*fc[7]+1.224744871391589*fc[1]+0.7071067811865475*fc[0])*wv+(0.4564354645876384*fc[11]+0.3535533905932737*fc[4]+0.2041241452319315*fc[2])*dv; 
  Ghat_r[1] = (1.58113883008419*fc[11]+1.224744871391589*fc[4]+0.7071067811865475*fc[2])*wv+(0.408248290463863*fc[20]+0.3162277660168379*fc[12]+0.1825741858350554*fc[8]+0.4564354645876384*fc[7]+0.3535533905932737*fc[1]+0.2041241452319315*fc[0])*dv; 
  Ghat_r[2] = (1.58113883008419*fc[13]+1.224744871391589*fc[5]+0.7071067811865475*fc[3])*wv+(0.4564354645876384*fc[17]+0.3535533905932737*fc[10]+0.2041241452319315*fc[6])*dv; 
  Ghat_r[3] = (1.58113883008419*fc[17]+1.224744871391589*fc[10]+0.7071067811865475*fc[6])*wv+(0.408248290463863*fc[23]+0.3162277660168379*fc[18]+0.1825741858350554*fc[14]+0.4564354645876384*fc[13]+0.3535533905932737*fc[5]+0.2041241452319315*fc[3])*dv; 
  Ghat_r[4] = (1.58113883008419*fc[20]+1.224744871391589*fc[12]+0.7071067811865475*fc[8])*wv+(0.408248290463863*fc[11]+0.3162277660168379*fc[4]+0.1825741858350554*fc[2])*dv; 
  Ghat_r[5] = (1.58113883008419*fc[21]+1.224744871391589*fc[15]+0.7071067811865475*fc[9])*wv+(0.4564354645876384*fc[24]+0.3535533905932737*fc[19]+0.2041241452319315*fc[16])*dv; 
  Ghat_r[6] = (1.58113883008419*fc[23]+1.224744871391589*fc[18]+0.7071067811865475*fc[14])*wv+(0.408248290463863*fc[17]+0.3162277660168379*fc[10]+0.1825741858350554*fc[6])*dv; 
  Ghat_r[7] = (1.58113883008419*fc[24]+1.224744871391589*fc[19]+0.7071067811865475*fc[16])*wv+(0.408248290463863*fc[26]+0.3162277660168379*fc[25]+0.1825741858350554*fc[22]+0.4564354645876384*fc[21]+0.3535533905932737*fc[15]+0.2041241452319315*fc[9])*dv; 
  Ghat_r[8] = (1.58113883008419*fc[26]+1.224744871391589*fc[25]+0.7071067811865475*fc[22])*wv+(0.408248290463863*fc[24]+0.3162277660168379*fc[19]+0.1825741858350554*fc[16])*dv; 

  Ghat_l[0] = (1.58113883008419*fl[7]+1.224744871391589*fl[1]+0.7071067811865475*fl[0])*wv+(0.4564354645876384*fl[11]+0.3535533905932737*fl[4]+0.2041241452319315*fl[2])*dv; 
  Ghat_l[1] = (1.58113883008419*fl[11]+1.224744871391589*fl[4]+0.7071067811865475*fl[2])*wv+(0.408248290463863*fl[20]+0.3162277660168379*fl[12]+0.1825741858350554*fl[8]+0.4564354645876384*fl[7]+0.3535533905932737*fl[1]+0.2041241452319315*fl[0])*dv; 
  Ghat_l[2] = (1.58113883008419*fl[13]+1.224744871391589*fl[5]+0.7071067811865475*fl[3])*wv+(0.4564354645876384*fl[17]+0.3535533905932737*fl[10]+0.2041241452319315*fl[6])*dv; 
  Ghat_l[3] = (1.58113883008419*fl[17]+1.224744871391589*fl[10]+0.7071067811865475*fl[6])*wv+(0.408248290463863*fl[23]+0.3162277660168379*fl[18]+0.1825741858350554*fl[14]+0.4564354645876384*fl[13]+0.3535533905932737*fl[5]+0.2041241452319315*fl[3])*dv; 
  Ghat_l[4] = (1.58113883008419*fl[20]+1.224744871391589*fl[12]+0.7071067811865475*fl[8])*wv+(0.408248290463863*fl[11]+0.3162277660168379*fl[4]+0.1825741858350554*fl[2])*dv; 
  Ghat_l[5] = (1.58113883008419*fl[21]+1.224744871391589*fl[15]+0.7071067811865475*fl[9])*wv+(0.4564354645876384*fl[24]+0.3535533905932737*fl[19]+0.2041241452319315*fl[16])*dv; 
  Ghat_l[6] = (1.58113883008419*fl[23]+1.224744871391589*fl[18]+0.7071067811865475*fl[14])*wv+(0.408248290463863*fl[17]+0.3162277660168379*fl[10]+0.1825741858350554*fl[6])*dv; 
  Ghat_l[7] = (1.58113883008419*fl[24]+1.224744871391589*fl[19]+0.7071067811865475*fl[16])*wv+(0.408248290463863*fl[26]+0.3162277660168379*fl[25]+0.1825741858350554*fl[22]+0.4564354645876384*fl[21]+0.3535533905932737*fl[15]+0.2041241452319315*fl[9])*dv; 
  Ghat_l[8] = (1.58113883008419*fl[26]+1.224744871391589*fl[25]+0.7071067811865475*fl[22])*wv+(0.408248290463863*fl[24]+0.3162277660168379*fl[19]+0.1825741858350554*fl[16])*dv; 

  } else { 

  Ghat_r[0] = 1.58113883008419*fr[7]*wv-1.224744871391589*fr[1]*wv+0.7071067811865475*fr[0]*wv+0.4564354645876383*fr[11]*dv-0.3535533905932737*fr[4]*dv+0.2041241452319315*fr[2]*dv; 
  Ghat_r[1] = 1.58113883008419*fr[11]*wv-1.224744871391589*fr[4]*wv+0.7071067811865475*fr[2]*wv+0.408248290463863*fr[20]*dv-0.3162277660168379*fr[12]*dv+0.1825741858350554*fr[8]*dv+0.4564354645876384*fr[7]*dv-0.3535533905932737*fr[1]*dv+0.2041241452319315*fr[0]*dv; 
  Ghat_r[2] = 1.58113883008419*fr[13]*wv-1.224744871391589*fr[5]*wv+0.7071067811865475*fr[3]*wv+0.4564354645876384*fr[17]*dv-0.3535533905932737*fr[10]*dv+0.2041241452319315*fr[6]*dv; 
  Ghat_r[3] = 1.58113883008419*fr[17]*wv-1.224744871391589*fr[10]*wv+0.7071067811865475*fr[6]*wv+0.408248290463863*fr[23]*dv-0.3162277660168379*fr[18]*dv+0.1825741858350553*fr[14]*dv+0.4564354645876383*fr[13]*dv-0.3535533905932737*fr[5]*dv+0.2041241452319315*fr[3]*dv; 
  Ghat_r[4] = 1.58113883008419*fr[20]*wv-1.224744871391589*fr[12]*wv+0.7071067811865475*fr[8]*wv+0.408248290463863*fr[11]*dv-0.3162277660168379*fr[4]*dv+0.1825741858350554*fr[2]*dv; 
  Ghat_r[5] = 1.58113883008419*fr[21]*wv-1.224744871391589*fr[15]*wv+0.7071067811865475*fr[9]*wv+0.4564354645876384*fr[24]*dv-0.3535533905932737*fr[19]*dv+0.2041241452319315*fr[16]*dv; 
  Ghat_r[6] = 1.58113883008419*fr[23]*wv-1.224744871391589*fr[18]*wv+0.7071067811865475*fr[14]*wv+0.408248290463863*fr[17]*dv-0.3162277660168379*fr[10]*dv+0.1825741858350553*fr[6]*dv; 
  Ghat_r[7] = 1.58113883008419*fr[24]*wv-1.224744871391589*fr[19]*wv+0.7071067811865475*fr[16]*wv+0.408248290463863*fr[26]*dv-0.3162277660168379*fr[25]*dv+0.1825741858350553*fr[22]*dv+0.4564354645876383*fr[21]*dv-0.3535533905932737*fr[15]*dv+0.2041241452319315*fr[9]*dv; 
  Ghat_r[8] = 1.58113883008419*fr[26]*wv-1.224744871391589*fr[25]*wv+0.7071067811865475*fr[22]*wv+0.408248290463863*fr[24]*dv-0.3162277660168379*fr[19]*dv+0.1825741858350553*fr[16]*dv; 

  Ghat_l[0] = 1.58113883008419*fc[7]*wv-1.224744871391589*fc[1]*wv+0.7071067811865475*fc[0]*wv+0.4564354645876383*fc[11]*dv-0.3535533905932737*fc[4]*dv+0.2041241452319315*fc[2]*dv; 
  Ghat_l[1] = 1.58113883008419*fc[11]*wv-1.224744871391589*fc[4]*wv+0.7071067811865475*fc[2]*wv+0.408248290463863*fc[20]*dv-0.3162277660168379*fc[12]*dv+0.1825741858350554*fc[8]*dv+0.4564354645876384*fc[7]*dv-0.3535533905932737*fc[1]*dv+0.2041241452319315*fc[0]*dv; 
  Ghat_l[2] = 1.58113883008419*fc[13]*wv-1.224744871391589*fc[5]*wv+0.7071067811865475*fc[3]*wv+0.4564354645876384*fc[17]*dv-0.3535533905932737*fc[10]*dv+0.2041241452319315*fc[6]*dv; 
  Ghat_l[3] = 1.58113883008419*fc[17]*wv-1.224744871391589*fc[10]*wv+0.7071067811865475*fc[6]*wv+0.408248290463863*fc[23]*dv-0.3162277660168379*fc[18]*dv+0.1825741858350553*fc[14]*dv+0.4564354645876383*fc[13]*dv-0.3535533905932737*fc[5]*dv+0.2041241452319315*fc[3]*dv; 
  Ghat_l[4] = 1.58113883008419*fc[20]*wv-1.224744871391589*fc[12]*wv+0.7071067811865475*fc[8]*wv+0.408248290463863*fc[11]*dv-0.3162277660168379*fc[4]*dv+0.1825741858350554*fc[2]*dv; 
  Ghat_l[5] = 1.58113883008419*fc[21]*wv-1.224744871391589*fc[15]*wv+0.7071067811865475*fc[9]*wv+0.4564354645876384*fc[24]*dv-0.3535533905932737*fc[19]*dv+0.2041241452319315*fc[16]*dv; 
  Ghat_l[6] = 1.58113883008419*fc[23]*wv-1.224744871391589*fc[18]*wv+0.7071067811865475*fc[14]*wv+0.408248290463863*fc[17]*dv-0.3162277660168379*fc[10]*dv+0.1825741858350553*fc[6]*dv; 
  Ghat_l[7] = 1.58113883008419*fc[24]*wv-1.224744871391589*fc[19]*wv+0.7071067811865475*fc[16]*wv+0.408248290463863*fc[26]*dv-0.3162277660168379*fc[25]*dv+0.1825741858350553*fc[22]*dv+0.4564354645876383*fc[21]*dv-0.3535533905932737*fc[15]*dv+0.2041241452319315*fc[9]*dv; 
  Ghat_l[8] = 1.58113883008419*fc[26]*wv-1.224744871391589*fc[25]*wv+0.7071067811865475*fc[22]*wv+0.408248290463863*fc[24]*dv-0.3162277660168379*fc[19]*dv+0.1825741858350553*fc[16]*dv; 

  } 
  out[0] += (0.7071067811865475*Ghat_l[0]-0.7071067811865475*Ghat_r[0])*dx10; 
  out[1] += -1.224744871391589*(Ghat_r[0]+Ghat_l[0])*dx10; 
  out[2] += (0.7071067811865475*Ghat_l[1]-0.7071067811865475*Ghat_r[1])*dx10; 
  out[3] += (0.7071067811865475*Ghat_l[2]-0.7071067811865475*Ghat_r[2])*dx10; 
  out[4] += -1.224744871391589*(Ghat_r[1]+Ghat_l[1])*dx10; 
  out[5] += -1.224744871391589*(Ghat_r[2]+Ghat_l[2])*dx10; 
  out[6] += (0.7071067811865475*Ghat_l[3]-0.7071067811865475*Ghat_r[3])*dx10; 
  out[7] += (1.58113883008419*Ghat_l[0]-1.58113883008419*Ghat_r[0])*dx10; 
  out[8] += (0.7071067811865475*Ghat_l[4]-0.7071067811865475*Ghat_r[4])*dx10; 
  out[9] += (0.7071067811865475*Ghat_l[5]-0.7071067811865475*Ghat_r[5])*dx10; 
  out[10] += -1.224744871391589*(Ghat_r[3]+Ghat_l[3])*dx10; 
  out[11] += (1.58113883008419*Ghat_l[1]-1.58113883008419*Ghat_r[1])*dx10; 
  out[12] += -1.224744871391589*(Ghat_r[4]+Ghat_l[4])*dx10; 
  out[13] += (1.58113883008419*Ghat_l[2]-1.58113883008419*Ghat_r[2])*dx10; 
  out[14] += (0.7071067811865475*Ghat_l[6]-0.7071067811865475*Ghat_r[6])*dx10; 
  out[15] += -1.224744871391589*(Ghat_r[5]+Ghat_l[5])*dx10; 
  out[16] += (0.7071067811865475*Ghat_l[7]-0.7071067811865475*Ghat_r[7])*dx10; 
  out[17] += (1.58113883008419*Ghat_l[3]-1.58113883008419*Ghat_r[3])*dx10; 
  out[18] += -1.224744871391589*(Ghat_r[6]+Ghat_l[6])*dx10; 
  out[19] += -1.224744871391589*(Ghat_r[7]+Ghat_l[7])*dx10; 
  out[20] += (1.58113883008419*Ghat_l[4]-1.58113883008419*Ghat_r[4])*dx10; 
  out[21] += (1.58113883008419*Ghat_l[5]-1.58113883008419*Ghat_r[5])*dx10; 
  out[22] += (0.7071067811865475*Ghat_l[8]-0.7071067811865475*Ghat_r[8])*dx10; 
  out[23] += (1.58113883008419*Ghat_l[6]-1.58113883008419*Ghat_r[6])*dx10; 
  out[24] += (1.58113883008419*Ghat_l[7]-1.58113883008419*Ghat_r[7])*dx10; 
  out[25] += -1.224744871391589*(Ghat_r[8]+Ghat_l[8])*dx10; 
  out[26] += (1.58113883008419*Ghat_l[8]-1.58113883008419*Ghat_r[8])*dx10; 

  return 0.;

} 