#include <gkyl_vlasov_kernels.h> 
GKYL_CU_DH void vlasov_sr_surfx_1x1v_ser_p2(const double *w, const double *dxv, const double *p_over_gamma, const double *fl, const double *fc, const double *fr, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]:   Cell-center coordinates.
  // dxv[NDIM]: Cell spacing.
  // p_over_gamma: p/gamma (velocity).
  // fl/fc/fr:  Input Distribution function in left/center/right cells.
  // out:       Incremented distribution function in center cell.
  const double *p0_over_gamma = &p_over_gamma[0]; 
  double alpha[3] = {0.0}; 
  alpha[0] = p0_over_gamma[0]; 
  alpha[1] = p0_over_gamma[1]; 
  alpha[2] = p0_over_gamma[2]; 

  const double dx10 = 2/dxv[0]; 
  const double dv = dxv[1], wv = w[1]; 
  double Ghat_r[3]; 
  double Ghat_l[3]; 
  if (wv>0) { 

  Ghat_r[0] = 0.8660254037844386*alpha[2]*fc[7]+1.118033988749895*alpha[1]*fc[6]+0.5*alpha[2]*fc[5]+1.118033988749895*alpha[0]*fc[4]+alpha[1]*(0.8660254037844386*fc[3]+0.5*fc[2])+alpha[0]*(0.8660254037844386*fc[1]+0.5*fc[0]); 
  Ghat_r[1] = 0.7745966692414833*alpha[1]*fc[7]+(alpha[2]+1.118033988749895*alpha[0])*fc[6]+alpha[1]*(0.4472135954999579*fc[5]+1.118033988749895*fc[4])+(0.7745966692414833*alpha[2]+0.8660254037844386*alpha[0])*fc[3]+(0.4472135954999579*alpha[2]+0.5*alpha[0])*fc[2]+alpha[1]*(0.8660254037844386*fc[1]+0.5*fc[0]); 
  Ghat_r[2] = (0.5532833351724881*alpha[2]+0.8660254037844386*alpha[0])*fc[7]+alpha[1]*fc[6]+(0.31943828249997*alpha[2]+0.5*alpha[0])*fc[5]+1.118033988749895*alpha[2]*fc[4]+alpha[1]*(0.7745966692414833*fc[3]+0.4472135954999579*fc[2])+(0.8660254037844386*fc[1]+0.5*fc[0])*alpha[2]; 

  Ghat_l[0] = 0.8660254037844386*alpha[2]*fl[7]+1.118033988749895*alpha[1]*fl[6]+0.5*alpha[2]*fl[5]+1.118033988749895*alpha[0]*fl[4]+alpha[1]*(0.8660254037844386*fl[3]+0.5*fl[2])+alpha[0]*(0.8660254037844386*fl[1]+0.5*fl[0]); 
  Ghat_l[1] = 0.7745966692414833*alpha[1]*fl[7]+(alpha[2]+1.118033988749895*alpha[0])*fl[6]+alpha[1]*(0.4472135954999579*fl[5]+1.118033988749895*fl[4])+(0.7745966692414833*alpha[2]+0.8660254037844386*alpha[0])*fl[3]+(0.4472135954999579*alpha[2]+0.5*alpha[0])*fl[2]+alpha[1]*(0.8660254037844386*fl[1]+0.5*fl[0]); 
  Ghat_l[2] = (0.5532833351724881*alpha[2]+0.8660254037844386*alpha[0])*fl[7]+alpha[1]*fl[6]+(0.31943828249997*alpha[2]+0.5*alpha[0])*fl[5]+1.118033988749895*alpha[2]*fl[4]+alpha[1]*(0.7745966692414833*fl[3]+0.4472135954999579*fl[2])+(0.8660254037844386*fl[1]+0.5*fl[0])*alpha[2]; 

  } else { 

  Ghat_r[0] = (-0.8660254037844387*alpha[2]*fr[7])+1.118033988749895*alpha[1]*fr[6]+0.5*alpha[2]*fr[5]+1.118033988749895*alpha[0]*fr[4]-0.8660254037844386*alpha[1]*fr[3]+0.5*alpha[1]*fr[2]-0.8660254037844386*alpha[0]*fr[1]+0.5*alpha[0]*fr[0]; 
  Ghat_r[1] = (-0.7745966692414834*alpha[1]*fr[7])+1.0*alpha[2]*fr[6]+1.118033988749895*alpha[0]*fr[6]+0.4472135954999579*alpha[1]*fr[5]+1.118033988749895*alpha[1]*fr[4]-0.7745966692414833*alpha[2]*fr[3]-0.8660254037844386*alpha[0]*fr[3]+0.4472135954999579*alpha[2]*fr[2]+0.5*alpha[0]*fr[2]-0.8660254037844386*alpha[1]*fr[1]+0.5*fr[0]*alpha[1]; 
  Ghat_r[2] = (-0.5532833351724881*alpha[2]*fr[7])-0.8660254037844387*alpha[0]*fr[7]+1.0*alpha[1]*fr[6]+0.31943828249997*alpha[2]*fr[5]+0.5*alpha[0]*fr[5]+1.118033988749895*alpha[2]*fr[4]-0.7745966692414833*alpha[1]*fr[3]+0.4472135954999579*alpha[1]*fr[2]-0.8660254037844386*fr[1]*alpha[2]+0.5*fr[0]*alpha[2]; 

  Ghat_l[0] = (-0.8660254037844387*alpha[2]*fc[7])+1.118033988749895*alpha[1]*fc[6]+0.5*alpha[2]*fc[5]+1.118033988749895*alpha[0]*fc[4]-0.8660254037844386*alpha[1]*fc[3]+0.5*alpha[1]*fc[2]-0.8660254037844386*alpha[0]*fc[1]+0.5*alpha[0]*fc[0]; 
  Ghat_l[1] = (-0.7745966692414834*alpha[1]*fc[7])+1.0*alpha[2]*fc[6]+1.118033988749895*alpha[0]*fc[6]+0.4472135954999579*alpha[1]*fc[5]+1.118033988749895*alpha[1]*fc[4]-0.7745966692414833*alpha[2]*fc[3]-0.8660254037844386*alpha[0]*fc[3]+0.4472135954999579*alpha[2]*fc[2]+0.5*alpha[0]*fc[2]-0.8660254037844386*alpha[1]*fc[1]+0.5*fc[0]*alpha[1]; 
  Ghat_l[2] = (-0.5532833351724881*alpha[2]*fc[7])-0.8660254037844387*alpha[0]*fc[7]+1.0*alpha[1]*fc[6]+0.31943828249997*alpha[2]*fc[5]+0.5*alpha[0]*fc[5]+1.118033988749895*alpha[2]*fc[4]-0.7745966692414833*alpha[1]*fc[3]+0.4472135954999579*alpha[1]*fc[2]-0.8660254037844386*fc[1]*alpha[2]+0.5*fc[0]*alpha[2]; 

  } 
  out[0] += (0.7071067811865475*Ghat_l[0]-0.7071067811865475*Ghat_r[0])*dx10; 
  out[1] += -1.224744871391589*(Ghat_r[0]+Ghat_l[0])*dx10; 
  out[2] += (0.7071067811865475*Ghat_l[1]-0.7071067811865475*Ghat_r[1])*dx10; 
  out[3] += -1.224744871391589*(Ghat_r[1]+Ghat_l[1])*dx10; 
  out[4] += (1.58113883008419*Ghat_l[0]-1.58113883008419*Ghat_r[0])*dx10; 
  out[5] += (0.7071067811865475*Ghat_l[2]-0.7071067811865475*Ghat_r[2])*dx10; 
  out[6] += (1.58113883008419*Ghat_l[1]-1.58113883008419*Ghat_r[1])*dx10; 
  out[7] += -1.224744871391589*(Ghat_r[2]+Ghat_l[2])*dx10; 
} 
