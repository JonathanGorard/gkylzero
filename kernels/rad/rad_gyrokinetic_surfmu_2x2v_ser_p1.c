#include <gkyl_rad_gyrokinetic_kernels.h> 
GKYL_CU_DH double rad_gyrokinetic_surfmu_2x2v_ser_p1(const double *w, const double *dxv, 
  const double *nvnu_l, const double *nvnu_r, const double *nvsqnu_l, const double *nvsqnu_r, 
  const double *fl, const double *fc, const double *fr, 
  double* GKYL_RESTRICT out) 
{ 
  // w[4]:     cell-center coordinates. 
  // dxv[4]:   cell spacing. 
  // nvnu_l: Surface expansion sum_s n_s*nu_s(v) in vparallel direction on the left.
  // nvnu_r: Surface expansion sum_s n_s*nu_s(v) in vparallel direction on the right.
  // nvsqnu_l: Surface expansion sum_s n_s*nu_s(v) in mu direction on the left.
  // nvsqnu_r: Surface expansion sum_s n_s*nu_s(v) in mu direction on the right.
  // fl/fc/fr:  distribution function in cells 
  // out:       incremented distribution function in cell 

  double rdv2 = 2.0/dxv[3]; 

  double Ghat_r[12] = {0.0}; 
  double Ghat_l[12] = {0.0}; 
  Ghat_l[0] = (-0.4330127018922194*(nvsqnu_l[11]*fc[23]+nvsqnu_l[10]*fc[22]+nvsqnu_l[9]*fc[21]))+0.25*nvsqnu_l[11]*fc[20]-0.4330127018922194*nvsqnu_l[8]*fc[19]+0.25*(nvsqnu_l[10]*fc[18]+nvsqnu_l[9]*fc[17]+nvsqnu_l[8]*fc[16])-0.4330127018922193*(nvsqnu_l[7]*fc[15]+nvsqnu_l[6]*fc[14]+nvsqnu_l[5]*fc[13]+nvsqnu_l[4]*fc[12])+0.25*nvsqnu_l[7]*fc[11]-0.4330127018922193*(nvsqnu_l[3]*fc[10]+nvsqnu_l[2]*fc[9]+nvsqnu_l[1]*fc[8])+0.25*(nvsqnu_l[6]*fc[7]+nvsqnu_l[5]*fc[6]+nvsqnu_l[4]*fc[5])-0.4330127018922193*nvsqnu_l[0]*fc[4]+0.25*(fc[3]*nvsqnu_l[3]+fc[2]*nvsqnu_l[2]+fc[1]*nvsqnu_l[1]+fc[0]*nvsqnu_l[0]); 
  Ghat_l[1] = (-0.4330127018922193*(nvsqnu_l[10]*fc[23]+nvsqnu_l[11]*fc[22]+nvsqnu_l[8]*fc[21]))+0.2500000000000001*nvsqnu_l[10]*fc[20]-0.4330127018922193*nvsqnu_l[9]*fc[19]+0.2500000000000001*(nvsqnu_l[11]*fc[18]+nvsqnu_l[8]*fc[17]+nvsqnu_l[9]*fc[16])-0.4330127018922193*(nvsqnu_l[6]*fc[15]+nvsqnu_l[7]*fc[14]+nvsqnu_l[3]*fc[13]+nvsqnu_l[2]*fc[12])+0.25*nvsqnu_l[6]*fc[11]-0.4330127018922193*(nvsqnu_l[5]*fc[10]+nvsqnu_l[4]*fc[9]+nvsqnu_l[0]*fc[8])+0.25*(fc[7]*nvsqnu_l[7]+nvsqnu_l[3]*fc[6]+fc[3]*nvsqnu_l[5]+nvsqnu_l[2]*fc[5]+fc[2]*nvsqnu_l[4])-0.4330127018922193*nvsqnu_l[1]*fc[4]+0.25*(fc[0]*nvsqnu_l[1]+nvsqnu_l[0]*fc[1]); 
  Ghat_l[2] = (-0.4330127018922193*(nvsqnu_l[9]*fc[23]+nvsqnu_l[8]*fc[22]+nvsqnu_l[11]*fc[21]))+0.2500000000000001*nvsqnu_l[9]*fc[20]-0.4330127018922193*nvsqnu_l[10]*fc[19]+0.2500000000000001*(nvsqnu_l[8]*fc[18]+nvsqnu_l[11]*fc[17]+nvsqnu_l[10]*fc[16])-0.4330127018922193*(nvsqnu_l[5]*fc[15]+nvsqnu_l[3]*fc[14]+nvsqnu_l[7]*fc[13]+nvsqnu_l[1]*fc[12])+0.25*nvsqnu_l[5]*fc[11]-0.4330127018922193*(nvsqnu_l[6]*fc[10]+nvsqnu_l[0]*fc[9]+nvsqnu_l[4]*fc[8])+0.25*(fc[6]*nvsqnu_l[7]+nvsqnu_l[3]*fc[7]+fc[3]*nvsqnu_l[6]+nvsqnu_l[1]*fc[5]+fc[1]*nvsqnu_l[4])-0.4330127018922193*nvsqnu_l[2]*fc[4]+0.25*(fc[0]*nvsqnu_l[2]+nvsqnu_l[0]*fc[2]); 
  Ghat_l[3] = (-0.3872983346207417*nvsqnu_l[7]*fc[23])-0.3872983346207416*(nvsqnu_l[6]*fc[22]+nvsqnu_l[5]*fc[21])+0.223606797749979*nvsqnu_l[7]*fc[20]-0.3872983346207417*nvsqnu_l[3]*fc[19]+0.223606797749979*(nvsqnu_l[6]*fc[18]+nvsqnu_l[5]*fc[17])+0.223606797749979*nvsqnu_l[3]*fc[16]+((-0.3872983346207416*nvsqnu_l[11])-0.4330127018922193*nvsqnu_l[4])*fc[15]+((-0.3872983346207417*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[14]-0.3872983346207417*nvsqnu_l[9]*fc[13]-0.4330127018922193*(nvsqnu_l[1]*fc[13]+nvsqnu_l[7]*fc[12])+fc[11]*(0.223606797749979*nvsqnu_l[11]+0.25*nvsqnu_l[4])+0.223606797749979*fc[7]*nvsqnu_l[10]+((-0.3872983346207416*nvsqnu_l[8])-0.4330127018922193*nvsqnu_l[0])*fc[10]+0.223606797749979*fc[6]*nvsqnu_l[9]-0.4330127018922193*nvsqnu_l[6]*fc[9]+0.223606797749979*fc[3]*nvsqnu_l[8]-0.4330127018922193*nvsqnu_l[5]*fc[8]+0.25*(fc[5]*nvsqnu_l[7]+nvsqnu_l[2]*fc[7]+fc[2]*nvsqnu_l[6]+nvsqnu_l[1]*fc[6]+fc[1]*nvsqnu_l[5])-0.4330127018922193*nvsqnu_l[3]*fc[4]+0.25*(fc[0]*nvsqnu_l[3]+nvsqnu_l[0]*fc[3]); 
  Ghat_l[4] = (-0.4330127018922194*(nvsqnu_l[8]*fc[23]+nvsqnu_l[9]*fc[22]+nvsqnu_l[10]*fc[21]))+0.25*nvsqnu_l[8]*fc[20]-0.4330127018922194*nvsqnu_l[11]*fc[19]+0.25*(nvsqnu_l[9]*fc[18]+nvsqnu_l[10]*fc[17]+nvsqnu_l[11]*fc[16])-0.4330127018922193*(nvsqnu_l[3]*fc[15]+nvsqnu_l[5]*fc[14]+nvsqnu_l[6]*fc[13]+nvsqnu_l[0]*fc[12])+0.25*nvsqnu_l[3]*fc[11]-0.4330127018922193*(nvsqnu_l[7]*fc[10]+nvsqnu_l[1]*fc[9]+nvsqnu_l[2]*fc[8])+0.25*(fc[3]*nvsqnu_l[7]+nvsqnu_l[5]*fc[7]+fc[6]*nvsqnu_l[6]+nvsqnu_l[0]*fc[5])-0.4330127018922193*fc[4]*nvsqnu_l[4]+0.25*(fc[0]*nvsqnu_l[4]+fc[1]*nvsqnu_l[2]+nvsqnu_l[1]*fc[2]); 
  Ghat_l[5] = (-0.3872983346207417*nvsqnu_l[6]*fc[23])-0.3872983346207416*(nvsqnu_l[7]*fc[22]+nvsqnu_l[3]*fc[21])+0.223606797749979*nvsqnu_l[6]*fc[20]-0.3872983346207417*nvsqnu_l[5]*fc[19]+0.223606797749979*(nvsqnu_l[7]*fc[18]+nvsqnu_l[3]*fc[17])+0.223606797749979*nvsqnu_l[5]*fc[16]+((-0.3872983346207417*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[15]+((-0.3872983346207416*nvsqnu_l[11])-0.4330127018922193*nvsqnu_l[4])*fc[14]-0.3872983346207416*nvsqnu_l[8]*fc[13]-0.4330127018922193*(nvsqnu_l[0]*fc[13]+nvsqnu_l[6]*fc[12])+0.223606797749979*fc[7]*nvsqnu_l[11]+(0.223606797749979*nvsqnu_l[10]+0.25*nvsqnu_l[2])*fc[11]+((-0.3872983346207417*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[10]+0.223606797749979*fc[3]*nvsqnu_l[9]-0.4330127018922193*nvsqnu_l[7]*fc[9]+0.223606797749979*fc[6]*nvsqnu_l[8]-0.4330127018922193*nvsqnu_l[3]*fc[8]+0.25*(fc[2]*nvsqnu_l[7]+nvsqnu_l[4]*fc[7]+fc[5]*nvsqnu_l[6]+nvsqnu_l[0]*fc[6])-0.4330127018922193*fc[4]*nvsqnu_l[5]+0.25*(fc[0]*nvsqnu_l[5]+fc[1]*nvsqnu_l[3]+nvsqnu_l[1]*fc[3]); 
  Ghat_l[6] = (-0.3872983346207417*nvsqnu_l[5]*fc[23])-0.3872983346207416*(nvsqnu_l[3]*fc[22]+nvsqnu_l[7]*fc[21])+0.223606797749979*nvsqnu_l[5]*fc[20]-0.3872983346207417*nvsqnu_l[6]*fc[19]+0.223606797749979*(nvsqnu_l[3]*fc[18]+nvsqnu_l[7]*fc[17])+0.223606797749979*nvsqnu_l[6]*fc[16]+((-0.3872983346207417*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[15]+((-0.3872983346207416*nvsqnu_l[8])-0.4330127018922193*nvsqnu_l[0])*fc[14]-0.3872983346207416*nvsqnu_l[11]*fc[13]-0.4330127018922193*(nvsqnu_l[4]*fc[13]+nvsqnu_l[5]*fc[12])+0.223606797749979*fc[6]*nvsqnu_l[11]+(0.223606797749979*nvsqnu_l[9]+0.25*nvsqnu_l[1])*fc[11]+(0.223606797749979*fc[3]-0.3872983346207417*fc[10])*nvsqnu_l[10]-0.4330127018922193*(nvsqnu_l[2]*fc[10]+nvsqnu_l[3]*fc[9])+0.223606797749979*fc[7]*nvsqnu_l[8]-0.4330127018922193*nvsqnu_l[7]*fc[8]+0.25*(fc[1]*nvsqnu_l[7]+nvsqnu_l[0]*fc[7])-0.4330127018922193*fc[4]*nvsqnu_l[6]+0.25*(fc[0]*nvsqnu_l[6]+nvsqnu_l[4]*fc[6]+fc[5]*nvsqnu_l[5]+fc[2]*nvsqnu_l[3]+nvsqnu_l[2]*fc[3]); 
  Ghat_l[7] = (-0.3872983346207417*nvsqnu_l[3]*fc[23])-0.3872983346207416*(nvsqnu_l[5]*fc[22]+nvsqnu_l[6]*fc[21])+0.223606797749979*nvsqnu_l[3]*fc[20]-0.3872983346207417*nvsqnu_l[7]*fc[19]+0.223606797749979*(nvsqnu_l[5]*fc[18]+nvsqnu_l[6]*fc[17])+0.223606797749979*nvsqnu_l[7]*fc[16]+((-0.3872983346207416*nvsqnu_l[8])-0.4330127018922193*nvsqnu_l[0])*fc[15]+((-0.3872983346207417*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[14]-0.3872983346207417*nvsqnu_l[10]*fc[13]-0.4330127018922193*(nvsqnu_l[2]*fc[13]+nvsqnu_l[3]*fc[12])+(0.223606797749979*fc[3]-0.3872983346207416*fc[10])*nvsqnu_l[11]+(0.223606797749979*nvsqnu_l[8]+0.25*nvsqnu_l[0])*fc[11]+0.223606797749979*fc[6]*nvsqnu_l[10]-0.4330127018922193*nvsqnu_l[4]*fc[10]+0.223606797749979*fc[7]*nvsqnu_l[9]-0.4330127018922193*(nvsqnu_l[5]*fc[9]+nvsqnu_l[6]*fc[8]+fc[4]*nvsqnu_l[7])+0.25*(fc[0]*nvsqnu_l[7]+nvsqnu_l[1]*fc[7]+fc[1]*nvsqnu_l[6]+nvsqnu_l[2]*fc[6]+fc[2]*nvsqnu_l[5]+nvsqnu_l[3]*fc[5]+fc[3]*nvsqnu_l[4]); 
  Ghat_l[8] = ((-0.276641667586244*nvsqnu_l[11])-0.4330127018922194*nvsqnu_l[4])*fc[23]+((-0.276641667586244*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[22]+((-0.276641667586244*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[21]+(0.159719141249985*nvsqnu_l[11]+0.25*nvsqnu_l[4])*fc[20]+((-0.276641667586244*nvsqnu_l[8])-0.4330127018922194*nvsqnu_l[0])*fc[19]+(0.159719141249985*nvsqnu_l[10]+0.2500000000000001*nvsqnu_l[2])*fc[18]+(0.159719141249985*nvsqnu_l[9]+0.2500000000000001*nvsqnu_l[1])*fc[17]+(0.159719141249985*nvsqnu_l[8]+0.25*nvsqnu_l[0])*fc[16]-0.3872983346207416*(nvsqnu_l[7]*fc[15]+nvsqnu_l[6]*fc[14]+nvsqnu_l[5]*fc[13])+nvsqnu_l[11]*(0.25*fc[5]-0.4330127018922193*fc[12])+0.223606797749979*nvsqnu_l[7]*fc[11]+(0.2500000000000001*fc[2]-0.4330127018922194*fc[9])*nvsqnu_l[10]-0.3872983346207416*nvsqnu_l[3]*fc[10]+(0.2500000000000001*fc[1]-0.4330127018922194*fc[8])*nvsqnu_l[9]+(0.25*fc[0]-0.4330127018922193*fc[4])*nvsqnu_l[8]+0.223606797749979*(nvsqnu_l[6]*fc[7]+nvsqnu_l[5]*fc[6]+fc[3]*nvsqnu_l[3]); 
  Ghat_l[9] = ((-0.276641667586244*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[23]+((-0.276641667586244*nvsqnu_l[11])-0.4330127018922194*nvsqnu_l[4])*fc[22]+((-0.276641667586244*nvsqnu_l[8])-0.4330127018922194*nvsqnu_l[0])*fc[21]+(0.159719141249985*nvsqnu_l[10]+0.2500000000000001*nvsqnu_l[2])*fc[20]+((-0.276641667586244*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[19]+(0.159719141249985*nvsqnu_l[11]+0.25*nvsqnu_l[4])*fc[18]+(0.159719141249985*nvsqnu_l[8]+0.25*nvsqnu_l[0])*fc[17]+(0.159719141249985*nvsqnu_l[9]+0.2500000000000001*nvsqnu_l[1])*fc[16]-0.3872983346207417*(nvsqnu_l[6]*fc[15]+nvsqnu_l[7]*fc[14]+nvsqnu_l[3]*fc[13])-0.4330127018922193*nvsqnu_l[10]*fc[12]+(0.2500000000000001*fc[2]-0.4330127018922194*fc[9])*nvsqnu_l[11]+0.223606797749979*nvsqnu_l[6]*fc[11]+0.25*fc[5]*nvsqnu_l[10]-0.3872983346207417*nvsqnu_l[5]*fc[10]+(0.25*fc[0]-0.4330127018922193*fc[4])*nvsqnu_l[9]+(0.2500000000000001*fc[1]-0.4330127018922194*fc[8])*nvsqnu_l[8]+0.223606797749979*(fc[7]*nvsqnu_l[7]+nvsqnu_l[3]*fc[6]+fc[3]*nvsqnu_l[5]); 
  Ghat_l[10] = ((-0.276641667586244*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[23]+((-0.276641667586244*nvsqnu_l[8])-0.4330127018922194*nvsqnu_l[0])*fc[22]+((-0.276641667586244*nvsqnu_l[11])-0.4330127018922194*nvsqnu_l[4])*fc[21]+(0.159719141249985*nvsqnu_l[9]+0.2500000000000001*nvsqnu_l[1])*fc[20]+((-0.276641667586244*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[19]+(0.159719141249985*nvsqnu_l[8]+0.25*nvsqnu_l[0])*fc[18]+(0.159719141249985*nvsqnu_l[11]+0.25*nvsqnu_l[4])*fc[17]+(0.159719141249985*nvsqnu_l[10]+0.2500000000000001*nvsqnu_l[2])*fc[16]-0.3872983346207417*(nvsqnu_l[5]*fc[15]+nvsqnu_l[3]*fc[14]+nvsqnu_l[7]*fc[13])-0.4330127018922193*nvsqnu_l[9]*fc[12]+(0.2500000000000001*fc[1]-0.4330127018922194*fc[8])*nvsqnu_l[11]+0.223606797749979*nvsqnu_l[5]*fc[11]+(0.25*fc[0]-0.4330127018922193*fc[4])*nvsqnu_l[10]-0.3872983346207417*nvsqnu_l[6]*fc[10]+0.25*fc[5]*nvsqnu_l[9]+nvsqnu_l[8]*(0.2500000000000001*fc[2]-0.4330127018922194*fc[9])+0.223606797749979*(fc[6]*nvsqnu_l[7]+nvsqnu_l[3]*fc[7]+fc[3]*nvsqnu_l[6]); 
  Ghat_l[11] = ((-0.276641667586244*nvsqnu_l[8])-0.4330127018922194*nvsqnu_l[0])*fc[23]+((-0.276641667586244*nvsqnu_l[9])-0.4330127018922193*nvsqnu_l[1])*fc[22]+((-0.276641667586244*nvsqnu_l[10])-0.4330127018922193*nvsqnu_l[2])*fc[21]+(0.159719141249985*nvsqnu_l[8]+0.25*nvsqnu_l[0])*fc[20]+((-0.276641667586244*nvsqnu_l[11])-0.4330127018922194*nvsqnu_l[4])*fc[19]+(0.159719141249985*nvsqnu_l[9]+0.2500000000000001*nvsqnu_l[1])*fc[18]+(0.159719141249985*nvsqnu_l[10]+0.2500000000000001*nvsqnu_l[2])*fc[17]+(0.159719141249985*nvsqnu_l[11]+0.25*nvsqnu_l[4])*fc[16]-0.3872983346207416*(nvsqnu_l[3]*fc[15]+nvsqnu_l[5]*fc[14]+nvsqnu_l[6]*fc[13])-0.4330127018922193*nvsqnu_l[8]*fc[12]+(0.25*fc[0]-0.4330127018922193*fc[4])*nvsqnu_l[11]+0.223606797749979*nvsqnu_l[3]*fc[11]+(0.2500000000000001*fc[1]-0.4330127018922194*fc[8])*nvsqnu_l[10]-0.3872983346207416*nvsqnu_l[7]*fc[10]+(0.2500000000000001*fc[2]-0.4330127018922194*fc[9])*nvsqnu_l[9]+0.25*fc[5]*nvsqnu_l[8]+0.223606797749979*(fc[3]*nvsqnu_l[7]+nvsqnu_l[5]*fc[7]+fc[6]*nvsqnu_l[6]); 

  Ghat_r[0] = (-0.4330127018922194*(nvsqnu_r[11]*fr[23]+nvsqnu_r[10]*fr[22]+nvsqnu_r[9]*fr[21]))+0.25*nvsqnu_r[11]*fr[20]-0.4330127018922194*nvsqnu_r[8]*fr[19]+0.25*(nvsqnu_r[10]*fr[18]+nvsqnu_r[9]*fr[17]+nvsqnu_r[8]*fr[16])-0.4330127018922193*(nvsqnu_r[7]*fr[15]+nvsqnu_r[6]*fr[14]+nvsqnu_r[5]*fr[13]+nvsqnu_r[4]*fr[12])+0.25*nvsqnu_r[7]*fr[11]-0.4330127018922193*(nvsqnu_r[3]*fr[10]+nvsqnu_r[2]*fr[9]+nvsqnu_r[1]*fr[8])+0.25*(nvsqnu_r[6]*fr[7]+nvsqnu_r[5]*fr[6]+nvsqnu_r[4]*fr[5])-0.4330127018922193*nvsqnu_r[0]*fr[4]+0.25*(fr[3]*nvsqnu_r[3]+fr[2]*nvsqnu_r[2]+fr[1]*nvsqnu_r[1]+fr[0]*nvsqnu_r[0]); 
  Ghat_r[1] = (-0.4330127018922193*(nvsqnu_r[10]*fr[23]+nvsqnu_r[11]*fr[22]+nvsqnu_r[8]*fr[21]))+0.2500000000000001*nvsqnu_r[10]*fr[20]-0.4330127018922193*nvsqnu_r[9]*fr[19]+0.2500000000000001*(nvsqnu_r[11]*fr[18]+nvsqnu_r[8]*fr[17]+nvsqnu_r[9]*fr[16])-0.4330127018922193*(nvsqnu_r[6]*fr[15]+nvsqnu_r[7]*fr[14]+nvsqnu_r[3]*fr[13]+nvsqnu_r[2]*fr[12])+0.25*nvsqnu_r[6]*fr[11]-0.4330127018922193*(nvsqnu_r[5]*fr[10]+nvsqnu_r[4]*fr[9]+nvsqnu_r[0]*fr[8])+0.25*(fr[7]*nvsqnu_r[7]+nvsqnu_r[3]*fr[6]+fr[3]*nvsqnu_r[5]+nvsqnu_r[2]*fr[5]+fr[2]*nvsqnu_r[4])-0.4330127018922193*nvsqnu_r[1]*fr[4]+0.25*(fr[0]*nvsqnu_r[1]+nvsqnu_r[0]*fr[1]); 
  Ghat_r[2] = (-0.4330127018922193*(nvsqnu_r[9]*fr[23]+nvsqnu_r[8]*fr[22]+nvsqnu_r[11]*fr[21]))+0.2500000000000001*nvsqnu_r[9]*fr[20]-0.4330127018922193*nvsqnu_r[10]*fr[19]+0.2500000000000001*(nvsqnu_r[8]*fr[18]+nvsqnu_r[11]*fr[17]+nvsqnu_r[10]*fr[16])-0.4330127018922193*(nvsqnu_r[5]*fr[15]+nvsqnu_r[3]*fr[14]+nvsqnu_r[7]*fr[13]+nvsqnu_r[1]*fr[12])+0.25*nvsqnu_r[5]*fr[11]-0.4330127018922193*(nvsqnu_r[6]*fr[10]+nvsqnu_r[0]*fr[9]+nvsqnu_r[4]*fr[8])+0.25*(fr[6]*nvsqnu_r[7]+nvsqnu_r[3]*fr[7]+fr[3]*nvsqnu_r[6]+nvsqnu_r[1]*fr[5]+fr[1]*nvsqnu_r[4])-0.4330127018922193*nvsqnu_r[2]*fr[4]+0.25*(fr[0]*nvsqnu_r[2]+nvsqnu_r[0]*fr[2]); 
  Ghat_r[3] = (-0.3872983346207417*nvsqnu_r[7]*fr[23])-0.3872983346207416*(nvsqnu_r[6]*fr[22]+nvsqnu_r[5]*fr[21])+0.223606797749979*nvsqnu_r[7]*fr[20]-0.3872983346207417*nvsqnu_r[3]*fr[19]+0.223606797749979*(nvsqnu_r[6]*fr[18]+nvsqnu_r[5]*fr[17])+0.223606797749979*nvsqnu_r[3]*fr[16]+((-0.3872983346207416*nvsqnu_r[11])-0.4330127018922193*nvsqnu_r[4])*fr[15]+((-0.3872983346207417*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[14]-0.3872983346207417*nvsqnu_r[9]*fr[13]-0.4330127018922193*(nvsqnu_r[1]*fr[13]+nvsqnu_r[7]*fr[12])+fr[11]*(0.223606797749979*nvsqnu_r[11]+0.25*nvsqnu_r[4])+0.223606797749979*fr[7]*nvsqnu_r[10]+((-0.3872983346207416*nvsqnu_r[8])-0.4330127018922193*nvsqnu_r[0])*fr[10]+0.223606797749979*fr[6]*nvsqnu_r[9]-0.4330127018922193*nvsqnu_r[6]*fr[9]+0.223606797749979*fr[3]*nvsqnu_r[8]-0.4330127018922193*nvsqnu_r[5]*fr[8]+0.25*(fr[5]*nvsqnu_r[7]+nvsqnu_r[2]*fr[7]+fr[2]*nvsqnu_r[6]+nvsqnu_r[1]*fr[6]+fr[1]*nvsqnu_r[5])-0.4330127018922193*nvsqnu_r[3]*fr[4]+0.25*(fr[0]*nvsqnu_r[3]+nvsqnu_r[0]*fr[3]); 
  Ghat_r[4] = (-0.4330127018922194*(nvsqnu_r[8]*fr[23]+nvsqnu_r[9]*fr[22]+nvsqnu_r[10]*fr[21]))+0.25*nvsqnu_r[8]*fr[20]-0.4330127018922194*nvsqnu_r[11]*fr[19]+0.25*(nvsqnu_r[9]*fr[18]+nvsqnu_r[10]*fr[17]+nvsqnu_r[11]*fr[16])-0.4330127018922193*(nvsqnu_r[3]*fr[15]+nvsqnu_r[5]*fr[14]+nvsqnu_r[6]*fr[13]+nvsqnu_r[0]*fr[12])+0.25*nvsqnu_r[3]*fr[11]-0.4330127018922193*(nvsqnu_r[7]*fr[10]+nvsqnu_r[1]*fr[9]+nvsqnu_r[2]*fr[8])+0.25*(fr[3]*nvsqnu_r[7]+nvsqnu_r[5]*fr[7]+fr[6]*nvsqnu_r[6]+nvsqnu_r[0]*fr[5])-0.4330127018922193*fr[4]*nvsqnu_r[4]+0.25*(fr[0]*nvsqnu_r[4]+fr[1]*nvsqnu_r[2]+nvsqnu_r[1]*fr[2]); 
  Ghat_r[5] = (-0.3872983346207417*nvsqnu_r[6]*fr[23])-0.3872983346207416*(nvsqnu_r[7]*fr[22]+nvsqnu_r[3]*fr[21])+0.223606797749979*nvsqnu_r[6]*fr[20]-0.3872983346207417*nvsqnu_r[5]*fr[19]+0.223606797749979*(nvsqnu_r[7]*fr[18]+nvsqnu_r[3]*fr[17])+0.223606797749979*nvsqnu_r[5]*fr[16]+((-0.3872983346207417*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[15]+((-0.3872983346207416*nvsqnu_r[11])-0.4330127018922193*nvsqnu_r[4])*fr[14]-0.3872983346207416*nvsqnu_r[8]*fr[13]-0.4330127018922193*(nvsqnu_r[0]*fr[13]+nvsqnu_r[6]*fr[12])+0.223606797749979*fr[7]*nvsqnu_r[11]+(0.223606797749979*nvsqnu_r[10]+0.25*nvsqnu_r[2])*fr[11]+((-0.3872983346207417*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[10]+0.223606797749979*fr[3]*nvsqnu_r[9]-0.4330127018922193*nvsqnu_r[7]*fr[9]+0.223606797749979*fr[6]*nvsqnu_r[8]-0.4330127018922193*nvsqnu_r[3]*fr[8]+0.25*(fr[2]*nvsqnu_r[7]+nvsqnu_r[4]*fr[7]+fr[5]*nvsqnu_r[6]+nvsqnu_r[0]*fr[6])-0.4330127018922193*fr[4]*nvsqnu_r[5]+0.25*(fr[0]*nvsqnu_r[5]+fr[1]*nvsqnu_r[3]+nvsqnu_r[1]*fr[3]); 
  Ghat_r[6] = (-0.3872983346207417*nvsqnu_r[5]*fr[23])-0.3872983346207416*(nvsqnu_r[3]*fr[22]+nvsqnu_r[7]*fr[21])+0.223606797749979*nvsqnu_r[5]*fr[20]-0.3872983346207417*nvsqnu_r[6]*fr[19]+0.223606797749979*(nvsqnu_r[3]*fr[18]+nvsqnu_r[7]*fr[17])+0.223606797749979*nvsqnu_r[6]*fr[16]+((-0.3872983346207417*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[15]+((-0.3872983346207416*nvsqnu_r[8])-0.4330127018922193*nvsqnu_r[0])*fr[14]-0.3872983346207416*nvsqnu_r[11]*fr[13]-0.4330127018922193*(nvsqnu_r[4]*fr[13]+nvsqnu_r[5]*fr[12])+0.223606797749979*fr[6]*nvsqnu_r[11]+(0.223606797749979*nvsqnu_r[9]+0.25*nvsqnu_r[1])*fr[11]+(0.223606797749979*fr[3]-0.3872983346207417*fr[10])*nvsqnu_r[10]-0.4330127018922193*(nvsqnu_r[2]*fr[10]+nvsqnu_r[3]*fr[9])+0.223606797749979*fr[7]*nvsqnu_r[8]-0.4330127018922193*nvsqnu_r[7]*fr[8]+0.25*(fr[1]*nvsqnu_r[7]+nvsqnu_r[0]*fr[7])-0.4330127018922193*fr[4]*nvsqnu_r[6]+0.25*(fr[0]*nvsqnu_r[6]+nvsqnu_r[4]*fr[6]+fr[5]*nvsqnu_r[5]+fr[2]*nvsqnu_r[3]+nvsqnu_r[2]*fr[3]); 
  Ghat_r[7] = (-0.3872983346207417*nvsqnu_r[3]*fr[23])-0.3872983346207416*(nvsqnu_r[5]*fr[22]+nvsqnu_r[6]*fr[21])+0.223606797749979*nvsqnu_r[3]*fr[20]-0.3872983346207417*nvsqnu_r[7]*fr[19]+0.223606797749979*(nvsqnu_r[5]*fr[18]+nvsqnu_r[6]*fr[17])+0.223606797749979*nvsqnu_r[7]*fr[16]+((-0.3872983346207416*nvsqnu_r[8])-0.4330127018922193*nvsqnu_r[0])*fr[15]+((-0.3872983346207417*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[14]-0.3872983346207417*nvsqnu_r[10]*fr[13]-0.4330127018922193*(nvsqnu_r[2]*fr[13]+nvsqnu_r[3]*fr[12])+(0.223606797749979*fr[3]-0.3872983346207416*fr[10])*nvsqnu_r[11]+(0.223606797749979*nvsqnu_r[8]+0.25*nvsqnu_r[0])*fr[11]+0.223606797749979*fr[6]*nvsqnu_r[10]-0.4330127018922193*nvsqnu_r[4]*fr[10]+0.223606797749979*fr[7]*nvsqnu_r[9]-0.4330127018922193*(nvsqnu_r[5]*fr[9]+nvsqnu_r[6]*fr[8]+fr[4]*nvsqnu_r[7])+0.25*(fr[0]*nvsqnu_r[7]+nvsqnu_r[1]*fr[7]+fr[1]*nvsqnu_r[6]+nvsqnu_r[2]*fr[6]+fr[2]*nvsqnu_r[5]+nvsqnu_r[3]*fr[5]+fr[3]*nvsqnu_r[4]); 
  Ghat_r[8] = ((-0.276641667586244*nvsqnu_r[11])-0.4330127018922194*nvsqnu_r[4])*fr[23]+((-0.276641667586244*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[22]+((-0.276641667586244*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[21]+(0.159719141249985*nvsqnu_r[11]+0.25*nvsqnu_r[4])*fr[20]+((-0.276641667586244*nvsqnu_r[8])-0.4330127018922194*nvsqnu_r[0])*fr[19]+(0.159719141249985*nvsqnu_r[10]+0.2500000000000001*nvsqnu_r[2])*fr[18]+(0.159719141249985*nvsqnu_r[9]+0.2500000000000001*nvsqnu_r[1])*fr[17]+(0.159719141249985*nvsqnu_r[8]+0.25*nvsqnu_r[0])*fr[16]-0.3872983346207416*(nvsqnu_r[7]*fr[15]+nvsqnu_r[6]*fr[14]+nvsqnu_r[5]*fr[13])+nvsqnu_r[11]*(0.25*fr[5]-0.4330127018922193*fr[12])+0.223606797749979*nvsqnu_r[7]*fr[11]+(0.2500000000000001*fr[2]-0.4330127018922194*fr[9])*nvsqnu_r[10]-0.3872983346207416*nvsqnu_r[3]*fr[10]+(0.2500000000000001*fr[1]-0.4330127018922194*fr[8])*nvsqnu_r[9]+(0.25*fr[0]-0.4330127018922193*fr[4])*nvsqnu_r[8]+0.223606797749979*(nvsqnu_r[6]*fr[7]+nvsqnu_r[5]*fr[6]+fr[3]*nvsqnu_r[3]); 
  Ghat_r[9] = ((-0.276641667586244*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[23]+((-0.276641667586244*nvsqnu_r[11])-0.4330127018922194*nvsqnu_r[4])*fr[22]+((-0.276641667586244*nvsqnu_r[8])-0.4330127018922194*nvsqnu_r[0])*fr[21]+(0.159719141249985*nvsqnu_r[10]+0.2500000000000001*nvsqnu_r[2])*fr[20]+((-0.276641667586244*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[19]+(0.159719141249985*nvsqnu_r[11]+0.25*nvsqnu_r[4])*fr[18]+(0.159719141249985*nvsqnu_r[8]+0.25*nvsqnu_r[0])*fr[17]+(0.159719141249985*nvsqnu_r[9]+0.2500000000000001*nvsqnu_r[1])*fr[16]-0.3872983346207417*(nvsqnu_r[6]*fr[15]+nvsqnu_r[7]*fr[14]+nvsqnu_r[3]*fr[13])-0.4330127018922193*nvsqnu_r[10]*fr[12]+(0.2500000000000001*fr[2]-0.4330127018922194*fr[9])*nvsqnu_r[11]+0.223606797749979*nvsqnu_r[6]*fr[11]+0.25*fr[5]*nvsqnu_r[10]-0.3872983346207417*nvsqnu_r[5]*fr[10]+(0.25*fr[0]-0.4330127018922193*fr[4])*nvsqnu_r[9]+(0.2500000000000001*fr[1]-0.4330127018922194*fr[8])*nvsqnu_r[8]+0.223606797749979*(fr[7]*nvsqnu_r[7]+nvsqnu_r[3]*fr[6]+fr[3]*nvsqnu_r[5]); 
  Ghat_r[10] = ((-0.276641667586244*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[23]+((-0.276641667586244*nvsqnu_r[8])-0.4330127018922194*nvsqnu_r[0])*fr[22]+((-0.276641667586244*nvsqnu_r[11])-0.4330127018922194*nvsqnu_r[4])*fr[21]+(0.159719141249985*nvsqnu_r[9]+0.2500000000000001*nvsqnu_r[1])*fr[20]+((-0.276641667586244*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[19]+(0.159719141249985*nvsqnu_r[8]+0.25*nvsqnu_r[0])*fr[18]+(0.159719141249985*nvsqnu_r[11]+0.25*nvsqnu_r[4])*fr[17]+(0.159719141249985*nvsqnu_r[10]+0.2500000000000001*nvsqnu_r[2])*fr[16]-0.3872983346207417*(nvsqnu_r[5]*fr[15]+nvsqnu_r[3]*fr[14]+nvsqnu_r[7]*fr[13])-0.4330127018922193*nvsqnu_r[9]*fr[12]+(0.2500000000000001*fr[1]-0.4330127018922194*fr[8])*nvsqnu_r[11]+0.223606797749979*nvsqnu_r[5]*fr[11]+(0.25*fr[0]-0.4330127018922193*fr[4])*nvsqnu_r[10]-0.3872983346207417*nvsqnu_r[6]*fr[10]+0.25*fr[5]*nvsqnu_r[9]+nvsqnu_r[8]*(0.2500000000000001*fr[2]-0.4330127018922194*fr[9])+0.223606797749979*(fr[6]*nvsqnu_r[7]+nvsqnu_r[3]*fr[7]+fr[3]*nvsqnu_r[6]); 
  Ghat_r[11] = ((-0.276641667586244*nvsqnu_r[8])-0.4330127018922194*nvsqnu_r[0])*fr[23]+((-0.276641667586244*nvsqnu_r[9])-0.4330127018922193*nvsqnu_r[1])*fr[22]+((-0.276641667586244*nvsqnu_r[10])-0.4330127018922193*nvsqnu_r[2])*fr[21]+(0.159719141249985*nvsqnu_r[8]+0.25*nvsqnu_r[0])*fr[20]+((-0.276641667586244*nvsqnu_r[11])-0.4330127018922194*nvsqnu_r[4])*fr[19]+(0.159719141249985*nvsqnu_r[9]+0.2500000000000001*nvsqnu_r[1])*fr[18]+(0.159719141249985*nvsqnu_r[10]+0.2500000000000001*nvsqnu_r[2])*fr[17]+(0.159719141249985*nvsqnu_r[11]+0.25*nvsqnu_r[4])*fr[16]-0.3872983346207416*(nvsqnu_r[3]*fr[15]+nvsqnu_r[5]*fr[14]+nvsqnu_r[6]*fr[13])-0.4330127018922193*nvsqnu_r[8]*fr[12]+(0.25*fr[0]-0.4330127018922193*fr[4])*nvsqnu_r[11]+0.223606797749979*nvsqnu_r[3]*fr[11]+(0.2500000000000001*fr[1]-0.4330127018922194*fr[8])*nvsqnu_r[10]-0.3872983346207416*nvsqnu_r[7]*fr[10]+(0.2500000000000001*fr[2]-0.4330127018922194*fr[9])*nvsqnu_r[9]+0.25*fr[5]*nvsqnu_r[8]+0.223606797749979*(fr[3]*nvsqnu_r[7]+nvsqnu_r[5]*fr[7]+fr[6]*nvsqnu_r[6]); 

  out[0] += (0.7071067811865475*Ghat_r[0]-0.7071067811865475*Ghat_l[0])*rdv2; 
  out[1] += (0.7071067811865475*Ghat_r[1]-0.7071067811865475*Ghat_l[1])*rdv2; 
  out[2] += (0.7071067811865475*Ghat_r[2]-0.7071067811865475*Ghat_l[2])*rdv2; 
  out[3] += (0.7071067811865475*Ghat_r[3]-0.7071067811865475*Ghat_l[3])*rdv2; 
  out[4] += 1.224744871391589*(Ghat_r[0]+Ghat_l[0])*rdv2; 
  out[5] += (0.7071067811865475*Ghat_r[4]-0.7071067811865475*Ghat_l[4])*rdv2; 
  out[6] += (0.7071067811865475*Ghat_r[5]-0.7071067811865475*Ghat_l[5])*rdv2; 
  out[7] += (0.7071067811865475*Ghat_r[6]-0.7071067811865475*Ghat_l[6])*rdv2; 
  out[8] += 1.224744871391589*(Ghat_r[1]+Ghat_l[1])*rdv2; 
  out[9] += 1.224744871391589*(Ghat_r[2]+Ghat_l[2])*rdv2; 
  out[10] += 1.224744871391589*(Ghat_r[3]+Ghat_l[3])*rdv2; 
  out[11] += (0.7071067811865475*Ghat_r[7]-0.7071067811865475*Ghat_l[7])*rdv2; 
  out[12] += 1.224744871391589*(Ghat_r[4]+Ghat_l[4])*rdv2; 
  out[13] += 1.224744871391589*(Ghat_r[5]+Ghat_l[5])*rdv2; 
  out[14] += 1.224744871391589*(Ghat_r[6]+Ghat_l[6])*rdv2; 
  out[15] += 1.224744871391589*(Ghat_r[7]+Ghat_l[7])*rdv2; 
  out[16] += (0.7071067811865475*Ghat_r[8]-0.7071067811865475*Ghat_l[8])*rdv2; 
  out[17] += (0.7071067811865475*Ghat_r[9]-0.7071067811865475*Ghat_l[9])*rdv2; 
  out[18] += (0.7071067811865475*Ghat_r[10]-0.7071067811865475*Ghat_l[10])*rdv2; 
  out[19] += 1.224744871391589*(Ghat_r[8]+Ghat_l[8])*rdv2; 
  out[20] += (0.7071067811865475*Ghat_r[11]-0.7071067811865475*Ghat_l[11])*rdv2; 
  out[21] += 1.224744871391589*(Ghat_r[9]+Ghat_l[9])*rdv2; 
  out[22] += 1.224744871391589*(Ghat_r[10]+Ghat_l[10])*rdv2; 
  out[23] += 1.224744871391589*(Ghat_r[11]+Ghat_l[11])*rdv2; 

  double cflFreq = fmax(fabs(nvsqnu_l[0]), fabs(nvsqnu_r[0])); 
  return 0.5303300858899105*rdv2*cflFreq; 

} 