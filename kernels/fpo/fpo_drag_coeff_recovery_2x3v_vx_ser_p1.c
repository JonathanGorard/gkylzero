#include <gkyl_fpo_vlasov_kernels.h> 
GKYL_CU_DH void fpo_drag_coeff_recov_vx_2x3v_ser_p1(const double *dxv, const double *H_l, const double *H_c, const double *H_r, double *drag_coeff) {
  // dxv[NDIM]: Cell spacing. 
  // H_l/c/r:   Input potential in left/center/right cells in recovery direction.
  
  const double dv1 = 2.0/dxv[2]; 
  double *drag_coeff_x = &drag_coeff[0]; 
  double *drag_coeff_y = &drag_coeff[32]; 
  double *drag_coeff_z = &drag_coeff[64]; 
  drag_coeff_x[0] = (-0.2886751345948129*H_r[3]*dv1)-0.2886751345948129*H_l[3]*dv1+0.5773502691896258*H_c[3]*dv1+0.25*H_r[0]*dv1-0.25*H_l[0]*dv1; 
  drag_coeff_x[1] = (-0.2886751345948129*H_r[7]*dv1)-0.2886751345948129*H_l[7]*dv1+0.5773502691896258*H_c[7]*dv1+0.25*H_r[1]*dv1-0.25*H_l[1]*dv1; 
  drag_coeff_x[2] = (-0.2886751345948129*H_r[8]*dv1)-0.2886751345948129*H_l[8]*dv1+0.5773502691896258*H_c[8]*dv1+0.25*H_r[2]*dv1-0.25*H_l[2]*dv1; 
  drag_coeff_x[3] = (-0.5*H_r[3]*dv1)+0.5*H_l[3]*dv1+0.4330127018922193*H_r[0]*dv1+0.4330127018922193*H_l[0]*dv1-0.8660254037844386*H_c[0]*dv1; 
  drag_coeff_x[4] = (-0.2886751345948129*H_r[11]*dv1)-0.2886751345948129*H_l[11]*dv1+0.5773502691896258*H_c[11]*dv1+0.25*H_r[4]*dv1-0.25*H_l[4]*dv1; 
  drag_coeff_x[5] = (-0.2886751345948129*H_r[14]*dv1)-0.2886751345948129*H_l[14]*dv1+0.5773502691896258*H_c[14]*dv1+0.25*H_r[5]*dv1-0.25*H_l[5]*dv1; 
  drag_coeff_x[6] = (-0.2886751345948129*H_r[16]*dv1)-0.2886751345948129*H_l[16]*dv1+0.5773502691896258*H_c[16]*dv1+0.25*H_r[6]*dv1-0.25*H_l[6]*dv1; 
  drag_coeff_x[7] = (-0.5*H_r[7]*dv1)+0.5*H_l[7]*dv1+0.4330127018922193*H_r[1]*dv1+0.4330127018922193*H_l[1]*dv1-0.8660254037844386*H_c[1]*dv1; 
  drag_coeff_x[8] = (-0.5*H_r[8]*dv1)+0.5*H_l[8]*dv1+0.4330127018922193*H_r[2]*dv1+0.4330127018922193*H_l[2]*dv1-0.8660254037844386*H_c[2]*dv1; 
  drag_coeff_x[9] = (-0.2886751345948129*H_r[18]*dv1)-0.2886751345948129*H_l[18]*dv1+0.5773502691896258*H_c[18]*dv1+0.25*H_r[9]*dv1-0.25*H_l[9]*dv1; 
  drag_coeff_x[10] = (-0.2886751345948129*H_r[19]*dv1)-0.2886751345948129*H_l[19]*dv1+0.5773502691896258*H_c[19]*dv1+0.25*H_r[10]*dv1-0.25*H_l[10]*dv1; 
  drag_coeff_x[11] = (-0.5*H_r[11]*dv1)+0.5*H_l[11]*dv1+0.4330127018922193*H_r[4]*dv1+0.4330127018922193*H_l[4]*dv1-0.8660254037844386*H_c[4]*dv1; 
  drag_coeff_x[12] = (-0.2886751345948129*H_r[21]*dv1)-0.2886751345948129*H_l[21]*dv1+0.5773502691896258*H_c[21]*dv1+0.25*H_r[12]*dv1-0.25*H_l[12]*dv1; 
  drag_coeff_x[13] = (-0.2886751345948129*H_r[22]*dv1)-0.2886751345948129*H_l[22]*dv1+0.5773502691896258*H_c[22]*dv1+0.25*H_r[13]*dv1-0.25*H_l[13]*dv1; 
  drag_coeff_x[14] = (-0.5*H_r[14]*dv1)+0.5*H_l[14]*dv1+0.4330127018922193*H_r[5]*dv1+0.4330127018922193*H_l[5]*dv1-0.8660254037844386*H_c[5]*dv1; 
  drag_coeff_x[15] = (-0.2886751345948129*H_r[25]*dv1)-0.2886751345948129*H_l[25]*dv1+0.5773502691896258*H_c[25]*dv1+0.25*H_r[15]*dv1-0.25*H_l[15]*dv1; 
  drag_coeff_x[16] = (-0.5*H_r[16]*dv1)+0.5*H_l[16]*dv1+0.4330127018922193*H_r[6]*dv1+0.4330127018922193*H_l[6]*dv1-0.8660254037844386*H_c[6]*dv1; 
  drag_coeff_x[17] = (-0.2886751345948129*H_r[26]*dv1)-0.2886751345948129*H_l[26]*dv1+0.5773502691896258*H_c[26]*dv1+0.25*H_r[17]*dv1-0.25*H_l[17]*dv1; 
  drag_coeff_x[18] = (-0.5*H_r[18]*dv1)+0.5*H_l[18]*dv1+0.4330127018922193*H_r[9]*dv1+0.4330127018922193*H_l[9]*dv1-0.8660254037844386*H_c[9]*dv1; 
  drag_coeff_x[19] = (-0.5*H_r[19]*dv1)+0.5*H_l[19]*dv1+0.4330127018922193*H_r[10]*dv1+0.4330127018922193*H_l[10]*dv1-0.8660254037844386*H_c[10]*dv1; 
  drag_coeff_x[20] = (-0.2886751345948129*H_r[27]*dv1)-0.2886751345948129*H_l[27]*dv1+0.5773502691896258*H_c[27]*dv1+0.25*H_r[20]*dv1-0.25*H_l[20]*dv1; 
  drag_coeff_x[21] = (-0.5*H_r[21]*dv1)+0.5*H_l[21]*dv1+0.4330127018922193*H_r[12]*dv1+0.4330127018922193*H_l[12]*dv1-0.8660254037844386*H_c[12]*dv1; 
  drag_coeff_x[22] = (-0.5*H_r[22]*dv1)+0.5*H_l[22]*dv1+0.4330127018922193*H_r[13]*dv1+0.4330127018922193*H_l[13]*dv1-0.8660254037844386*H_c[13]*dv1; 
  drag_coeff_x[23] = (-0.2886751345948129*H_r[29]*dv1)-0.2886751345948129*H_l[29]*dv1+0.5773502691896258*H_c[29]*dv1+0.25*H_r[23]*dv1-0.25*H_l[23]*dv1; 
  drag_coeff_x[24] = (-0.2886751345948129*H_r[30]*dv1)-0.2886751345948129*H_l[30]*dv1+0.5773502691896258*H_c[30]*dv1+0.25*H_r[24]*dv1-0.25*H_l[24]*dv1; 
  drag_coeff_x[25] = (-0.5*H_r[25]*dv1)+0.5*H_l[25]*dv1+0.4330127018922193*H_r[15]*dv1+0.4330127018922193*H_l[15]*dv1-0.8660254037844386*H_c[15]*dv1; 
  drag_coeff_x[26] = (-0.5*H_r[26]*dv1)+0.5*H_l[26]*dv1+0.4330127018922193*H_r[17]*dv1+0.4330127018922193*H_l[17]*dv1-0.8660254037844386*H_c[17]*dv1; 
  drag_coeff_x[27] = (-0.5*H_r[27]*dv1)+0.5*H_l[27]*dv1+0.4330127018922193*H_r[20]*dv1+0.4330127018922193*H_l[20]*dv1-0.8660254037844386*H_c[20]*dv1; 
  drag_coeff_x[28] = (-0.2886751345948129*H_r[31]*dv1)-0.2886751345948129*H_l[31]*dv1+0.5773502691896258*H_c[31]*dv1+0.25*H_r[28]*dv1-0.25*H_l[28]*dv1; 
  drag_coeff_x[29] = (-0.5*H_r[29]*dv1)+0.5*H_l[29]*dv1+0.4330127018922193*H_r[23]*dv1+0.4330127018922193*H_l[23]*dv1-0.8660254037844386*H_c[23]*dv1; 
  drag_coeff_x[30] = (-0.5*H_r[30]*dv1)+0.5*H_l[30]*dv1+0.4330127018922193*H_r[24]*dv1+0.4330127018922193*H_l[24]*dv1-0.8660254037844386*H_c[24]*dv1; 
  drag_coeff_x[31] = (-0.5*H_r[31]*dv1)+0.5*H_l[31]*dv1+0.4330127018922193*H_r[28]*dv1+0.4330127018922193*H_l[28]*dv1-0.8660254037844386*H_c[28]*dv1; 
} 
