#include <gkyl_dg_diffusion_gyrokinetic_kernels.h>

GKYL_CU_DH double dg_diffusion_gyrokinetic_order6_surfy_2x2v_ser_p2_constcoeff(const double *w, const double *dx, const double *coeff, const double *jacobgeo_inv, const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{
  // w[NDIM]: Cell-center coordinate.
  // dxv[NDIM]: Cell length.
  // coeff: Diffusion coefficient.
  // jacobgeo_inv: one divided by the configuration space Jacobian.
  // ql: Input field in the left cell.
  // qc: Input field in the center cell.
  // qr: Input field in the right cell.
  // out: Incremented output.

  const double rdx2Sq = pow(2./dx[1],6.);

  out[0] += 0.0625*(563.489130329947*coeff[1]*qr[12]+563.489130329947*coeff[1]*ql[12]-1126.978260659894*coeff[1]*qc[12]-545.5960043841961*coeff[1]*qr[2]+545.5960043841961*coeff[1]*ql[2]+(315.0*qr[0]+315.0*ql[0]-630.0*qc[0])*coeff[1])*rdx2Sq; 
  out[1] += 0.0625*(563.4891303299469*coeff[1]*qr[20]+563.4891303299469*coeff[1]*ql[20]-1126.978260659894*coeff[1]*qc[20]-545.5960043841961*coeff[1]*qr[5]+545.5960043841961*coeff[1]*ql[5]+315.0*coeff[1]*qr[1]+315.0*coeff[1]*ql[1]-630.0*coeff[1]*qc[1])*rdx2Sq; 
  out[2] += 0.0078125*(6587.944671898812*coeff[1]*qr[12]-6587.944671898812*coeff[1]*ql[12]-7245.0*coeff[1]*qr[2]-7245.0*coeff[1]*ql[2]-15750.0*coeff[1]*qc[2]+(4364.768035073569*qr[0]-4364.768035073569*ql[0])*coeff[1])*rdx2Sq; 
  out[3] += 0.0625*(563.4891303299469*coeff[1]*qr[22]+563.4891303299469*coeff[1]*ql[22]-1126.978260659894*coeff[1]*qc[22]-545.5960043841961*coeff[1]*qr[7]+545.5960043841961*coeff[1]*ql[7]+315.0*coeff[1]*qr[3]+315.0*coeff[1]*ql[3]-630.0*coeff[1]*qc[3])*rdx2Sq; 
  out[4] += 0.0625*(563.4891303299469*coeff[1]*qr[26]+563.4891303299469*coeff[1]*ql[26]-1126.978260659894*coeff[1]*qc[26]-545.5960043841961*coeff[1]*qr[9]+545.5960043841961*coeff[1]*ql[9]+315.0*coeff[1]*qr[4]+315.0*coeff[1]*ql[4]-630.0*coeff[1]*qc[4])*rdx2Sq; 
  out[5] += 0.0078125*(6587.944671898817*coeff[1]*qr[20]-6587.944671898817*coeff[1]*ql[20]-7245.0*coeff[1]*qr[5]-7245.0*coeff[1]*ql[5]-15750.0*coeff[1]*qc[5]+4364.768035073569*coeff[1]*qr[1]-4364.768035073569*coeff[1]*ql[1])*rdx2Sq; 
  out[6] += 0.0625*(563.489130329947*coeff[1]*qr[33]+563.489130329947*coeff[1]*ql[33]-1126.978260659894*coeff[1]*qc[33]-545.5960043841961*coeff[1]*qr[15]+545.5960043841961*coeff[1]*ql[15]+315.0*coeff[1]*qr[6]+315.0*coeff[1]*ql[6]-630.0*coeff[1]*qc[6])*rdx2Sq; 
  out[7] += 0.0078125*(6587.944671898817*coeff[1]*qr[22]-6587.944671898817*coeff[1]*ql[22]-7245.0*coeff[1]*qr[7]-7245.0*coeff[1]*ql[7]-15750.0*coeff[1]*qc[7]+4364.768035073569*coeff[1]*qr[3]-4364.768035073569*coeff[1]*ql[3])*rdx2Sq; 
  out[8] += 0.0625*(563.489130329947*coeff[1]*qr[36]+563.489130329947*coeff[1]*ql[36]-1126.978260659894*coeff[1]*qc[36]-545.5960043841961*coeff[1]*qr[16]+545.5960043841961*coeff[1]*ql[16]+315.0*coeff[1]*qr[8]+315.0*coeff[1]*ql[8]-630.0*coeff[1]*qc[8])*rdx2Sq; 
  out[9] += 0.0078125*(6587.944671898817*coeff[1]*qr[26]-6587.944671898817*coeff[1]*ql[26]-7245.0*coeff[1]*qr[9]-7245.0*coeff[1]*ql[9]-15750.0*coeff[1]*qc[9]+4364.768035073569*coeff[1]*qr[4]-4364.768035073569*coeff[1]*ql[4])*rdx2Sq; 
  out[10] += 0.0625*(563.489130329947*coeff[1]*qr[38]+563.489130329947*coeff[1]*ql[38]-1126.978260659894*coeff[1]*qc[38]-545.5960043841961*coeff[1]*qr[18]+545.5960043841961*coeff[1]*ql[18]+315.0*coeff[1]*qr[10]+315.0*coeff[1]*ql[10]-630.0*coeff[1]*qc[10])*rdx2Sq; 
  out[11] += -0.0625*(545.5960043841964*coeff[1]*qr[19]-545.5960043841964*coeff[1]*ql[19]-315.0*coeff[1]*qr[11]-315.0*coeff[1]*ql[11]+630.0*coeff[1]*qc[11])*rdx2Sq; 
  out[12] += -0.0078125*(405.0*coeff[1]*qr[12]+405.0*coeff[1]*ql[12]+18090.0*coeff[1]*qc[12]+1568.558255214003*coeff[1]*qr[2]-1568.558255214003*coeff[1]*ql[2]+((-1609.968943799849*qr[0])-1609.968943799849*ql[0]+3219.937887599698*qc[0])*coeff[1])*rdx2Sq; 
  out[13] += -0.0625*(545.5960043841964*coeff[1]*qr[24]-545.5960043841964*coeff[1]*ql[24]-315.0*coeff[1]*qr[13]-315.0*coeff[1]*ql[13]+630.0*coeff[1]*qc[13])*rdx2Sq; 
  out[14] += -0.0625*(545.5960043841964*coeff[1]*qr[29]-545.5960043841964*coeff[1]*ql[29]-315.0*coeff[1]*qr[14]-315.0*coeff[1]*ql[14]+630.0*coeff[1]*qc[14])*rdx2Sq; 
  out[15] += 0.0078125*(6587.944671898812*coeff[1]*qr[33]-6587.944671898812*coeff[1]*ql[33]-7245.0*coeff[1]*qr[15]-7245.0*coeff[1]*ql[15]-15750.0*coeff[1]*qc[15]+4364.768035073569*coeff[1]*qr[6]-4364.768035073569*coeff[1]*ql[6])*rdx2Sq; 
  out[16] += 0.0078125*(6587.944671898812*coeff[1]*qr[36]-6587.944671898812*coeff[1]*ql[36]-7245.0*coeff[1]*qr[16]-7245.0*coeff[1]*ql[16]-15750.0*coeff[1]*qc[16]+4364.768035073569*coeff[1]*qr[8]-4364.768035073569*coeff[1]*ql[8])*rdx2Sq; 
  out[17] += 0.0625*(563.4891303299469*coeff[1]*qr[45]+563.4891303299469*coeff[1]*ql[45]-1126.978260659894*coeff[1]*qc[45]-545.5960043841961*coeff[1]*qr[31]+545.5960043841961*coeff[1]*ql[31]+315.0*coeff[1]*qr[17]+315.0*coeff[1]*ql[17]-630.0*coeff[1]*qc[17])*rdx2Sq; 
  out[18] += 0.0078125*(6587.944671898812*coeff[1]*qr[38]-6587.944671898812*coeff[1]*ql[38]-7245.0*coeff[1]*qr[18]-7245.0*coeff[1]*ql[18]-15750.0*coeff[1]*qc[18]+4364.768035073569*coeff[1]*qr[10]-4364.768035073569*coeff[1]*ql[10])*rdx2Sq; 
  out[19] += -0.0078125*(7245.0*coeff[1]*qr[19]+7245.0*coeff[1]*ql[19]+15750.0*coeff[1]*qc[19]-4364.768035073571*coeff[1]*qr[11]+4364.768035073571*coeff[1]*ql[11])*rdx2Sq; 
  out[20] += -0.0078125*(405.0*coeff[1]*qr[20]+405.0*coeff[1]*ql[20]+18090.0*coeff[1]*qc[20]+1568.558255214004*coeff[1]*qr[5]-1568.558255214004*coeff[1]*ql[5]-1609.968943799848*coeff[1]*qr[1]-1609.968943799848*coeff[1]*ql[1]+3219.937887599697*coeff[1]*qc[1])*rdx2Sq; 
  out[21] += -0.0625*(545.5960043841964*coeff[1]*qr[32]-545.5960043841964*coeff[1]*ql[32]-315.0*coeff[1]*qr[21]-315.0*coeff[1]*ql[21]+630.0*coeff[1]*qc[21])*rdx2Sq; 
  out[22] += -0.0078125*(405.0*coeff[1]*qr[22]+405.0*coeff[1]*ql[22]+18090.0*coeff[1]*qc[22]+1568.558255214004*coeff[1]*qr[7]-1568.558255214004*coeff[1]*ql[7]-1609.968943799848*coeff[1]*qr[3]-1609.968943799848*coeff[1]*ql[3]+3219.937887599697*coeff[1]*qc[3])*rdx2Sq; 
  out[23] += -0.0625*(545.5960043841964*coeff[1]*qr[34]-545.5960043841964*coeff[1]*ql[34]-315.0*coeff[1]*qr[23]-315.0*coeff[1]*ql[23]+630.0*coeff[1]*qc[23])*rdx2Sq; 
  out[24] += -0.0078125*(7245.0*coeff[1]*qr[24]+7245.0*coeff[1]*ql[24]+15750.0*coeff[1]*qc[24]-4364.768035073571*coeff[1]*qr[13]+4364.768035073571*coeff[1]*ql[13])*rdx2Sq; 
  out[25] += -0.0625*(545.5960043841964*coeff[1]*qr[35]-545.5960043841964*coeff[1]*ql[35]-315.0*coeff[1]*qr[25]-315.0*coeff[1]*ql[25]+630.0*coeff[1]*qc[25])*rdx2Sq; 
  out[26] += -0.0078125*(405.0*coeff[1]*qr[26]+405.0*coeff[1]*ql[26]+18090.0*coeff[1]*qc[26]+1568.558255214004*coeff[1]*qr[9]-1568.558255214004*coeff[1]*ql[9]-1609.968943799848*coeff[1]*qr[4]-1609.968943799848*coeff[1]*ql[4]+3219.937887599697*coeff[1]*qc[4])*rdx2Sq; 
  out[27] += -0.0625*(545.5960043841964*coeff[1]*qr[40]-545.5960043841964*coeff[1]*ql[40]-315.0*coeff[1]*qr[27]-315.0*coeff[1]*ql[27]+630.0*coeff[1]*qc[27])*rdx2Sq; 
  out[28] += -0.0625*(545.5960043841964*coeff[1]*qr[41]-545.5960043841964*coeff[1]*ql[41]-315.0*coeff[1]*qr[28]-315.0*coeff[1]*ql[28]+630.0*coeff[1]*qc[28])*rdx2Sq; 
  out[29] += -0.0078125*(7245.0*coeff[1]*qr[29]+7245.0*coeff[1]*ql[29]+15750.0*coeff[1]*qc[29]-4364.768035073571*coeff[1]*qr[14]+4364.768035073571*coeff[1]*ql[14])*rdx2Sq; 
  out[30] += -0.0625*(545.5960043841964*coeff[1]*qr[43]-545.5960043841964*coeff[1]*ql[43]-315.0*coeff[1]*qr[30]-315.0*coeff[1]*ql[30]+630.0*coeff[1]*qc[30])*rdx2Sq; 
  out[31] += 0.0078125*(6587.944671898817*coeff[1]*qr[45]-6587.944671898817*coeff[1]*ql[45]-7245.0*coeff[1]*qr[31]-7245.0*coeff[1]*ql[31]-15750.0*coeff[1]*qc[31]+4364.768035073569*coeff[1]*qr[17]-4364.768035073569*coeff[1]*ql[17])*rdx2Sq; 
  out[32] += -0.0078125*(7245.0*coeff[1]*qr[32]+7245.0*coeff[1]*ql[32]+15750.0*coeff[1]*qc[32]-4364.768035073571*coeff[1]*qr[21]+4364.768035073571*coeff[1]*ql[21])*rdx2Sq; 
  out[33] += -0.0078125*(405.0*coeff[1]*qr[33]+405.0*coeff[1]*ql[33]+18090.0*coeff[1]*qc[33]+1568.558255214003*coeff[1]*qr[15]-1568.558255214003*coeff[1]*ql[15]-1609.968943799849*coeff[1]*qr[6]-1609.968943799849*coeff[1]*ql[6]+3219.937887599698*coeff[1]*qc[6])*rdx2Sq; 
  out[34] += -0.0078125*(7245.0*coeff[1]*qr[34]+7245.0*coeff[1]*ql[34]+15750.0*coeff[1]*qc[34]-4364.768035073571*coeff[1]*qr[23]+4364.768035073571*coeff[1]*ql[23])*rdx2Sq; 
  out[35] += -0.0078125*(7245.0*coeff[1]*qr[35]+7245.0*coeff[1]*ql[35]+15750.0*coeff[1]*qc[35]-4364.768035073571*coeff[1]*qr[25]+4364.768035073571*coeff[1]*ql[25])*rdx2Sq; 
  out[36] += -0.0078125*(405.0*coeff[1]*qr[36]+405.0*coeff[1]*ql[36]+18090.0*coeff[1]*qc[36]+1568.558255214003*coeff[1]*qr[16]-1568.558255214003*coeff[1]*ql[16]-1609.968943799849*coeff[1]*qr[8]-1609.968943799849*coeff[1]*ql[8]+3219.937887599698*coeff[1]*qc[8])*rdx2Sq; 
  out[37] += -0.0625*(545.5960043841964*coeff[1]*qr[44]-545.5960043841964*coeff[1]*ql[44]-315.0*coeff[1]*qr[37]-315.0*coeff[1]*ql[37]+630.0*coeff[1]*qc[37])*rdx2Sq; 
  out[38] += -0.0078125*(405.0*coeff[1]*qr[38]+405.0*coeff[1]*ql[38]+18090.0*coeff[1]*qc[38]+1568.558255214003*coeff[1]*qr[18]-1568.558255214003*coeff[1]*ql[18]-1609.968943799849*coeff[1]*qr[10]-1609.968943799849*coeff[1]*ql[10]+3219.937887599698*coeff[1]*qc[10])*rdx2Sq; 
  out[39] += -0.0625*(545.5960043841964*coeff[1]*qr[46]-545.5960043841964*coeff[1]*ql[46]-315.0*coeff[1]*qr[39]-315.0*coeff[1]*ql[39]+630.0*coeff[1]*qc[39])*rdx2Sq; 
  out[40] += -0.0078125*(7245.0*coeff[1]*qr[40]+7245.0*coeff[1]*ql[40]+15750.0*coeff[1]*qc[40]-4364.768035073571*coeff[1]*qr[27]+4364.768035073571*coeff[1]*ql[27])*rdx2Sq; 
  out[41] += -0.0078125*(7245.0*coeff[1]*qr[41]+7245.0*coeff[1]*ql[41]+15750.0*coeff[1]*qc[41]-4364.768035073571*coeff[1]*qr[28]+4364.768035073571*coeff[1]*ql[28])*rdx2Sq; 
  out[42] += -0.0625*(545.5960043841964*coeff[1]*qr[47]-545.5960043841964*coeff[1]*ql[47]-315.0*coeff[1]*qr[42]-315.0*coeff[1]*ql[42]+630.0*coeff[1]*qc[42])*rdx2Sq; 
  out[43] += -0.0078125*(7245.0*coeff[1]*qr[43]+7245.0*coeff[1]*ql[43]+15750.0*coeff[1]*qc[43]-4364.768035073571*coeff[1]*qr[30]+4364.768035073571*coeff[1]*ql[30])*rdx2Sq; 
  out[44] += -0.0078125*(7245.0*coeff[1]*qr[44]+7245.0*coeff[1]*ql[44]+15750.0*coeff[1]*qc[44]-4364.768035073571*coeff[1]*qr[37]+4364.768035073571*coeff[1]*ql[37])*rdx2Sq; 
  out[45] += -0.0078125*(405.0*coeff[1]*qr[45]+405.0*coeff[1]*ql[45]+18090.0*coeff[1]*qc[45]+1568.558255214004*coeff[1]*qr[31]-1568.558255214004*coeff[1]*ql[31]-1609.968943799848*coeff[1]*qr[17]-1609.968943799848*coeff[1]*ql[17]+3219.937887599697*coeff[1]*qc[17])*rdx2Sq; 
  out[46] += -0.0078125*(7245.0*coeff[1]*qr[46]+7245.0*coeff[1]*ql[46]+15750.0*coeff[1]*qc[46]-4364.768035073571*coeff[1]*qr[39]+4364.768035073571*coeff[1]*ql[39])*rdx2Sq; 
  out[47] += -0.0078125*(7245.0*coeff[1]*qr[47]+7245.0*coeff[1]*ql[47]+15750.0*coeff[1]*qc[47]-4364.768035073571*coeff[1]*qr[42]+4364.768035073571*coeff[1]*ql[42])*rdx2Sq; 

  return 0.;

}

