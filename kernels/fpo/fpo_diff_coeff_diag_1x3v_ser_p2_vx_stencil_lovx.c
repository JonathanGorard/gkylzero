#include <gkyl_fpo_vlasov_kernels.h> 

GKYL_CU_DH void fpo_diff_coeff_diag_1x3v_vx_ser_p2_lovx(const double *dxv, const double *gamma, const double* fpo_g_stencil[3], const double* fpo_d2gdv2_surf, double *diff_coeff) {
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[3]: 3 cell stencil of Rosenbluth potential G. 
  // fpo_d2gdv2_surf: Surface projection of d2G/dv2 in center cell. 
  // diff_coeff: Output array for diffusion tensor. 

  // Use cell-average value for gamma. 
  double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1_sq = 4.0/dxv[1]/dxv[1]; 

  const double* G_C = fpo_g_stencil[0]; 
  const double* G_R = fpo_g_stencil[1]; 
  
  const double* d2G_surf_C = &fpo_d2gdv2_surf[0]; 
  
  double *out = &diff_coeff[0]; 
  
  out[0] = -(0.8594886288514817*G_R[12]*dv1_sq*gamma_avg)+5.848715303647888*G_C[12]*dv1_sq*gamma_avg+0.7848355221796475*G_R[2]*dv1_sq*gamma_avg+0.5142025834970104*G_C[2]*dv1_sq*gamma_avg-0.375*G_R[0]*dv1_sq*gamma_avg+0.375*G_C[0]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[0]*gamma_avg; 
  out[1] = -(0.8594886288514816*G_R[20]*dv1_sq*gamma_avg)+5.848715303647886*G_C[20]*dv1_sq*gamma_avg+0.7848355221796475*G_R[5]*dv1_sq*gamma_avg+0.5142025834970104*G_C[5]*dv1_sq*gamma_avg-0.375*G_R[1]*dv1_sq*gamma_avg+0.375*G_C[1]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[1]*gamma_avg; 
  out[2] = 2.8163850770702057*G_R[12]*dv1_sq*gamma_avg-9.577161630793526*G_C[12]*dv1_sq*gamma_avg-4.1296875*G_R[2]*dv1_sq*gamma_avg-6.1453125*G_C[2]*dv1_sq*gamma_avg+2.966137007961702*G_R[0]*dv1_sq*gamma_avg-2.966137007961702*G_C[0]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[0]*gamma_avg; 
  out[3] = -(0.8594886288514816*G_R[22]*dv1_sq*gamma_avg)+5.848715303647886*G_C[22]*dv1_sq*gamma_avg+0.7848355221796475*G_R[7]*dv1_sq*gamma_avg+0.5142025834970104*G_C[7]*dv1_sq*gamma_avg-0.375*G_R[3]*dv1_sq*gamma_avg+0.375*G_C[3]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[2]*gamma_avg; 
  out[4] = -(0.8594886288514816*G_R[26]*dv1_sq*gamma_avg)+5.848715303647886*G_C[26]*dv1_sq*gamma_avg+0.7848355221796475*G_R[9]*dv1_sq*gamma_avg+0.5142025834970104*G_C[9]*dv1_sq*gamma_avg-0.375*G_R[4]*dv1_sq*gamma_avg+0.375*G_C[4]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[3]*gamma_avg; 
  out[5] = 2.816385077070206*G_R[20]*dv1_sq*gamma_avg-9.577161630793528*G_C[20]*dv1_sq*gamma_avg-4.1296875*G_R[5]*dv1_sq*gamma_avg-6.1453125*G_C[5]*dv1_sq*gamma_avg+2.966137007961702*G_R[1]*dv1_sq*gamma_avg-2.966137007961702*G_C[1]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[1]*gamma_avg; 
  out[6] = -(0.8594886288514817*G_R[33]*dv1_sq*gamma_avg)+5.848715303647888*G_C[33]*dv1_sq*gamma_avg+0.7848355221796475*G_R[15]*dv1_sq*gamma_avg+0.5142025834970104*G_C[15]*dv1_sq*gamma_avg-0.375*G_R[6]*dv1_sq*gamma_avg+0.375*G_C[6]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[4]*gamma_avg; 
  out[7] = 2.816385077070206*G_R[22]*dv1_sq*gamma_avg-9.577161630793528*G_C[22]*dv1_sq*gamma_avg-4.1296875*G_R[7]*dv1_sq*gamma_avg-6.1453125*G_C[7]*dv1_sq*gamma_avg+2.966137007961702*G_R[3]*dv1_sq*gamma_avg-2.966137007961702*G_C[3]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[2]*gamma_avg; 
  out[8] = -(0.8594886288514817*G_R[36]*dv1_sq*gamma_avg)+5.848715303647888*G_C[36]*dv1_sq*gamma_avg+0.7848355221796475*G_R[16]*dv1_sq*gamma_avg+0.5142025834970104*G_C[16]*dv1_sq*gamma_avg-0.375*G_R[8]*dv1_sq*gamma_avg+0.375*G_C[8]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[5]*gamma_avg; 
  out[9] = 2.816385077070206*G_R[26]*dv1_sq*gamma_avg-9.577161630793528*G_C[26]*dv1_sq*gamma_avg-4.1296875*G_R[9]*dv1_sq*gamma_avg-6.1453125*G_C[9]*dv1_sq*gamma_avg+2.966137007961702*G_R[4]*dv1_sq*gamma_avg-2.966137007961702*G_C[4]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[3]*gamma_avg; 
  out[10] = -(0.8594886288514817*G_R[38]*dv1_sq*gamma_avg)+5.848715303647888*G_C[38]*dv1_sq*gamma_avg+0.7848355221796475*G_R[18]*dv1_sq*gamma_avg+0.5142025834970104*G_C[18]*dv1_sq*gamma_avg-0.375*G_R[10]*dv1_sq*gamma_avg+0.375*G_C[10]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[6]*gamma_avg; 
  out[11] = 0.7848355221796476*G_R[19]*dv1_sq*gamma_avg+0.5142025834970104*G_C[19]*dv1_sq*gamma_avg-0.375*G_R[11]*dv1_sq*gamma_avg+0.375*G_C[11]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[7]*gamma_avg; 
  out[12] = -(1.3453125*G_R[12]*dv1_sq*gamma_avg)-1.3453125*G_C[12]*dv1_sq*gamma_avg+1.228461905125165*G_R[2]*dv1_sq*gamma_avg+0.8048543516337288*G_C[2]*dv1_sq*gamma_avg-0.5869678440936947*G_R[0]*dv1_sq*gamma_avg+0.5869678440936947*G_C[0]*dv1_sq*gamma_avg+0.22135943621178647*d2G_surf_C[0]*gamma_avg; 
  out[13] = 0.7848355221796476*G_R[24]*dv1_sq*gamma_avg+0.5142025834970104*G_C[24]*dv1_sq*gamma_avg-0.375*G_R[13]*dv1_sq*gamma_avg+0.375*G_C[13]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[8]*gamma_avg; 
  out[14] = 0.7848355221796476*G_R[29]*dv1_sq*gamma_avg+0.5142025834970104*G_C[29]*dv1_sq*gamma_avg-0.375*G_R[14]*dv1_sq*gamma_avg+0.375*G_C[14]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[9]*gamma_avg; 
  out[15] = 2.8163850770702057*G_R[33]*dv1_sq*gamma_avg-9.577161630793526*G_C[33]*dv1_sq*gamma_avg-4.1296875*G_R[15]*dv1_sq*gamma_avg-6.1453125*G_C[15]*dv1_sq*gamma_avg+2.966137007961702*G_R[6]*dv1_sq*gamma_avg-2.966137007961702*G_C[6]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[4]*gamma_avg; 
  out[16] = 2.8163850770702057*G_R[36]*dv1_sq*gamma_avg-9.577161630793526*G_C[36]*dv1_sq*gamma_avg-4.1296875*G_R[16]*dv1_sq*gamma_avg-6.1453125*G_C[16]*dv1_sq*gamma_avg+2.966137007961702*G_R[8]*dv1_sq*gamma_avg-2.966137007961702*G_C[8]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[5]*gamma_avg; 
  out[17] = -(0.8594886288514816*G_R[45]*dv1_sq*gamma_avg)+5.848715303647886*G_C[45]*dv1_sq*gamma_avg+0.7848355221796475*G_R[31]*dv1_sq*gamma_avg+0.5142025834970104*G_C[31]*dv1_sq*gamma_avg-0.375*G_R[17]*dv1_sq*gamma_avg+0.375*G_C[17]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[10]*gamma_avg; 
  out[18] = 2.8163850770702057*G_R[38]*dv1_sq*gamma_avg-9.577161630793526*G_C[38]*dv1_sq*gamma_avg-4.1296875*G_R[18]*dv1_sq*gamma_avg-6.1453125*G_C[18]*dv1_sq*gamma_avg+2.966137007961702*G_R[10]*dv1_sq*gamma_avg-2.966137007961702*G_C[10]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[6]*gamma_avg; 
  out[19] = -(4.1296875*G_R[19]*dv1_sq*gamma_avg)-6.1453125*G_C[19]*dv1_sq*gamma_avg+2.966137007961702*G_R[11]*dv1_sq*gamma_avg-2.966137007961702*G_C[11]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[7]*gamma_avg; 
  out[20] = -(1.3453125*G_R[20]*dv1_sq*gamma_avg)-1.3453125*G_C[20]*dv1_sq*gamma_avg+1.228461905125165*G_R[5]*dv1_sq*gamma_avg+0.8048543516337289*G_C[5]*dv1_sq*gamma_avg-0.5869678440936947*G_R[1]*dv1_sq*gamma_avg+0.5869678440936947*G_C[1]*dv1_sq*gamma_avg+0.22135943621178653*d2G_surf_C[1]*gamma_avg; 
  out[21] = 0.7848355221796476*G_R[32]*dv1_sq*gamma_avg+0.5142025834970104*G_C[32]*dv1_sq*gamma_avg-0.375*G_R[21]*dv1_sq*gamma_avg+0.375*G_C[21]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[11]*gamma_avg; 
  out[22] = -(1.3453125*G_R[22]*dv1_sq*gamma_avg)-1.3453125*G_C[22]*dv1_sq*gamma_avg+1.228461905125165*G_R[7]*dv1_sq*gamma_avg+0.8048543516337289*G_C[7]*dv1_sq*gamma_avg-0.5869678440936947*G_R[3]*dv1_sq*gamma_avg+0.5869678440936947*G_C[3]*dv1_sq*gamma_avg+0.22135943621178653*d2G_surf_C[2]*gamma_avg; 
  out[23] = 0.7848355221796476*G_R[34]*dv1_sq*gamma_avg+0.5142025834970104*G_C[34]*dv1_sq*gamma_avg-0.375*G_R[23]*dv1_sq*gamma_avg+0.375*G_C[23]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[12]*gamma_avg; 
  out[24] = -(4.1296875*G_R[24]*dv1_sq*gamma_avg)-6.1453125*G_C[24]*dv1_sq*gamma_avg+2.966137007961702*G_R[13]*dv1_sq*gamma_avg-2.966137007961702*G_C[13]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[8]*gamma_avg; 
  out[25] = 0.7848355221796476*G_R[35]*dv1_sq*gamma_avg+0.5142025834970104*G_C[35]*dv1_sq*gamma_avg-0.375*G_R[25]*dv1_sq*gamma_avg+0.375*G_C[25]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[13]*gamma_avg; 
  out[26] = -(1.3453125*G_R[26]*dv1_sq*gamma_avg)-1.3453125*G_C[26]*dv1_sq*gamma_avg+1.228461905125165*G_R[9]*dv1_sq*gamma_avg+0.8048543516337289*G_C[9]*dv1_sq*gamma_avg-0.5869678440936947*G_R[4]*dv1_sq*gamma_avg+0.5869678440936947*G_C[4]*dv1_sq*gamma_avg+0.22135943621178653*d2G_surf_C[3]*gamma_avg; 
  out[27] = 0.7848355221796476*G_R[40]*dv1_sq*gamma_avg+0.5142025834970104*G_C[40]*dv1_sq*gamma_avg-0.375*G_R[27]*dv1_sq*gamma_avg+0.375*G_C[27]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[14]*gamma_avg; 
  out[28] = 0.7848355221796476*G_R[41]*dv1_sq*gamma_avg+0.5142025834970104*G_C[41]*dv1_sq*gamma_avg-0.375*G_R[28]*dv1_sq*gamma_avg+0.375*G_C[28]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[15]*gamma_avg; 
  out[29] = -(4.1296875*G_R[29]*dv1_sq*gamma_avg)-6.1453125*G_C[29]*dv1_sq*gamma_avg+2.966137007961702*G_R[14]*dv1_sq*gamma_avg-2.966137007961702*G_C[14]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[9]*gamma_avg; 
  out[30] = 0.7848355221796476*G_R[43]*dv1_sq*gamma_avg+0.5142025834970104*G_C[43]*dv1_sq*gamma_avg-0.375*G_R[30]*dv1_sq*gamma_avg+0.375*G_C[30]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[16]*gamma_avg; 
  out[31] = 2.816385077070206*G_R[45]*dv1_sq*gamma_avg-9.577161630793528*G_C[45]*dv1_sq*gamma_avg-4.1296875*G_R[31]*dv1_sq*gamma_avg-6.1453125*G_C[31]*dv1_sq*gamma_avg+2.966137007961702*G_R[17]*dv1_sq*gamma_avg-2.966137007961702*G_C[17]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[10]*gamma_avg; 
  out[32] = -(4.1296875*G_R[32]*dv1_sq*gamma_avg)-6.1453125*G_C[32]*dv1_sq*gamma_avg+2.966137007961702*G_R[21]*dv1_sq*gamma_avg-2.966137007961702*G_C[21]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[11]*gamma_avg; 
  out[33] = -(1.3453125*G_R[33]*dv1_sq*gamma_avg)-1.3453125*G_C[33]*dv1_sq*gamma_avg+1.228461905125165*G_R[15]*dv1_sq*gamma_avg+0.8048543516337288*G_C[15]*dv1_sq*gamma_avg-0.5869678440936947*G_R[6]*dv1_sq*gamma_avg+0.5869678440936947*G_C[6]*dv1_sq*gamma_avg+0.22135943621178647*d2G_surf_C[4]*gamma_avg; 
  out[34] = -(4.1296875*G_R[34]*dv1_sq*gamma_avg)-6.1453125*G_C[34]*dv1_sq*gamma_avg+2.966137007961702*G_R[23]*dv1_sq*gamma_avg-2.966137007961702*G_C[23]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[12]*gamma_avg; 
  out[35] = -(4.1296875*G_R[35]*dv1_sq*gamma_avg)-6.1453125*G_C[35]*dv1_sq*gamma_avg+2.966137007961702*G_R[25]*dv1_sq*gamma_avg-2.966137007961702*G_C[25]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[13]*gamma_avg; 
  out[36] = -(1.3453125*G_R[36]*dv1_sq*gamma_avg)-1.3453125*G_C[36]*dv1_sq*gamma_avg+1.228461905125165*G_R[16]*dv1_sq*gamma_avg+0.8048543516337288*G_C[16]*dv1_sq*gamma_avg-0.5869678440936947*G_R[8]*dv1_sq*gamma_avg+0.5869678440936947*G_C[8]*dv1_sq*gamma_avg+0.22135943621178647*d2G_surf_C[5]*gamma_avg; 
  out[37] = 0.7848355221796476*G_R[44]*dv1_sq*gamma_avg+0.5142025834970104*G_C[44]*dv1_sq*gamma_avg-0.375*G_R[37]*dv1_sq*gamma_avg+0.375*G_C[37]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[17]*gamma_avg; 
  out[38] = -(1.3453125*G_R[38]*dv1_sq*gamma_avg)-1.3453125*G_C[38]*dv1_sq*gamma_avg+1.228461905125165*G_R[18]*dv1_sq*gamma_avg+0.8048543516337288*G_C[18]*dv1_sq*gamma_avg-0.5869678440936947*G_R[10]*dv1_sq*gamma_avg+0.5869678440936947*G_C[10]*dv1_sq*gamma_avg+0.22135943621178647*d2G_surf_C[6]*gamma_avg; 
  out[39] = 0.7848355221796476*G_R[46]*dv1_sq*gamma_avg+0.5142025834970104*G_C[46]*dv1_sq*gamma_avg-0.375*G_R[39]*dv1_sq*gamma_avg+0.375*G_C[39]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[18]*gamma_avg; 
  out[40] = -(4.1296875*G_R[40]*dv1_sq*gamma_avg)-6.1453125*G_C[40]*dv1_sq*gamma_avg+2.966137007961702*G_R[27]*dv1_sq*gamma_avg-2.966137007961702*G_C[27]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[14]*gamma_avg; 
  out[41] = -(4.1296875*G_R[41]*dv1_sq*gamma_avg)-6.1453125*G_C[41]*dv1_sq*gamma_avg+2.966137007961702*G_R[28]*dv1_sq*gamma_avg-2.966137007961702*G_C[28]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[15]*gamma_avg; 
  out[42] = 0.7848355221796476*G_R[47]*dv1_sq*gamma_avg+0.5142025834970104*G_C[47]*dv1_sq*gamma_avg-0.375*G_R[42]*dv1_sq*gamma_avg+0.375*G_C[42]*dv1_sq*gamma_avg+0.1414213562373095*d2G_surf_C[19]*gamma_avg; 
  out[43] = -(4.1296875*G_R[43]*dv1_sq*gamma_avg)-6.1453125*G_C[43]*dv1_sq*gamma_avg+2.966137007961702*G_R[30]*dv1_sq*gamma_avg-2.966137007961702*G_C[30]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[16]*gamma_avg; 
  out[44] = -(4.1296875*G_R[44]*dv1_sq*gamma_avg)-6.1453125*G_C[44]*dv1_sq*gamma_avg+2.966137007961702*G_R[37]*dv1_sq*gamma_avg-2.966137007961702*G_C[37]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[17]*gamma_avg; 
  out[45] = -(1.3453125*G_R[45]*dv1_sq*gamma_avg)-1.3453125*G_C[45]*dv1_sq*gamma_avg+1.228461905125165*G_R[31]*dv1_sq*gamma_avg+0.8048543516337289*G_C[31]*dv1_sq*gamma_avg-0.5869678440936947*G_R[17]*dv1_sq*gamma_avg+0.5869678440936947*G_C[17]*dv1_sq*gamma_avg+0.22135943621178653*d2G_surf_C[10]*gamma_avg; 
  out[46] = -(4.1296875*G_R[46]*dv1_sq*gamma_avg)-6.1453125*G_C[46]*dv1_sq*gamma_avg+2.966137007961702*G_R[39]*dv1_sq*gamma_avg-2.966137007961702*G_C[39]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[18]*gamma_avg; 
  out[47] = -(4.1296875*G_R[47]*dv1_sq*gamma_avg)-6.1453125*G_C[47]*dv1_sq*gamma_avg+2.966137007961702*G_R[42]*dv1_sq*gamma_avg-2.966137007961702*G_C[42]*dv1_sq*gamma_avg-0.2204540768504859*d2G_surf_C[19]*gamma_avg; 
} 
