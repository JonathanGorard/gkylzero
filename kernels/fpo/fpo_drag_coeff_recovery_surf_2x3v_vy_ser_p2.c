#include <gkyl_fpo_vlasov_kernels.h> 
GKYL_CU_DH void fpo_drag_coeff_recov_surf_vy_2x3v_ser_p2(const int edge, const double *dxv, const double *H_skin, const double *H_edge, double *drag_coeff) {
  // dxv[NDIM]: Cell spacing. 
  // H_skin/edge:   Input potential in skin/edge cells in recovery direction.
  
  const double dv1 = 2.0/dxv[3]; 
  double *drag_coeff_x = &drag_coeff[0]; 
  double *drag_coeff_y = &drag_coeff[112]; 
  double *drag_coeff_z = &drag_coeff[224]; 
  if (edge == 1) {
  drag_coeff_y[0] = (-0.2445699350390395*H_skin[19]*dv1)+0.8734640537108556*H_edge[19]*dv1-0.3518228202874282*H_skin[4]*dv1+1.217848224071866*H_edge[4]*dv1-0.25*H_skin[0]*dv1+0.25*H_edge[0]*dv1; 
  drag_coeff_y[1] = (-0.2445699350390395*H_skin[40]*dv1)+0.8734640537108554*H_edge[40]*dv1-0.3518228202874282*H_skin[9]*dv1+1.217848224071866*H_edge[9]*dv1-0.25*H_skin[1]*dv1+0.25*H_edge[1]*dv1; 
  drag_coeff_y[2] = (-0.2445699350390395*H_skin[41]*dv1)+0.8734640537108554*H_edge[41]*dv1-0.3518228202874282*H_skin[10]*dv1+1.217848224071866*H_edge[10]*dv1-0.25*H_skin[2]*dv1+0.25*H_edge[2]*dv1; 
  drag_coeff_y[3] = (-0.2445699350390395*H_skin[42]*dv1)+0.8734640537108554*H_edge[42]*dv1-0.3518228202874282*H_skin[11]*dv1+1.217848224071866*H_edge[11]*dv1-0.25*H_skin[3]*dv1+0.25*H_edge[3]*dv1; 
  drag_coeff_y[4] = 0.4236075534914363*H_skin[19]*dv1+2.360099226595144*H_edge[19]*dv1+0.609375*H_skin[4]*dv1+0.890625*H_edge[4]*dv1+0.4330127018922193*H_skin[0]*dv1-0.4330127018922193*H_edge[0]*dv1; 
  drag_coeff_y[5] = (-0.2445699350390395*H_skin[46]*dv1)+0.8734640537108554*H_edge[46]*dv1-0.3518228202874282*H_skin[15]*dv1+1.217848224071866*H_edge[15]*dv1-0.25*H_skin[5]*dv1+0.25*H_edge[5]*dv1; 
  drag_coeff_y[6] = (-0.2445699350390395*H_skin[65]*dv1)+0.8734640537108556*H_edge[65]*dv1-0.3518228202874282*H_skin[22]*dv1+1.217848224071866*H_edge[22]*dv1-0.25*H_skin[6]*dv1+0.25*H_edge[6]*dv1; 
  drag_coeff_y[7] = (-0.2445699350390395*H_skin[66]*dv1)+0.8734640537108556*H_edge[66]*dv1-0.3518228202874282*H_skin[23]*dv1+1.217848224071866*H_edge[23]*dv1-0.25*H_skin[7]*dv1+0.25*H_edge[7]*dv1; 
  drag_coeff_y[8] = (-0.2445699350390395*H_skin[67]*dv1)+0.8734640537108556*H_edge[67]*dv1-0.3518228202874282*H_skin[24]*dv1+1.217848224071866*H_edge[24]*dv1-0.25*H_skin[8]*dv1+0.25*H_edge[8]*dv1; 
  drag_coeff_y[9] = 0.4236075534914363*H_skin[40]*dv1+2.360099226595145*H_edge[40]*dv1+0.609375*H_skin[9]*dv1+0.890625*H_edge[9]*dv1+0.4330127018922193*H_skin[1]*dv1-0.4330127018922193*H_edge[1]*dv1; 
  drag_coeff_y[10] = 0.4236075534914363*H_skin[41]*dv1+2.360099226595145*H_edge[41]*dv1+0.609375*H_skin[10]*dv1+0.890625*H_edge[10]*dv1+0.4330127018922193*H_skin[2]*dv1-0.4330127018922193*H_edge[2]*dv1; 
  drag_coeff_y[11] = 0.4236075534914363*H_skin[42]*dv1+2.360099226595145*H_edge[42]*dv1+0.609375*H_skin[11]*dv1+0.890625*H_edge[11]*dv1+0.4330127018922193*H_skin[3]*dv1-0.4330127018922193*H_edge[3]*dv1; 
  drag_coeff_y[12] = (-0.2445699350390395*H_skin[77]*dv1)+0.8734640537108556*H_edge[77]*dv1-0.3518228202874282*H_skin[28]*dv1+1.217848224071866*H_edge[28]*dv1-0.25*H_skin[12]*dv1+0.25*H_edge[12]*dv1; 
  drag_coeff_y[13] = (-0.2445699350390395*H_skin[78]*dv1)+0.8734640537108556*H_edge[78]*dv1-0.3518228202874282*H_skin[29]*dv1+1.217848224071866*H_edge[29]*dv1-0.25*H_skin[13]*dv1+0.25*H_edge[13]*dv1; 
  drag_coeff_y[14] = (-0.2445699350390395*H_skin[79]*dv1)+0.8734640537108556*H_edge[79]*dv1-0.3518228202874282*H_skin[30]*dv1+1.217848224071866*H_edge[30]*dv1-0.25*H_skin[14]*dv1+0.25*H_edge[14]*dv1; 
  drag_coeff_y[15] = 0.4236075534914363*H_skin[46]*dv1+2.360099226595145*H_edge[46]*dv1+0.609375*H_skin[15]*dv1+0.890625*H_edge[15]*dv1+0.4330127018922193*H_skin[5]*dv1-0.4330127018922193*H_edge[5]*dv1; 
  drag_coeff_y[16] = (-0.3518228202874282*H_skin[37]*dv1)+1.217848224071867*H_edge[37]*dv1-0.25*H_skin[16]*dv1+0.25*H_edge[16]*dv1; 
  drag_coeff_y[17] = (-0.3518228202874282*H_skin[38]*dv1)+1.217848224071867*H_edge[38]*dv1-0.25*H_skin[17]*dv1+0.25*H_edge[17]*dv1; 
  drag_coeff_y[18] = (-0.3518228202874282*H_skin[39]*dv1)+1.217848224071867*H_edge[39]*dv1-0.25*H_skin[18]*dv1+0.25*H_edge[18]*dv1; 
  drag_coeff_y[19] = (-0.546875*H_skin[19]*dv1)+1.953125*H_edge[19]*dv1-0.7866997421983816*H_skin[4]*dv1-1.149791930905327*H_edge[4]*dv1-0.5590169943749475*H_skin[0]*dv1+0.5590169943749475*H_edge[0]*dv1; 
  drag_coeff_y[20] = (-0.3518228202874282*H_skin[50]*dv1)+1.217848224071867*H_edge[50]*dv1-0.25*H_skin[20]*dv1+0.25*H_edge[20]*dv1; 
  drag_coeff_y[21] = (-0.2445699350390395*H_skin[90]*dv1)+0.8734640537108554*H_edge[90]*dv1-0.3518228202874282*H_skin[51]*dv1+1.217848224071866*H_edge[51]*dv1-0.25*H_skin[21]*dv1+0.25*H_edge[21]*dv1; 
  drag_coeff_y[22] = 0.4236075534914363*H_skin[65]*dv1+2.360099226595144*H_edge[65]*dv1+0.609375*H_skin[22]*dv1+0.890625*H_edge[22]*dv1+0.4330127018922193*H_skin[6]*dv1-0.4330127018922193*H_edge[6]*dv1; 
  drag_coeff_y[23] = 0.4236075534914363*H_skin[66]*dv1+2.360099226595144*H_edge[66]*dv1+0.609375*H_skin[23]*dv1+0.890625*H_edge[23]*dv1+0.4330127018922193*H_skin[7]*dv1-0.4330127018922193*H_edge[7]*dv1; 
  drag_coeff_y[24] = 0.4236075534914363*H_skin[67]*dv1+2.360099226595144*H_edge[67]*dv1+0.609375*H_skin[24]*dv1+0.890625*H_edge[24]*dv1+0.4330127018922193*H_skin[8]*dv1-0.4330127018922193*H_edge[8]*dv1; 
  drag_coeff_y[25] = (-0.2445699350390395*H_skin[100]*dv1)+0.8734640537108554*H_edge[100]*dv1-0.3518228202874282*H_skin[53]*dv1+1.217848224071866*H_edge[53]*dv1-0.25*H_skin[25]*dv1+0.25*H_edge[25]*dv1; 
  drag_coeff_y[26] = (-0.2445699350390395*H_skin[101]*dv1)+0.8734640537108554*H_edge[101]*dv1-0.3518228202874282*H_skin[54]*dv1+1.217848224071866*H_edge[54]*dv1-0.25*H_skin[26]*dv1+0.25*H_edge[26]*dv1; 
  drag_coeff_y[27] = (-0.2445699350390395*H_skin[102]*dv1)+0.8734640537108554*H_edge[102]*dv1-0.3518228202874282*H_skin[55]*dv1+1.217848224071866*H_edge[55]*dv1-0.25*H_skin[27]*dv1+0.25*H_edge[27]*dv1; 
  drag_coeff_y[28] = 0.4236075534914363*H_skin[77]*dv1+2.360099226595144*H_edge[77]*dv1+0.609375*H_skin[28]*dv1+0.890625*H_edge[28]*dv1+0.4330127018922193*H_skin[12]*dv1-0.4330127018922193*H_edge[12]*dv1; 
  drag_coeff_y[29] = 0.4236075534914363*H_skin[78]*dv1+2.360099226595144*H_edge[78]*dv1+0.609375*H_skin[29]*dv1+0.890625*H_edge[29]*dv1+0.4330127018922193*H_skin[13]*dv1-0.4330127018922193*H_edge[13]*dv1; 
  drag_coeff_y[30] = 0.4236075534914363*H_skin[79]*dv1+2.360099226595144*H_edge[79]*dv1+0.609375*H_skin[30]*dv1+0.890625*H_edge[30]*dv1+0.4330127018922193*H_skin[14]*dv1-0.4330127018922193*H_edge[14]*dv1; 
  drag_coeff_y[31] = (-0.3518228202874282*H_skin[59]*dv1)+1.217848224071867*H_edge[59]*dv1-0.25*H_skin[31]*dv1+0.25*H_edge[31]*dv1; 
  drag_coeff_y[32] = (-0.3518228202874282*H_skin[60]*dv1)+1.217848224071867*H_edge[60]*dv1-0.25*H_skin[32]*dv1+0.25*H_edge[32]*dv1; 
  drag_coeff_y[33] = (-0.3518228202874282*H_skin[61]*dv1)+1.217848224071867*H_edge[61]*dv1-0.25*H_skin[33]*dv1+0.25*H_edge[33]*dv1; 
  drag_coeff_y[34] = (-0.3518228202874282*H_skin[62]*dv1)+1.217848224071867*H_edge[62]*dv1-0.25*H_skin[34]*dv1+0.25*H_edge[34]*dv1; 
  drag_coeff_y[35] = (-0.3518228202874282*H_skin[63]*dv1)+1.217848224071867*H_edge[63]*dv1-0.25*H_skin[35]*dv1+0.25*H_edge[35]*dv1; 
  drag_coeff_y[36] = (-0.3518228202874282*H_skin[64]*dv1)+1.217848224071867*H_edge[64]*dv1-0.25*H_skin[36]*dv1+0.25*H_edge[36]*dv1; 
  drag_coeff_y[37] = 0.609375*H_skin[37]*dv1+0.890625*H_edge[37]*dv1+0.4330127018922194*H_skin[16]*dv1-0.4330127018922194*H_edge[16]*dv1; 
  drag_coeff_y[38] = 0.609375*H_skin[38]*dv1+0.890625*H_edge[38]*dv1+0.4330127018922194*H_skin[17]*dv1-0.4330127018922194*H_edge[17]*dv1; 
  drag_coeff_y[39] = 0.609375*H_skin[39]*dv1+0.890625*H_edge[39]*dv1+0.4330127018922194*H_skin[18]*dv1-0.4330127018922194*H_edge[18]*dv1; 
  drag_coeff_y[40] = (-0.546875*H_skin[40]*dv1)+1.953125*H_edge[40]*dv1-0.7866997421983816*H_skin[9]*dv1-1.149791930905327*H_edge[9]*dv1-0.5590169943749476*H_skin[1]*dv1+0.5590169943749476*H_edge[1]*dv1; 
  drag_coeff_y[41] = (-0.546875*H_skin[41]*dv1)+1.953125*H_edge[41]*dv1-0.7866997421983816*H_skin[10]*dv1-1.149791930905327*H_edge[10]*dv1-0.5590169943749476*H_skin[2]*dv1+0.5590169943749476*H_edge[2]*dv1; 
  drag_coeff_y[42] = (-0.546875*H_skin[42]*dv1)+1.953125*H_edge[42]*dv1-0.7866997421983816*H_skin[11]*dv1-1.149791930905327*H_edge[11]*dv1-0.5590169943749476*H_skin[3]*dv1+0.5590169943749476*H_edge[3]*dv1; 
  drag_coeff_y[43] = (-0.3518228202874282*H_skin[74]*dv1)+1.217848224071867*H_edge[74]*dv1-0.25*H_skin[43]*dv1+0.25*H_edge[43]*dv1; 
  drag_coeff_y[44] = (-0.3518228202874282*H_skin[75]*dv1)+1.217848224071867*H_edge[75]*dv1-0.25*H_skin[44]*dv1+0.25*H_edge[44]*dv1; 
  drag_coeff_y[45] = (-0.3518228202874282*H_skin[76]*dv1)+1.217848224071867*H_edge[76]*dv1-0.25*H_skin[45]*dv1+0.25*H_edge[45]*dv1; 
  drag_coeff_y[46] = (-0.546875*H_skin[46]*dv1)+1.953125*H_edge[46]*dv1-0.7866997421983816*H_skin[15]*dv1-1.149791930905327*H_edge[15]*dv1-0.5590169943749476*H_skin[5]*dv1+0.5590169943749476*H_edge[5]*dv1; 
  drag_coeff_y[47] = (-0.3518228202874282*H_skin[83]*dv1)+1.217848224071867*H_edge[83]*dv1-0.25*H_skin[47]*dv1+0.25*H_edge[47]*dv1; 
  drag_coeff_y[48] = (-0.3518228202874282*H_skin[84]*dv1)+1.217848224071867*H_edge[84]*dv1-0.25*H_skin[48]*dv1+0.25*H_edge[48]*dv1; 
  drag_coeff_y[49] = (-0.3518228202874282*H_skin[85]*dv1)+1.217848224071867*H_edge[85]*dv1-0.25*H_skin[49]*dv1+0.25*H_edge[49]*dv1; 
  drag_coeff_y[50] = 0.609375*H_skin[50]*dv1+0.890625*H_edge[50]*dv1+0.4330127018922194*H_skin[20]*dv1-0.4330127018922194*H_edge[20]*dv1; 
  drag_coeff_y[51] = 0.4236075534914363*H_skin[90]*dv1+2.360099226595145*H_edge[90]*dv1+0.609375*H_skin[51]*dv1+0.890625*H_edge[51]*dv1+0.4330127018922193*H_skin[21]*dv1-0.4330127018922193*H_edge[21]*dv1; 
  drag_coeff_y[52] = (-0.2445699350390395*H_skin[110]*dv1)+0.8734640537108556*H_edge[110]*dv1-0.3518228202874282*H_skin[86]*dv1+1.217848224071866*H_edge[86]*dv1-0.25*H_skin[52]*dv1+0.25*H_edge[52]*dv1; 
  drag_coeff_y[53] = 0.4236075534914363*H_skin[100]*dv1+2.360099226595145*H_edge[100]*dv1+0.609375*H_skin[53]*dv1+0.890625*H_edge[53]*dv1+0.4330127018922193*H_skin[25]*dv1-0.4330127018922193*H_edge[25]*dv1; 
  drag_coeff_y[54] = 0.4236075534914363*H_skin[101]*dv1+2.360099226595145*H_edge[101]*dv1+0.609375*H_skin[54]*dv1+0.890625*H_edge[54]*dv1+0.4330127018922193*H_skin[26]*dv1-0.4330127018922193*H_edge[26]*dv1; 
  drag_coeff_y[55] = 0.4236075534914363*H_skin[102]*dv1+2.360099226595145*H_edge[102]*dv1+0.609375*H_skin[55]*dv1+0.890625*H_edge[55]*dv1+0.4330127018922193*H_skin[27]*dv1-0.4330127018922193*H_edge[27]*dv1; 
  drag_coeff_y[56] = (-0.3518228202874282*H_skin[87]*dv1)+1.217848224071867*H_edge[87]*dv1-0.25*H_skin[56]*dv1+0.25*H_edge[56]*dv1; 
  drag_coeff_y[57] = (-0.3518228202874282*H_skin[88]*dv1)+1.217848224071867*H_edge[88]*dv1-0.25*H_skin[57]*dv1+0.25*H_edge[57]*dv1; 
  drag_coeff_y[58] = (-0.3518228202874282*H_skin[89]*dv1)+1.217848224071867*H_edge[89]*dv1-0.25*H_skin[58]*dv1+0.25*H_edge[58]*dv1; 
  drag_coeff_y[59] = 0.609375*H_skin[59]*dv1+0.890625*H_edge[59]*dv1+0.4330127018922194*H_skin[31]*dv1-0.4330127018922194*H_edge[31]*dv1; 
  drag_coeff_y[60] = 0.609375*H_skin[60]*dv1+0.890625*H_edge[60]*dv1+0.4330127018922194*H_skin[32]*dv1-0.4330127018922194*H_edge[32]*dv1; 
  drag_coeff_y[61] = 0.609375*H_skin[61]*dv1+0.890625*H_edge[61]*dv1+0.4330127018922194*H_skin[33]*dv1-0.4330127018922194*H_edge[33]*dv1; 
  drag_coeff_y[62] = 0.609375*H_skin[62]*dv1+0.890625*H_edge[62]*dv1+0.4330127018922194*H_skin[34]*dv1-0.4330127018922194*H_edge[34]*dv1; 
  drag_coeff_y[63] = 0.609375*H_skin[63]*dv1+0.890625*H_edge[63]*dv1+0.4330127018922194*H_skin[35]*dv1-0.4330127018922194*H_edge[35]*dv1; 
  drag_coeff_y[64] = 0.609375*H_skin[64]*dv1+0.890625*H_edge[64]*dv1+0.4330127018922194*H_skin[36]*dv1-0.4330127018922194*H_edge[36]*dv1; 
  drag_coeff_y[65] = (-0.546875*H_skin[65]*dv1)+1.953125*H_edge[65]*dv1-0.7866997421983816*H_skin[22]*dv1-1.149791930905327*H_edge[22]*dv1-0.5590169943749475*H_skin[6]*dv1+0.5590169943749475*H_edge[6]*dv1; 
  drag_coeff_y[66] = (-0.546875*H_skin[66]*dv1)+1.953125*H_edge[66]*dv1-0.7866997421983816*H_skin[23]*dv1-1.149791930905327*H_edge[23]*dv1-0.5590169943749475*H_skin[7]*dv1+0.5590169943749475*H_edge[7]*dv1; 
  drag_coeff_y[67] = (-0.546875*H_skin[67]*dv1)+1.953125*H_edge[67]*dv1-0.7866997421983816*H_skin[24]*dv1-1.149791930905327*H_edge[24]*dv1-0.5590169943749475*H_skin[8]*dv1+0.5590169943749475*H_edge[8]*dv1; 
  drag_coeff_y[68] = (-0.3518228202874282*H_skin[94]*dv1)+1.217848224071867*H_edge[94]*dv1-0.25*H_skin[68]*dv1+0.25*H_edge[68]*dv1; 
  drag_coeff_y[69] = (-0.3518228202874282*H_skin[95]*dv1)+1.217848224071867*H_edge[95]*dv1-0.25*H_skin[69]*dv1+0.25*H_edge[69]*dv1; 
  drag_coeff_y[70] = (-0.3518228202874282*H_skin[96]*dv1)+1.217848224071867*H_edge[96]*dv1-0.25*H_skin[70]*dv1+0.25*H_edge[70]*dv1; 
  drag_coeff_y[71] = (-0.3518228202874282*H_skin[97]*dv1)+1.217848224071867*H_edge[97]*dv1-0.25*H_skin[71]*dv1+0.25*H_edge[71]*dv1; 
  drag_coeff_y[72] = (-0.3518228202874282*H_skin[98]*dv1)+1.217848224071867*H_edge[98]*dv1-0.25*H_skin[72]*dv1+0.25*H_edge[72]*dv1; 
  drag_coeff_y[73] = (-0.3518228202874282*H_skin[99]*dv1)+1.217848224071867*H_edge[99]*dv1-0.25*H_skin[73]*dv1+0.25*H_edge[73]*dv1; 
  drag_coeff_y[74] = 0.609375*H_skin[74]*dv1+0.890625*H_edge[74]*dv1+0.4330127018922194*H_skin[43]*dv1-0.4330127018922194*H_edge[43]*dv1; 
  drag_coeff_y[75] = 0.609375*H_skin[75]*dv1+0.890625*H_edge[75]*dv1+0.4330127018922194*H_skin[44]*dv1-0.4330127018922194*H_edge[44]*dv1; 
  drag_coeff_y[76] = 0.609375*H_skin[76]*dv1+0.890625*H_edge[76]*dv1+0.4330127018922194*H_skin[45]*dv1-0.4330127018922194*H_edge[45]*dv1; 
  drag_coeff_y[77] = (-0.546875*H_skin[77]*dv1)+1.953125*H_edge[77]*dv1-0.7866997421983816*H_skin[28]*dv1-1.149791930905327*H_edge[28]*dv1-0.5590169943749475*H_skin[12]*dv1+0.5590169943749475*H_edge[12]*dv1; 
  drag_coeff_y[78] = (-0.546875*H_skin[78]*dv1)+1.953125*H_edge[78]*dv1-0.7866997421983816*H_skin[29]*dv1-1.149791930905327*H_edge[29]*dv1-0.5590169943749475*H_skin[13]*dv1+0.5590169943749475*H_edge[13]*dv1; 
  drag_coeff_y[79] = (-0.546875*H_skin[79]*dv1)+1.953125*H_edge[79]*dv1-0.7866997421983816*H_skin[30]*dv1-1.149791930905327*H_edge[30]*dv1-0.5590169943749475*H_skin[14]*dv1+0.5590169943749475*H_edge[14]*dv1; 
  drag_coeff_y[80] = (-0.3518228202874282*H_skin[104]*dv1)+1.217848224071867*H_edge[104]*dv1-0.25*H_skin[80]*dv1+0.25*H_edge[80]*dv1; 
  drag_coeff_y[81] = (-0.3518228202874282*H_skin[105]*dv1)+1.217848224071867*H_edge[105]*dv1-0.25*H_skin[81]*dv1+0.25*H_edge[81]*dv1; 
  drag_coeff_y[82] = (-0.3518228202874282*H_skin[106]*dv1)+1.217848224071867*H_edge[106]*dv1-0.25*H_skin[82]*dv1+0.25*H_edge[82]*dv1; 
  drag_coeff_y[83] = 0.609375*H_skin[83]*dv1+0.890625*H_edge[83]*dv1+0.4330127018922194*H_skin[47]*dv1-0.4330127018922194*H_edge[47]*dv1; 
  drag_coeff_y[84] = 0.609375*H_skin[84]*dv1+0.890625*H_edge[84]*dv1+0.4330127018922194*H_skin[48]*dv1-0.4330127018922194*H_edge[48]*dv1; 
  drag_coeff_y[85] = 0.609375*H_skin[85]*dv1+0.890625*H_edge[85]*dv1+0.4330127018922194*H_skin[49]*dv1-0.4330127018922194*H_edge[49]*dv1; 
  drag_coeff_y[86] = 0.4236075534914363*H_skin[110]*dv1+2.360099226595144*H_edge[110]*dv1+0.609375*H_skin[86]*dv1+0.890625*H_edge[86]*dv1+0.4330127018922193*H_skin[52]*dv1-0.4330127018922193*H_edge[52]*dv1; 
  drag_coeff_y[87] = 0.609375*H_skin[87]*dv1+0.890625*H_edge[87]*dv1+0.4330127018922194*H_skin[56]*dv1-0.4330127018922194*H_edge[56]*dv1; 
  drag_coeff_y[88] = 0.609375*H_skin[88]*dv1+0.890625*H_edge[88]*dv1+0.4330127018922194*H_skin[57]*dv1-0.4330127018922194*H_edge[57]*dv1; 
  drag_coeff_y[89] = 0.609375*H_skin[89]*dv1+0.890625*H_edge[89]*dv1+0.4330127018922194*H_skin[58]*dv1-0.4330127018922194*H_edge[58]*dv1; 
  drag_coeff_y[90] = (-0.546875*H_skin[90]*dv1)+1.953125*H_edge[90]*dv1-0.7866997421983816*H_skin[51]*dv1-1.149791930905327*H_edge[51]*dv1-0.5590169943749476*H_skin[21]*dv1+0.5590169943749476*H_edge[21]*dv1; 
  drag_coeff_y[91] = (-0.3518228202874282*H_skin[107]*dv1)+1.217848224071867*H_edge[107]*dv1-0.25*H_skin[91]*dv1+0.25*H_edge[91]*dv1; 
  drag_coeff_y[92] = (-0.3518228202874282*H_skin[108]*dv1)+1.217848224071867*H_edge[108]*dv1-0.25*H_skin[92]*dv1+0.25*H_edge[92]*dv1; 
  drag_coeff_y[93] = (-0.3518228202874282*H_skin[109]*dv1)+1.217848224071867*H_edge[109]*dv1-0.25*H_skin[93]*dv1+0.25*H_edge[93]*dv1; 
  drag_coeff_y[94] = 0.609375*H_skin[94]*dv1+0.890625*H_edge[94]*dv1+0.4330127018922194*H_skin[68]*dv1-0.4330127018922194*H_edge[68]*dv1; 
  drag_coeff_y[95] = 0.609375*H_skin[95]*dv1+0.890625*H_edge[95]*dv1+0.4330127018922194*H_skin[69]*dv1-0.4330127018922194*H_edge[69]*dv1; 
  drag_coeff_y[96] = 0.609375*H_skin[96]*dv1+0.890625*H_edge[96]*dv1+0.4330127018922194*H_skin[70]*dv1-0.4330127018922194*H_edge[70]*dv1; 
  drag_coeff_y[97] = 0.609375*H_skin[97]*dv1+0.890625*H_edge[97]*dv1+0.4330127018922194*H_skin[71]*dv1-0.4330127018922194*H_edge[71]*dv1; 
  drag_coeff_y[98] = 0.609375*H_skin[98]*dv1+0.890625*H_edge[98]*dv1+0.4330127018922194*H_skin[72]*dv1-0.4330127018922194*H_edge[72]*dv1; 
  drag_coeff_y[99] = 0.609375*H_skin[99]*dv1+0.890625*H_edge[99]*dv1+0.4330127018922194*H_skin[73]*dv1-0.4330127018922194*H_edge[73]*dv1; 
  drag_coeff_y[100] = (-0.546875*H_skin[100]*dv1)+1.953125*H_edge[100]*dv1-0.7866997421983816*H_skin[53]*dv1-1.149791930905327*H_edge[53]*dv1-0.5590169943749476*H_skin[25]*dv1+0.5590169943749476*H_edge[25]*dv1; 
  drag_coeff_y[101] = (-0.546875*H_skin[101]*dv1)+1.953125*H_edge[101]*dv1-0.7866997421983816*H_skin[54]*dv1-1.149791930905327*H_edge[54]*dv1-0.5590169943749476*H_skin[26]*dv1+0.5590169943749476*H_edge[26]*dv1; 
  drag_coeff_y[102] = (-0.546875*H_skin[102]*dv1)+1.953125*H_edge[102]*dv1-0.7866997421983816*H_skin[55]*dv1-1.149791930905327*H_edge[55]*dv1-0.5590169943749476*H_skin[27]*dv1+0.5590169943749476*H_edge[27]*dv1; 
  drag_coeff_y[103] = (-0.3518228202874282*H_skin[111]*dv1)+1.217848224071867*H_edge[111]*dv1-0.25*H_skin[103]*dv1+0.25*H_edge[103]*dv1; 
  drag_coeff_y[104] = 0.609375*H_skin[104]*dv1+0.890625*H_edge[104]*dv1+0.4330127018922194*H_skin[80]*dv1-0.4330127018922194*H_edge[80]*dv1; 
  drag_coeff_y[105] = 0.609375*H_skin[105]*dv1+0.890625*H_edge[105]*dv1+0.4330127018922194*H_skin[81]*dv1-0.4330127018922194*H_edge[81]*dv1; 
  drag_coeff_y[106] = 0.609375*H_skin[106]*dv1+0.890625*H_edge[106]*dv1+0.4330127018922194*H_skin[82]*dv1-0.4330127018922194*H_edge[82]*dv1; 
  drag_coeff_y[107] = 0.609375*H_skin[107]*dv1+0.890625*H_edge[107]*dv1+0.4330127018922194*H_skin[91]*dv1-0.4330127018922194*H_edge[91]*dv1; 
  drag_coeff_y[108] = 0.609375*H_skin[108]*dv1+0.890625*H_edge[108]*dv1+0.4330127018922194*H_skin[92]*dv1-0.4330127018922194*H_edge[92]*dv1; 
  drag_coeff_y[109] = 0.609375*H_skin[109]*dv1+0.890625*H_edge[109]*dv1+0.4330127018922194*H_skin[93]*dv1-0.4330127018922194*H_edge[93]*dv1; 
  drag_coeff_y[110] = (-0.546875*H_skin[110]*dv1)+1.953125*H_edge[110]*dv1-0.7866997421983816*H_skin[86]*dv1-1.149791930905327*H_edge[86]*dv1-0.5590169943749475*H_skin[52]*dv1+0.5590169943749475*H_edge[52]*dv1; 
  drag_coeff_y[111] = 0.609375*H_skin[111]*dv1+0.890625*H_edge[111]*dv1+0.4330127018922194*H_skin[103]*dv1-0.4330127018922194*H_edge[103]*dv1; 
  } else {
  drag_coeff_y[0] = 0.2445699350390395*H_skin[19]*dv1-0.8734640537108556*H_edge[19]*dv1-0.3518228202874282*H_skin[4]*dv1+1.217848224071866*H_edge[4]*dv1+0.25*H_skin[0]*dv1-0.25*H_edge[0]*dv1; 
  drag_coeff_y[1] = 0.2445699350390395*H_skin[40]*dv1-0.8734640537108554*H_edge[40]*dv1-0.3518228202874282*H_skin[9]*dv1+1.217848224071866*H_edge[9]*dv1+0.25*H_skin[1]*dv1-0.25*H_edge[1]*dv1; 
  drag_coeff_y[2] = 0.2445699350390395*H_skin[41]*dv1-0.8734640537108554*H_edge[41]*dv1-0.3518228202874282*H_skin[10]*dv1+1.217848224071866*H_edge[10]*dv1+0.25*H_skin[2]*dv1-0.25*H_edge[2]*dv1; 
  drag_coeff_y[3] = 0.2445699350390395*H_skin[42]*dv1-0.8734640537108554*H_edge[42]*dv1-0.3518228202874282*H_skin[11]*dv1+1.217848224071866*H_edge[11]*dv1+0.25*H_skin[3]*dv1-0.25*H_edge[3]*dv1; 
  drag_coeff_y[4] = 0.4236075534914363*H_skin[19]*dv1+2.360099226595144*H_edge[19]*dv1-0.609375*H_skin[4]*dv1-0.890625*H_edge[4]*dv1+0.4330127018922193*H_skin[0]*dv1-0.4330127018922193*H_edge[0]*dv1; 
  drag_coeff_y[5] = 0.2445699350390395*H_skin[46]*dv1-0.8734640537108554*H_edge[46]*dv1-0.3518228202874282*H_skin[15]*dv1+1.217848224071866*H_edge[15]*dv1+0.25*H_skin[5]*dv1-0.25*H_edge[5]*dv1; 
  drag_coeff_y[6] = 0.2445699350390395*H_skin[65]*dv1-0.8734640537108556*H_edge[65]*dv1-0.3518228202874282*H_skin[22]*dv1+1.217848224071866*H_edge[22]*dv1+0.25*H_skin[6]*dv1-0.25*H_edge[6]*dv1; 
  drag_coeff_y[7] = 0.2445699350390395*H_skin[66]*dv1-0.8734640537108556*H_edge[66]*dv1-0.3518228202874282*H_skin[23]*dv1+1.217848224071866*H_edge[23]*dv1+0.25*H_skin[7]*dv1-0.25*H_edge[7]*dv1; 
  drag_coeff_y[8] = 0.2445699350390395*H_skin[67]*dv1-0.8734640537108556*H_edge[67]*dv1-0.3518228202874282*H_skin[24]*dv1+1.217848224071866*H_edge[24]*dv1+0.25*H_skin[8]*dv1-0.25*H_edge[8]*dv1; 
  drag_coeff_y[9] = 0.4236075534914363*H_skin[40]*dv1+2.360099226595145*H_edge[40]*dv1-0.609375*H_skin[9]*dv1-0.890625*H_edge[9]*dv1+0.4330127018922193*H_skin[1]*dv1-0.4330127018922193*H_edge[1]*dv1; 
  drag_coeff_y[10] = 0.4236075534914363*H_skin[41]*dv1+2.360099226595145*H_edge[41]*dv1-0.609375*H_skin[10]*dv1-0.890625*H_edge[10]*dv1+0.4330127018922193*H_skin[2]*dv1-0.4330127018922193*H_edge[2]*dv1; 
  drag_coeff_y[11] = 0.4236075534914363*H_skin[42]*dv1+2.360099226595145*H_edge[42]*dv1-0.609375*H_skin[11]*dv1-0.890625*H_edge[11]*dv1+0.4330127018922193*H_skin[3]*dv1-0.4330127018922193*H_edge[3]*dv1; 
  drag_coeff_y[12] = 0.2445699350390395*H_skin[77]*dv1-0.8734640537108556*H_edge[77]*dv1-0.3518228202874282*H_skin[28]*dv1+1.217848224071866*H_edge[28]*dv1+0.25*H_skin[12]*dv1-0.25*H_edge[12]*dv1; 
  drag_coeff_y[13] = 0.2445699350390395*H_skin[78]*dv1-0.8734640537108556*H_edge[78]*dv1-0.3518228202874282*H_skin[29]*dv1+1.217848224071866*H_edge[29]*dv1+0.25*H_skin[13]*dv1-0.25*H_edge[13]*dv1; 
  drag_coeff_y[14] = 0.2445699350390395*H_skin[79]*dv1-0.8734640537108556*H_edge[79]*dv1-0.3518228202874282*H_skin[30]*dv1+1.217848224071866*H_edge[30]*dv1+0.25*H_skin[14]*dv1-0.25*H_edge[14]*dv1; 
  drag_coeff_y[15] = 0.4236075534914363*H_skin[46]*dv1+2.360099226595145*H_edge[46]*dv1-0.609375*H_skin[15]*dv1-0.890625*H_edge[15]*dv1+0.4330127018922193*H_skin[5]*dv1-0.4330127018922193*H_edge[5]*dv1; 
  drag_coeff_y[16] = (-0.3518228202874282*H_skin[37]*dv1)+1.217848224071867*H_edge[37]*dv1+0.25*H_skin[16]*dv1-0.25*H_edge[16]*dv1; 
  drag_coeff_y[17] = (-0.3518228202874282*H_skin[38]*dv1)+1.217848224071867*H_edge[38]*dv1+0.25*H_skin[17]*dv1-0.25*H_edge[17]*dv1; 
  drag_coeff_y[18] = (-0.3518228202874282*H_skin[39]*dv1)+1.217848224071867*H_edge[39]*dv1+0.25*H_skin[18]*dv1-0.25*H_edge[18]*dv1; 
  drag_coeff_y[19] = 0.546875*H_skin[19]*dv1-1.953125*H_edge[19]*dv1-0.7866997421983816*H_skin[4]*dv1-1.149791930905327*H_edge[4]*dv1+0.5590169943749475*H_skin[0]*dv1-0.5590169943749475*H_edge[0]*dv1; 
  drag_coeff_y[20] = (-0.3518228202874282*H_skin[50]*dv1)+1.217848224071867*H_edge[50]*dv1+0.25*H_skin[20]*dv1-0.25*H_edge[20]*dv1; 
  drag_coeff_y[21] = 0.2445699350390395*H_skin[90]*dv1-0.8734640537108554*H_edge[90]*dv1-0.3518228202874282*H_skin[51]*dv1+1.217848224071866*H_edge[51]*dv1+0.25*H_skin[21]*dv1-0.25*H_edge[21]*dv1; 
  drag_coeff_y[22] = 0.4236075534914363*H_skin[65]*dv1+2.360099226595144*H_edge[65]*dv1-0.609375*H_skin[22]*dv1-0.890625*H_edge[22]*dv1+0.4330127018922193*H_skin[6]*dv1-0.4330127018922193*H_edge[6]*dv1; 
  drag_coeff_y[23] = 0.4236075534914363*H_skin[66]*dv1+2.360099226595144*H_edge[66]*dv1-0.609375*H_skin[23]*dv1-0.890625*H_edge[23]*dv1+0.4330127018922193*H_skin[7]*dv1-0.4330127018922193*H_edge[7]*dv1; 
  drag_coeff_y[24] = 0.4236075534914363*H_skin[67]*dv1+2.360099226595144*H_edge[67]*dv1-0.609375*H_skin[24]*dv1-0.890625*H_edge[24]*dv1+0.4330127018922193*H_skin[8]*dv1-0.4330127018922193*H_edge[8]*dv1; 
  drag_coeff_y[25] = 0.2445699350390395*H_skin[100]*dv1-0.8734640537108554*H_edge[100]*dv1-0.3518228202874282*H_skin[53]*dv1+1.217848224071866*H_edge[53]*dv1+0.25*H_skin[25]*dv1-0.25*H_edge[25]*dv1; 
  drag_coeff_y[26] = 0.2445699350390395*H_skin[101]*dv1-0.8734640537108554*H_edge[101]*dv1-0.3518228202874282*H_skin[54]*dv1+1.217848224071866*H_edge[54]*dv1+0.25*H_skin[26]*dv1-0.25*H_edge[26]*dv1; 
  drag_coeff_y[27] = 0.2445699350390395*H_skin[102]*dv1-0.8734640537108554*H_edge[102]*dv1-0.3518228202874282*H_skin[55]*dv1+1.217848224071866*H_edge[55]*dv1+0.25*H_skin[27]*dv1-0.25*H_edge[27]*dv1; 
  drag_coeff_y[28] = 0.4236075534914363*H_skin[77]*dv1+2.360099226595144*H_edge[77]*dv1-0.609375*H_skin[28]*dv1-0.890625*H_edge[28]*dv1+0.4330127018922193*H_skin[12]*dv1-0.4330127018922193*H_edge[12]*dv1; 
  drag_coeff_y[29] = 0.4236075534914363*H_skin[78]*dv1+2.360099226595144*H_edge[78]*dv1-0.609375*H_skin[29]*dv1-0.890625*H_edge[29]*dv1+0.4330127018922193*H_skin[13]*dv1-0.4330127018922193*H_edge[13]*dv1; 
  drag_coeff_y[30] = 0.4236075534914363*H_skin[79]*dv1+2.360099226595144*H_edge[79]*dv1-0.609375*H_skin[30]*dv1-0.890625*H_edge[30]*dv1+0.4330127018922193*H_skin[14]*dv1-0.4330127018922193*H_edge[14]*dv1; 
  drag_coeff_y[31] = (-0.3518228202874282*H_skin[59]*dv1)+1.217848224071867*H_edge[59]*dv1+0.25*H_skin[31]*dv1-0.25*H_edge[31]*dv1; 
  drag_coeff_y[32] = (-0.3518228202874282*H_skin[60]*dv1)+1.217848224071867*H_edge[60]*dv1+0.25*H_skin[32]*dv1-0.25*H_edge[32]*dv1; 
  drag_coeff_y[33] = (-0.3518228202874282*H_skin[61]*dv1)+1.217848224071867*H_edge[61]*dv1+0.25*H_skin[33]*dv1-0.25*H_edge[33]*dv1; 
  drag_coeff_y[34] = (-0.3518228202874282*H_skin[62]*dv1)+1.217848224071867*H_edge[62]*dv1+0.25*H_skin[34]*dv1-0.25*H_edge[34]*dv1; 
  drag_coeff_y[35] = (-0.3518228202874282*H_skin[63]*dv1)+1.217848224071867*H_edge[63]*dv1+0.25*H_skin[35]*dv1-0.25*H_edge[35]*dv1; 
  drag_coeff_y[36] = (-0.3518228202874282*H_skin[64]*dv1)+1.217848224071867*H_edge[64]*dv1+0.25*H_skin[36]*dv1-0.25*H_edge[36]*dv1; 
  drag_coeff_y[37] = (-0.609375*H_skin[37]*dv1)-0.890625*H_edge[37]*dv1+0.4330127018922194*H_skin[16]*dv1-0.4330127018922194*H_edge[16]*dv1; 
  drag_coeff_y[38] = (-0.609375*H_skin[38]*dv1)-0.890625*H_edge[38]*dv1+0.4330127018922194*H_skin[17]*dv1-0.4330127018922194*H_edge[17]*dv1; 
  drag_coeff_y[39] = (-0.609375*H_skin[39]*dv1)-0.890625*H_edge[39]*dv1+0.4330127018922194*H_skin[18]*dv1-0.4330127018922194*H_edge[18]*dv1; 
  drag_coeff_y[40] = 0.546875*H_skin[40]*dv1-1.953125*H_edge[40]*dv1-0.7866997421983816*H_skin[9]*dv1-1.149791930905327*H_edge[9]*dv1+0.5590169943749476*H_skin[1]*dv1-0.5590169943749476*H_edge[1]*dv1; 
  drag_coeff_y[41] = 0.546875*H_skin[41]*dv1-1.953125*H_edge[41]*dv1-0.7866997421983816*H_skin[10]*dv1-1.149791930905327*H_edge[10]*dv1+0.5590169943749476*H_skin[2]*dv1-0.5590169943749476*H_edge[2]*dv1; 
  drag_coeff_y[42] = 0.546875*H_skin[42]*dv1-1.953125*H_edge[42]*dv1-0.7866997421983816*H_skin[11]*dv1-1.149791930905327*H_edge[11]*dv1+0.5590169943749476*H_skin[3]*dv1-0.5590169943749476*H_edge[3]*dv1; 
  drag_coeff_y[43] = (-0.3518228202874282*H_skin[74]*dv1)+1.217848224071867*H_edge[74]*dv1+0.25*H_skin[43]*dv1-0.25*H_edge[43]*dv1; 
  drag_coeff_y[44] = (-0.3518228202874282*H_skin[75]*dv1)+1.217848224071867*H_edge[75]*dv1+0.25*H_skin[44]*dv1-0.25*H_edge[44]*dv1; 
  drag_coeff_y[45] = (-0.3518228202874282*H_skin[76]*dv1)+1.217848224071867*H_edge[76]*dv1+0.25*H_skin[45]*dv1-0.25*H_edge[45]*dv1; 
  drag_coeff_y[46] = 0.546875*H_skin[46]*dv1-1.953125*H_edge[46]*dv1-0.7866997421983816*H_skin[15]*dv1-1.149791930905327*H_edge[15]*dv1+0.5590169943749476*H_skin[5]*dv1-0.5590169943749476*H_edge[5]*dv1; 
  drag_coeff_y[47] = (-0.3518228202874282*H_skin[83]*dv1)+1.217848224071867*H_edge[83]*dv1+0.25*H_skin[47]*dv1-0.25*H_edge[47]*dv1; 
  drag_coeff_y[48] = (-0.3518228202874282*H_skin[84]*dv1)+1.217848224071867*H_edge[84]*dv1+0.25*H_skin[48]*dv1-0.25*H_edge[48]*dv1; 
  drag_coeff_y[49] = (-0.3518228202874282*H_skin[85]*dv1)+1.217848224071867*H_edge[85]*dv1+0.25*H_skin[49]*dv1-0.25*H_edge[49]*dv1; 
  drag_coeff_y[50] = (-0.609375*H_skin[50]*dv1)-0.890625*H_edge[50]*dv1+0.4330127018922194*H_skin[20]*dv1-0.4330127018922194*H_edge[20]*dv1; 
  drag_coeff_y[51] = 0.4236075534914363*H_skin[90]*dv1+2.360099226595145*H_edge[90]*dv1-0.609375*H_skin[51]*dv1-0.890625*H_edge[51]*dv1+0.4330127018922193*H_skin[21]*dv1-0.4330127018922193*H_edge[21]*dv1; 
  drag_coeff_y[52] = 0.2445699350390395*H_skin[110]*dv1-0.8734640537108556*H_edge[110]*dv1-0.3518228202874282*H_skin[86]*dv1+1.217848224071866*H_edge[86]*dv1+0.25*H_skin[52]*dv1-0.25*H_edge[52]*dv1; 
  drag_coeff_y[53] = 0.4236075534914363*H_skin[100]*dv1+2.360099226595145*H_edge[100]*dv1-0.609375*H_skin[53]*dv1-0.890625*H_edge[53]*dv1+0.4330127018922193*H_skin[25]*dv1-0.4330127018922193*H_edge[25]*dv1; 
  drag_coeff_y[54] = 0.4236075534914363*H_skin[101]*dv1+2.360099226595145*H_edge[101]*dv1-0.609375*H_skin[54]*dv1-0.890625*H_edge[54]*dv1+0.4330127018922193*H_skin[26]*dv1-0.4330127018922193*H_edge[26]*dv1; 
  drag_coeff_y[55] = 0.4236075534914363*H_skin[102]*dv1+2.360099226595145*H_edge[102]*dv1-0.609375*H_skin[55]*dv1-0.890625*H_edge[55]*dv1+0.4330127018922193*H_skin[27]*dv1-0.4330127018922193*H_edge[27]*dv1; 
  drag_coeff_y[56] = (-0.3518228202874282*H_skin[87]*dv1)+1.217848224071867*H_edge[87]*dv1+0.25*H_skin[56]*dv1-0.25*H_edge[56]*dv1; 
  drag_coeff_y[57] = (-0.3518228202874282*H_skin[88]*dv1)+1.217848224071867*H_edge[88]*dv1+0.25*H_skin[57]*dv1-0.25*H_edge[57]*dv1; 
  drag_coeff_y[58] = (-0.3518228202874282*H_skin[89]*dv1)+1.217848224071867*H_edge[89]*dv1+0.25*H_skin[58]*dv1-0.25*H_edge[58]*dv1; 
  drag_coeff_y[59] = (-0.609375*H_skin[59]*dv1)-0.890625*H_edge[59]*dv1+0.4330127018922194*H_skin[31]*dv1-0.4330127018922194*H_edge[31]*dv1; 
  drag_coeff_y[60] = (-0.609375*H_skin[60]*dv1)-0.890625*H_edge[60]*dv1+0.4330127018922194*H_skin[32]*dv1-0.4330127018922194*H_edge[32]*dv1; 
  drag_coeff_y[61] = (-0.609375*H_skin[61]*dv1)-0.890625*H_edge[61]*dv1+0.4330127018922194*H_skin[33]*dv1-0.4330127018922194*H_edge[33]*dv1; 
  drag_coeff_y[62] = (-0.609375*H_skin[62]*dv1)-0.890625*H_edge[62]*dv1+0.4330127018922194*H_skin[34]*dv1-0.4330127018922194*H_edge[34]*dv1; 
  drag_coeff_y[63] = (-0.609375*H_skin[63]*dv1)-0.890625*H_edge[63]*dv1+0.4330127018922194*H_skin[35]*dv1-0.4330127018922194*H_edge[35]*dv1; 
  drag_coeff_y[64] = (-0.609375*H_skin[64]*dv1)-0.890625*H_edge[64]*dv1+0.4330127018922194*H_skin[36]*dv1-0.4330127018922194*H_edge[36]*dv1; 
  drag_coeff_y[65] = 0.546875*H_skin[65]*dv1-1.953125*H_edge[65]*dv1-0.7866997421983816*H_skin[22]*dv1-1.149791930905327*H_edge[22]*dv1+0.5590169943749475*H_skin[6]*dv1-0.5590169943749475*H_edge[6]*dv1; 
  drag_coeff_y[66] = 0.546875*H_skin[66]*dv1-1.953125*H_edge[66]*dv1-0.7866997421983816*H_skin[23]*dv1-1.149791930905327*H_edge[23]*dv1+0.5590169943749475*H_skin[7]*dv1-0.5590169943749475*H_edge[7]*dv1; 
  drag_coeff_y[67] = 0.546875*H_skin[67]*dv1-1.953125*H_edge[67]*dv1-0.7866997421983816*H_skin[24]*dv1-1.149791930905327*H_edge[24]*dv1+0.5590169943749475*H_skin[8]*dv1-0.5590169943749475*H_edge[8]*dv1; 
  drag_coeff_y[68] = (-0.3518228202874282*H_skin[94]*dv1)+1.217848224071867*H_edge[94]*dv1+0.25*H_skin[68]*dv1-0.25*H_edge[68]*dv1; 
  drag_coeff_y[69] = (-0.3518228202874282*H_skin[95]*dv1)+1.217848224071867*H_edge[95]*dv1+0.25*H_skin[69]*dv1-0.25*H_edge[69]*dv1; 
  drag_coeff_y[70] = (-0.3518228202874282*H_skin[96]*dv1)+1.217848224071867*H_edge[96]*dv1+0.25*H_skin[70]*dv1-0.25*H_edge[70]*dv1; 
  drag_coeff_y[71] = (-0.3518228202874282*H_skin[97]*dv1)+1.217848224071867*H_edge[97]*dv1+0.25*H_skin[71]*dv1-0.25*H_edge[71]*dv1; 
  drag_coeff_y[72] = (-0.3518228202874282*H_skin[98]*dv1)+1.217848224071867*H_edge[98]*dv1+0.25*H_skin[72]*dv1-0.25*H_edge[72]*dv1; 
  drag_coeff_y[73] = (-0.3518228202874282*H_skin[99]*dv1)+1.217848224071867*H_edge[99]*dv1+0.25*H_skin[73]*dv1-0.25*H_edge[73]*dv1; 
  drag_coeff_y[74] = (-0.609375*H_skin[74]*dv1)-0.890625*H_edge[74]*dv1+0.4330127018922194*H_skin[43]*dv1-0.4330127018922194*H_edge[43]*dv1; 
  drag_coeff_y[75] = (-0.609375*H_skin[75]*dv1)-0.890625*H_edge[75]*dv1+0.4330127018922194*H_skin[44]*dv1-0.4330127018922194*H_edge[44]*dv1; 
  drag_coeff_y[76] = (-0.609375*H_skin[76]*dv1)-0.890625*H_edge[76]*dv1+0.4330127018922194*H_skin[45]*dv1-0.4330127018922194*H_edge[45]*dv1; 
  drag_coeff_y[77] = 0.546875*H_skin[77]*dv1-1.953125*H_edge[77]*dv1-0.7866997421983816*H_skin[28]*dv1-1.149791930905327*H_edge[28]*dv1+0.5590169943749475*H_skin[12]*dv1-0.5590169943749475*H_edge[12]*dv1; 
  drag_coeff_y[78] = 0.546875*H_skin[78]*dv1-1.953125*H_edge[78]*dv1-0.7866997421983816*H_skin[29]*dv1-1.149791930905327*H_edge[29]*dv1+0.5590169943749475*H_skin[13]*dv1-0.5590169943749475*H_edge[13]*dv1; 
  drag_coeff_y[79] = 0.546875*H_skin[79]*dv1-1.953125*H_edge[79]*dv1-0.7866997421983816*H_skin[30]*dv1-1.149791930905327*H_edge[30]*dv1+0.5590169943749475*H_skin[14]*dv1-0.5590169943749475*H_edge[14]*dv1; 
  drag_coeff_y[80] = (-0.3518228202874282*H_skin[104]*dv1)+1.217848224071867*H_edge[104]*dv1+0.25*H_skin[80]*dv1-0.25*H_edge[80]*dv1; 
  drag_coeff_y[81] = (-0.3518228202874282*H_skin[105]*dv1)+1.217848224071867*H_edge[105]*dv1+0.25*H_skin[81]*dv1-0.25*H_edge[81]*dv1; 
  drag_coeff_y[82] = (-0.3518228202874282*H_skin[106]*dv1)+1.217848224071867*H_edge[106]*dv1+0.25*H_skin[82]*dv1-0.25*H_edge[82]*dv1; 
  drag_coeff_y[83] = (-0.609375*H_skin[83]*dv1)-0.890625*H_edge[83]*dv1+0.4330127018922194*H_skin[47]*dv1-0.4330127018922194*H_edge[47]*dv1; 
  drag_coeff_y[84] = (-0.609375*H_skin[84]*dv1)-0.890625*H_edge[84]*dv1+0.4330127018922194*H_skin[48]*dv1-0.4330127018922194*H_edge[48]*dv1; 
  drag_coeff_y[85] = (-0.609375*H_skin[85]*dv1)-0.890625*H_edge[85]*dv1+0.4330127018922194*H_skin[49]*dv1-0.4330127018922194*H_edge[49]*dv1; 
  drag_coeff_y[86] = 0.4236075534914363*H_skin[110]*dv1+2.360099226595144*H_edge[110]*dv1-0.609375*H_skin[86]*dv1-0.890625*H_edge[86]*dv1+0.4330127018922193*H_skin[52]*dv1-0.4330127018922193*H_edge[52]*dv1; 
  drag_coeff_y[87] = (-0.609375*H_skin[87]*dv1)-0.890625*H_edge[87]*dv1+0.4330127018922194*H_skin[56]*dv1-0.4330127018922194*H_edge[56]*dv1; 
  drag_coeff_y[88] = (-0.609375*H_skin[88]*dv1)-0.890625*H_edge[88]*dv1+0.4330127018922194*H_skin[57]*dv1-0.4330127018922194*H_edge[57]*dv1; 
  drag_coeff_y[89] = (-0.609375*H_skin[89]*dv1)-0.890625*H_edge[89]*dv1+0.4330127018922194*H_skin[58]*dv1-0.4330127018922194*H_edge[58]*dv1; 
  drag_coeff_y[90] = 0.546875*H_skin[90]*dv1-1.953125*H_edge[90]*dv1-0.7866997421983816*H_skin[51]*dv1-1.149791930905327*H_edge[51]*dv1+0.5590169943749476*H_skin[21]*dv1-0.5590169943749476*H_edge[21]*dv1; 
  drag_coeff_y[91] = (-0.3518228202874282*H_skin[107]*dv1)+1.217848224071867*H_edge[107]*dv1+0.25*H_skin[91]*dv1-0.25*H_edge[91]*dv1; 
  drag_coeff_y[92] = (-0.3518228202874282*H_skin[108]*dv1)+1.217848224071867*H_edge[108]*dv1+0.25*H_skin[92]*dv1-0.25*H_edge[92]*dv1; 
  drag_coeff_y[93] = (-0.3518228202874282*H_skin[109]*dv1)+1.217848224071867*H_edge[109]*dv1+0.25*H_skin[93]*dv1-0.25*H_edge[93]*dv1; 
  drag_coeff_y[94] = (-0.609375*H_skin[94]*dv1)-0.890625*H_edge[94]*dv1+0.4330127018922194*H_skin[68]*dv1-0.4330127018922194*H_edge[68]*dv1; 
  drag_coeff_y[95] = (-0.609375*H_skin[95]*dv1)-0.890625*H_edge[95]*dv1+0.4330127018922194*H_skin[69]*dv1-0.4330127018922194*H_edge[69]*dv1; 
  drag_coeff_y[96] = (-0.609375*H_skin[96]*dv1)-0.890625*H_edge[96]*dv1+0.4330127018922194*H_skin[70]*dv1-0.4330127018922194*H_edge[70]*dv1; 
  drag_coeff_y[97] = (-0.609375*H_skin[97]*dv1)-0.890625*H_edge[97]*dv1+0.4330127018922194*H_skin[71]*dv1-0.4330127018922194*H_edge[71]*dv1; 
  drag_coeff_y[98] = (-0.609375*H_skin[98]*dv1)-0.890625*H_edge[98]*dv1+0.4330127018922194*H_skin[72]*dv1-0.4330127018922194*H_edge[72]*dv1; 
  drag_coeff_y[99] = (-0.609375*H_skin[99]*dv1)-0.890625*H_edge[99]*dv1+0.4330127018922194*H_skin[73]*dv1-0.4330127018922194*H_edge[73]*dv1; 
  drag_coeff_y[100] = 0.546875*H_skin[100]*dv1-1.953125*H_edge[100]*dv1-0.7866997421983816*H_skin[53]*dv1-1.149791930905327*H_edge[53]*dv1+0.5590169943749476*H_skin[25]*dv1-0.5590169943749476*H_edge[25]*dv1; 
  drag_coeff_y[101] = 0.546875*H_skin[101]*dv1-1.953125*H_edge[101]*dv1-0.7866997421983816*H_skin[54]*dv1-1.149791930905327*H_edge[54]*dv1+0.5590169943749476*H_skin[26]*dv1-0.5590169943749476*H_edge[26]*dv1; 
  drag_coeff_y[102] = 0.546875*H_skin[102]*dv1-1.953125*H_edge[102]*dv1-0.7866997421983816*H_skin[55]*dv1-1.149791930905327*H_edge[55]*dv1+0.5590169943749476*H_skin[27]*dv1-0.5590169943749476*H_edge[27]*dv1; 
  drag_coeff_y[103] = (-0.3518228202874282*H_skin[111]*dv1)+1.217848224071867*H_edge[111]*dv1+0.25*H_skin[103]*dv1-0.25*H_edge[103]*dv1; 
  drag_coeff_y[104] = (-0.609375*H_skin[104]*dv1)-0.890625*H_edge[104]*dv1+0.4330127018922194*H_skin[80]*dv1-0.4330127018922194*H_edge[80]*dv1; 
  drag_coeff_y[105] = (-0.609375*H_skin[105]*dv1)-0.890625*H_edge[105]*dv1+0.4330127018922194*H_skin[81]*dv1-0.4330127018922194*H_edge[81]*dv1; 
  drag_coeff_y[106] = (-0.609375*H_skin[106]*dv1)-0.890625*H_edge[106]*dv1+0.4330127018922194*H_skin[82]*dv1-0.4330127018922194*H_edge[82]*dv1; 
  drag_coeff_y[107] = (-0.609375*H_skin[107]*dv1)-0.890625*H_edge[107]*dv1+0.4330127018922194*H_skin[91]*dv1-0.4330127018922194*H_edge[91]*dv1; 
  drag_coeff_y[108] = (-0.609375*H_skin[108]*dv1)-0.890625*H_edge[108]*dv1+0.4330127018922194*H_skin[92]*dv1-0.4330127018922194*H_edge[92]*dv1; 
  drag_coeff_y[109] = (-0.609375*H_skin[109]*dv1)-0.890625*H_edge[109]*dv1+0.4330127018922194*H_skin[93]*dv1-0.4330127018922194*H_edge[93]*dv1; 
  drag_coeff_y[110] = 0.546875*H_skin[110]*dv1-1.953125*H_edge[110]*dv1-0.7866997421983816*H_skin[86]*dv1-1.149791930905327*H_edge[86]*dv1+0.5590169943749475*H_skin[52]*dv1-0.5590169943749475*H_edge[52]*dv1; 
  drag_coeff_y[111] = (-0.609375*H_skin[111]*dv1)-0.890625*H_edge[111]*dv1+0.4330127018922194*H_skin[103]*dv1-0.4330127018922194*H_edge[103]*dv1; 
  } 
}