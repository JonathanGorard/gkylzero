#include <gkyl_binop_cross_mul_ser.h> 
 
GKYL_CU_DH
void
binop_cross_mul_accumulate_1d_3d_ser_p3(double a, const double *f, const double *g, double *fg) 
{ 
  // f:  First input DG field. 
  // g:  Second input DG field. 
  // fg: Output DG field f*g using weak multiplication. 
 
  fg[0] += a*(0.7071067811865475*f[3]*g[17]+0.7071067811865475*f[2]*g[7]+0.7071067811865475*f[1]*g[1]+0.7071067811865475*f[0]*g[0]); 
  fg[1] += a*(0.6210590034081186*f[2]*g[17]+0.6210590034081186*f[3]*g[7]+0.6324555320336759*f[1]*g[7]+0.6324555320336759*g[1]*f[2]+0.7071067811865475*f[0]*g[1]+0.7071067811865475*g[0]*f[1]); 
  fg[2] += a*(0.7071067811865474*f[3]*g[23]+0.7071067811865475*f[2]*g[11]+0.7071067811865475*f[1]*g[4]+0.7071067811865475*f[0]*g[2]); 
  fg[3] += a*(0.7071067811865474*f[3]*g[25]+0.7071067811865475*f[2]*g[13]+0.7071067811865475*f[1]*g[5]+0.7071067811865475*f[0]*g[3]); 
  fg[4] += a*(0.6210590034081187*f[2]*g[23]+0.6210590034081187*f[3]*g[11]+0.632455532033676*f[1]*g[11]+0.6324555320336759*f[2]*g[4]+0.7071067811865475*f[0]*g[4]+0.7071067811865475*f[1]*g[2]); 
  fg[5] += a*(0.6210590034081187*f[2]*g[25]+0.6210590034081187*f[3]*g[13]+0.632455532033676*f[1]*g[13]+0.6324555320336759*f[2]*g[5]+0.7071067811865475*f[0]*g[5]+0.7071067811865475*f[1]*g[3]); 
  fg[6] += a*(0.7071067811865475*f[3]*g[29]+0.7071067811865475*f[2]*g[20]+0.7071067811865475*f[1]*g[10]+0.7071067811865475*f[0]*g[6]); 
  fg[7] += a*(0.421637021355784*f[3]*g[17]+0.6210590034081186*f[1]*g[17]+0.45175395145262565*f[2]*g[7]+0.7071067811865475*f[0]*g[7]+0.6210590034081186*g[1]*f[3]+0.7071067811865475*g[0]*f[2]+0.6324555320336759*f[1]*g[1]); 
  fg[8] += a*(0.7071067811865475*f[1]*g[12]+0.7071067811865475*f[0]*g[8]); 
  fg[9] += a*(0.7071067811865475*f[1]*g[15]+0.7071067811865475*f[0]*g[9]); 
  fg[10] += a*(0.6210590034081186*f[2]*g[29]+0.6210590034081186*f[3]*g[20]+0.6324555320336759*f[1]*g[20]+0.6324555320336759*f[2]*g[10]+0.7071067811865475*f[0]*g[10]+0.7071067811865475*f[1]*g[6]); 
  fg[11] += a*(0.4216370213557839*f[3]*g[23]+0.6210590034081187*f[1]*g[23]+0.45175395145262565*f[2]*g[11]+0.7071067811865475*f[0]*g[11]+0.6210590034081187*f[3]*g[4]+0.632455532033676*f[1]*g[4]+0.7071067811865475*f[2]*g[2]); 
  fg[12] += a*(0.6324555320336759*f[2]*g[12]+0.7071067811865475*f[0]*g[12]+0.7071067811865475*f[1]*g[8]); 
  fg[13] += a*(0.4216370213557839*f[3]*g[25]+0.6210590034081187*f[1]*g[25]+0.45175395145262565*f[2]*g[13]+0.7071067811865475*f[0]*g[13]+0.6210590034081187*f[3]*g[5]+0.632455532033676*f[1]*g[5]+0.7071067811865475*f[2]*g[3]); 
  fg[14] += a*(0.7071067811865475*f[1]*g[21]+0.7071067811865475*f[0]*g[14]); 
  fg[15] += a*(0.6324555320336759*f[2]*g[15]+0.7071067811865475*f[0]*g[15]+0.7071067811865475*f[1]*g[9]); 
  fg[16] += a*(0.7071067811865475*f[1]*g[22]+0.7071067811865475*f[0]*g[16]); 
  fg[17] += a*(0.421637021355784*f[2]*g[17]+0.7071067811865475*f[0]*g[17]+0.421637021355784*f[3]*g[7]+0.6210590034081186*f[1]*g[7]+0.7071067811865475*g[0]*f[3]+0.6210590034081186*g[1]*f[2]); 
  fg[18] += a*(0.7071067811865474*f[1]*g[24]+0.7071067811865475*f[0]*g[18]); 
  fg[19] += a*(0.7071067811865474*f[1]*g[27]+0.7071067811865475*f[0]*g[19]); 
  fg[20] += a*(0.421637021355784*f[3]*g[29]+0.6210590034081186*f[1]*g[29]+0.45175395145262565*f[2]*g[20]+0.7071067811865475*f[0]*g[20]+0.6210590034081186*f[3]*g[10]+0.6324555320336759*f[1]*g[10]+0.7071067811865475*f[2]*g[6]); 
  fg[21] += a*(0.6324555320336759*f[2]*g[21]+0.7071067811865475*f[0]*g[21]+0.7071067811865475*f[1]*g[14]); 
  fg[22] += a*(0.6324555320336759*f[2]*g[22]+0.7071067811865475*f[0]*g[22]+0.7071067811865475*f[1]*g[16]); 
  fg[23] += a*(0.421637021355784*f[2]*g[23]+0.7071067811865475*f[0]*g[23]+0.4216370213557839*f[3]*g[11]+0.6210590034081187*f[1]*g[11]+0.6210590034081187*f[2]*g[4]+0.7071067811865474*g[2]*f[3]); 
  fg[24] += a*(0.6324555320336759*f[2]*g[24]+0.7071067811865475*f[0]*g[24]+0.7071067811865474*f[1]*g[18]); 
  fg[25] += a*(0.421637021355784*f[2]*g[25]+0.7071067811865475*f[0]*g[25]+0.4216370213557839*f[3]*g[13]+0.6210590034081187*f[1]*g[13]+0.6210590034081187*f[2]*g[5]+0.7071067811865474*f[3]*g[3]); 
  fg[26] += a*(0.7071067811865474*f[1]*g[30]+0.7071067811865475*f[0]*g[26]); 
  fg[27] += a*(0.6324555320336759*f[2]*g[27]+0.7071067811865475*f[0]*g[27]+0.7071067811865474*f[1]*g[19]); 
  fg[28] += a*(0.7071067811865474*f[1]*g[31]+0.7071067811865475*f[0]*g[28]); 
  fg[29] += a*(0.421637021355784*f[2]*g[29]+0.7071067811865475*f[0]*g[29]+0.421637021355784*f[3]*g[20]+0.6210590034081186*f[1]*g[20]+0.6210590034081186*f[2]*g[10]+0.7071067811865475*f[3]*g[6]); 
  fg[30] += a*(0.6324555320336759*f[2]*g[30]+0.7071067811865475*f[0]*g[30]+0.7071067811865474*f[1]*g[26]); 
  fg[31] += a*(0.6324555320336759*f[2]*g[31]+0.7071067811865475*f[0]*g[31]+0.7071067811865474*f[1]*g[28]); 
} 
