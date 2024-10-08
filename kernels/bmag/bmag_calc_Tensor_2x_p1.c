#include "gkyl_calc_bmag_kernels.h"

static inline double magnitude(double B_R, double B_Z, double B_PHI) 
{ double mag = 0;  mag = sqrt(B_R*B_R + B_Z*B_Z + B_PHI*B_PHI); return mag; } 

GKYL_CU_DH void bmag_2x_Tensor_p1( const double **psibyr, const double *psibyr2, const double *bphi, double *bmagout, double scale_factorR, double scale_factorZ) 
{ 
double B_R_n[4], B_Z_n[4], B_Z1_n[4], B_Z2_n[4], B_PHI_n[4]; 
const double *psibyrI = psibyr[0];
const double *psibyrL = psibyr[1];
const double *psibyrR = psibyr[2];
const double *psibyrB = psibyr[3];
const double *psibyrT = psibyr[4];
double bmag_n[4]; 
  B_R_n[0] = 0.9375*psibyrI[3]+0.9375*psibyrB[3]-0.5412658773652748*psibyrI[2]-0.5412658773652739*psibyrB[2]-0.9742785792574933*psibyrI[1]+0.9742785792574913*psibyrB[1]+0.5625*psibyrI[0]-0.5625*psibyrB[0]; 
  B_R_n[0] = B_R_n[0]*scale_factorZ; 
  B_Z1_n[0] = (-0.9375*psibyrL[3])-0.9375*psibyrI[3]-0.9742785792574913*psibyrL[2]+0.9742785792574933*psibyrI[2]+0.5412658773652739*psibyrL[1]+0.5412658773652748*psibyrI[1]+0.5625*psibyrL[0]-0.5625*psibyrI[0]; 
  B_Z2_n[0] = (-1.5*psibyr2[3])+0.8660254037844386*psibyr2[2]+0.8660254037844386*psibyr2[1]-0.5*psibyr2[0]; 
  B_Z_n[0] = B_Z1_n[0]*scale_factorR + B_Z2_n[0]; 
  B_PHI_n[0] = 1.5*bphi[3]-0.8660254037844386*bphi[2]-0.8660254037844386*bphi[1]+0.5*bphi[0]; 
  bmag_n[0] = magnitude(B_R_n[0], B_Z_n[0], B_PHI_n[0]); 
  B_R_n[1] = (-0.9375*psibyrI[3])-0.9375*psibyrB[3]-0.5412658773652748*psibyrI[2]-0.5412658773652739*psibyrB[2]+0.9742785792574933*psibyrI[1]-0.9742785792574913*psibyrB[1]+0.5625*psibyrI[0]-0.5625*psibyrB[0]; 
  B_R_n[1] = B_R_n[1]*scale_factorZ; 
  B_Z1_n[1] = (-0.9375*psibyrR[3])-0.9375*psibyrI[3]+0.9742785792574913*psibyrR[2]-0.9742785792574933*psibyrI[2]+0.5412658773652739*psibyrR[1]+0.5412658773652748*psibyrI[1]-0.5625*psibyrR[0]+0.5625*psibyrI[0]; 
  B_Z2_n[1] = 1.5*psibyr2[3]+0.8660254037844386*psibyr2[2]-0.8660254037844386*psibyr2[1]-0.5*psibyr2[0]; 
  B_Z_n[1] = B_Z1_n[1]*scale_factorR + B_Z2_n[1]; 
  B_PHI_n[1] = (-1.5*bphi[3])-0.8660254037844386*bphi[2]+0.8660254037844386*bphi[1]+0.5*bphi[0]; 
  bmag_n[1] = magnitude(B_R_n[1], B_Z_n[1], B_PHI_n[1]); 
  B_R_n[2] = 0.9375*psibyrT[3]+0.9375*psibyrI[3]-0.5412658773652739*psibyrT[2]-0.5412658773652748*psibyrI[2]-0.9742785792574913*psibyrT[1]+0.9742785792574933*psibyrI[1]+0.5625*psibyrT[0]-0.5625*psibyrI[0]; 
  B_R_n[2] = B_R_n[2]*scale_factorZ; 
  B_Z1_n[2] = 0.9375*psibyrL[3]+0.9375*psibyrI[3]+0.9742785792574913*psibyrL[2]-0.9742785792574933*psibyrI[2]+0.5412658773652739*psibyrL[1]+0.5412658773652748*psibyrI[1]+0.5625*psibyrL[0]-0.5625*psibyrI[0]; 
  B_Z2_n[2] = 1.5*psibyr2[3]-0.8660254037844386*psibyr2[2]+0.8660254037844386*psibyr2[1]-0.5*psibyr2[0]; 
  B_Z_n[2] = B_Z1_n[2]*scale_factorR + B_Z2_n[2]; 
  B_PHI_n[2] = (-1.5*bphi[3])+0.8660254037844386*bphi[2]-0.8660254037844386*bphi[1]+0.5*bphi[0]; 
  bmag_n[2] = magnitude(B_R_n[2], B_Z_n[2], B_PHI_n[2]); 
  B_R_n[3] = (-0.9375*psibyrT[3])-0.9375*psibyrI[3]-0.5412658773652739*psibyrT[2]-0.5412658773652748*psibyrI[2]+0.9742785792574913*psibyrT[1]-0.9742785792574933*psibyrI[1]+0.5625*psibyrT[0]-0.5625*psibyrI[0]; 
  B_R_n[3] = B_R_n[3]*scale_factorZ; 
  B_Z1_n[3] = 0.9375*psibyrR[3]+0.9375*psibyrI[3]-0.9742785792574913*psibyrR[2]+0.9742785792574933*psibyrI[2]+0.5412658773652739*psibyrR[1]+0.5412658773652748*psibyrI[1]-0.5625*psibyrR[0]+0.5625*psibyrI[0]; 
  B_Z2_n[3] = (-1.5*psibyr2[3])-0.8660254037844386*psibyr2[2]-0.8660254037844386*psibyr2[1]-0.5*psibyr2[0]; 
  B_Z_n[3] = B_Z1_n[3]*scale_factorR + B_Z2_n[3]; 
  B_PHI_n[3] = 1.5*bphi[3]+0.8660254037844386*bphi[2]+0.8660254037844386*bphi[1]+0.5*bphi[0]; 
  bmag_n[3] = magnitude(B_R_n[3], B_Z_n[3], B_PHI_n[3]); 
  bmagout[0] = 0.5*bmag_n[3]+0.5*bmag_n[2]+0.5*bmag_n[1]+0.5*bmag_n[0]; 
  bmagout[1] = 0.2886751345948129*bmag_n[3]-0.2886751345948129*bmag_n[2]+0.2886751345948129*bmag_n[1]-0.2886751345948129*bmag_n[0]; 
  bmagout[2] = 0.2886751345948129*bmag_n[3]+0.2886751345948129*bmag_n[2]-0.2886751345948129*bmag_n[1]-0.2886751345948129*bmag_n[0]; 
  bmagout[3] = 0.1666666666666667*bmag_n[3]-0.1666666666666667*bmag_n[2]-0.1666666666666667*bmag_n[1]+0.1666666666666667*bmag_n[0]; 
 
}
