#pragma once 
#include <math.h> 
#include <gkyl_util.h> 
EXTERN_C_BEG 

GKYL_CU_DH void derived_geo_3x_Ser_p1(const double *gij, const double *bmag, double *J, double *Jinv, double *grij, double *bi, double *cmag, double *Jtot, double *Jtotinv, double *bmaginv, double *bmaginvsq, double *gxxJ, double *gxyJ, double *gyyJ); 
GKYL_CU_DH void adjust_bmag_3x_Ser_p1(const double *cmag, const double *gzz, const double *J, double *bmag); 
GKYL_CU_DH void derived_geo_3x_Ser_p2(const double *gij, const double *bmag, double *J, double *Jinv, double *grij, double *bi, double *cmag, double *Jtot, double *Jtotinv, double *bmaginv, double *bmaginvsq, double *gxxJ, double *gxyJ, double *gyyJ); 
GKYL_CU_DH void adjust_bmag_3x_Ser_p2(const double *cmag, const double *gzz, const double *J, double *bmag); 
EXTERN_C_END 