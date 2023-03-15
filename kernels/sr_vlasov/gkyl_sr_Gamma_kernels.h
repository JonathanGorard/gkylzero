#pragma once 
#include <math.h> 
#include <gkyl_util.h> 
EXTERN_C_BEG 

GKYL_CU_DH void sr_Gamma2_1x1v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x1v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x1v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_1x1v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x1v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x1v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_1x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_1x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_1x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_1x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_1x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_1x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_2x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_2x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_2x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_2x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_2x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_2x2v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_2x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_2x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_2x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_2x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_2x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_2x3v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv); 

GKYL_CU_DH void sr_Gamma2_3x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2); 
GKYL_CU_DH void sr_Gamma_3x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma); 
GKYL_CU_DH void sr_Gamma_inv_3x3v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma_inv); 

EXTERN_C_END 
