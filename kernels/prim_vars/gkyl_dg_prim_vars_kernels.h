#pragma once 
#include <math.h> 
#include <gkyl_mat.h> 
#include <gkyl_util.h> 
 
EXTERN_C_BEG 

GKYL_CU_DH void vlasov_prim_vars_1x1v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_1x1v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_1x1v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_1x1v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_1x1v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 

GKYL_CU_DH void vlasov_prim_vars_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_1x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_1x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_1x2v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 

GKYL_CU_DH void vlasov_prim_vars_1x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_1x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_1x3v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_1x3v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_1x3v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 

GKYL_CU_DH void vlasov_prim_vars_2x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_2x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_2x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_2x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_2x2v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 

GKYL_CU_DH void vlasov_prim_vars_2x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_2x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_2x3v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_2x3v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_2x3v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 

GKYL_CU_DH void vlasov_prim_vars_3x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_u_i_3x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void vlasov_prim_vars_vtSq_3x3v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_vlasov_3x3v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_i_3x3v_ser_p1(const double *b_i, const double *moms, double* u_par_i); 


GKYL_CU_DH void gyrokinetic_prim_vars_1x1v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_1x1v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_1x1v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_1x1v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_1x1v_ser_p1(const double *b_i, const double *moms, double* u_par); 

GKYL_CU_DH void gyrokinetic_prim_vars_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_1x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_1x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_1x2v_ser_p1(const double *b_i, const double *moms, double* u_par); 

GKYL_CU_DH void gyrokinetic_prim_vars_1x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_1x3v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_1x3v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_1x3v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_1x3v_ser_p1(const double *b_i, const double *moms, double* u_par); 

GKYL_CU_DH void gyrokinetic_prim_vars_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_1x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_1x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_1x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_1x2v_ser_p1(const double *b_i, const double *moms, double* u_par); 

GKYL_CU_DH void gyrokinetic_prim_vars_2x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_2x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_2x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_2x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_2x2v_ser_p1(const double *b_i, const double *moms, double* u_par); 

GKYL_CU_DH void gyrokinetic_prim_vars_3x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_upar_3x2v_ser_p1(const double *moms, double* prim_vars); 
GKYL_CU_DH void gyrokinetic_prim_vars_vtSq_3x2v_ser_p1(const double *moms, double* prim_vars); 

GKYL_CU_DH void transform_prim_vars_gk_3x2v_ser_p1(const double *b_i, const double *moms, double* prim_vars); 
GKYL_CU_DH void transform_u_par_3x2v_ser_p1(const double *b_i, const double *moms, double* u_par); 

EXTERN_C_END 
