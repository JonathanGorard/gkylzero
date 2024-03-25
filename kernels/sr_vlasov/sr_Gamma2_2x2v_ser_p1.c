#include <gkyl_sr_Gamma_kernels.h> 
#include <gkyl_basis_ser_2x_p1_exp_sq.h> 
#include <gkyl_basis_ser_2x_p1_inv.h> 
GKYL_CU_DH void sr_Gamma2_2x2v_ser_p1(const double *V, double* GKYL_RESTRICT Gamma2) 
{ 
  // V:      Input velocity. 
  // Gamma2: Gamma^2 = 1/(1 - V^2/c^2). 
 
  const double *V_0 = &V[0]; 
  double V_0_sq[4] = {0.0}; 
  ser_2x_p1_exp_sq(V_0, V_0_sq); 
 
  const double *V_1 = &V[4]; 
  double V_1_sq[4] = {0.0}; 
  ser_2x_p1_exp_sq(V_1, V_1_sq); 
 
  double Gamma2_inv[4] = {0.0}; 
 
  Gamma2_inv[0] = (-1.0*V_1_sq[0])-1.0*V_0_sq[0]+2.0; 
  Gamma2_inv[1] = (-1.0*V_1_sq[1])-1.0*V_0_sq[1]; 
  Gamma2_inv[2] = (-1.0*V_1_sq[2])-1.0*V_0_sq[2]; 
  Gamma2_inv[3] = (-1.0*V_1_sq[3])-1.0*V_0_sq[3]; 

  int cell_avg = 0;
  // Check if Gamma^{-2} = 1 - V^2/c^2 < 0 at control points. 
  if (1.5*Gamma2_inv[3]-0.8660254037844386*Gamma2_inv[2]-0.8660254037844386*Gamma2_inv[1]+0.5*Gamma2_inv[0] < 0.0) cell_avg = 1; 
  if ((-1.5*Gamma2_inv[3])-0.8660254037844386*Gamma2_inv[2]+0.8660254037844386*Gamma2_inv[1]+0.5*Gamma2_inv[0] < 0.0) cell_avg = 1; 
  if ((-1.5*Gamma2_inv[3])+0.8660254037844386*Gamma2_inv[2]-0.8660254037844386*Gamma2_inv[1]+0.5*Gamma2_inv[0] < 0.0) cell_avg = 1; 
  if (1.5*Gamma2_inv[3]+0.8660254037844386*Gamma2_inv[2]+0.8660254037844386*Gamma2_inv[1]+0.5*Gamma2_inv[0] < 0.0) cell_avg = 1; 
 
  if (cell_avg) { 
    Gamma2[0] = 4.0/Gamma2_inv[0]; 
    Gamma2[1] = 0.0; 
    Gamma2[2] = 0.0; 
    Gamma2[3] = 0.0; 
  } else { 
    ser_2x_p1_inv(Gamma2_inv, Gamma2); 
  } 
} 
 
