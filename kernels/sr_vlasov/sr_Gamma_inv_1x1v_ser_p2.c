#include <gkyl_sr_Gamma_kernels.h> 
#include <gkyl_basis_ser_1x_p2_exp_sq.h> 
#include <gkyl_basis_ser_1x_p2_sqrt.h> 
GKYL_CU_DH void sr_Gamma_inv_1x1v_ser_p2(const double *V, double* GKYL_RESTRICT Gamma_inv) 
{ 
  // V:     Input velocity. 
  // Gamma: Gamma = 1/sqrt(1 - V^2/c^2). 
 
  const double *V_0 = &V[0]; 
  double V_0_sq[3] = {0.0}; 
  ser_1x_p2_exp_sq(V_0, V_0_sq); 
 
  double V_sq_avg = (V_0_sq[0])/(1.414213562373095); 
 
  double Gamma2_inv[3] = {0.0}; 
  Gamma2_inv[0] = 1.414213562373095-1.0*V_0_sq[0]; 
  Gamma2_inv[1] = -1.0*V_0_sq[1]; 
  Gamma2_inv[2] = -1.0*V_0_sq[2]; 

  // Check if cell average of Gamma^{-2} = 1 - V^2/c^2 < 0. 
  if (V_sq_avg > 1.0) { 
    Gamma2_inv[0] = 1.414213562373095 - V_sq_avg; 
    Gamma2_inv[1] = 0.0; 
    Gamma2_inv[2] = 0.0; 
  } 
 
  int cell_avg = 0;
  // Check if Gamma^{-2} = 1 - V^2/c^2 < 0 at control points. 
  if (1.58113883008419*Gamma2_inv[2]-1.224744871391589*Gamma2_inv[1]+0.7071067811865475*Gamma2_inv[0] < 0.0) cell_avg = 1; 
  if (0.7071067811865475*Gamma2_inv[0]-0.7905694150420947*Gamma2_inv[2] < 0.0) cell_avg = 1; 
  if (1.58113883008419*Gamma2_inv[2]+1.224744871391589*Gamma2_inv[1]+0.7071067811865475*Gamma2_inv[0] < 0.0) cell_avg = 1; 
 
  if (cell_avg) { 
    Gamma2_inv[1] = 0.0; 
    Gamma2_inv[2] = 0.0; 
    ser_1x_p2_sqrt(Gamma2_inv, Gamma_inv); 
  } else { 
    ser_1x_p2_sqrt(Gamma2_inv, Gamma_inv); 
  } 
} 
 
