#include <gkyl_prim_lbo_vlasov_kernels.h> 
 
GKYL_CU_DH void vlasov_pkpm_self_prim_moments_1x1v_ser_p2(struct gkyl_mat *A, struct gkyl_mat *rhs, const double *vlasov_pkpm_moms, const double *boundary_corrections) 
{ 
  // A:                    Matrix to be inverted to solve Ax = rhs (set by this function). 
  // rhs:                  right-hand side of Ax = rhs (set by this function). 
  // vlasov_pkpm_moms:     [rho, p_parallel, p_perp], Moments computed from kinetic equation in pkpm model. 
  // boundary_corrections: boundary corrections to vtSq. 
 
  const double *rho = &vlasov_pkpm_moms[0]; 
  const double *p_parallel = &vlasov_pkpm_moms[3]; 
  const double *p_perp = &vlasov_pkpm_moms[6]; 
  // If a corner value is below zero, use cell average.
  bool cellAvg = false;
  if (0.7071067811865475*(2.23606797749979*rho[2]-1.732050807568877*rho[1]+rho[0]) < 0) cellAvg = true; 
  if (0.7071067811865475*(2.23606797749979*p_parallel[2]-1.732050807568877*p_parallel[1]+p_parallel[0]) < 0) cellAvg = true; 
  if (0.7071067811865475*(2.23606797749979*p_perp[2]-1.732050807568877*p_perp[1]+p_perp[0]) < 0) cellAvg = true; 
  if (0.7071067811865475*(2.23606797749979*rho[2]+1.732050807568877*rho[1]+rho[0]) < 0) cellAvg = true; 
  if (0.7071067811865475*(2.23606797749979*p_parallel[2]+1.732050807568877*p_parallel[1]+p_parallel[0]) < 0) cellAvg = true; 
  if (0.7071067811865475*(2.23606797749979*p_perp[2]+1.732050807568877*p_perp[1]+p_perp[0]) < 0) cellAvg = true; 
 
  double m0r[3] = {0.0}; 
  double cEr[3] = {0.0}; 
  if (cellAvg) { 
    m0r[0] = rho[0]; 
    m0r[1] = 0.0; 
    m0r[2] = 0.0; 
    cEr[0] = boundary_corrections[0]; 
    cEr[1] = 0.0; 
    cEr[2] = 0.0; 
    gkyl_mat_set(rhs,0,0,2.0*p_perp[0]+p_parallel[0]); 
    gkyl_mat_set(rhs,1,0,0.0); 
    gkyl_mat_set(rhs,2,0,0.0); 
  } else { 
    m0r[0] = rho[0]; 
    m0r[1] = rho[1]; 
    m0r[2] = rho[2]; 
    cEr[0] = boundary_corrections[0]; 
    cEr[1] = boundary_corrections[1]; 
    cEr[2] = boundary_corrections[2]; 
    gkyl_mat_set(rhs,0,0,2.0*p_perp[0]+p_parallel[0]); 
    gkyl_mat_set(rhs,1,0,2.0*p_perp[1]+p_parallel[1]); 
    gkyl_mat_set(rhs,2,0,2.0*p_perp[2]+p_parallel[2]); 
  } 
 
  // ....... Block from correction to vtSq .......... // 
  gkyl_mat_set(A,0,0,2.121320343559642*m0r[0]-0.7071067811865475*cEr[0]); 
  gkyl_mat_set(A,0,1,2.121320343559642*m0r[1]-0.7071067811865475*cEr[1]); 
  gkyl_mat_set(A,0,2,2.121320343559642*m0r[2]-0.7071067811865475*cEr[2]); 
  gkyl_mat_set(A,1,0,2.121320343559642*m0r[1]-0.7071067811865475*cEr[1]); 
  gkyl_mat_set(A,1,1,1.897366596101028*m0r[2]-0.6324555320336759*cEr[2]+2.121320343559642*m0r[0]-0.7071067811865475*cEr[0]); 
  gkyl_mat_set(A,1,2,1.897366596101028*m0r[1]-0.6324555320336759*cEr[1]); 
  gkyl_mat_set(A,2,0,2.121320343559642*m0r[2]-0.7071067811865475*cEr[2]); 
  gkyl_mat_set(A,2,1,1.897366596101028*m0r[1]-0.6324555320336759*cEr[1]); 
  gkyl_mat_set(A,2,2,1.355261854357877*m0r[2]-0.4517539514526256*cEr[2]+2.121320343559642*m0r[0]-0.7071067811865475*cEr[0]); 
 
} 
 
