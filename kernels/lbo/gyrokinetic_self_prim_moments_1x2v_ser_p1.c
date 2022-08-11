#include <gkyl_prim_lbo_gyrokinetic_kernels.h> 
 
GKYL_CU_DH void gyrokinetic_self_prim_moments_1x2v_ser_p1(struct gkyl_mat *A, struct gkyl_mat *rhs, const double *moms, const double *boundary_corrections) 
{ 
  // A:                    Matrix to be inverted to solve Ax = rhs (set by this function). 
  // rhs:                  right-hand side of Ax = rhs (set by this function). 
  // moms:                 moments of the distribution function (Zeroth, First, and Second in single array). 
  // boundary_corrections: boundary corrections to u and vtSq. 
 
  // If a corner value is below zero, use cell average m0.
  bool cellAvg = false;
  if (-0.5*(2.449489742783178*moms[1]-1.414213562373095*moms[0]) < 0) cellAvg = true; 
  if (0.5*(2.449489742783178*moms[1]+1.414213562373095*moms[0]) < 0) cellAvg = true; 
 
  double m0r[2] = {0.0}; 
  double m1r[2] = {0.0}; 
  double cMr[2] = {0.0}; 
  double cEr[2] = {0.0}; 
  if (cellAvg) { 
    m0r[0] = moms[0]; 
    m0r[1] = 0.0; 
    m1r[0] = moms[2]; 
    m1r[1] = 0.0; 
    gkyl_mat_set(rhs,0,0,moms[2]); 
    gkyl_mat_set(rhs,1,0,0.0); 
    cMr[0] = boundary_corrections[0]; 
    cMr[1] = 0.0; 
    cEr[0] = boundary_corrections[2]; 
    cEr[1] = 0.0; 
    gkyl_mat_set(rhs,2,0,moms[4]); 
    gkyl_mat_set(rhs,3,0,0.0); 
  } else { 
    m0r[0] = moms[0]; 
    m0r[1] = moms[1]; 
    m1r[0] = moms[2]; 
    m1r[1] = moms[3]; 
    gkyl_mat_set(rhs,0,0,moms[2]); 
    gkyl_mat_set(rhs,1,0,moms[3]); 
    cMr[0] = boundary_corrections[0]; 
    cMr[1] = boundary_corrections[1]; 
    cEr[0] = boundary_corrections[2]; 
    cEr[1] = boundary_corrections[3]; 
    gkyl_mat_set(rhs,2,0,moms[4]); 
    gkyl_mat_set(rhs,3,0,moms[5]); 
  } 
 
  // ....... Block from weak multiply of ux and m0  .......... // 
  gkyl_mat_set(A,0,0,0.7071067811865475*m0r[0]); 
  gkyl_mat_set(A,0,1,0.7071067811865475*m0r[1]); 
  gkyl_mat_set(A,1,0,0.7071067811865475*m0r[1]); 
  gkyl_mat_set(A,1,1,0.7071067811865475*m0r[0]); 
 
  // ....... Block from correction to ux .......... // 
  gkyl_mat_set(A,0,2,-0.7071067811865475*cMr[0]); 
  gkyl_mat_set(A,0,3,-0.7071067811865475*cMr[1]); 
  gkyl_mat_set(A,1,2,-0.7071067811865475*cMr[1]); 
  gkyl_mat_set(A,1,3,-0.7071067811865475*cMr[0]); 
 
  // ....... Block from weak multiply of ux and m1x  .......... // 
  gkyl_mat_set(A,2,0,0.7071067811865475*m1r[0]); 
  gkyl_mat_set(A,2,1,0.7071067811865475*m1r[1]); 
  gkyl_mat_set(A,3,0,0.7071067811865475*m1r[1]); 
  gkyl_mat_set(A,3,1,0.7071067811865475*m1r[0]); 
 
  // ....... Block from correction to vtSq .......... // 
  gkyl_mat_set(A,2,2,2.121320343559642*m0r[0]-0.7071067811865475*cEr[0]); 
  gkyl_mat_set(A,2,3,2.121320343559642*m0r[1]-0.7071067811865475*cEr[1]); 
  gkyl_mat_set(A,3,2,2.121320343559642*m0r[1]-0.7071067811865475*cEr[1]); 
  gkyl_mat_set(A,3,3,2.121320343559642*m0r[0]-0.7071067811865475*cEr[0]); 
 
} 
 