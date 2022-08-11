#include <gkyl_prim_lbo_vlasov_kernels.h> 
 
GKYL_CU_DH void vlasov_cross_prim_moments_2x2v_ser_p1(struct gkyl_mat *A, struct gkyl_mat *rhs, const double *greene, const double m_self, const double *u_self, const double *vtsq_self, const double m_other, const double *u_other, const double *vtsq_other, const double *moms, const double *boundary_corrections) 
{ 
  // greene:               Greene's factor. 
  // m_:                   mass. 
  // moms:                 moments of the distribution function. 
  // u,vtSq:               self primitive moments: mean flow velocity and thermal speed squared. 
  // boundary_corrections: corrections to momentum and energy conservation due to finite velocity space. 
  // uCross,vtSqCross:     cross primitive moments: mean flow velocity and thermal speed squared. 
 
  // If a corner value is below zero, use cell average m0.
  bool cellAvg = false;
  if (0.5*(3.0*moms[3]-1.732050807568877*(moms[2]+moms[1])+moms[0]) < 0) cellAvg = true; 
  if (-0.5*(3.0*moms[3]+1.732050807568877*moms[2]-1.732050807568877*moms[1]-1.0*moms[0]) < 0) cellAvg = true; 
  if (-0.5*(3.0*moms[3]-1.732050807568877*moms[2]+1.732050807568877*moms[1]-1.0*moms[0]) < 0) cellAvg = true; 
  if (0.5*(3.0*moms[3]+1.732050807568877*(moms[2]+moms[1])+moms[0]) < 0) cellAvg = true; 
 
  double m0r[4] = {0.0}; 
  double m1r[8] = {0.0}; 
  double m2r[4] = {0.0}; 
  double cMr[8] = {0.0}; 
  double cEr[4] = {0.0}; 
  if (cellAvg) { 
    m0r[0] = moms[0]; 
    m0r[1] = 0.0; 
    m0r[2] = 0.0; 
    m0r[3] = 0.0; 
    m1r[0] = moms[4]; 
    m1r[1] = 0.0; 
    m1r[2] = 0.0; 
    m1r[3] = 0.0; 
    cMr[0] = boundary_corrections[0]; 
    cMr[1] = 0.0; 
    cMr[2] = 0.0; 
    cMr[3] = 0.0; 
    m1r[4] = moms[8]; 
    m1r[5] = 0.0; 
    m1r[6] = 0.0; 
    m1r[7] = 0.0; 
    cMr[4] = boundary_corrections[4]; 
    cMr[5] = 0.0; 
    cMr[6] = 0.0; 
    cMr[7] = 0.0; 
    m2r[0] = moms[12]; 
    m2r[1] = 0.0; 
    m2r[2] = 0.0; 
    m2r[3] = 0.0; 
    cEr[0] = boundary_corrections[8]; 
    cEr[1] = 0.0; 
    cEr[2] = 0.0; 
    cEr[3] = 0.0; 
  } else { 
    m0r[0] = moms[0]; 
    m0r[1] = moms[1]; 
    m0r[2] = moms[2]; 
    m0r[3] = moms[3]; 
    m1r[0] = moms[4]; 
    m1r[1] = moms[5]; 
    m1r[2] = moms[6]; 
    m1r[3] = moms[7]; 
    m1r[4] = moms[8]; 
    m1r[5] = moms[9]; 
    m1r[6] = moms[10]; 
    m1r[7] = moms[11]; 
    m2r[0] = moms[12]; 
    m2r[1] = moms[13]; 
    m2r[2] = moms[14]; 
    m2r[3] = moms[15]; 
    cMr[0] = boundary_corrections[0]; 
    cMr[1] = boundary_corrections[1]; 
    cMr[2] = boundary_corrections[2]; 
    cMr[3] = boundary_corrections[3]; 
    cMr[4] = boundary_corrections[4]; 
    cMr[5] = boundary_corrections[5]; 
    cMr[6] = boundary_corrections[6]; 
    cMr[7] = boundary_corrections[7]; 
    cEr[0] = boundary_corrections[8]; 
    cEr[1] = boundary_corrections[9]; 
    cEr[2] = boundary_corrections[10]; 
    cEr[3] = boundary_corrections[11]; 
  } 
 
  double momRHS[8] = {0.0}; 
 
  // ... Block from weak multiply of m_self, nu, M0 and uCrossX ... // 
  gkyl_mat_set(A,0,0,m0r[0]); 
  gkyl_mat_set(A,0,1,m0r[1]); 
  gkyl_mat_set(A,0,2,m0r[2]); 
  gkyl_mat_set(A,0,3,m0r[3]); 
  gkyl_mat_set(A,1,0,m0r[1]); 
  gkyl_mat_set(A,1,1,m0r[0]); 
  gkyl_mat_set(A,1,2,m0r[3]); 
  gkyl_mat_set(A,1,3,m0r[2]); 
  gkyl_mat_set(A,2,0,m0r[2]); 
  gkyl_mat_set(A,2,1,m0r[3]); 
  gkyl_mat_set(A,2,2,m0r[0]); 
  gkyl_mat_set(A,2,3,m0r[1]); 
  gkyl_mat_set(A,3,0,m0r[3]); 
  gkyl_mat_set(A,3,1,m0r[2]); 
  gkyl_mat_set(A,3,2,m0r[1]); 
  gkyl_mat_set(A,3,3,m0r[0]); 
 
  // ... Block from correction to momentum conservation (self) ... // 
  gkyl_mat_set(A,0,8,-1.0*cMr[0]); 
  gkyl_mat_set(A,0,9,-1.0*cMr[1]); 
  gkyl_mat_set(A,0,10,-1.0*cMr[2]); 
  gkyl_mat_set(A,0,11,-1.0*cMr[3]); 
  gkyl_mat_set(A,1,8,-1.0*cMr[1]); 
  gkyl_mat_set(A,1,9,-1.0*cMr[0]); 
  gkyl_mat_set(A,1,10,-1.0*cMr[3]); 
  gkyl_mat_set(A,1,11,-1.0*cMr[2]); 
  gkyl_mat_set(A,2,8,-1.0*cMr[2]); 
  gkyl_mat_set(A,2,9,-1.0*cMr[3]); 
  gkyl_mat_set(A,2,10,-1.0*cMr[0]); 
  gkyl_mat_set(A,2,11,-1.0*cMr[1]); 
  gkyl_mat_set(A,3,8,-1.0*cMr[3]); 
  gkyl_mat_set(A,3,9,-1.0*cMr[2]); 
  gkyl_mat_set(A,3,10,-1.0*cMr[1]); 
  gkyl_mat_set(A,3,11,-1.0*cMr[0]); 
 
  // ... Block from weak multiply of m_self, nu, m1X and uCrossX ... // 
  gkyl_mat_set(A,8,0,(-0.25*m0r[3]*u_self[3])-0.25*m0r[3]*u_other[3]-0.25*m0r[2]*u_self[2]-0.25*m0r[2]*u_other[2]-0.25*m0r[1]*u_self[1]-0.25*m0r[1]*u_other[1]-0.25*m0r[0]*u_self[0]-0.25*m0r[0]*u_other[0]+m1r[0]); 
  gkyl_mat_set(A,8,1,(-0.25*m0r[2]*u_self[3])-0.25*m0r[2]*u_other[3]-0.25*u_self[2]*m0r[3]-0.25*u_other[2]*m0r[3]-0.25*m0r[0]*u_self[1]-0.25*m0r[0]*u_other[1]+m1r[1]-0.25*u_self[0]*m0r[1]-0.25*u_other[0]*m0r[1]); 
  gkyl_mat_set(A,8,2,(-0.25*m0r[1]*u_self[3])-0.25*m0r[1]*u_other[3]-0.25*u_self[1]*m0r[3]-0.25*u_other[1]*m0r[3]-0.25*m0r[0]*u_self[2]-0.25*m0r[0]*u_other[2]+m1r[2]-0.25*u_self[0]*m0r[2]-0.25*u_other[0]*m0r[2]); 
  gkyl_mat_set(A,8,3,(-0.25*m0r[0]*u_self[3])-0.25*m0r[0]*u_other[3]+m1r[3]-0.25*u_self[0]*m0r[3]-0.25*u_other[0]*m0r[3]-0.25*m0r[1]*u_self[2]-0.25*m0r[1]*u_other[2]-0.25*u_self[1]*m0r[2]-0.25*u_other[1]*m0r[2]); 
  gkyl_mat_set(A,9,0,(-0.25*m0r[2]*u_self[3])-0.25*m0r[2]*u_other[3]-0.25*u_self[2]*m0r[3]-0.25*u_other[2]*m0r[3]-0.25*m0r[0]*u_self[1]-0.25*m0r[0]*u_other[1]+m1r[1]-0.25*u_self[0]*m0r[1]-0.25*u_other[0]*m0r[1]); 
  gkyl_mat_set(A,9,1,(-0.45*m0r[3]*u_self[3])-0.45*m0r[3]*u_other[3]-0.25*m0r[2]*u_self[2]-0.25*m0r[2]*u_other[2]-0.45*m0r[1]*u_self[1]-0.45*m0r[1]*u_other[1]-0.25*m0r[0]*u_self[0]-0.25*m0r[0]*u_other[0]+m1r[0]); 
  gkyl_mat_set(A,9,2,(-0.25*m0r[0]*u_self[3])-0.25*m0r[0]*u_other[3]+m1r[3]-0.25*u_self[0]*m0r[3]-0.25*u_other[0]*m0r[3]-0.25*m0r[1]*u_self[2]-0.25*m0r[1]*u_other[2]-0.25*u_self[1]*m0r[2]-0.25*u_other[1]*m0r[2]); 
  gkyl_mat_set(A,9,3,(-0.45*m0r[1]*u_self[3])-0.45*m0r[1]*u_other[3]-0.45*u_self[1]*m0r[3]-0.45*u_other[1]*m0r[3]-0.25*m0r[0]*u_self[2]-0.25*m0r[0]*u_other[2]+m1r[2]-0.25*u_self[0]*m0r[2]-0.25*u_other[0]*m0r[2]); 
  gkyl_mat_set(A,10,0,(-0.25*m0r[1]*u_self[3])-0.25*m0r[1]*u_other[3]-0.25*u_self[1]*m0r[3]-0.25*u_other[1]*m0r[3]-0.25*m0r[0]*u_self[2]-0.25*m0r[0]*u_other[2]+m1r[2]-0.25*u_self[0]*m0r[2]-0.25*u_other[0]*m0r[2]); 
  gkyl_mat_set(A,10,1,(-0.25*m0r[0]*u_self[3])-0.25*m0r[0]*u_other[3]+m1r[3]-0.25*u_self[0]*m0r[3]-0.25*u_other[0]*m0r[3]-0.25*m0r[1]*u_self[2]-0.25*m0r[1]*u_other[2]-0.25*u_self[1]*m0r[2]-0.25*u_other[1]*m0r[2]); 
  gkyl_mat_set(A,10,2,(-0.45*m0r[3]*u_self[3])-0.45*m0r[3]*u_other[3]-0.45*m0r[2]*u_self[2]-0.45*m0r[2]*u_other[2]-0.25*m0r[1]*u_self[1]-0.25*m0r[1]*u_other[1]-0.25*m0r[0]*u_self[0]-0.25*m0r[0]*u_other[0]+m1r[0]); 
  gkyl_mat_set(A,10,3,(-0.45*m0r[2]*u_self[3])-0.45*m0r[2]*u_other[3]-0.45*u_self[2]*m0r[3]-0.45*u_other[2]*m0r[3]-0.25*m0r[0]*u_self[1]-0.25*m0r[0]*u_other[1]+m1r[1]-0.25*u_self[0]*m0r[1]-0.25*u_other[0]*m0r[1]); 
  gkyl_mat_set(A,11,0,(-0.25*m0r[0]*u_self[3])-0.25*m0r[0]*u_other[3]+m1r[3]-0.25*u_self[0]*m0r[3]-0.25*u_other[0]*m0r[3]-0.25*m0r[1]*u_self[2]-0.25*m0r[1]*u_other[2]-0.25*u_self[1]*m0r[2]-0.25*u_other[1]*m0r[2]); 
  gkyl_mat_set(A,11,1,(-0.45*m0r[1]*u_self[3])-0.45*m0r[1]*u_other[3]-0.45*u_self[1]*m0r[3]-0.45*u_other[1]*m0r[3]-0.25*m0r[0]*u_self[2]-0.25*m0r[0]*u_other[2]+m1r[2]-0.25*u_self[0]*m0r[2]-0.25*u_other[0]*m0r[2]); 
  gkyl_mat_set(A,11,2,(-0.45*m0r[2]*u_self[3])-0.45*m0r[2]*u_other[3]-0.45*u_self[2]*m0r[3]-0.45*u_other[2]*m0r[3]-0.25*m0r[0]*u_self[1]-0.25*m0r[0]*u_other[1]+m1r[1]-0.25*u_self[0]*m0r[1]-0.25*u_other[0]*m0r[1]); 
  gkyl_mat_set(A,11,3,(-0.81*m0r[3]*u_self[3])-0.81*m0r[3]*u_other[3]-0.45*m0r[2]*u_self[2]-0.45*m0r[2]*u_other[2]-0.45*m0r[1]*u_self[1]-0.45*m0r[1]*u_other[1]-0.25*m0r[0]*u_self[0]-0.25*m0r[0]*u_other[0]+m1r[0]); 
 
  momRHS[0] += -0.5*(greene[3]*u_self[3]-1.0*greene[3]*u_other[3]+greene[2]*u_self[2]-1.0*greene[2]*u_other[2]+greene[1]*u_self[1]-1.0*greene[1]*u_other[1]+greene[0]*u_self[0]-1.0*greene[0]*u_other[0]-4.0*m1r[0]); 
  momRHS[1] += -0.5*(greene[2]*u_self[3]-1.0*greene[2]*u_other[3]+(u_self[2]-1.0*u_other[2])*greene[3]+greene[0]*u_self[1]-1.0*greene[0]*u_other[1]-4.0*m1r[1]+(u_self[0]-1.0*u_other[0])*greene[1]); 
  momRHS[2] += -0.5*(greene[1]*u_self[3]-1.0*greene[1]*u_other[3]+(u_self[1]-1.0*u_other[1])*greene[3]+greene[0]*u_self[2]-1.0*greene[0]*u_other[2]-4.0*m1r[2]+(u_self[0]-1.0*u_other[0])*greene[2]); 
  momRHS[3] += -0.5*(greene[0]*u_self[3]-1.0*greene[0]*u_other[3]-4.0*m1r[3]+(u_self[0]-1.0*u_other[0])*greene[3]+greene[1]*u_self[2]-1.0*greene[1]*u_other[2]+(u_self[1]-1.0*u_other[1])*greene[2]); 
 
  // ... Block from weak multiply of m_self, nu, M0 and uCrossY ... // 
  gkyl_mat_set(A,4,4,m0r[0]); 
  gkyl_mat_set(A,4,5,m0r[1]); 
  gkyl_mat_set(A,4,6,m0r[2]); 
  gkyl_mat_set(A,4,7,m0r[3]); 
  gkyl_mat_set(A,5,4,m0r[1]); 
  gkyl_mat_set(A,5,5,m0r[0]); 
  gkyl_mat_set(A,5,6,m0r[3]); 
  gkyl_mat_set(A,5,7,m0r[2]); 
  gkyl_mat_set(A,6,4,m0r[2]); 
  gkyl_mat_set(A,6,5,m0r[3]); 
  gkyl_mat_set(A,6,6,m0r[0]); 
  gkyl_mat_set(A,6,7,m0r[1]); 
  gkyl_mat_set(A,7,4,m0r[3]); 
  gkyl_mat_set(A,7,5,m0r[2]); 
  gkyl_mat_set(A,7,6,m0r[1]); 
  gkyl_mat_set(A,7,7,m0r[0]); 
 
  // ... Block from correction to momentum conservation (self) ... // 
  gkyl_mat_set(A,4,8,-1.0*cMr[4]); 
  gkyl_mat_set(A,4,9,-1.0*cMr[5]); 
  gkyl_mat_set(A,4,10,-1.0*cMr[6]); 
  gkyl_mat_set(A,4,11,-1.0*cMr[7]); 
  gkyl_mat_set(A,5,8,-1.0*cMr[5]); 
  gkyl_mat_set(A,5,9,-1.0*cMr[4]); 
  gkyl_mat_set(A,5,10,-1.0*cMr[7]); 
  gkyl_mat_set(A,5,11,-1.0*cMr[6]); 
  gkyl_mat_set(A,6,8,-1.0*cMr[6]); 
  gkyl_mat_set(A,6,9,-1.0*cMr[7]); 
  gkyl_mat_set(A,6,10,-1.0*cMr[4]); 
  gkyl_mat_set(A,6,11,-1.0*cMr[5]); 
  gkyl_mat_set(A,7,8,-1.0*cMr[7]); 
  gkyl_mat_set(A,7,9,-1.0*cMr[6]); 
  gkyl_mat_set(A,7,10,-1.0*cMr[5]); 
  gkyl_mat_set(A,7,11,-1.0*cMr[4]); 
 
  // ... Block from weak multiply of m_self, nu, m1Y and uCrossY ... // 
  gkyl_mat_set(A,8,4,(-0.25*m0r[3]*u_self[7])-0.25*m0r[3]*u_other[7]-0.25*m0r[2]*u_self[6]-0.25*m0r[2]*u_other[6]-0.25*m0r[1]*u_self[5]-0.25*m0r[1]*u_other[5]-0.25*m0r[0]*u_self[4]-0.25*m0r[0]*u_other[4]+m1r[4]); 
  gkyl_mat_set(A,8,5,(-0.25*m0r[2]*u_self[7])-0.25*m0r[2]*u_other[7]-0.25*m0r[3]*u_self[6]-0.25*m0r[3]*u_other[6]-0.25*m0r[0]*u_self[5]-0.25*m0r[0]*u_other[5]+m1r[5]-0.25*m0r[1]*u_self[4]-0.25*m0r[1]*u_other[4]); 
  gkyl_mat_set(A,8,6,(-0.25*m0r[1]*u_self[7])-0.25*m0r[1]*u_other[7]-0.25*m0r[0]*u_self[6]-0.25*m0r[0]*u_other[6]+m1r[6]-0.25*m0r[3]*u_self[5]-0.25*m0r[3]*u_other[5]-0.25*m0r[2]*u_self[4]-0.25*m0r[2]*u_other[4]); 
  gkyl_mat_set(A,8,7,(-0.25*m0r[0]*u_self[7])-0.25*m0r[0]*u_other[7]+m1r[7]-0.25*m0r[1]*u_self[6]-0.25*m0r[1]*u_other[6]-0.25*m0r[2]*u_self[5]-0.25*m0r[2]*u_other[5]-0.25*m0r[3]*u_self[4]-0.25*m0r[3]*u_other[4]); 
  gkyl_mat_set(A,9,4,(-0.25*m0r[2]*u_self[7])-0.25*m0r[2]*u_other[7]-0.25*m0r[3]*u_self[6]-0.25*m0r[3]*u_other[6]-0.25*m0r[0]*u_self[5]-0.25*m0r[0]*u_other[5]+m1r[5]-0.25*m0r[1]*u_self[4]-0.25*m0r[1]*u_other[4]); 
  gkyl_mat_set(A,9,5,(-0.45*m0r[3]*u_self[7])-0.45*m0r[3]*u_other[7]-0.25*m0r[2]*u_self[6]-0.25*m0r[2]*u_other[6]-0.45*m0r[1]*u_self[5]-0.45*m0r[1]*u_other[5]-0.25*m0r[0]*u_self[4]-0.25*m0r[0]*u_other[4]+m1r[4]); 
  gkyl_mat_set(A,9,6,(-0.25*m0r[0]*u_self[7])-0.25*m0r[0]*u_other[7]+m1r[7]-0.25*m0r[1]*u_self[6]-0.25*m0r[1]*u_other[6]-0.25*m0r[2]*u_self[5]-0.25*m0r[2]*u_other[5]-0.25*m0r[3]*u_self[4]-0.25*m0r[3]*u_other[4]); 
  gkyl_mat_set(A,9,7,(-0.45*m0r[1]*u_self[7])-0.45*m0r[1]*u_other[7]-0.25*m0r[0]*u_self[6]-0.25*m0r[0]*u_other[6]+m1r[6]-0.45*m0r[3]*u_self[5]-0.45*m0r[3]*u_other[5]-0.25*m0r[2]*u_self[4]-0.25*m0r[2]*u_other[4]); 
  gkyl_mat_set(A,10,4,(-0.25*m0r[1]*u_self[7])-0.25*m0r[1]*u_other[7]-0.25*m0r[0]*u_self[6]-0.25*m0r[0]*u_other[6]+m1r[6]-0.25*m0r[3]*u_self[5]-0.25*m0r[3]*u_other[5]-0.25*m0r[2]*u_self[4]-0.25*m0r[2]*u_other[4]); 
  gkyl_mat_set(A,10,5,(-0.25*m0r[0]*u_self[7])-0.25*m0r[0]*u_other[7]+m1r[7]-0.25*m0r[1]*u_self[6]-0.25*m0r[1]*u_other[6]-0.25*m0r[2]*u_self[5]-0.25*m0r[2]*u_other[5]-0.25*m0r[3]*u_self[4]-0.25*m0r[3]*u_other[4]); 
  gkyl_mat_set(A,10,6,(-0.45*m0r[3]*u_self[7])-0.45*m0r[3]*u_other[7]-0.45*m0r[2]*u_self[6]-0.45*m0r[2]*u_other[6]-0.25*m0r[1]*u_self[5]-0.25*m0r[1]*u_other[5]-0.25*m0r[0]*u_self[4]-0.25*m0r[0]*u_other[4]+m1r[4]); 
  gkyl_mat_set(A,10,7,(-0.45*m0r[2]*u_self[7])-0.45*m0r[2]*u_other[7]-0.45*m0r[3]*u_self[6]-0.45*m0r[3]*u_other[6]-0.25*m0r[0]*u_self[5]-0.25*m0r[0]*u_other[5]+m1r[5]-0.25*m0r[1]*u_self[4]-0.25*m0r[1]*u_other[4]); 
  gkyl_mat_set(A,11,4,(-0.25*m0r[0]*u_self[7])-0.25*m0r[0]*u_other[7]+m1r[7]-0.25*m0r[1]*u_self[6]-0.25*m0r[1]*u_other[6]-0.25*m0r[2]*u_self[5]-0.25*m0r[2]*u_other[5]-0.25*m0r[3]*u_self[4]-0.25*m0r[3]*u_other[4]); 
  gkyl_mat_set(A,11,5,(-0.45*m0r[1]*u_self[7])-0.45*m0r[1]*u_other[7]-0.25*m0r[0]*u_self[6]-0.25*m0r[0]*u_other[6]+m1r[6]-0.45*m0r[3]*u_self[5]-0.45*m0r[3]*u_other[5]-0.25*m0r[2]*u_self[4]-0.25*m0r[2]*u_other[4]); 
  gkyl_mat_set(A,11,6,(-0.45*m0r[2]*u_self[7])-0.45*m0r[2]*u_other[7]-0.45*m0r[3]*u_self[6]-0.45*m0r[3]*u_other[6]-0.25*m0r[0]*u_self[5]-0.25*m0r[0]*u_other[5]+m1r[5]-0.25*m0r[1]*u_self[4]-0.25*m0r[1]*u_other[4]); 
  gkyl_mat_set(A,11,7,(-0.81*m0r[3]*u_self[7])-0.81*m0r[3]*u_other[7]-0.45*m0r[2]*u_self[6]-0.45*m0r[2]*u_other[6]-0.45*m0r[1]*u_self[5]-0.45*m0r[1]*u_other[5]-0.25*m0r[0]*u_self[4]-0.25*m0r[0]*u_other[4]+m1r[4]); 
 
  momRHS[4] += -0.5*(greene[3]*u_self[7]-1.0*greene[3]*u_other[7]+greene[2]*u_self[6]-1.0*greene[2]*u_other[6]+greene[1]*u_self[5]-1.0*greene[1]*u_other[5]+greene[0]*u_self[4]-1.0*greene[0]*u_other[4]-4.0*m1r[4]); 
  momRHS[5] += -0.5*(greene[2]*u_self[7]-1.0*greene[2]*u_other[7]+greene[3]*u_self[6]-1.0*greene[3]*u_other[6]+greene[0]*u_self[5]-1.0*greene[0]*u_other[5]-4.0*m1r[5]+greene[1]*u_self[4]-1.0*greene[1]*u_other[4]); 
  momRHS[6] += -0.5*(greene[1]*u_self[7]-1.0*greene[1]*u_other[7]+greene[0]*u_self[6]-1.0*greene[0]*u_other[6]-4.0*m1r[6]+greene[3]*u_self[5]-1.0*greene[3]*u_other[5]+greene[2]*u_self[4]-1.0*greene[2]*u_other[4]); 
  momRHS[7] += -0.5*(greene[0]*u_self[7]-1.0*greene[0]*u_other[7]-4.0*m1r[7]+greene[1]*u_self[6]-1.0*greene[1]*u_other[6]+greene[2]*u_self[5]-1.0*greene[2]*u_other[5]+greene[3]*u_self[4]-1.0*greene[3]*u_other[4]); 
 
  double ucMSelf[4] = {0.0}; 
  double ucMOther[4] = {0.0}; 
  for (int vd=0; vd<2; vd++) 
  { 
    int a0 = 4*vd; 
    // Contribution to dot-product from weak multiplication of vd component. 
    ucMSelf[0] += 0.5*cMr[a0+3]*u_self[a0+3]+0.5*cMr[a0+2]*u_self[a0+2]+0.5*cMr[a0+1]*u_self[a0+1]+0.5*cMr[a0]*u_self[a0]; 
    ucMSelf[1] += 0.5*cMr[a0+2]*u_self[a0+3]+0.5*u_self[a0+2]*cMr[a0+3]+0.5*cMr[a0]*u_self[a0+1]+0.5*u_self[a0]*cMr[a0+1]; 
    ucMSelf[2] += 0.5*cMr[a0+1]*u_self[a0+3]+0.5*u_self[a0+1]*cMr[a0+3]+0.5*cMr[a0]*u_self[a0+2]+0.5*u_self[a0]*cMr[a0+2]; 
    ucMSelf[3] += 0.5*cMr[a0]*u_self[a0+3]+0.5*u_self[a0]*cMr[a0+3]+0.5*cMr[a0+1]*u_self[a0+2]+0.5*u_self[a0+1]*cMr[a0+2]; 
    ucMOther[0] += 0.5*cMr[a0+3]*u_other[a0+3]+0.5*cMr[a0+2]*u_other[a0+2]+0.5*cMr[a0+1]*u_other[a0+1]+0.5*cMr[a0]*u_other[a0]; 
    ucMOther[1] += 0.5*cMr[a0+2]*u_other[a0+3]+0.5*u_other[a0+2]*cMr[a0+3]+0.5*cMr[a0]*u_other[a0+1]+0.5*u_other[a0]*cMr[a0+1]; 
    ucMOther[2] += 0.5*cMr[a0+1]*u_other[a0+3]+0.5*u_other[a0+1]*cMr[a0+3]+0.5*cMr[a0]*u_other[a0+2]+0.5*u_other[a0]*cMr[a0+2]; 
    ucMOther[3] += 0.5*cMr[a0]*u_other[a0+3]+0.5*u_other[a0]*cMr[a0+3]+0.5*cMr[a0+1]*u_other[a0+2]+0.5*u_other[a0+1]*cMr[a0+2]; 
  } 
 
  // ... Block from correction to (self) 2nd moment of collision operator ... // 
  gkyl_mat_set(A,8,8,0.5*ucMSelf[0]+0.5*ucMOther[0]+2.0*m0r[0]-1.0*cEr[0]); 
  gkyl_mat_set(A,8,9,0.5*ucMSelf[1]+0.5*ucMOther[1]+2.0*m0r[1]-1.0*cEr[1]); 
  gkyl_mat_set(A,8,10,0.5*ucMSelf[2]+0.5*ucMOther[2]+2.0*m0r[2]-1.0*cEr[2]); 
  gkyl_mat_set(A,8,11,0.5*ucMSelf[3]+0.5*ucMOther[3]+2.0*m0r[3]-1.0*cEr[3]); 
  gkyl_mat_set(A,9,8,0.5*ucMSelf[1]+0.5*ucMOther[1]+2.0*m0r[1]-1.0*cEr[1]); 
  gkyl_mat_set(A,9,9,0.5*ucMSelf[0]+0.5*ucMOther[0]+2.0*m0r[0]-1.0*cEr[0]); 
  gkyl_mat_set(A,9,10,0.5*ucMSelf[3]+0.5*ucMOther[3]+2.0*m0r[3]-1.0*cEr[3]); 
  gkyl_mat_set(A,9,11,0.5*ucMSelf[2]+0.5*ucMOther[2]+2.0*m0r[2]-1.0*cEr[2]); 
  gkyl_mat_set(A,10,8,0.5*ucMSelf[2]+0.5*ucMOther[2]+2.0*m0r[2]-1.0*cEr[2]); 
  gkyl_mat_set(A,10,9,0.5*ucMSelf[3]+0.5*ucMOther[3]+2.0*m0r[3]-1.0*cEr[3]); 
  gkyl_mat_set(A,10,10,0.5*ucMSelf[0]+0.5*ucMOther[0]+2.0*m0r[0]-1.0*cEr[0]); 
  gkyl_mat_set(A,10,11,0.5*ucMSelf[1]+0.5*ucMOther[1]+2.0*m0r[1]-1.0*cEr[1]); 
  gkyl_mat_set(A,11,8,0.5*ucMSelf[3]+0.5*ucMOther[3]+2.0*m0r[3]-1.0*cEr[3]); 
  gkyl_mat_set(A,11,9,0.5*ucMSelf[2]+0.5*ucMOther[2]+2.0*m0r[2]-1.0*cEr[2]); 
  gkyl_mat_set(A,11,10,0.5*ucMSelf[1]+0.5*ucMOther[1]+2.0*m0r[1]-1.0*cEr[1]); 
  gkyl_mat_set(A,11,11,0.5*ucMSelf[0]+0.5*ucMOther[0]+2.0*m0r[0]-1.0*cEr[0]); 
 
  double uM1Self[4] = {0.0}; 
  double uM1Other[4] = {0.0}; 
  double uSumSq[4] = {0.0}; 
  for (int vd=0; vd<2; vd++) 
  { 
    int a0 = 4*vd; 
    // Dot product terms in energy equation RHS. 
    uM1Self[0] += 0.5*m1r[7]*u_self[a0+3]+0.5*m1r[6]*u_self[a0+2]+0.5*m1r[5]*u_self[a0+1]+0.5*m1r[4]*u_self[a0]; 
    uM1Self[1] += 0.5*m1r[6]*u_self[a0+3]+0.5*m1r[7]*u_self[a0+2]+0.5*m1r[4]*u_self[a0+1]+0.5*m1r[5]*u_self[a0]; 
    uM1Self[2] += 0.5*m1r[5]*u_self[a0+3]+0.5*m1r[4]*u_self[a0+2]+0.5*m1r[7]*u_self[a0+1]+0.5*m1r[6]*u_self[a0]; 
    uM1Self[3] += 0.5*m1r[4]*u_self[a0+3]+0.5*m1r[5]*u_self[a0+2]+0.5*m1r[6]*u_self[a0+1]+0.5*m1r[7]*u_self[a0]; 
    uM1Other[0] += 0.5*m1r[7]*u_other[a0+3]+0.5*m1r[6]*u_other[a0+2]+0.5*m1r[5]*u_other[a0+1]+0.5*m1r[4]*u_other[a0]; 
    uM1Other[1] += 0.5*m1r[6]*u_other[a0+3]+0.5*m1r[7]*u_other[a0+2]+0.5*m1r[4]*u_other[a0+1]+0.5*m1r[5]*u_other[a0]; 
    uM1Other[2] += 0.5*m1r[5]*u_other[a0+3]+0.5*m1r[4]*u_other[a0+2]+0.5*m1r[7]*u_other[a0+1]+0.5*m1r[6]*u_other[a0]; 
    uM1Other[3] += 0.5*m1r[4]*u_other[a0+3]+0.5*m1r[5]*u_other[a0+2]+0.5*m1r[6]*u_other[a0+1]+0.5*m1r[7]*u_other[a0]; 
  const double u_self0R2 = pow(u_self[a0],2);
  const double u_self1R2 = pow(u_self[a0+1],2);
  const double u_self2R2 = pow(u_self[a0+2],2);
  const double u_self3R2 = pow(u_self[a0+3],2);
  const double u_other0R2 = pow(u_other[a0],2);
  const double u_other1R2 = pow(u_other[a0+1],2);
  const double u_other2R2 = pow(u_other[a0+2],2);
  const double u_other3R2 = pow(u_other[a0+3],2);

  uSumSq[0] += 0.5*u_self3R2-1.0*u_other[a0+3]*u_self[a0+3]+0.5*u_other3R2+0.5*u_self2R2-1.0*u_other[a0+2]*u_self[a0+2]+0.5*u_other2R2+0.5*u_self1R2-1.0*u_other[a0+1]*u_self[a0+1]+0.5*u_other1R2+0.5*u_self0R2-1.0*u_other[a0]*u_self[a0]+0.5*u_other0R2; 
  uSumSq[1] += u_self[a0+2]*u_self[a0+3]-1.0*u_other[a0+2]*u_self[a0+3]-1.0*u_self[a0+2]*u_other[a0+3]+u_other[a0+2]*u_other[a0+3]+u_self[a0]*u_self[a0+1]-1.0*u_other[a0]*u_self[a0+1]-1.0*u_self[a0]*u_other[a0+1]+u_other[a0]*u_other[a0+1]; 
  uSumSq[2] += u_self[a0+1]*u_self[a0+3]-1.0*u_other[a0+1]*u_self[a0+3]-1.0*u_self[a0+1]*u_other[a0+3]+u_other[a0+1]*u_other[a0+3]+u_self[a0]*u_self[a0+2]-1.0*u_other[a0]*u_self[a0+2]-1.0*u_self[a0]*u_other[a0+2]+u_other[a0]*u_other[a0+2]; 
  uSumSq[3] += u_self[a0]*u_self[a0+3]-1.0*u_other[a0]*u_self[a0+3]-1.0*u_self[a0]*u_other[a0+3]+u_other[a0]*u_other[a0+3]+u_self[a0+1]*u_self[a0+2]-1.0*u_other[a0+1]*u_self[a0+2]-1.0*u_self[a0+1]*u_other[a0+2]+u_other[a0+1]*u_other[a0+2]; 
  } 
 
  double enRHS[4] = {0.0}; 
  enRHS[0] = (-(4.0*greene[3]*vtsq_self[3]*m_self)/(4.0*m_self+4.0*m_other))-(1.0*greene[3]*uSumSq[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[2]*vtsq_self[2]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[2]*uSumSq[2]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[1]*vtsq_self[1]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[1]*uSumSq[1]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[0]*vtsq_self[0]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[0]*uSumSq[0]*m_self)/(4.0*m_self+4.0*m_other)+(4.0*greene[3]*vtsq_other[3]*m_other)/(4.0*m_self+4.0*m_other)+(greene[3]*uSumSq[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[2]*vtsq_other[2]*m_other)/(4.0*m_self+4.0*m_other)+(greene[2]*uSumSq[2]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[1]*vtsq_other[1]*m_other)/(4.0*m_self+4.0*m_other)+(greene[1]*uSumSq[1]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[0]*vtsq_other[0]*m_other)/(4.0*m_self+4.0*m_other)+(greene[0]*uSumSq[0]*m_other)/(4.0*m_self+4.0*m_other)-1.0*uM1Self[0]-1.0*uM1Other[0]+2.0*m2r[0]; 
  enRHS[1] = (-(4.0*greene[2]*vtsq_self[3]*m_self)/(4.0*m_self+4.0*m_other))-(1.0*greene[2]*uSumSq[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[2]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[2]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[0]*vtsq_self[1]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[0]*uSumSq[1]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[0]*greene[1]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[0]*greene[1]*m_self)/(4.0*m_self+4.0*m_other)+(4.0*greene[2]*vtsq_other[3]*m_other)/(4.0*m_self+4.0*m_other)+(greene[2]*uSumSq[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[2]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[2]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[0]*vtsq_other[1]*m_other)/(4.0*m_self+4.0*m_other)+(greene[0]*uSumSq[1]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[0]*greene[1]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[0]*greene[1]*m_other)/(4.0*m_self+4.0*m_other)-1.0*uM1Self[1]-1.0*uM1Other[1]+2.0*m2r[1]; 
  enRHS[2] = (-(4.0*greene[1]*vtsq_self[3]*m_self)/(4.0*m_self+4.0*m_other))-(1.0*greene[1]*uSumSq[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[1]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[1]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[0]*vtsq_self[2]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[0]*uSumSq[2]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[0]*greene[2]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[0]*greene[2]*m_self)/(4.0*m_self+4.0*m_other)+(4.0*greene[1]*vtsq_other[3]*m_other)/(4.0*m_self+4.0*m_other)+(greene[1]*uSumSq[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[1]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[1]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[0]*vtsq_other[2]*m_other)/(4.0*m_self+4.0*m_other)+(greene[0]*uSumSq[2]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[0]*greene[2]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[0]*greene[2]*m_other)/(4.0*m_self+4.0*m_other)-1.0*uM1Self[2]-1.0*uM1Other[2]+2.0*m2r[2]; 
  enRHS[3] = (-(4.0*greene[0]*vtsq_self[3]*m_self)/(4.0*m_self+4.0*m_other))-(1.0*greene[0]*uSumSq[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[0]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[0]*greene[3]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*greene[1]*vtsq_self[2]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*greene[1]*uSumSq[2]*m_self)/(4.0*m_self+4.0*m_other)-(4.0*vtsq_self[1]*greene[2]*m_self)/(4.0*m_self+4.0*m_other)-(1.0*uSumSq[1]*greene[2]*m_self)/(4.0*m_self+4.0*m_other)+(4.0*greene[0]*vtsq_other[3]*m_other)/(4.0*m_self+4.0*m_other)+(greene[0]*uSumSq[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[0]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[0]*greene[3]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*greene[1]*vtsq_other[2]*m_other)/(4.0*m_self+4.0*m_other)+(greene[1]*uSumSq[2]*m_other)/(4.0*m_self+4.0*m_other)+(4.0*vtsq_other[1]*greene[2]*m_other)/(4.0*m_self+4.0*m_other)+(uSumSq[1]*greene[2]*m_other)/(4.0*m_self+4.0*m_other)-1.0*uM1Self[3]-1.0*uM1Other[3]+2.0*m2r[3]; 
 
  gkyl_mat_set(rhs,0,0,momRHS[0]); 
  gkyl_mat_set(rhs,1,0,momRHS[1]); 
  gkyl_mat_set(rhs,2,0,momRHS[2]); 
  gkyl_mat_set(rhs,3,0,momRHS[3]); 
  gkyl_mat_set(rhs,4,0,momRHS[4]); 
  gkyl_mat_set(rhs,5,0,momRHS[5]); 
  gkyl_mat_set(rhs,6,0,momRHS[6]); 
  gkyl_mat_set(rhs,7,0,momRHS[7]); 
  gkyl_mat_set(rhs,8,0,enRHS[0]); 
  gkyl_mat_set(rhs,9,0,enRHS[1]); 
  gkyl_mat_set(rhs,10,0,enRHS[2]); 
  gkyl_mat_set(rhs,11,0,enRHS[3]); 
} 
 