#include <gkyl_gyrokinetic_kernels.h> 
GKYL_CU_DH int gyrokinetic_alpha_surfx_2x2v_ser_p1(const double *w, const double *dxv, const double q_, const double m_, 
  const double *bmag, const double *jacobtot_inv, const double *cmag, const double *b_i, 
  const double *phi, double* GKYL_RESTRICT alpha_surf, double* GKYL_RESTRICT sgn_alpha_surf) 
{ 
  // w[NDIM]: cell-center.
  // dxv[NDIM]: cell length.
  // q_,m_: species charge and mass.
  // bmag: magnetic field amplitude.
  // jacobtot_inv: reciprocal of the conf-space jacobian time the guiding center coordinate Jacobian.
  // cmag: coefficient multiplying parallel gradient.
  // b_i: covariant components of the field aligned unit vector.
  // phi: electrostatic potential.
  // alpha_surf: output surface phase space flux in each direction (cdim + 1 components).
  //             Note: Each cell owns their *lower* edge surface evaluation.
  // sgn_alpha_surf: output sign(alpha_surf) in each direction at quadrature points (cdim + 1 components).
  //                 Note: Each cell owns their *lower* edge sign(alpha_surf).
  // returns int const_sgn_alpha (true if sign(alpha_surf) is only one sign, either +1 or -1).

  double wx = w[0];
  double rdx2 = 2.0/dxv[0];
  double wz = w[1];
  double rdz2 = 2.0/dxv[1];
  double wvpar = w[2];
  double rdvpar2 = 2.0/dxv[2];
  double wmu = w[3];
  double rdmu2 = 2.0/dxv[3];

  double wvparSq = wvpar*wvpar;
  double rdvpar2Sq = rdvpar2*rdvpar2;

  const double *b_x = &b_i[0];
  const double *b_y = &b_i[4];
  const double *b_z = &b_i[8];

  double hamil[24] = {0.}; 
  hamil[0] = 2.0*(m_*wvparSq+bmag[0]*wmu)+(0.6666666666666666*m_)/rdvpar2Sq+2.0*phi[0]*q_; 
  hamil[1] = 2.0*(bmag[1]*wmu+phi[1]*q_); 
  hamil[2] = 2.0*(bmag[2]*wmu+phi[2]*q_); 
  hamil[3] = (2.309401076758503*m_*wvpar)/rdvpar2; 
  hamil[4] = (1.154700538379252*bmag[0])/rdmu2; 
  hamil[5] = 2.0*(bmag[3]*wmu+phi[3]*q_); 
  hamil[8] = (1.154700538379252*bmag[1])/rdmu2; 
  hamil[9] = (1.154700538379252*bmag[2])/rdmu2; 
  hamil[12] = (1.154700538379252*bmag[3])/rdmu2; 
  hamil[16] = (0.5962847939999438*m_)/rdvpar2Sq; 

  double *alphaL = &alpha_surf[0];
  double *sgn_alpha_surfL = &sgn_alpha_surf[0];
  alphaL[0] = (((-1.590990257669731*jacobtot_inv[1]*b_y[3]*hamil[3])+0.9185586535436913*jacobtot_inv[0]*b_y[3]*hamil[3]+0.9185586535436913*jacobtot_inv[1]*b_y[2]*hamil[3]-0.5303300858899105*jacobtot_inv[0]*b_y[2]*hamil[3])*rdvpar2*rdz2*wvpar)/q_+(((-2.053959590644372*jacobtot_inv[1]*b_y[3]*hamil[16])+1.185854122563142*jacobtot_inv[0]*b_y[3]*hamil[16]+1.185854122563142*jacobtot_inv[1]*b_y[2]*hamil[16]-0.6846531968814573*jacobtot_inv[0]*b_y[2]*hamil[16]-1.590990257669731*b_y[3]*jacobtot_inv[3]*hamil[5]+0.9185586535436913*b_y[2]*jacobtot_inv[3]*hamil[5]+0.9185586535436913*jacobtot_inv[2]*b_y[3]*hamil[5]-0.5303300858899105*b_y[2]*jacobtot_inv[2]*hamil[5]-1.590990257669731*b_y[1]*jacobtot_inv[1]*hamil[5]+0.9185586535436913*b_y[0]*jacobtot_inv[1]*hamil[5]+0.9185586535436913*jacobtot_inv[0]*b_y[1]*hamil[5]-0.5303300858899105*b_y[0]*jacobtot_inv[0]*hamil[5]+0.9185586535436913*hamil[2]*b_y[3]*jacobtot_inv[3]-0.5303300858899105*b_y[2]*hamil[2]*jacobtot_inv[3]-0.5303300858899105*hamil[2]*jacobtot_inv[2]*b_y[3]+0.3061862178478971*b_y[2]*hamil[2]*jacobtot_inv[2]+0.9185586535436913*b_y[1]*jacobtot_inv[1]*hamil[2]-0.5303300858899105*b_y[0]*jacobtot_inv[1]*hamil[2]-0.5303300858899105*jacobtot_inv[0]*b_y[1]*hamil[2]+0.3061862178478971*b_y[0]*jacobtot_inv[0]*hamil[2])*rdz2)/q_; 
  alphaL[1] = (((-1.590990257669731*b_y[3]*hamil[3]*jacobtot_inv[3])+0.9185586535436913*b_y[2]*hamil[3]*jacobtot_inv[3]+0.9185586535436913*jacobtot_inv[2]*b_y[3]*hamil[3]-0.5303300858899105*b_y[2]*jacobtot_inv[2]*hamil[3])*rdvpar2*rdz2*wvpar)/q_+(((-2.053959590644372*b_y[3]*jacobtot_inv[3]*hamil[16])+1.185854122563142*b_y[2]*jacobtot_inv[3]*hamil[16]+1.185854122563142*jacobtot_inv[2]*b_y[3]*hamil[16]-0.6846531968814573*b_y[2]*jacobtot_inv[2]*hamil[16]-1.590990257669731*b_y[1]*jacobtot_inv[3]*hamil[5]+0.9185586535436913*b_y[0]*jacobtot_inv[3]*hamil[5]-1.590990257669731*jacobtot_inv[1]*b_y[3]*hamil[5]+0.9185586535436913*jacobtot_inv[0]*b_y[3]*hamil[5]+0.9185586535436913*b_y[1]*jacobtot_inv[2]*hamil[5]-0.5303300858899105*b_y[0]*jacobtot_inv[2]*hamil[5]+0.9185586535436913*jacobtot_inv[1]*b_y[2]*hamil[5]-0.5303300858899105*jacobtot_inv[0]*b_y[2]*hamil[5]+0.9185586535436913*b_y[1]*hamil[2]*jacobtot_inv[3]-0.5303300858899105*b_y[0]*hamil[2]*jacobtot_inv[3]+0.9185586535436913*jacobtot_inv[1]*hamil[2]*b_y[3]-0.5303300858899105*jacobtot_inv[0]*hamil[2]*b_y[3]-0.5303300858899105*b_y[1]*hamil[2]*jacobtot_inv[2]+0.3061862178478971*b_y[0]*hamil[2]*jacobtot_inv[2]-0.5303300858899105*jacobtot_inv[1]*b_y[2]*hamil[2]+0.3061862178478971*jacobtot_inv[0]*b_y[2]*hamil[2])*rdz2)/q_; 
  alphaL[2] = (((-3.557562367689425*jacobtot_inv[1]*b_y[3]*hamil[16])+2.053959590644372*jacobtot_inv[0]*b_y[3]*hamil[16]+2.053959590644372*jacobtot_inv[1]*b_y[2]*hamil[16]-1.185854122563142*jacobtot_inv[0]*b_y[2]*hamil[16])*rdvpar2*rdz2*wvpar)/q_+(((-0.9185586535436913*jacobtot_inv[1]*b_y[3]*hamil[3])+0.5303300858899105*jacobtot_inv[0]*b_y[3]*hamil[3]+0.5303300858899105*jacobtot_inv[1]*b_y[2]*hamil[3]-0.3061862178478971*jacobtot_inv[0]*b_y[2]*hamil[3])*rdz2)/q_; 
  alphaL[3] = (((-1.590990257669731*b_y[3]*jacobtot_inv[3]*hamil[12])+0.9185586535436913*b_y[2]*jacobtot_inv[3]*hamil[12]+0.9185586535436913*jacobtot_inv[2]*b_y[3]*hamil[12]-0.5303300858899105*b_y[2]*jacobtot_inv[2]*hamil[12]-1.590990257669731*b_y[1]*jacobtot_inv[1]*hamil[12]+0.9185586535436913*b_y[0]*jacobtot_inv[1]*hamil[12]+0.9185586535436913*jacobtot_inv[0]*b_y[1]*hamil[12]-0.5303300858899105*b_y[0]*jacobtot_inv[0]*hamil[12]+0.9185586535436913*b_y[3]*jacobtot_inv[3]*hamil[9]-0.5303300858899105*b_y[2]*jacobtot_inv[3]*hamil[9]-0.5303300858899105*jacobtot_inv[2]*b_y[3]*hamil[9]+0.3061862178478971*b_y[2]*jacobtot_inv[2]*hamil[9]+0.9185586535436913*b_y[1]*jacobtot_inv[1]*hamil[9]-0.5303300858899105*b_y[0]*jacobtot_inv[1]*hamil[9]-0.5303300858899105*jacobtot_inv[0]*b_y[1]*hamil[9]+0.3061862178478971*b_y[0]*jacobtot_inv[0]*hamil[9])*rdz2)/q_; 
  alphaL[4] = (((-3.557562367689425*b_y[3]*jacobtot_inv[3]*hamil[16])+2.053959590644372*b_y[2]*jacobtot_inv[3]*hamil[16]+2.053959590644372*jacobtot_inv[2]*b_y[3]*hamil[16]-1.185854122563142*b_y[2]*jacobtot_inv[2]*hamil[16])*rdvpar2*rdz2*wvpar)/q_+(((-0.9185586535436913*b_y[3]*hamil[3]*jacobtot_inv[3])+0.5303300858899105*b_y[2]*hamil[3]*jacobtot_inv[3]+0.5303300858899105*jacobtot_inv[2]*b_y[3]*hamil[3]-0.3061862178478971*b_y[2]*jacobtot_inv[2]*hamil[3])*rdz2)/q_; 
  alphaL[5] = (((-1.590990257669731*b_y[1]*jacobtot_inv[3]*hamil[12])+0.9185586535436913*b_y[0]*jacobtot_inv[3]*hamil[12]-1.590990257669731*jacobtot_inv[1]*b_y[3]*hamil[12]+0.9185586535436913*jacobtot_inv[0]*b_y[3]*hamil[12]+0.9185586535436913*b_y[1]*jacobtot_inv[2]*hamil[12]-0.5303300858899105*b_y[0]*jacobtot_inv[2]*hamil[12]+0.9185586535436913*jacobtot_inv[1]*b_y[2]*hamil[12]-0.5303300858899105*jacobtot_inv[0]*b_y[2]*hamil[12]+0.9185586535436913*b_y[1]*jacobtot_inv[3]*hamil[9]-0.5303300858899105*b_y[0]*jacobtot_inv[3]*hamil[9]+0.9185586535436913*jacobtot_inv[1]*b_y[3]*hamil[9]-0.5303300858899105*jacobtot_inv[0]*b_y[3]*hamil[9]-0.5303300858899105*b_y[1]*jacobtot_inv[2]*hamil[9]+0.3061862178478971*b_y[0]*jacobtot_inv[2]*hamil[9]-0.5303300858899105*jacobtot_inv[1]*b_y[2]*hamil[9]+0.3061862178478971*jacobtot_inv[0]*b_y[2]*hamil[9])*rdz2)/q_; 
  alphaL[8] = (((-1.837117307087383*jacobtot_inv[1]*b_y[3]*hamil[16])+1.060660171779821*jacobtot_inv[0]*b_y[3]*hamil[16]+1.060660171779821*jacobtot_inv[1]*b_y[2]*hamil[16]-0.6123724356957944*jacobtot_inv[0]*b_y[2]*hamil[16])*rdz2)/q_; 
  alphaL[9] = (((-1.837117307087383*b_y[3]*jacobtot_inv[3]*hamil[16])+1.060660171779821*b_y[2]*jacobtot_inv[3]*hamil[16]+1.060660171779821*jacobtot_inv[2]*b_y[3]*hamil[16]-0.6123724356957944*b_y[2]*jacobtot_inv[2]*hamil[16])*rdz2)/q_; 

  int const_sgn_alpha_surf = 1;  
  
  if ((-0.3162277660168378*alphaL[9])+0.3162277660168378*alphaL[8]+0.3535533905932734*alphaL[5]+0.4743416490252568*alphaL[4]-0.3535533905932734*alphaL[3]-0.4743416490252568*alphaL[2]-0.3535533905932734*alphaL[1]+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[0] = 1.0; 
  else  
    sgn_alpha_surfL[0] = -1.0; 
  
  if (0.3952847075210473*alphaL[9]-0.3952847075210471*alphaL[8]+0.3535533905932734*alphaL[5]-0.3535533905932734*(alphaL[3]+alphaL[1])+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[1] = 1.0; 
  else  
    sgn_alpha_surfL[1] = -1.0; 
  
  if (sgn_alpha_surfL[1] == sgn_alpha_surfL[0]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if ((-0.3162277660168378*alphaL[9])+0.3162277660168378*alphaL[8]+0.3535533905932734*alphaL[5]-0.4743416490252568*alphaL[4]-0.3535533905932734*alphaL[3]+0.4743416490252568*alphaL[2]-0.3535533905932734*alphaL[1]+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[2] = 1.0; 
  else  
    sgn_alpha_surfL[2] = -1.0; 
  
  if (sgn_alpha_surfL[2] == sgn_alpha_surfL[1]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if ((-0.3162277660168378*alphaL[9])+0.3162277660168378*alphaL[8]-0.3535533905932734*alphaL[5]+0.4743416490252568*alphaL[4]+0.3535533905932734*alphaL[3]-0.4743416490252568*alphaL[2]-0.3535533905932734*alphaL[1]+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[3] = 1.0; 
  else  
    sgn_alpha_surfL[3] = -1.0; 
  
  if (sgn_alpha_surfL[3] == sgn_alpha_surfL[2]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if (0.3952847075210473*alphaL[9]-0.3952847075210471*alphaL[8]-0.3535533905932734*alphaL[5]+0.3535533905932734*alphaL[3]-0.3535533905932734*alphaL[1]+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[4] = 1.0; 
  else  
    sgn_alpha_surfL[4] = -1.0; 
  
  if (sgn_alpha_surfL[4] == sgn_alpha_surfL[3]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if ((-0.3162277660168378*alphaL[9])+0.3162277660168378*alphaL[8]-0.3535533905932734*alphaL[5]-0.4743416490252568*alphaL[4]+0.3535533905932734*alphaL[3]+0.4743416490252568*alphaL[2]-0.3535533905932734*alphaL[1]+0.3535533905932734*alphaL[0] > 0.) 
    sgn_alpha_surfL[5] = 1.0; 
  else  
    sgn_alpha_surfL[5] = -1.0; 
  
  if (sgn_alpha_surfL[5] == sgn_alpha_surfL[4]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if (0.3162277660168378*(alphaL[9]+alphaL[8])-0.3535533905932734*alphaL[5]-0.4743416490252568*alphaL[4]-0.3535533905932734*alphaL[3]-0.4743416490252568*alphaL[2]+0.3535533905932734*(alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[6] = 1.0; 
  else  
    sgn_alpha_surfL[6] = -1.0; 
  
  if (sgn_alpha_surfL[6] == sgn_alpha_surfL[5]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if ((-0.3952847075210473*alphaL[9])-0.3952847075210471*alphaL[8]-0.3535533905932734*(alphaL[5]+alphaL[3])+0.3535533905932734*(alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[7] = 1.0; 
  else  
    sgn_alpha_surfL[7] = -1.0; 
  
  if (sgn_alpha_surfL[7] == sgn_alpha_surfL[6]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if (0.3162277660168378*(alphaL[9]+alphaL[8])-0.3535533905932734*alphaL[5]+0.4743416490252568*alphaL[4]-0.3535533905932734*alphaL[3]+0.4743416490252568*alphaL[2]+0.3535533905932734*(alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[8] = 1.0; 
  else  
    sgn_alpha_surfL[8] = -1.0; 
  
  if (sgn_alpha_surfL[8] == sgn_alpha_surfL[7]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if (0.3162277660168378*(alphaL[9]+alphaL[8])+0.3535533905932734*alphaL[5]-0.4743416490252568*alphaL[4]+0.3535533905932734*alphaL[3]-0.4743416490252568*alphaL[2]+0.3535533905932734*(alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[9] = 1.0; 
  else  
    sgn_alpha_surfL[9] = -1.0; 
  
  if (sgn_alpha_surfL[9] == sgn_alpha_surfL[8]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if ((-0.3952847075210473*alphaL[9])-0.3952847075210471*alphaL[8]+0.3535533905932734*(alphaL[5]+alphaL[3]+alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[10] = 1.0; 
  else  
    sgn_alpha_surfL[10] = -1.0; 
  
  if (sgn_alpha_surfL[10] == sgn_alpha_surfL[9]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  if (0.3162277660168378*(alphaL[9]+alphaL[8])+0.3535533905932734*alphaL[5]+0.4743416490252568*alphaL[4]+0.3535533905932734*alphaL[3]+0.4743416490252568*alphaL[2]+0.3535533905932734*(alphaL[1]+alphaL[0]) > 0.) 
    sgn_alpha_surfL[11] = 1.0; 
  else  
    sgn_alpha_surfL[11] = -1.0; 
  
  if (sgn_alpha_surfL[11] == sgn_alpha_surfL[10]) 
    const_sgn_alpha_surf = const_sgn_alpha_surf ? 1 : 0; 
  else  
    const_sgn_alpha_surf = 0; 
  
  return const_sgn_alpha_surf; 

} 