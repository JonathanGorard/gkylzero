#include <gkyl_rad_gyrokinetic_kernels.h> 
GKYL_CU_DH void rad_gyrokinetic_drag_numu_1x2v_ser_p1(const double *w, const double *dxv, 
  double charge, double mass, double a, double alpha, double beta, double gamma, double v0, 
  const double *bmag, double* GKYL_RESTRICT drag_rad_nu_surf, double* GKYL_RESTRICT drag_rad_nu) 
{ 
  // w[NDIM]: cell-center.
  // dxv[NDIM]: cell length.
  // charge: elementary charge.
  // mass: mass of electron.
  // a, alpha, beta, gamma, v0: Radiation fitting parameters.
  // bmag: magnetic field amplitude.
  // drag_rad_nu_surf: Radiation drag in direction dir at corresponding surfaces.
  //                   Note: Each cell owns their *lower* edge surface evaluation.
  // drag_rad_nu: Radiation drag direction dir volume expansion.

  const double wvpar = w[1], dvpar = dxv[1]; 
  const double wvmu = w[2], dvmu = dxv[2]; 
  const double wvpar_sq = wvpar*wvpar, dvpar_sq = dvpar*dvpar; 

  double scaled_v0 = v0/sqrt(mass/(2.0*fabs(charge)));
  double c_const = 8.0*fabs(charge)/sqrt(M_PI)*pow(2*fabs(charge)/mass,gamma/2.0); 
  double const_mult = a*(alpha+beta)/c_const;
  double vmag = 0.0;
  double vmag_surf = 0.0;

  double vpar_p[2] = {0.0};
  double vpar_sq_p[3] = {0.0};
  double mu_p[2] = {0.0};
  vpar_p[0] = 1.414213562373095*wvpar; 
  vpar_p[1] = 0.408248290463863*dvpar; 
  vpar_sq_p[0] = 1.414213562373095*wvpar_sq+0.1178511301977578*dvpar_sq; 
  vpar_sq_p[1] = 0.8164965809277262*dvpar*wvpar; 
  vpar_sq_p[2] = 0.1054092553389459*dvpar_sq; 
  mu_p[0] = 1.414213562373095*wvmu; 
  mu_p[1] = 0.408248290463863*dvmu; 

  double vsqnu_surf_nodal[6] = {0.0};
  double vsqnu_nodal[12] = {0.0};
  vmag_surf = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[0] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));
  vmag_surf = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[1] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));
  vmag_surf = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[2] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));
  vmag_surf = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[3] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));
  vmag_surf = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[4] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));
  vmag_surf = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1]))/mass);
  vsqnu_surf_nodal[5] = 2.0*(0.7071067811865475*mu_p[0]-1.224744871391589*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag_surf,gamma))/(beta*pow(vmag_surf/scaled_v0,-alpha)+alpha*pow(vmag_surf/scaled_v0,beta)));

  drag_rad_nu_surf[0] = 0.2777777777777778*vsqnu_surf_nodal[5]+0.4444444444444444*vsqnu_surf_nodal[4]+0.2777777777777778*vsqnu_surf_nodal[3]+0.2777777777777778*vsqnu_surf_nodal[2]+0.4444444444444444*vsqnu_surf_nodal[1]+0.2777777777777778*vsqnu_surf_nodal[0]; 
  drag_rad_nu_surf[1] = 0.2777777777777778*vsqnu_surf_nodal[5]+0.4444444444444444*vsqnu_surf_nodal[4]+0.2777777777777778*vsqnu_surf_nodal[3]-0.2777777777777778*vsqnu_surf_nodal[2]-0.4444444444444444*vsqnu_surf_nodal[1]-0.2777777777777778*vsqnu_surf_nodal[0]; 
  drag_rad_nu_surf[2] = 0.3726779962499649*vsqnu_surf_nodal[5]-0.3726779962499649*vsqnu_surf_nodal[3]+0.3726779962499649*vsqnu_surf_nodal[2]-0.3726779962499649*vsqnu_surf_nodal[0]; 
  drag_rad_nu_surf[3] = 0.3726779962499649*vsqnu_surf_nodal[5]-0.3726779962499649*vsqnu_surf_nodal[3]-0.3726779962499649*vsqnu_surf_nodal[2]+0.3726779962499649*vsqnu_surf_nodal[0]; 
  drag_rad_nu_surf[4] = 0.2484519974999766*vsqnu_surf_nodal[5]-0.4969039949999532*vsqnu_surf_nodal[4]+0.2484519974999766*vsqnu_surf_nodal[3]+0.2484519974999766*vsqnu_surf_nodal[2]-0.4969039949999532*vsqnu_surf_nodal[1]+0.2484519974999766*vsqnu_surf_nodal[0]; 
  drag_rad_nu_surf[5] = 0.2484519974999766*vsqnu_surf_nodal[5]-0.4969039949999533*vsqnu_surf_nodal[4]+0.2484519974999766*vsqnu_surf_nodal[3]-0.2484519974999766*vsqnu_surf_nodal[2]+0.4969039949999533*vsqnu_surf_nodal[1]-0.2484519974999766*vsqnu_surf_nodal[0]; 

  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[0] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[1] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*((0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[2] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[3] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*(0.7071067811865475*(0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[4] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(0.7071067811865475*bmag[0]-0.7071067811865475*bmag[1])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[5] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[6] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[7] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.7071067811865475*(bmag[1]+bmag[0])*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1]))/mass);
  vsqnu_nodal[8] = 2.0*(0.7071067811865475*mu_p[0]-0.7071067811865475*mu_p[1])*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]-0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.4999999999999999*(bmag[1]+bmag[0])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[9] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.7071067811865475*vpar_sq_p[0]-0.7905694150420947*vpar_sq_p[2])+2.0*(0.4999999999999999*(bmag[1]+bmag[0])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[10] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));
  vmag = sqrt((0.6324555320336759*vpar_sq_p[2]+0.9486832980505137*vpar_sq_p[1]+0.7071067811865475*vpar_sq_p[0])+2.0*(0.4999999999999999*(bmag[1]+bmag[0])*(mu_p[1]+mu_p[0]))/mass);
  vsqnu_nodal[11] = 2.0*(0.7071067811865475*(mu_p[1]+mu_p[0]))*((a/c_const*(alpha+beta)*pow(vmag,gamma))/(beta*pow(vmag/scaled_v0,-alpha)+alpha*pow(vmag/scaled_v0,beta)));

  drag_rad_nu[0] = 0.1964185503295965*vsqnu_nodal[11]+0.3142696805273545*vsqnu_nodal[10]+0.1964185503295965*vsqnu_nodal[9]+0.1964185503295965*vsqnu_nodal[8]+0.3142696805273545*vsqnu_nodal[7]+0.1964185503295965*vsqnu_nodal[6]+0.1964185503295965*vsqnu_nodal[5]+0.3142696805273545*vsqnu_nodal[4]+0.1964185503295965*vsqnu_nodal[3]+0.1964185503295965*vsqnu_nodal[2]+0.3142696805273545*vsqnu_nodal[1]+0.1964185503295965*vsqnu_nodal[0]; 
  drag_rad_nu[1] = 0.1964185503295965*vsqnu_nodal[11]+0.3142696805273545*vsqnu_nodal[10]+0.1964185503295965*vsqnu_nodal[9]+0.1964185503295965*vsqnu_nodal[8]+0.3142696805273545*vsqnu_nodal[7]+0.1964185503295965*vsqnu_nodal[6]-0.1964185503295965*vsqnu_nodal[5]-0.3142696805273545*vsqnu_nodal[4]-0.1964185503295965*vsqnu_nodal[3]-0.1964185503295965*vsqnu_nodal[2]-0.3142696805273545*vsqnu_nodal[1]-0.1964185503295965*vsqnu_nodal[0]; 
  drag_rad_nu[2] = 0.2635231383473649*vsqnu_nodal[11]-0.2635231383473649*vsqnu_nodal[9]+0.2635231383473649*vsqnu_nodal[8]-0.2635231383473649*vsqnu_nodal[6]+0.2635231383473649*vsqnu_nodal[5]-0.2635231383473649*vsqnu_nodal[3]+0.2635231383473649*vsqnu_nodal[2]-0.2635231383473649*vsqnu_nodal[0]; 
  drag_rad_nu[3] = 0.1964185503295965*vsqnu_nodal[11]+0.3142696805273545*vsqnu_nodal[10]+0.1964185503295965*vsqnu_nodal[9]-0.1964185503295965*vsqnu_nodal[8]-0.3142696805273545*vsqnu_nodal[7]-0.1964185503295965*vsqnu_nodal[6]+0.1964185503295965*vsqnu_nodal[5]+0.3142696805273545*vsqnu_nodal[4]+0.1964185503295965*vsqnu_nodal[3]-0.1964185503295965*vsqnu_nodal[2]-0.3142696805273545*vsqnu_nodal[1]-0.1964185503295965*vsqnu_nodal[0]; 
  drag_rad_nu[4] = 0.2635231383473649*vsqnu_nodal[11]-0.2635231383473649*vsqnu_nodal[9]+0.2635231383473649*vsqnu_nodal[8]-0.2635231383473649*vsqnu_nodal[6]-0.2635231383473649*vsqnu_nodal[5]+0.2635231383473649*vsqnu_nodal[3]-0.2635231383473649*vsqnu_nodal[2]+0.2635231383473649*vsqnu_nodal[0]; 
  drag_rad_nu[5] = 0.1964185503295965*vsqnu_nodal[11]+0.3142696805273545*vsqnu_nodal[10]+0.1964185503295965*vsqnu_nodal[9]-0.1964185503295965*vsqnu_nodal[8]-0.3142696805273545*vsqnu_nodal[7]-0.1964185503295965*vsqnu_nodal[6]-0.1964185503295965*vsqnu_nodal[5]-0.3142696805273545*vsqnu_nodal[4]-0.1964185503295965*vsqnu_nodal[3]+0.1964185503295965*vsqnu_nodal[2]+0.3142696805273545*vsqnu_nodal[1]+0.1964185503295965*vsqnu_nodal[0]; 
  drag_rad_nu[6] = 0.2635231383473649*vsqnu_nodal[11]-0.2635231383473649*vsqnu_nodal[9]-0.2635231383473649*vsqnu_nodal[8]+0.2635231383473649*vsqnu_nodal[6]+0.2635231383473649*vsqnu_nodal[5]-0.2635231383473649*vsqnu_nodal[3]-0.2635231383473649*vsqnu_nodal[2]+0.2635231383473649*vsqnu_nodal[0]; 
  drag_rad_nu[7] = 0.2635231383473649*vsqnu_nodal[11]-0.2635231383473649*vsqnu_nodal[9]-0.2635231383473649*vsqnu_nodal[8]+0.2635231383473649*vsqnu_nodal[6]-0.2635231383473649*vsqnu_nodal[5]+0.2635231383473649*vsqnu_nodal[3]+0.2635231383473649*vsqnu_nodal[2]-0.2635231383473649*vsqnu_nodal[0]; 
  drag_rad_nu[8] = 0.1756820922315766*vsqnu_nodal[11]-0.3513641844631532*vsqnu_nodal[10]+0.1756820922315766*vsqnu_nodal[9]+0.1756820922315766*vsqnu_nodal[8]-0.3513641844631532*vsqnu_nodal[7]+0.1756820922315766*vsqnu_nodal[6]+0.1756820922315766*vsqnu_nodal[5]-0.3513641844631532*vsqnu_nodal[4]+0.1756820922315766*vsqnu_nodal[3]+0.1756820922315766*vsqnu_nodal[2]-0.3513641844631532*vsqnu_nodal[1]+0.1756820922315766*vsqnu_nodal[0]; 
  drag_rad_nu[9] = 0.1756820922315766*vsqnu_nodal[11]-0.3513641844631533*vsqnu_nodal[10]+0.1756820922315766*vsqnu_nodal[9]+0.1756820922315766*vsqnu_nodal[8]-0.3513641844631533*vsqnu_nodal[7]+0.1756820922315766*vsqnu_nodal[6]-0.1756820922315766*vsqnu_nodal[5]+0.3513641844631533*vsqnu_nodal[4]-0.1756820922315766*vsqnu_nodal[3]-0.1756820922315766*vsqnu_nodal[2]+0.3513641844631533*vsqnu_nodal[1]-0.1756820922315766*vsqnu_nodal[0]; 
  drag_rad_nu[10] = 0.1756820922315766*vsqnu_nodal[11]-0.3513641844631533*vsqnu_nodal[10]+0.1756820922315766*vsqnu_nodal[9]-0.1756820922315766*vsqnu_nodal[8]+0.3513641844631533*vsqnu_nodal[7]-0.1756820922315766*vsqnu_nodal[6]+0.1756820922315766*vsqnu_nodal[5]-0.3513641844631533*vsqnu_nodal[4]+0.1756820922315766*vsqnu_nodal[3]-0.1756820922315766*vsqnu_nodal[2]+0.3513641844631533*vsqnu_nodal[1]-0.1756820922315766*vsqnu_nodal[0]; 
  drag_rad_nu[11] = 0.1756820922315766*vsqnu_nodal[11]-0.3513641844631532*vsqnu_nodal[10]+0.1756820922315766*vsqnu_nodal[9]-0.1756820922315766*vsqnu_nodal[8]+0.3513641844631532*vsqnu_nodal[7]-0.1756820922315766*vsqnu_nodal[6]-0.1756820922315766*vsqnu_nodal[5]+0.3513641844631532*vsqnu_nodal[4]-0.1756820922315766*vsqnu_nodal[3]+0.1756820922315766*vsqnu_nodal[2]-0.3513641844631532*vsqnu_nodal[1]+0.1756820922315766*vsqnu_nodal[0]; 

} 