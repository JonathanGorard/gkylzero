#include <gkyl_fpo_vlasov_kernels.h> 
 
GKYL_CU_DH void fpo_diff_coeff_cross_1x3v_vyvx_ser_p2_upvy_lovx(const double *dxv, const double *gamma, const double* fpo_g_stencil[9], const double* fpo_g_surf_stencil[9], const double* fpo_dgdv_surf, double *diff_coeff) { 
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_g_stencil[9]: 9 cell stencil of Rosenbluth potential G. 
  // fpo_g_surf_stencil[9]: 9 cell stencil of surface projection of G. 
  // fpo_dgdv_surf: Surface expansion of dG/dv in center cell. 
  // diff_coeff: Output array for diffusion tensor. 

  // Use cell-average value for gamma. 
 double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1_pv1 = 2.0/dxv[2]; 
  double dv1_pv2 = 2.0/dxv[1]; 
  double dv1_sq = 4.0/dxv[2]/dxv[1]; 
 
  const double* GCL = fpo_g_stencil[0]; 
  const double* GCC = fpo_g_stencil[1]; 
  const double* GTL = fpo_g_stencil[2]; 
  const double* GTC = fpo_g_stencil[3]; 

  const double* g_surf_CL = fpo_g_surf_stencil[0]; 
  const double* g_surf_CL_pv2 = &g_surf_CL[0]; 
  const double* g_surf_CC = fpo_g_surf_stencil[1]; 
  const double* g_surf_CC_pv2 = &g_surf_CC[0]; 
  
  const double* g_surf_CC_pv1 = &g_surf_CC[20]; 
  const double* dgdpv1_surf_CC_pv2 = &fpo_dgdv_surf[20]; 
  const double* dgdpv2_surf_CC_pv1 = &fpo_dgdv_surf[60]; 
  const double* dgdpv1_surf_CC_pv1 = &fpo_dgdv_surf[80]; 
  
  double surft1_upper[20], surft1_lower[20]; 
  double surft2_upper[20], surft2_lower[20]; 
  
  double *diff_coeff_vxvy = &diff_coeff[48]; 
  double *diff_coeff_vxvz = &diff_coeff[96]; 
  double *diff_coeff_vyvx = &diff_coeff[144]; 
  double *diff_coeff_vyvz = &diff_coeff[240]; 
  double *diff_coeff_vzvx = &diff_coeff[288]; 
  double *diff_coeff_vzvy = &diff_coeff[336]; 
  
  double *out = diff_coeff_vyvx; 
  
  surft1_upper[0] = (0.11180339887498948*dgdpv1_surf_CC_pv1[8])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[2])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[0])/dv1_pv1+0.20999597273098428*GTL[24]-0.5020433520257306*GTC[24]-0.20999597273098428*GCL[24]+0.5020433520257306*GCC[24]-0.20218664720405521*GTL[22]+0.2845589849538555*GTC[22]-0.20218664720405521*GCL[22]+0.2845589849538555*GCC[22]-0.14921997708919524*GTL[13]+0.35674444853774495*GTC[13]-0.14921997708919524*GCL[13]+0.35674444853774495*GCC[13]-0.14051136087662214*GTL[12]+0.14051136087662214*GTC[12]-0.14051136087662214*GCL[12]+0.14051136087662214*GCC[12]+0.2908529064802475*GTL[7]-0.4093485350462743*GTC[7]-0.2908529064802475*GCL[7]+0.4093485350462743*GCC[7]-0.20667569704733044*GTL[3]+0.29087690695550217*GTC[3]-0.20667569704733044*GCL[3]+0.29087690695550217*GCC[3]+0.2021307453761506*GTL[2]-0.2021307453761506*GTC[2]-0.2021307453761506*GCL[2]+0.2021307453761506*GCC[2]-0.14363106492851735*GTL[0]+0.14363106492851735*GTC[0]-0.14363106492851735*GCL[0]+0.14363106492851735*GCC[0]; 
  surft1_upper[1] = (0.11180339887498951*dgdpv1_surf_CC_pv1[12])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[4])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[1])/dv1_pv1+0.20999597273098425*GTL[34]-0.5020433520257305*GTC[34]-0.20999597273098425*GCL[34]+0.5020433520257305*GCC[34]-0.2021866472040551*GTL[33]+0.28455898495385545*GTC[33]-0.2021866472040551*GCL[33]+0.28455898495385545*GCC[33]-0.1492199770891953*GTL[23]+0.35674444853774506*GTC[23]-0.1492199770891953*GCL[23]+0.35674444853774506*GCC[23]-0.14051136087662214*GTL[20]+0.14051136087662214*GTC[20]-0.14051136087662214*GCL[20]+0.14051136087662214*GCC[20]+0.2908529064802475*GTL[15]-0.4093485350462743*GTC[15]-0.2908529064802475*GCL[15]+0.4093485350462743*GCC[15]-0.20667569704733044*GTL[6]+0.29087690695550217*GTC[6]-0.20667569704733044*GCL[6]+0.29087690695550217*GCC[6]+0.2021307453761506*GTL[5]-0.2021307453761506*GTC[5]-0.2021307453761506*GCL[5]+0.2021307453761506*GCC[5]-0.14363106492851735*GTL[1]+0.14363106492851735*GTC[1]-0.14363106492851735*GCL[1]+0.14363106492851735*GCC[1]; 
  surft1_upper[2] = (0.19364916731037082*dgdpv1_surf_CC_pv1[8])/dv1_pv1+(0.15*dgdpv1_surf_CC_pv1[2])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[0])/dv1_pv1-0.05781038847495312*GTL[24]-1.2910986759406198*GTC[24]+0.05781038847495312*GCL[24]+1.2910986759406198*GCC[24]+0.07133653706043892*GTL[22]+0.07133653706043892*GTC[22]+0.07133653706043892*GCL[22]+0.07133653706043892*GCC[22]+0.04107919181288743*GTL[13]+0.9174352838211527*GTC[13]+0.04107919181288743*GCL[13]+0.9174352838211527*GCC[13]+0.05616295755668199*GTL[12]-0.05616295755668199*GTC[12]+0.05616295755668199*GCL[12]-0.05616295755668199*GCC[12]-0.10262022457558416*GTL[7]-0.10262022457558416*GTC[7]+0.10262022457558416*GCL[7]+0.10262022457558416*GCC[7]+0.07292038680986264*GTL[3]+0.07292038680986264*GTC[3]+0.07292038680986264*GCL[3]+0.07292038680986264*GCC[3]-0.08079247402229095*GTL[2]+0.08079247402229095*GTC[2]+0.08079247402229095*GCL[2]-0.08079247402229095*GCC[2]+0.057409915846480676*GTL[0]-0.057409915846480676*GTC[0]+0.057409915846480676*GCL[0]-0.057409915846480676*GCC[0]; 
  surft1_upper[3] = (0.11180339887498951*dgdpv1_surf_CC_pv1[14])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[6])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[3])/dv1_pv1+0.20999597273098425*GTL[40]-0.5020433520257305*GTC[40]-0.20999597273098425*GCL[40]+0.5020433520257305*GCC[40]-0.2021866472040551*GTL[38]+0.28455898495385545*GTC[38]-0.2021866472040551*GCL[38]+0.28455898495385545*GCC[38]-0.1492199770891953*GTL[27]+0.35674444853774506*GTC[27]-0.1492199770891953*GCL[27]+0.35674444853774506*GCC[27]-0.14051136087662214*GTL[26]+0.14051136087662214*GTC[26]-0.14051136087662214*GCL[26]+0.14051136087662214*GCC[26]+0.2908529064802475*GTL[18]-0.4093485350462743*GTC[18]-0.2908529064802475*GCL[18]+0.4093485350462743*GCC[18]-0.20667569704733044*GTL[10]+0.29087690695550217*GTC[10]-0.20667569704733044*GCL[10]+0.29087690695550217*GCC[10]+0.2021307453761506*GTL[9]-0.2021307453761506*GTC[9]-0.2021307453761506*GCL[9]+0.2021307453761506*GCC[9]-0.14363106492851735*GTL[4]+0.14363106492851735*GTC[4]-0.14363106492851735*GCL[4]+0.14363106492851735*GCC[4]; 
  surft1_upper[4] = (0.19364916731037085*dgdpv1_surf_CC_pv1[12])/dv1_pv1+(0.15*dgdpv1_surf_CC_pv1[4])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[1])/dv1_pv1-0.057810388474953116*GTL[34]-1.2910986759406196*GTC[34]+0.057810388474953116*GCL[34]+1.2910986759406196*GCC[34]+0.07133653706043892*GTL[33]+0.07133653706043892*GTC[33]+0.07133653706043892*GCL[33]+0.07133653706043892*GCC[33]+0.04107919181288744*GTL[23]+0.9174352838211529*GTC[23]+0.04107919181288744*GCL[23]+0.9174352838211529*GCC[23]+0.05616295755668199*GTL[20]-0.05616295755668199*GTC[20]+0.05616295755668199*GCL[20]-0.05616295755668199*GCC[20]-0.10262022457558416*GTL[15]-0.10262022457558416*GTC[15]+0.10262022457558416*GCL[15]+0.10262022457558416*GCC[15]+0.07292038680986264*GTL[6]+0.07292038680986264*GTC[6]+0.07292038680986264*GCL[6]+0.07292038680986264*GCC[6]-0.08079247402229095*GTL[5]+0.08079247402229095*GTC[5]+0.08079247402229095*GCL[5]-0.08079247402229095*GCC[5]+0.057409915846480676*GTL[1]-0.057409915846480676*GTC[1]+0.057409915846480676*GCL[1]-0.057409915846480676*GCC[1]; 
  surft1_upper[5] = (0.11180339887498948*dgdpv1_surf_CC_pv1[18])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[10])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[5])/dv1_pv1+0.20999597273098428*GTL[46]-0.5020433520257306*GTC[46]-0.20999597273098428*GCL[46]+0.5020433520257306*GCC[46]-0.20218664720405521*GTL[45]+0.2845589849538555*GTC[45]-0.20218664720405521*GCL[45]+0.2845589849538555*GCC[45]-0.14921997708919524*GTL[39]+0.35674444853774495*GTC[39]-0.14921997708919524*GCL[39]+0.35674444853774495*GCC[39]-0.14051136087662214*GTL[36]+0.14051136087662214*GTC[36]-0.14051136087662214*GCL[36]+0.14051136087662214*GCC[36]+0.2908529064802475*GTL[31]-0.4093485350462743*GTC[31]-0.2908529064802475*GCL[31]+0.4093485350462743*GCC[31]-0.20667569704733044*GTL[17]+0.29087690695550217*GTC[17]-0.20667569704733044*GCL[17]+0.29087690695550217*GCC[17]+0.2021307453761506*GTL[16]-0.2021307453761506*GTC[16]-0.2021307453761506*GCL[16]+0.2021307453761506*GCC[16]-0.14363106492851735*GTL[8]+0.14363106492851735*GTC[8]-0.14363106492851735*GCL[8]+0.14363106492851735*GCC[8]; 
  surft1_upper[6] = (0.19364916731037085*dgdpv1_surf_CC_pv1[14])/dv1_pv1+(0.15*dgdpv1_surf_CC_pv1[6])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[3])/dv1_pv1-0.057810388474953116*GTL[40]-1.2910986759406196*GTC[40]+0.057810388474953116*GCL[40]+1.2910986759406196*GCC[40]+0.07133653706043892*GTL[38]+0.07133653706043892*GTC[38]+0.07133653706043892*GCL[38]+0.07133653706043892*GCC[38]+0.04107919181288744*GTL[27]+0.9174352838211529*GTC[27]+0.04107919181288744*GCL[27]+0.9174352838211529*GCC[27]+0.05616295755668199*GTL[26]-0.05616295755668199*GTC[26]+0.05616295755668199*GCL[26]-0.05616295755668199*GCC[26]-0.10262022457558416*GTL[18]-0.10262022457558416*GTC[18]+0.10262022457558416*GCL[18]+0.10262022457558416*GCC[18]+0.07292038680986264*GTL[10]+0.07292038680986264*GTC[10]+0.07292038680986264*GCL[10]+0.07292038680986264*GCC[10]-0.08079247402229095*GTL[9]+0.08079247402229095*GTC[9]+0.08079247402229095*GCL[9]-0.08079247402229095*GCC[9]+0.057409915846480676*GTL[4]-0.057409915846480676*GTC[4]+0.057409915846480676*GCL[4]-0.057409915846480676*GCC[4]; 
  surft1_upper[7] = (0.08660254037844385*dgdpv1_surf_CC_pv1[11])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[7])/dv1_pv1+0.2908529064802475*GTL[32]-0.4093485350462743*GTC[32]-0.2908529064802475*GCL[32]+0.4093485350462743*GCC[32]-0.2066756970473305*GTL[21]+0.2908769069555022*GTC[21]-0.2066756970473305*GCL[21]+0.2908769069555022*GCC[21]+0.2021307453761506*GTL[19]-0.2021307453761506*GTC[19]-0.2021307453761506*GCL[19]+0.2021307453761506*GCC[19]-0.14363106492851735*GTL[11]+0.14363106492851735*GTC[11]-0.14363106492851735*GCL[11]+0.14363106492851735*GCC[11]; 
  surft1_upper[8] = (0.25*dgdpv1_surf_CC_pv1[8])/dv1_pv1+(0.19364916731037082*dgdpv1_surf_CC_pv1[2])/dv1_pv1+(0.11180339887498948*dgdpv1_surf_CC_pv1[0])/dv1_pv1+0.4695652700276729*GTL[24]-1.1226030627813903*GTC[24]-0.4695652700276729*GCL[24]+1.1226030627813903*GCC[24]-0.4521030872910352*GTL[22]-0.7032714691193882*GTC[22]-0.4521030872910352*GCL[22]-0.7032714691193882*GCC[22]-0.3336660123724018*GTL[13]+0.7977048375260732*GTC[13]-0.3336660123724018*GCL[13]+0.7977048375260732*GCC[13]-0.3141929545311315*GTL[12]+0.3141929545311315*GTC[12]-0.3141929545311315*GCL[12]+0.3141929545311315*GCC[12]+0.6503668703432224*GTL[7]+1.0116817983116795*GTC[7]-0.6503668703432224*GCL[7]-1.0116817983116795*GCC[7]-0.46214090789498335*GTL[3]-0.71888585672553*GTC[3]-0.46214090789498335*GCL[3]-0.71888585672553*GCC[3]+0.45197808700377406*GTL[2]-0.45197808700377406*GTC[2]-0.45197808700377406*GCL[2]+0.45197808700377406*GCC[2]-0.3211688248608508*GTL[0]+0.3211688248608508*GTC[0]-0.3211688248608508*GCL[0]+0.3211688248608508*GCC[0]; 
  surft1_upper[9] = (0.08660254037844385*dgdpv1_surf_CC_pv1[16])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[9])/dv1_pv1+0.2908529064802475*GTL[43]-0.4093485350462743*GTC[43]-0.2908529064802475*GCL[43]+0.4093485350462743*GCC[43]-0.2066756970473305*GTL[30]+0.2908769069555022*GTC[30]-0.2066756970473305*GCL[30]+0.2908769069555022*GCC[30]+0.2021307453761506*GTL[29]-0.2021307453761506*GTC[29]-0.2021307453761506*GCL[29]+0.2021307453761506*GCC[29]-0.14363106492851735*GTL[14]+0.14363106492851735*GTC[14]-0.14363106492851735*GCL[14]+0.14363106492851735*GCC[14]; 
  surft1_upper[10] = (0.19364916731037082*dgdpv1_surf_CC_pv1[18])/dv1_pv1+(0.15*dgdpv1_surf_CC_pv1[10])/dv1_pv1+(0.08660254037844387*dgdpv1_surf_CC_pv1[5])/dv1_pv1-0.05781038847495312*GTL[46]-1.2910986759406198*GTC[46]+0.05781038847495312*GCL[46]+1.2910986759406198*GCC[46]+0.07133653706043892*GTL[45]+0.07133653706043892*GTC[45]+0.07133653706043892*GCL[45]+0.07133653706043892*GCC[45]+0.04107919181288743*GTL[39]+0.9174352838211527*GTC[39]+0.04107919181288743*GCL[39]+0.9174352838211527*GCC[39]+0.05616295755668199*GTL[36]-0.05616295755668199*GTC[36]+0.05616295755668199*GCL[36]-0.05616295755668199*GCC[36]-0.10262022457558416*GTL[31]-0.10262022457558416*GTC[31]+0.10262022457558416*GCL[31]+0.10262022457558416*GCC[31]+0.07292038680986264*GTL[17]+0.07292038680986264*GTC[17]+0.07292038680986264*GCL[17]+0.07292038680986264*GCC[17]-0.08079247402229095*GTL[16]+0.08079247402229095*GTC[16]+0.08079247402229095*GCL[16]-0.08079247402229095*GCC[16]+0.057409915846480676*GTL[8]-0.057409915846480676*GTC[8]+0.057409915846480676*GCL[8]-0.057409915846480676*GCC[8]; 
  surft1_upper[11] = (0.15*dgdpv1_surf_CC_pv1[11])/dv1_pv1+(0.08660254037844385*dgdpv1_surf_CC_pv1[7])/dv1_pv1-0.10262022457558415*GTL[32]-0.10262022457558415*GTC[32]+0.10262022457558415*GCL[32]+0.10262022457558415*GCC[32]+0.07292038680986264*GTL[21]+0.07292038680986264*GTC[21]+0.07292038680986264*GCL[21]+0.07292038680986264*GCC[21]-0.08079247402229095*GTL[19]+0.08079247402229095*GTC[19]+0.08079247402229095*GCL[19]-0.08079247402229095*GCC[19]+0.05740991584648068*GTL[11]-0.05740991584648068*GTC[11]+0.05740991584648068*GCL[11]-0.05740991584648068*GCC[11]; 
  surft1_upper[12] = (0.25*dgdpv1_surf_CC_pv1[12])/dv1_pv1+(0.19364916731037085*dgdpv1_surf_CC_pv1[4])/dv1_pv1+(0.11180339887498951*dgdpv1_surf_CC_pv1[1])/dv1_pv1+0.4695652700276729*GTL[34]-1.1226030627813903*GTC[34]-0.4695652700276729*GCL[34]+1.1226030627813903*GCC[34]-0.4521030872910352*GTL[33]-0.7032714691193882*GTC[33]-0.4521030872910352*GCL[33]-0.7032714691193882*GCC[33]-0.3336660123724018*GTL[23]+0.7977048375260732*GTC[23]-0.3336660123724018*GCL[23]+0.7977048375260732*GCC[23]-0.3141929545311315*GTL[20]+0.3141929545311315*GTC[20]-0.3141929545311315*GCL[20]+0.3141929545311315*GCC[20]+0.6503668703432222*GTL[15]+1.0116817983116793*GTC[15]-0.6503668703432222*GCL[15]-1.0116817983116793*GCC[15]-0.46214090789498363*GTL[6]-0.7188858567255303*GTC[6]-0.46214090789498363*GCL[6]-0.7188858567255303*GCC[6]+0.45197808700377406*GTL[5]-0.45197808700377406*GTC[5]-0.45197808700377406*GCL[5]+0.45197808700377406*GCC[5]-0.3211688248608508*GTL[1]+0.3211688248608508*GTC[1]-0.3211688248608508*GCL[1]+0.3211688248608508*GCC[1]; 
  surft1_upper[13] = (0.08660254037844385*dgdpv1_surf_CC_pv1[17])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[13])/dv1_pv1+0.2908529064802475*GTL[44]-0.4093485350462743*GTC[44]-0.2908529064802475*GCL[44]+0.4093485350462743*GCC[44]-0.2066756970473305*GTL[37]+0.2908769069555022*GTC[37]-0.2066756970473305*GCL[37]+0.2908769069555022*GCC[37]+0.2021307453761506*GTL[35]-0.2021307453761506*GTC[35]-0.2021307453761506*GCL[35]+0.2021307453761506*GCC[35]-0.14363106492851735*GTL[25]+0.14363106492851735*GTC[25]-0.14363106492851735*GCL[25]+0.14363106492851735*GCC[25]; 
  surft1_upper[14] = (0.25*dgdpv1_surf_CC_pv1[14])/dv1_pv1+(0.19364916731037085*dgdpv1_surf_CC_pv1[6])/dv1_pv1+(0.11180339887498951*dgdpv1_surf_CC_pv1[3])/dv1_pv1+0.4695652700276729*GTL[40]-1.1226030627813903*GTC[40]-0.4695652700276729*GCL[40]+1.1226030627813903*GCC[40]-0.4521030872910352*GTL[38]-0.7032714691193882*GTC[38]-0.4521030872910352*GCL[38]-0.7032714691193882*GCC[38]-0.3336660123724018*GTL[27]+0.7977048375260732*GTC[27]-0.3336660123724018*GCL[27]+0.7977048375260732*GCC[27]-0.3141929545311315*GTL[26]+0.3141929545311315*GTC[26]-0.3141929545311315*GCL[26]+0.3141929545311315*GCC[26]+0.6503668703432222*GTL[18]+1.0116817983116793*GTC[18]-0.6503668703432222*GCL[18]-1.0116817983116793*GCC[18]-0.46214090789498363*GTL[10]-0.7188858567255303*GTC[10]-0.46214090789498363*GCL[10]-0.7188858567255303*GCC[10]+0.45197808700377406*GTL[9]-0.45197808700377406*GTC[9]-0.45197808700377406*GCL[9]+0.45197808700377406*GCC[9]-0.3211688248608508*GTL[4]+0.3211688248608508*GTC[4]-0.3211688248608508*GCL[4]+0.3211688248608508*GCC[4]; 
  surft1_upper[15] = (0.08660254037844385*dgdpv1_surf_CC_pv1[19])/dv1_pv1+(0.05*dgdpv1_surf_CC_pv1[15])/dv1_pv1+0.2908529064802475*GTL[47]-0.4093485350462743*GTC[47]-0.2908529064802475*GCL[47]+0.4093485350462743*GCC[47]-0.2066756970473305*GTL[42]+0.2908769069555022*GTC[42]-0.2066756970473305*GCL[42]+0.2908769069555022*GCC[42]+0.2021307453761506*GTL[41]-0.2021307453761506*GTC[41]-0.2021307453761506*GCL[41]+0.2021307453761506*GCC[41]-0.14363106492851735*GTL[28]+0.14363106492851735*GTC[28]-0.14363106492851735*GCL[28]+0.14363106492851735*GCC[28]; 
  surft1_upper[16] = (0.15*dgdpv1_surf_CC_pv1[16])/dv1_pv1+(0.08660254037844385*dgdpv1_surf_CC_pv1[9])/dv1_pv1-0.10262022457558415*GTL[43]-0.10262022457558415*GTC[43]+0.10262022457558415*GCL[43]+0.10262022457558415*GCC[43]+0.07292038680986264*GTL[30]+0.07292038680986264*GTC[30]+0.07292038680986264*GCL[30]+0.07292038680986264*GCC[30]-0.08079247402229095*GTL[29]+0.08079247402229095*GTC[29]+0.08079247402229095*GCL[29]-0.08079247402229095*GCC[29]+0.05740991584648068*GTL[14]-0.05740991584648068*GTC[14]+0.05740991584648068*GCL[14]-0.05740991584648068*GCC[14]; 
  surft1_upper[17] = (0.15*dgdpv1_surf_CC_pv1[17])/dv1_pv1+(0.08660254037844385*dgdpv1_surf_CC_pv1[13])/dv1_pv1-0.10262022457558415*GTL[44]-0.10262022457558415*GTC[44]+0.10262022457558415*GCL[44]+0.10262022457558415*GCC[44]+0.07292038680986264*GTL[37]+0.07292038680986264*GTC[37]+0.07292038680986264*GCL[37]+0.07292038680986264*GCC[37]-0.08079247402229095*GTL[35]+0.08079247402229095*GTC[35]+0.08079247402229095*GCL[35]-0.08079247402229095*GCC[35]+0.05740991584648068*GTL[25]-0.05740991584648068*GTC[25]+0.05740991584648068*GCL[25]-0.05740991584648068*GCC[25]; 
  surft1_upper[18] = (0.25*dgdpv1_surf_CC_pv1[18])/dv1_pv1+(0.19364916731037082*dgdpv1_surf_CC_pv1[10])/dv1_pv1+(0.11180339887498948*dgdpv1_surf_CC_pv1[5])/dv1_pv1+0.4695652700276729*GTL[46]-1.1226030627813903*GTC[46]-0.4695652700276729*GCL[46]+1.1226030627813903*GCC[46]-0.4521030872910352*GTL[45]-0.7032714691193882*GTC[45]-0.4521030872910352*GCL[45]-0.7032714691193882*GCC[45]-0.3336660123724018*GTL[39]+0.7977048375260732*GTC[39]-0.3336660123724018*GCL[39]+0.7977048375260732*GCC[39]-0.3141929545311315*GTL[36]+0.3141929545311315*GTC[36]-0.3141929545311315*GCL[36]+0.3141929545311315*GCC[36]+0.6503668703432224*GTL[31]+1.0116817983116795*GTC[31]-0.6503668703432224*GCL[31]-1.0116817983116795*GCC[31]-0.46214090789498335*GTL[17]-0.71888585672553*GTC[17]-0.46214090789498335*GCL[17]-0.71888585672553*GCC[17]+0.45197808700377406*GTL[16]-0.45197808700377406*GTC[16]-0.45197808700377406*GCL[16]+0.45197808700377406*GCC[16]-0.3211688248608508*GTL[8]+0.3211688248608508*GTC[8]-0.3211688248608508*GCL[8]+0.3211688248608508*GCC[8]; 
  surft1_upper[19] = (0.15*dgdpv1_surf_CC_pv1[19])/dv1_pv1+(0.08660254037844385*dgdpv1_surf_CC_pv1[15])/dv1_pv1-0.10262022457558415*GTL[47]-0.10262022457558415*GTC[47]+0.10262022457558415*GCL[47]+0.10262022457558415*GCC[47]+0.07292038680986264*GTL[42]+0.07292038680986264*GTC[42]+0.07292038680986264*GCL[42]+0.07292038680986264*GCC[42]-0.08079247402229095*GTL[41]+0.08079247402229095*GTC[41]+0.08079247402229095*GCL[41]-0.08079247402229095*GCC[41]+0.05740991584648068*GTL[28]-0.05740991584648068*GTC[28]+0.05740991584648068*GCL[28]-0.05740991584648068*GCC[28]; 
  surft1_lower[0] = dgdpv1_surf_CC_pv2[0]/dv1_pv1; 
  surft1_lower[1] = dgdpv1_surf_CC_pv2[1]/dv1_pv1; 
  surft1_lower[2] = dgdpv1_surf_CC_pv2[2]/dv1_pv1; 
  surft1_lower[3] = dgdpv1_surf_CC_pv2[3]/dv1_pv1; 
  surft1_lower[4] = dgdpv1_surf_CC_pv2[4]/dv1_pv1; 
  surft1_lower[5] = dgdpv1_surf_CC_pv2[5]/dv1_pv1; 
  surft1_lower[6] = dgdpv1_surf_CC_pv2[6]/dv1_pv1; 
  surft1_lower[7] = dgdpv1_surf_CC_pv2[7]/dv1_pv1; 
  surft1_lower[8] = dgdpv1_surf_CC_pv2[8]/dv1_pv1; 
  surft1_lower[9] = dgdpv1_surf_CC_pv2[9]/dv1_pv1; 
  surft1_lower[10] = dgdpv1_surf_CC_pv2[10]/dv1_pv1; 
  surft1_lower[11] = dgdpv1_surf_CC_pv2[11]/dv1_pv1; 
  surft1_lower[12] = dgdpv1_surf_CC_pv2[12]/dv1_pv1; 
  surft1_lower[13] = dgdpv1_surf_CC_pv2[13]/dv1_pv1; 
  surft1_lower[14] = dgdpv1_surf_CC_pv2[14]/dv1_pv1; 
  surft1_lower[15] = dgdpv1_surf_CC_pv2[15]/dv1_pv1; 
  surft1_lower[16] = dgdpv1_surf_CC_pv2[16]/dv1_pv1; 
  surft1_lower[17] = dgdpv1_surf_CC_pv2[17]/dv1_pv1; 
  surft1_lower[18] = dgdpv1_surf_CC_pv2[18]/dv1_pv1; 
  surft1_lower[19] = dgdpv1_surf_CC_pv2[19]/dv1_pv1; 

  surft2_upper[0] = g_surf_CC_pv1[0]; 
  surft2_upper[1] = g_surf_CC_pv1[1]; 
  surft2_upper[2] = g_surf_CC_pv1[2]; 
  surft2_upper[3] = g_surf_CC_pv1[3]; 
  surft2_upper[4] = g_surf_CC_pv1[4]; 
  surft2_upper[5] = g_surf_CC_pv1[5]; 
  surft2_upper[6] = g_surf_CC_pv1[6]; 
  surft2_upper[7] = g_surf_CC_pv1[7]; 
  surft2_upper[8] = g_surf_CC_pv1[8]; 
  surft2_upper[9] = g_surf_CC_pv1[9]; 
  surft2_upper[10] = g_surf_CC_pv1[10]; 
  surft2_upper[11] = g_surf_CC_pv1[11]; 
  surft2_upper[12] = g_surf_CC_pv1[12]; 
  surft2_upper[13] = g_surf_CC_pv1[13]; 
  surft2_upper[14] = g_surf_CC_pv1[14]; 
  surft2_upper[15] = g_surf_CC_pv1[15]; 
  surft2_upper[16] = g_surf_CC_pv1[16]; 
  surft2_upper[17] = g_surf_CC_pv1[17]; 
  surft2_upper[18] = g_surf_CC_pv1[18]; 
  surft2_upper[19] = g_surf_CC_pv1[19]; 
  surft2_lower[0] = 0.34587411908091625*GCL[13]+0.34587411908091625*GCC[13]+0.49755260400283263*GCL[3]-0.49755260400283263*GCC[3]+0.3535533905932737*GCL[0]+0.3535533905932737*GCC[0]; 
  surft2_lower[1] = 0.34587411908091625*GCL[23]+0.34587411908091625*GCC[23]+0.49755260400283263*GCL[6]-0.49755260400283263*GCC[6]+0.3535533905932737*GCL[1]+0.3535533905932737*GCC[1]; 
  surft2_lower[2] = 0.34587411908091625*GCL[24]+0.34587411908091625*GCC[24]+0.49755260400283263*GCL[7]-0.49755260400283263*GCC[7]+0.3535533905932737*GCL[2]+0.3535533905932737*GCC[2]; 
  surft2_lower[3] = 0.34587411908091625*GCL[27]+0.34587411908091625*GCC[27]+0.49755260400283263*GCL[10]-0.49755260400283263*GCC[10]+0.3535533905932737*GCL[4]+0.3535533905932737*GCC[4]; 
  surft2_lower[4] = 0.34587411908091625*GCL[34]+0.34587411908091625*GCC[34]+0.49755260400283263*GCL[15]-0.49755260400283263*GCC[15]+0.3535533905932737*GCL[5]+0.3535533905932737*GCC[5]; 
  surft2_lower[5] = 0.34587411908091625*GCL[39]+0.34587411908091625*GCC[39]+0.49755260400283263*GCL[17]-0.49755260400283263*GCC[17]+0.3535533905932737*GCL[8]+0.3535533905932737*GCC[8]; 
  surft2_lower[6] = 0.34587411908091625*GCL[40]+0.34587411908091625*GCC[40]+0.49755260400283263*GCL[18]-0.49755260400283263*GCC[18]+0.3535533905932737*GCL[9]+0.3535533905932737*GCC[9]; 
  surft2_lower[7] = 0.49755260400283263*GCL[21]-0.49755260400283263*GCC[21]+0.3535533905932737*GCL[11]+0.3535533905932737*GCC[11]; 
  surft2_lower[8] = 0.49755260400283263*GCL[22]-0.49755260400283263*GCC[22]+0.3535533905932737*GCL[12]+0.3535533905932737*GCC[12]; 
  surft2_lower[9] = 0.49755260400283263*GCL[30]-0.49755260400283263*GCC[30]+0.3535533905932737*GCL[14]+0.3535533905932737*GCC[14]; 
  surft2_lower[10] = 0.34587411908091625*GCL[46]+0.34587411908091625*GCC[46]+0.49755260400283263*GCL[31]-0.49755260400283263*GCC[31]+0.3535533905932737*GCL[16]+0.3535533905932737*GCC[16]; 
  surft2_lower[11] = 0.49755260400283263*GCL[32]-0.49755260400283263*GCC[32]+0.3535533905932737*GCL[19]+0.3535533905932737*GCC[19]; 
  surft2_lower[12] = 0.49755260400283263*GCL[33]-0.49755260400283263*GCC[33]+0.3535533905932737*GCL[20]+0.3535533905932737*GCC[20]; 
  surft2_lower[13] = 0.49755260400283263*GCL[37]-0.49755260400283263*GCC[37]+0.3535533905932737*GCL[25]+0.3535533905932737*GCC[25]; 
  surft2_lower[14] = 0.49755260400283263*GCL[38]-0.49755260400283263*GCC[38]+0.3535533905932737*GCL[26]+0.3535533905932737*GCC[26]; 
  surft2_lower[15] = 0.49755260400283263*GCL[42]-0.49755260400283263*GCC[42]+0.3535533905932737*GCL[28]+0.3535533905932737*GCC[28]; 
  surft2_lower[16] = 0.49755260400283263*GCL[43]-0.49755260400283263*GCC[43]+0.3535533905932737*GCL[29]+0.3535533905932737*GCC[29]; 
  surft2_lower[17] = 0.49755260400283263*GCL[44]-0.49755260400283263*GCC[44]+0.3535533905932737*GCL[35]+0.3535533905932737*GCC[35]; 
  surft2_lower[18] = 0.49755260400283263*GCL[45]-0.49755260400283263*GCC[45]+0.3535533905932737*GCL[36]+0.3535533905932737*GCC[36]; 
  surft2_lower[19] = 0.49755260400283263*GCL[47]-0.49755260400283263*GCC[47]+0.3535533905932737*GCL[41]+0.3535533905932737*GCC[41]; 

  out[0] = 0.7071067811865475*surft1_upper[0]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[1] = 0.7071067811865475*surft1_upper[1]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[2] = -(1.224744871391589*surft2_upper[0]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[0]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[3] = 0.7071067811865475*surft1_upper[2]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[2]*dv1_sq*gamma_avg; 
  out[4] = 0.7071067811865475*surft1_upper[3]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[5] = -(1.224744871391589*surft2_upper[1]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[1]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[6] = 0.7071067811865475*surft1_upper[4]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[7] = 1.224744871391589*surft1_upper[2]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[2]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[0]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[0]*dv1_sq*gamma_avg+3.0*GCC[0]*dv1_sq*gamma_avg; 
  out[8] = 0.7071067811865475*surft1_upper[5]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[9] = -(1.224744871391589*surft2_upper[3]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[3]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[3]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[10] = 0.7071067811865475*surft1_upper[6]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[11] = 0.7071067811865475*surft1_upper[7]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[7]*dv1_sq*gamma_avg; 
  out[12] = -(2.7386127875258306*surft2_upper[2]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[2]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[0]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[0]*dv1_sq*gamma_avg; 
  out[13] = 0.7071067811865475*surft1_upper[8]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[8]*dv1_sq*gamma_avg; 
  out[14] = 0.7071067811865475*surft1_upper[9]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[9]*dv1_sq*gamma_avg; 
  out[15] = 1.224744871391589*surft1_upper[4]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[4]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[1]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[1]*dv1_sq*gamma_avg+3.0*GCC[1]*dv1_sq*gamma_avg; 
  out[16] = -(1.224744871391589*surft2_upper[5]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[5]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[5]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[17] = 0.7071067811865475*surft1_upper[10]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[10]*dv1_sq*gamma_avg; 
  out[18] = 1.224744871391589*surft1_upper[6]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[6]*dv1_sq*gamma_avg+3.0*GCC[4]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[3]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[3]*dv1_sq*gamma_avg; 
  out[19] = -(1.224744871391589*surft2_upper[7]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[7]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[7]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[7]*dv1_sq*gamma_avg; 
  out[20] = -(2.7386127875258306*surft2_upper[4]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[4]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[1]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[1]*dv1_sq*gamma_avg; 
  out[21] = 0.7071067811865475*surft1_upper[11]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[11]*dv1_sq*gamma_avg; 
  out[22] = -(4.743416490252569*surft2_upper[2]*dv1_sq*gamma_avg)-4.743416490252569*surft2_lower[2]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[2]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[2]*dv1_sq*gamma_avg+6.7082039324993685*GCC[2]*dv1_sq*gamma_avg; 
  out[23] = 0.7071067811865475*surft1_upper[12]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[12]*dv1_sq*gamma_avg; 
  out[24] = 1.224744871391589*surft1_upper[8]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[8]*dv1_sq*gamma_avg+6.7082039324993685*GCC[3]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[0]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[0]*dv1_sq*gamma_avg; 
  out[25] = 0.7071067811865475*surft1_upper[13]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[13]*dv1_sq*gamma_avg; 
  out[26] = -(2.7386127875258306*surft2_upper[6]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[3]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[3]*dv1_sq*gamma_avg; 
  out[27] = 0.7071067811865475*surft1_upper[14]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[14]*dv1_sq*gamma_avg; 
  out[28] = 0.7071067811865475*surft1_upper[15]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[15]*dv1_sq*gamma_avg; 
  out[29] = -(1.224744871391589*surft2_upper[9]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[9]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[9]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[9]*dv1_sq*gamma_avg; 
  out[30] = 0.7071067811865475*surft1_upper[16]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[16]*dv1_sq*gamma_avg; 
  out[31] = 1.224744871391589*surft1_upper[10]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[10]*dv1_sq*gamma_avg+3.0*GCC[8]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[5]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[5]*dv1_sq*gamma_avg; 
  out[32] = 1.224744871391589*surft1_upper[11]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[11]*dv1_sq*gamma_avg+3.0*GCC[11]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[7]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[7]*dv1_sq*gamma_avg; 
  out[33] = 6.708203932499369*GCC[5]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[4]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[4]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[4]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[4]*dv1_sq*gamma_avg; 
  out[34] = 1.224744871391589*surft1_upper[12]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[12]*dv1_sq*gamma_avg+6.708203932499369*GCC[6]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[1]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[1]*dv1_sq*gamma_avg; 
  out[35] = -(1.224744871391589*surft2_upper[13]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[13]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[13]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[13]*dv1_sq*gamma_avg; 
  out[36] = -(2.7386127875258306*surft2_upper[10]*dv1_sq*gamma_avg)+2.7386127875258306*surft2_lower[10]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[5]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[5]*dv1_sq*gamma_avg; 
  out[37] = 0.7071067811865475*surft1_upper[17]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[17]*dv1_sq*gamma_avg; 
  out[38] = 6.708203932499369*GCC[9]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[6]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[6]*dv1_sq*gamma_avg+1.5811388300841895*surft1_upper[6]*dv1_sq*gamma_avg-1.5811388300841895*surft1_lower[6]*dv1_sq*gamma_avg; 
  out[39] = 0.7071067811865475*surft1_upper[18]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[18]*dv1_sq*gamma_avg; 
  out[40] = 1.224744871391589*surft1_upper[14]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[14]*dv1_sq*gamma_avg+6.708203932499369*GCC[10]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[3]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[3]*dv1_sq*gamma_avg; 
  out[41] = -(1.224744871391589*surft2_upper[15]*dv1_sq*gamma_avg)+1.224744871391589*surft2_lower[15]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[15]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[15]*dv1_sq*gamma_avg; 
  out[42] = 0.7071067811865475*surft1_upper[19]*dv1_sq*gamma_avg-0.7071067811865475*surft1_lower[19]*dv1_sq*gamma_avg; 
  out[43] = 1.224744871391589*surft1_upper[16]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[16]*dv1_sq*gamma_avg+3.0*GCC[14]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[9]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[9]*dv1_sq*gamma_avg; 
  out[44] = 3.0*GCC[25]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[17]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[17]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[13]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[13]*dv1_sq*gamma_avg; 
  out[45] = 6.7082039324993685*GCC[16]*dv1_sq*gamma_avg-4.743416490252569*surft2_upper[10]*dv1_sq*gamma_avg-4.743416490252569*surft2_lower[10]*dv1_sq*gamma_avg+1.5811388300841898*surft1_upper[10]*dv1_sq*gamma_avg-1.5811388300841898*surft1_lower[10]*dv1_sq*gamma_avg; 
  out[46] = 1.224744871391589*surft1_upper[18]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[18]*dv1_sq*gamma_avg+6.7082039324993685*GCC[17]*dv1_sq*gamma_avg-2.7386127875258306*surft2_upper[5]*dv1_sq*gamma_avg+2.7386127875258306*surft2_lower[5]*dv1_sq*gamma_avg; 
  out[47] = 3.0*GCC[28]*dv1_sq*gamma_avg+1.224744871391589*surft1_upper[19]*dv1_sq*gamma_avg+1.224744871391589*surft1_lower[19]*dv1_sq*gamma_avg-2.1213203435596424*surft2_upper[15]*dv1_sq*gamma_avg-2.1213203435596424*surft2_lower[15]*dv1_sq*gamma_avg; 
} 
