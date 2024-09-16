#include <gkyl_fpo_vlasov_kernels.h> 

GKYL_CU_DH void fpo_drag_coeff_1x3v_vz_ser_p1_lovz(const double *dxv, const double *gamma, 
    const double* fpo_h_stencil[3], const double* fpo_dhdv_surf, double *drag_coeff, 
    double *drag_coeff_surf) {
  // dxv[NDIM]: Cell spacing in each direction. 
  // gamma: Scalar factor gamma. 
  // fpo_h_stencil[3]: 3 cell stencil of Rosenbluth potential H. 
  // fpo_dhdv_surf: Surface projection of dH/dv in center cell. 
  // drag_coeff: Output array for drag coefficient. 
  // drag_coeff_surf: Surface projection of drag coefficient at lower boundary.

  // Use cell-average value for gamma. 
  double gamma_avg = gamma[0]/sqrt(pow(2, 1)); 
  double dv1 = 2.0/dxv[3]; 

  const double* H_L = fpo_h_stencil[0]; 
  const double* H_C = fpo_h_stencil[1]; 
  const double* H_R = fpo_h_stencil[2]; 
  
  const double *dHdv_surf_C = &fpo_dhdv_surf[16]; 
  
  double *out = &drag_coeff[80]; 
  double *out_surf = &drag_coeff_surf[16]; 
  
  out[0] = (-(0.180421959121758*H_R[4])+2.70632938682637*H_C[4]+0.1875*H_R[0]-0.1875*H_C[0])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[0]*gamma_avg; 
  out[1] = (-(0.180421959121758*H_R[8])+2.70632938682637*H_C[8]+0.1875*H_R[1]-0.1875*H_C[1])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[1]*gamma_avg; 
  out[2] = (-(0.180421959121758*H_R[9])+2.70632938682637*H_C[9]+0.1875*H_R[2]-0.1875*H_C[2])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[2]*gamma_avg; 
  out[3] = (-(0.180421959121758*H_R[10])+2.70632938682637*H_C[10]+0.1875*H_R[3]-0.1875*H_C[3])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[3]*gamma_avg; 
  out[4] = (-(1.6875*H_R[4])-2.6875*H_C[4]+1.4072912811497125*H_R[0]-1.4072912811497125*H_C[0])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[0]*gamma_avg; 
  out[5] = (-(0.180421959121758*H_R[12])+2.70632938682637*H_C[12]+0.1875*H_R[5]-0.1875*H_C[5])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[4]*gamma_avg; 
  out[6] = (-(0.180421959121758*H_R[13])+2.70632938682637*H_C[13]+0.1875*H_R[6]-0.1875*H_C[6])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[5]*gamma_avg; 
  out[7] = (-(0.180421959121758*H_R[14])+2.70632938682637*H_C[14]+0.1875*H_R[7]-0.1875*H_C[7])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[6]*gamma_avg; 
  out[8] = (-(1.6875*H_R[8])-2.6875*H_C[8]+1.4072912811497125*H_R[1]-1.4072912811497125*H_C[1])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[1]*gamma_avg; 
  out[9] = (-(1.6875*H_R[9])-2.6875*H_C[9]+1.4072912811497125*H_R[2]-1.4072912811497125*H_C[2])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[2]*gamma_avg; 
  out[10] = (-(1.6875*H_R[10])-2.6875*H_C[10]+1.4072912811497125*H_R[3]-1.4072912811497125*H_C[3])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[3]*gamma_avg; 
  out[11] = (-(0.180421959121758*H_R[15])+2.70632938682637*H_C[15]+0.1875*H_R[11]-0.1875*H_C[11])*dv1*gamma_avg+0.23570226039551595*dHdv_surf_C[7]*gamma_avg; 
  out[12] = (-(1.6875*H_R[12])-2.6875*H_C[12]+1.4072912811497125*H_R[5]-1.4072912811497125*H_C[5])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[4]*gamma_avg; 
  out[13] = (-(1.6875*H_R[13])-2.6875*H_C[13]+1.4072912811497125*H_R[6]-1.4072912811497125*H_C[6])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[5]*gamma_avg; 
  out[14] = (-(1.6875*H_R[14])-2.6875*H_C[14]+1.4072912811497125*H_R[7]-1.4072912811497125*H_C[7])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[6]*gamma_avg; 
  out[15] = (-(1.6875*H_R[15])-2.6875*H_C[15]+1.4072912811497125*H_R[11]-1.4072912811497125*H_C[11])*dv1*gamma_avg-0.4082482904638632*dHdv_surf_C[7]*gamma_avg; 
  out[32] = (-(0.4034357652299393*H_R[4])-1.6944302139657443*H_C[4]+0.4192627457812106*H_R[0]-0.4192627457812106*H_C[0])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[0]*gamma_avg; 
  out[33] = (-(0.4034357652299393*H_R[8])-1.694430213965745*H_C[8]+0.4192627457812105*H_R[1]-0.4192627457812105*H_C[1])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[1]*gamma_avg; 
  out[34] = (-(0.4034357652299393*H_R[9])-1.694430213965745*H_C[9]+0.4192627457812105*H_R[2]-0.4192627457812105*H_C[2])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[2]*gamma_avg; 
  out[35] = (-(0.4034357652299393*H_R[10])-1.694430213965745*H_C[10]+0.4192627457812105*H_R[3]-0.4192627457812105*H_C[3])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[3]*gamma_avg; 
  out[36] = (-(0.4034357652299393*H_R[12])-1.6944302139657443*H_C[12]+0.4192627457812106*H_R[5]-0.4192627457812106*H_C[5])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[4]*gamma_avg; 
  out[37] = (-(0.4034357652299393*H_R[13])-1.6944302139657443*H_C[13]+0.4192627457812106*H_R[6]-0.4192627457812106*H_C[6])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[5]*gamma_avg; 
  out[38] = (-(0.4034357652299393*H_R[14])-1.6944302139657443*H_C[14]+0.4192627457812106*H_R[7]-0.4192627457812106*H_C[7])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[6]*gamma_avg; 
  out[39] = (-(0.4034357652299393*H_R[15])-1.694430213965745*H_C[15]+0.4192627457812105*H_R[11]-0.4192627457812105*H_C[11])*dv1*gamma_avg+0.5270462766947301*dHdv_surf_C[7]*gamma_avg; 

  out_surf[0] = 0.0; 
  out_surf[1] = 0.0; 
  out_surf[2] = 0.0; 
  out_surf[3] = 0.0; 
  out_surf[4] = 0.0; 
  out_surf[5] = 0.0; 
  out_surf[6] = 0.0; 
  out_surf[7] = 0.0; 
}

