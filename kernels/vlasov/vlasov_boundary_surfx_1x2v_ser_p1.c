#include <gkyl_vlasov_kernels.h> 
GKYL_CU_DH double vlasov_boundary_surfx_1x2v_ser_p1(const double *w, const double *dxv, 
  const double *alpha_surf_edge, const double *alpha_surf_skin, 
  const double *sgn_alpha_surf_edge, const double *sgn_alpha_surf_skin, 
  const int *const_sgn_alpha_edge, const int *const_sgn_alpha_skin, 
  const int edge, const double *fedge, const double *fskin, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: cell-center.
  // dxv[NDIM]: cell length.
  // alpha_surf_edge: Surface expansion of phase space flux on the lower edges of the edge cell (used by general geometry version).
  // alpha_surf_skin: Surface expansion of phase space flux on the lower edges of the skin cell (used by general geometry version).
  // sgn_alpha_surf_edge: sign(alpha_surf_edge) at quadrature points (used by general geometry version).
  // sgn_alpha_surf_skin: sign(alpha_surf_skin) at quadrature points (used by general geometry version).
  // const_sgn_alpha_edge: Boolean array true if sign(alpha_surf_edge) is only one sign, either +1 or -1 (used by general geometry version).
  // const_sgn_alpha_skin: Boolean array true if sign(alpha_surf_skin) is only one sign, either +1 or -1 (used by general geometry version).
  // edge: determines if the update is for the left edge (-1) or right edge (+1).
  // fskin,fedge: distribution function in skin cell/last edge cell.
  // out: output increment in center cell.

  const double dx10 = 2/dxv[0]; 
  const double dv = dxv[1], wv = w[1]; 
  double Ghat[8]; 

  if (edge == -1) { 

  if (wv>0) { 

  Ghat[0] = (1.224744871391589*fskin[1]+0.7071067811865475*fskin[0])*wv+(0.3535533905932737*fskin[4]+0.2041241452319315*fskin[2])*dv; 
  Ghat[1] = (1.224744871391589*fskin[4]+0.7071067811865475*fskin[2])*wv+(0.3162277660168379*fskin[9]+0.1825741858350554*fskin[8]+0.3535533905932737*fskin[1]+0.2041241452319315*fskin[0])*dv; 
  Ghat[2] = (1.224744871391589*fskin[5]+0.7071067811865475*fskin[3])*wv+(0.3535533905932737*fskin[7]+0.2041241452319315*fskin[6])*dv; 
  Ghat[3] = (1.224744871391589*fskin[7]+0.7071067811865475*fskin[6])*wv+(0.3162277660168379*fskin[11]+0.1825741858350554*fskin[10]+0.3535533905932737*fskin[5]+0.2041241452319315*fskin[3])*dv; 
  Ghat[4] = (1.224744871391589*fskin[9]+0.7071067811865475*fskin[8])*wv+(0.3162277660168379*fskin[4]+0.1825741858350554*fskin[2])*dv; 
  Ghat[5] = (1.224744871391589*fskin[13]+0.7071067811865475*fskin[12])*wv+(0.3535533905932737*fskin[15]+0.2041241452319315*fskin[14])*dv; 
  Ghat[6] = (1.224744871391589*fskin[11]+0.7071067811865475*fskin[10])*wv+(0.3162277660168379*fskin[7]+0.1825741858350554*fskin[6])*dv; 
  Ghat[7] = (1.224744871391589*fskin[15]+0.7071067811865475*fskin[14])*wv+(0.3535533905932737*fskin[13]+0.2041241452319315*fskin[12])*dv; 

  } else { 

  Ghat[0] = -0.08333333333333333*((14.69693845669907*fedge[1]-8.485281374238571*fedge[0])*wv+(4.242640687119286*fedge[4]-2.449489742783178*fedge[2])*dv); 
  Ghat[1] = -0.01666666666666667*((73.48469228349535*fedge[4]-42.42640687119286*fedge[2])*wv+(18.97366596101028*fedge[9]-10.95445115010332*fedge[8]+21.21320343559643*fedge[1]-12.24744871391589*fedge[0])*dv); 
  Ghat[2] = -0.08333333333333333*((14.69693845669907*fedge[5]-8.485281374238571*fedge[3])*wv+(4.242640687119286*fedge[7]-2.449489742783178*fedge[6])*dv); 
  Ghat[3] = -0.01666666666666667*((73.48469228349535*fedge[7]-42.42640687119286*fedge[6])*wv+(18.97366596101028*fedge[11]-10.95445115010333*fedge[10]+21.21320343559643*fedge[5]-12.24744871391589*fedge[3])*dv); 
  Ghat[4] = -0.03333333333333333*((36.74234614174768*fedge[9]-21.21320343559643*fedge[8])*wv+(9.48683298050514*fedge[4]-5.477225575051662*fedge[2])*dv); 
  Ghat[5] = -0.01666666666666667*((73.48469228349536*fedge[13]-42.42640687119286*fedge[12])*wv+(21.21320343559643*fedge[15]-12.24744871391589*fedge[14])*dv); 
  Ghat[6] = -0.03333333333333333*((36.74234614174768*fedge[11]-21.21320343559643*fedge[10])*wv+(9.48683298050514*fedge[7]-5.477225575051662*fedge[6])*dv); 
  Ghat[7] = -0.01666666666666667*((73.48469228349536*fedge[15]-42.42640687119286*fedge[14])*wv+(21.21320343559643*fedge[13]-12.24744871391589*fedge[12])*dv); 

  } 

  out[0] += -0.7071067811865475*Ghat[0]*dx10; 
  out[1] += -1.224744871391589*Ghat[0]*dx10; 
  out[2] += -0.7071067811865475*Ghat[1]*dx10; 
  out[3] += -0.7071067811865475*Ghat[2]*dx10; 
  out[4] += -1.224744871391589*Ghat[1]*dx10; 
  out[5] += -1.224744871391589*Ghat[2]*dx10; 
  out[6] += -0.7071067811865475*Ghat[3]*dx10; 
  out[7] += -1.224744871391589*Ghat[3]*dx10; 
  out[8] += -0.7071067811865475*Ghat[4]*dx10; 
  out[9] += -1.224744871391589*Ghat[4]*dx10; 
  out[10] += -0.7071067811865475*Ghat[6]*dx10; 
  out[11] += -1.224744871391589*Ghat[6]*dx10; 
  out[12] += -0.7071067811865475*Ghat[5]*dx10; 
  out[13] += -1.224744871391589*Ghat[5]*dx10; 
  out[14] += -0.7071067811865475*Ghat[7]*dx10; 
  out[15] += -1.224744871391589*Ghat[7]*dx10; 

  } else { 

  if (wv>0) { 

  Ghat[0] = (1.224744871391589*fedge[1]+0.7071067811865475*fedge[0])*wv+(0.3535533905932737*fedge[4]+0.2041241452319315*fedge[2])*dv; 
  Ghat[1] = (1.224744871391589*fedge[4]+0.7071067811865475*fedge[2])*wv+(0.3162277660168379*fedge[9]+0.1825741858350554*fedge[8]+0.3535533905932737*fedge[1]+0.2041241452319315*fedge[0])*dv; 
  Ghat[2] = (1.224744871391589*fedge[5]+0.7071067811865475*fedge[3])*wv+(0.3535533905932737*fedge[7]+0.2041241452319315*fedge[6])*dv; 
  Ghat[3] = (1.224744871391589*fedge[7]+0.7071067811865475*fedge[6])*wv+(0.3162277660168379*fedge[11]+0.1825741858350554*fedge[10]+0.3535533905932737*fedge[5]+0.2041241452319315*fedge[3])*dv; 
  Ghat[4] = (1.224744871391589*fedge[9]+0.7071067811865475*fedge[8])*wv+(0.3162277660168379*fedge[4]+0.1825741858350554*fedge[2])*dv; 
  Ghat[5] = (1.224744871391589*fedge[13]+0.7071067811865475*fedge[12])*wv+(0.3535533905932737*fedge[15]+0.2041241452319315*fedge[14])*dv; 
  Ghat[6] = (1.224744871391589*fedge[11]+0.7071067811865475*fedge[10])*wv+(0.3162277660168379*fedge[7]+0.1825741858350554*fedge[6])*dv; 
  Ghat[7] = (1.224744871391589*fedge[15]+0.7071067811865475*fedge[14])*wv+(0.3535533905932737*fedge[13]+0.2041241452319315*fedge[12])*dv; 

  } else { 

  Ghat[0] = -0.08333333333333333*((14.69693845669907*fskin[1]-8.485281374238571*fskin[0])*wv+(4.242640687119286*fskin[4]-2.449489742783178*fskin[2])*dv); 
  Ghat[1] = -0.01666666666666667*((73.48469228349535*fskin[4]-42.42640687119286*fskin[2])*wv+(18.97366596101028*fskin[9]-10.95445115010332*fskin[8]+21.21320343559643*fskin[1]-12.24744871391589*fskin[0])*dv); 
  Ghat[2] = -0.08333333333333333*((14.69693845669907*fskin[5]-8.485281374238571*fskin[3])*wv+(4.242640687119286*fskin[7]-2.449489742783178*fskin[6])*dv); 
  Ghat[3] = -0.01666666666666667*((73.48469228349535*fskin[7]-42.42640687119286*fskin[6])*wv+(18.97366596101028*fskin[11]-10.95445115010333*fskin[10]+21.21320343559643*fskin[5]-12.24744871391589*fskin[3])*dv); 
  Ghat[4] = -0.03333333333333333*((36.74234614174768*fskin[9]-21.21320343559643*fskin[8])*wv+(9.48683298050514*fskin[4]-5.477225575051662*fskin[2])*dv); 
  Ghat[5] = -0.01666666666666667*((73.48469228349536*fskin[13]-42.42640687119286*fskin[12])*wv+(21.21320343559643*fskin[15]-12.24744871391589*fskin[14])*dv); 
  Ghat[6] = -0.03333333333333333*((36.74234614174768*fskin[11]-21.21320343559643*fskin[10])*wv+(9.48683298050514*fskin[7]-5.477225575051662*fskin[6])*dv); 
  Ghat[7] = -0.01666666666666667*((73.48469228349536*fskin[15]-42.42640687119286*fskin[14])*wv+(21.21320343559643*fskin[13]-12.24744871391589*fskin[12])*dv); 

  } 

  out[0] += 0.7071067811865475*Ghat[0]*dx10; 
  out[1] += -1.224744871391589*Ghat[0]*dx10; 
  out[2] += 0.7071067811865475*Ghat[1]*dx10; 
  out[3] += 0.7071067811865475*Ghat[2]*dx10; 
  out[4] += -1.224744871391589*Ghat[1]*dx10; 
  out[5] += -1.224744871391589*Ghat[2]*dx10; 
  out[6] += 0.7071067811865475*Ghat[3]*dx10; 
  out[7] += -1.224744871391589*Ghat[3]*dx10; 
  out[8] += 0.7071067811865475*Ghat[4]*dx10; 
  out[9] += -1.224744871391589*Ghat[4]*dx10; 
  out[10] += 0.7071067811865475*Ghat[6]*dx10; 
  out[11] += -1.224744871391589*Ghat[6]*dx10; 
  out[12] += 0.7071067811865475*Ghat[5]*dx10; 
  out[13] += -1.224744871391589*Ghat[5]*dx10; 
  out[14] += 0.7071067811865475*Ghat[7]*dx10; 
  out[15] += -1.224744871391589*Ghat[7]*dx10; 

  } 
  return 0.;

} 
