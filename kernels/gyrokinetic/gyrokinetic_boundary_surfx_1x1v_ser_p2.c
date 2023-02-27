#include <gkyl_gyrokinetic_kernels.h>
#include <gkyl_basis_ser_2x_p2_surfx1_eval_quad.h> 
#include <gkyl_basis_ser_2x_p2_upwind_quad_to_modal.h> 
GKYL_CU_DH void gyrokinetic_boundary_surfx_1x1v_ser_p2(const double *w, const double *dxv, const double q_, const double m_, const double *bmag, const double *jacobtot_inv, const double *cmag, const double *b_i, const double *phi, const double *apar, const double *apardot, const int edge, const double *fedge, const double *fskin, double* GKYL_RESTRICT out) 
{ 
  // w[NDIM]: cell-center.
  // dxv[NDIM]: cell length.
  // q_,m_: species charge and mass.
  // bmag: magnetic field amplitude.
  // jacobtot_inv: reciprocal of the conf-space jacobian time the guiding center coordinate Jacobian.
  // cmag: coefficient multiplying parallel gradient.
  // b_i: covariant components of the field aligned unit vector.
  // phi: electrostatic potential .
  // apar: parallel component of magnetic vector potential.
  // apardot: time derivative of Apar.
  // edge: determines if the update is for the left edge (-1) or right edge (+1).
  // fskin,fedge: distribution function in skin cell/last edge cell.
  // out: output increment in center cell.

  double wx = w[0];
  double rdx2 = 2.0/dxv[0];
  double wvpar = w[1];
  double rdvpar2 = 2.0/dxv[1];

  double wxSq = w[0]*w[0];
  double rdx2Sq = rdx2*rdx2;
  double wvparSq = w[1]*w[1];
  double rdvpar2Sq = rdvpar2*rdvpar2;

  const double *b_x = &b_i[0];
  const double *b_y = &b_i[3];
  const double *b_z = &b_i[6];

  double hamil[8] = {0.}; 
  hamil[0] = (0.3333333333333333*(3.0*rdvpar2Sq*(m_*wvparSq+1.414213562373095*phi[0]*q_)+m_))/rdvpar2Sq; 
  hamil[1] = 1.414213562373095*phi[1]*q_; 
  hamil[2] = (1.154700538379252*m_*wvpar)/rdvpar2; 
  hamil[4] = 1.414213562373095*phi[2]*q_; 
  hamil[5] = (0.2981423969999719*m_)/rdvpar2Sq; 

  double BstarZdBmag[8] = {0.}; 
  BstarZdBmag[0] = (1.732050807568877*(2.23606797749979*jacobtot_inv[1]*b_y[2]+jacobtot_inv[0]*b_y[1])*m_*rdx2*wvpar+(cmag[2]*jacobtot_inv[2]+cmag[1]*jacobtot_inv[1]+cmag[0]*jacobtot_inv[0])*q_)/q_; 
  BstarZdBmag[1] = (0.2*(1.732050807568877*(b_y[2]*(10.0*jacobtot_inv[2]+11.18033988749895*jacobtot_inv[0])+5.0*b_y[1]*jacobtot_inv[1])*m_*rdx2*wvpar+(4.47213595499958*(cmag[1]*jacobtot_inv[2]+jacobtot_inv[1]*cmag[2])+5.0*(cmag[0]*jacobtot_inv[1]+jacobtot_inv[0]*cmag[1]))*q_))/q_; 
  BstarZdBmag[2] = ((2.23606797749979*jacobtot_inv[1]*b_y[2]+jacobtot_inv[0]*b_y[1])*m_*rdx2)/(q_*rdvpar2); 
  BstarZdBmag[3] = ((b_y[2]*(2.0*jacobtot_inv[2]+2.23606797749979*jacobtot_inv[0])+b_y[1]*jacobtot_inv[1])*m_*rdx2)/(q_*rdvpar2); 
  BstarZdBmag[4] = (0.02857142857142857*(60.6217782649107*(b_y[1]*jacobtot_inv[2]+2.0*jacobtot_inv[1]*b_y[2])*m_*rdx2*wvpar+((22.3606797749979*cmag[2]+35.0*cmag[0])*jacobtot_inv[2]+7.0*(5.0*jacobtot_inv[0]*cmag[2]+4.47213595499958*cmag[1]*jacobtot_inv[1]))*q_))/q_; 
  BstarZdBmag[6] = (1.0*(b_y[1]*jacobtot_inv[2]+2.0*jacobtot_inv[1]*b_y[2])*m_*rdx2)/(q_*rdvpar2); 

  if (edge == -1) { 

  double alphaR[3] = {0.}; 
  alphaR[0] = (0.3535533905932737*(hamil[5]*(8.660254037844387*BstarZdBmag[6]+6.708203932499369*BstarZdBmag[3]+3.872983346207417*BstarZdBmag[2])+hamil[2]*(3.872983346207417*BstarZdBmag[4]+3.0*BstarZdBmag[1]+1.732050807568877*BstarZdBmag[0]))*rdvpar2)/m_; 
  alphaR[1] = (0.3535533905932737*(3.872983346207417*hamil[2]*BstarZdBmag[6]+(8.660254037844386*BstarZdBmag[4]+6.708203932499369*BstarZdBmag[1]+3.872983346207417*BstarZdBmag[0])*hamil[5]+hamil[2]*(3.0*BstarZdBmag[3]+1.732050807568877*BstarZdBmag[2]))*rdvpar2)/m_; 
  alphaR[2] = (0.7071067811865475*hamil[5]*(3.872983346207417*BstarZdBmag[6]+3.0*BstarZdBmag[3]+1.732050807568877*BstarZdBmag[2])*rdvpar2)/m_; 

  double fUpOrdR[3] = {0.};
  if (alphaR[2]-1.499999999999997*alphaR[1]+1.118033988749892*alphaR[0] > 0.) {
    fUpOrdR[0] = ser_2x_p2_surfx1_eval_quad_node_0_r(fskin); 
  } else { 
    fUpOrdR[0] = ser_2x_p2_surfx1_eval_quad_node_0_l(fedge); 
  } 
  if (alphaR[0]-1.118033988749896*alphaR[2] > 0.) {
    fUpOrdR[1] = ser_2x_p2_surfx1_eval_quad_node_1_r(fskin); 
  } else { 
    fUpOrdR[1] = ser_2x_p2_surfx1_eval_quad_node_1_l(fedge); 
  } 
  if (alphaR[2]+1.499999999999997*alphaR[1]+1.118033988749892*alphaR[0] > 0.) {
    fUpOrdR[2] = ser_2x_p2_surfx1_eval_quad_node_2_r(fskin); 
  } else { 
    fUpOrdR[2] = ser_2x_p2_surfx1_eval_quad_node_2_l(fedge); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  double fUpR[3] = {0.};
  ser_2x_p2_upwind_quad_to_modal(fUpOrdR, fUpR); 

  double GhatR[8] = {0.}; 
  GhatR[0] = 0.7071067811865475*alphaR[2]*fUpR[2]+0.7071067811865475*alphaR[1]*fUpR[1]+0.7071067811865475*alphaR[0]*fUpR[0]; 
  GhatR[1] = 0.6324555320336759*alphaR[1]*fUpR[2]+0.6324555320336759*fUpR[1]*alphaR[2]+0.7071067811865475*alphaR[0]*fUpR[1]+0.7071067811865475*fUpR[0]*alphaR[1]; 
  GhatR[2] = 0.4517539514526256*alphaR[2]*fUpR[2]+0.7071067811865475*alphaR[0]*fUpR[2]+0.7071067811865475*fUpR[0]*alphaR[2]+0.6324555320336759*alphaR[1]*fUpR[1]; 

  out[0] += -0.7071067811865475*GhatR[0]*rdx2; 
  out[1] += -1.224744871391589*GhatR[0]*rdx2; 
  out[2] += -0.7071067811865475*GhatR[1]*rdx2; 
  out[3] += -1.224744871391589*GhatR[1]*rdx2; 
  out[4] += -1.58113883008419*GhatR[0]*rdx2; 
  out[5] += -0.7071067811865475*GhatR[2]*rdx2; 
  out[6] += -1.58113883008419*GhatR[1]*rdx2; 
  out[7] += -1.224744871391589*GhatR[2]*rdx2; 

  } else { 

  double alphaL[3] = {0.}; 
  alphaL[0] = (0.3535533905932737*(hamil[5]*(8.660254037844387*BstarZdBmag[6]-6.708203932499369*BstarZdBmag[3]+3.872983346207417*BstarZdBmag[2])+hamil[2]*(3.872983346207417*BstarZdBmag[4]-3.0*BstarZdBmag[1]+1.732050807568877*BstarZdBmag[0]))*rdvpar2)/m_; 
  alphaL[1] = (0.3535533905932737*(3.872983346207417*hamil[2]*BstarZdBmag[6]+(8.660254037844386*BstarZdBmag[4]-6.708203932499369*BstarZdBmag[1]+3.872983346207417*BstarZdBmag[0])*hamil[5]+hamil[2]*(1.732050807568877*BstarZdBmag[2]-3.0*BstarZdBmag[3]))*rdvpar2)/m_; 
  alphaL[2] = (0.7071067811865475*hamil[5]*(3.872983346207417*BstarZdBmag[6]-3.0*BstarZdBmag[3]+1.732050807568877*BstarZdBmag[2])*rdvpar2)/m_; 

  double fUpOrdL[3] = {0.};
  if (alphaL[2]-1.499999999999997*alphaL[1]+1.118033988749892*alphaL[0] > 0.) {
    fUpOrdL[0] = ser_2x_p2_surfx1_eval_quad_node_0_r(fedge); 
  } else { 
    fUpOrdL[0] = ser_2x_p2_surfx1_eval_quad_node_0_l(fskin); 
  } 
  if (alphaL[0]-1.118033988749896*alphaL[2] > 0.) {
    fUpOrdL[1] = ser_2x_p2_surfx1_eval_quad_node_1_r(fedge); 
  } else { 
    fUpOrdL[1] = ser_2x_p2_surfx1_eval_quad_node_1_l(fskin); 
  } 
  if (alphaL[2]+1.499999999999997*alphaL[1]+1.118033988749892*alphaL[0] > 0.) {
    fUpOrdL[2] = ser_2x_p2_surfx1_eval_quad_node_2_r(fedge); 
  } else { 
    fUpOrdL[2] = ser_2x_p2_surfx1_eval_quad_node_2_l(fskin); 
  } 

  // Project tensor nodal quadrature basis back onto modal basis. 
  double fUpL[3] = {0.};
  ser_2x_p2_upwind_quad_to_modal(fUpOrdL, fUpL); 

  double GhatL[8] = {0.}; 
  GhatL[0] = 0.7071067811865475*alphaL[2]*fUpL[2]+0.7071067811865475*alphaL[1]*fUpL[1]+0.7071067811865475*alphaL[0]*fUpL[0]; 
  GhatL[1] = 0.6324555320336759*alphaL[1]*fUpL[2]+0.6324555320336759*fUpL[1]*alphaL[2]+0.7071067811865475*alphaL[0]*fUpL[1]+0.7071067811865475*fUpL[0]*alphaL[1]; 
  GhatL[2] = 0.4517539514526256*alphaL[2]*fUpL[2]+0.7071067811865475*alphaL[0]*fUpL[2]+0.7071067811865475*fUpL[0]*alphaL[2]+0.6324555320336759*alphaL[1]*fUpL[1]; 

  out[0] += 0.7071067811865475*GhatL[0]*rdx2; 
  out[1] += -1.224744871391589*GhatL[0]*rdx2; 
  out[2] += 0.7071067811865475*GhatL[1]*rdx2; 
  out[3] += -1.224744871391589*GhatL[1]*rdx2; 
  out[4] += 1.58113883008419*GhatL[0]*rdx2; 
  out[5] += 0.7071067811865475*GhatL[2]*rdx2; 
  out[6] += 1.58113883008419*GhatL[1]*rdx2; 
  out[7] += -1.224744871391589*GhatL[2]*rdx2; 

  } 

} 