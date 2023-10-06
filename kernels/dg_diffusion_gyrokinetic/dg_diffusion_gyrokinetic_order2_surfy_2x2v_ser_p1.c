#include <gkyl_dg_diffusion_gyrokinetic_kernels.h>

GKYL_CU_DH double dg_diffusion_gyrokinetic_order2_surfy_2x2v_ser_p1_constcoeff(const double *w, const double *dx, const double *coeff, const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{
  // w[NDIM]: Cell-center coordinate.
  // dxv[NDIM]: Cell length.
  // coeff: Diffusion coefficient.
  // ql: Input field in the left cell.
  // qc: Input field in the center cell.
  // qr: Input field in the right cell.
  // out: Incremented output.

  const double Jfac = pow(2./dx[1],2.);

  out[0] += -0.0625*(8.660254037844386*coeff[1]*qr[2]-8.660254037844386*coeff[1]*ql[2]+((-9.0*qr[0])-9.0*ql[0]+18.0*qc[0])*coeff[1])*Jfac; 
  out[1] += -0.0625*(8.660254037844386*coeff[1]*qr[5]-8.660254037844386*coeff[1]*ql[5]-9.0*coeff[1]*qr[1]-9.0*coeff[1]*ql[1]+18.0*coeff[1]*qc[1])*Jfac; 
  out[2] += -0.0625*(7.0*coeff[1]*qr[2]+7.0*coeff[1]*ql[2]+46.0*coeff[1]*qc[2]+(8.660254037844386*ql[0]-8.660254037844386*qr[0])*coeff[1])*Jfac; 
  out[3] += -0.0625*(8.660254037844386*coeff[1]*qr[7]-8.660254037844386*coeff[1]*ql[7]-9.0*coeff[1]*qr[3]-9.0*coeff[1]*ql[3]+18.0*coeff[1]*qc[3])*Jfac; 
  out[4] += -0.0625*(8.660254037844386*coeff[1]*qr[9]-8.660254037844386*coeff[1]*ql[9]-9.0*coeff[1]*qr[4]-9.0*coeff[1]*ql[4]+18.0*coeff[1]*qc[4])*Jfac; 
  out[5] += -0.0625*(7.0*coeff[1]*qr[5]+7.0*coeff[1]*ql[5]+46.0*coeff[1]*qc[5]-8.660254037844386*coeff[1]*qr[1]+8.660254037844386*coeff[1]*ql[1])*Jfac; 
  out[6] += -0.0625*(8.660254037844386*coeff[1]*qr[11]-8.660254037844386*coeff[1]*ql[11]-9.0*coeff[1]*qr[6]-9.0*coeff[1]*ql[6]+18.0*coeff[1]*qc[6])*Jfac; 
  out[7] += -0.0625*(7.0*coeff[1]*qr[7]+7.0*coeff[1]*ql[7]+46.0*coeff[1]*qc[7]-8.660254037844386*coeff[1]*qr[3]+8.660254037844386*coeff[1]*ql[3])*Jfac; 
  out[8] += -0.0625*(8.660254037844386*coeff[1]*qr[12]-8.660254037844386*coeff[1]*ql[12]-9.0*coeff[1]*qr[8]-9.0*coeff[1]*ql[8]+18.0*coeff[1]*qc[8])*Jfac; 
  out[9] += -0.0625*(7.0*coeff[1]*qr[9]+7.0*coeff[1]*ql[9]+46.0*coeff[1]*qc[9]-8.660254037844386*coeff[1]*qr[4]+8.660254037844386*coeff[1]*ql[4])*Jfac; 
  out[10] += -0.0625*(8.660254037844386*coeff[1]*qr[14]-8.660254037844386*coeff[1]*ql[14]-9.0*coeff[1]*qr[10]-9.0*coeff[1]*ql[10]+18.0*coeff[1]*qc[10])*Jfac; 
  out[11] += -0.0625*(7.0*coeff[1]*qr[11]+7.0*coeff[1]*ql[11]+46.0*coeff[1]*qc[11]-8.660254037844386*coeff[1]*qr[6]+8.660254037844386*coeff[1]*ql[6])*Jfac; 
  out[12] += -0.0625*(7.0*coeff[1]*qr[12]+7.0*coeff[1]*ql[12]+46.0*coeff[1]*qc[12]-8.660254037844386*coeff[1]*qr[8]+8.660254037844386*coeff[1]*ql[8])*Jfac; 
  out[13] += -0.0625*(8.660254037844386*coeff[1]*qr[15]-8.660254037844386*coeff[1]*ql[15]-9.0*coeff[1]*qr[13]-9.0*coeff[1]*ql[13]+18.0*coeff[1]*qc[13])*Jfac; 
  out[14] += -0.0625*(7.0*coeff[1]*qr[14]+7.0*coeff[1]*ql[14]+46.0*coeff[1]*qc[14]-8.660254037844386*coeff[1]*qr[10]+8.660254037844386*coeff[1]*ql[10])*Jfac; 
  out[15] += -0.0625*(7.0*coeff[1]*qr[15]+7.0*coeff[1]*ql[15]+46.0*coeff[1]*qc[15]-8.660254037844386*coeff[1]*qr[13]+8.660254037844386*coeff[1]*ql[13])*Jfac; 
  out[16] += -0.0625*(8.660254037844387*coeff[1]*qr[18]-8.660254037844387*coeff[1]*ql[18]-9.0*coeff[1]*qr[16]-9.0*coeff[1]*ql[16]+18.0*coeff[1]*qc[16])*Jfac; 
  out[17] += -0.0625*(8.660254037844387*coeff[1]*qr[20]-8.660254037844387*coeff[1]*ql[20]-9.0*coeff[1]*qr[17]-9.0*coeff[1]*ql[17]+18.0*coeff[1]*qc[17])*Jfac; 
  out[18] += -0.0625*(7.0*coeff[1]*qr[18]+7.0*coeff[1]*ql[18]+46.0*coeff[1]*qc[18]-8.660254037844387*coeff[1]*qr[16]+8.660254037844387*coeff[1]*ql[16])*Jfac; 
  out[19] += -0.0625*(8.660254037844387*coeff[1]*qr[22]-8.660254037844387*coeff[1]*ql[22]-9.0*coeff[1]*qr[19]-9.0*coeff[1]*ql[19]+18.0*coeff[1]*qc[19])*Jfac; 
  out[20] += -0.0625*(7.0*coeff[1]*qr[20]+7.0*coeff[1]*ql[20]+46.0*coeff[1]*qc[20]-8.660254037844387*coeff[1]*qr[17]+8.660254037844387*coeff[1]*ql[17])*Jfac; 
  out[21] += -0.0625*(8.660254037844387*coeff[1]*qr[23]-8.660254037844387*coeff[1]*ql[23]-9.0*coeff[1]*qr[21]-9.0*coeff[1]*ql[21]+18.0*coeff[1]*qc[21])*Jfac; 
  out[22] += -0.0625*(7.0*coeff[1]*qr[22]+7.0*coeff[1]*ql[22]+46.0*coeff[1]*qc[22]-8.660254037844387*coeff[1]*qr[19]+8.660254037844387*coeff[1]*ql[19])*Jfac; 
  out[23] += -0.0625*(7.0*coeff[1]*qr[23]+7.0*coeff[1]*ql[23]+46.0*coeff[1]*qc[23]-8.660254037844387*coeff[1]*qr[21]+8.660254037844387*coeff[1]*ql[21])*Jfac; 

  return 0.;

}

GKYL_CU_DH double dg_diffusion_gyrokinetic_order2_surfy_2x2v_ser_p1_varcoeff(const double *w, const double *dx, const double *coeff, const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{
  // w[NDIM]: Cell-center coordinate.
  // dxv[NDIM]: Cell length.
  // coeff: Diffusion coefficient.
  // ql: Input field in the left cell.
  // qc: Input field in the center cell.
  // qr: Input field in the right cell.
  // out: Incremented output.

  const double Jfac = pow(2./dx[1],2.);

  out[0] += -0.03125*((15.0*qr[5]+15.0*ql[5]+30.0*qc[5]-15.58845726811989*qr[1]+15.58845726811989*ql[1])*coeff[7]+(15.0*qr[2]+15.0*ql[2]+30.0*qc[2]-15.58845726811989*qr[0]+15.58845726811989*ql[0])*coeff[6]+8.660254037844386*coeff[5]*qr[5]-8.660254037844386*coeff[5]*ql[5]+((-9.0*qr[1])-9.0*ql[1]+18.0*qc[1])*coeff[5]+(8.660254037844386*qr[2]-8.660254037844386*ql[2]-9.0*qr[0]-9.0*ql[0]+18.0*qc[0])*coeff[4])*Jfac; 
  out[1] += -0.03125*((15.0*qr[2]+15.0*ql[2]+30.0*qc[2]-15.58845726811989*qr[0]+15.58845726811989*ql[0])*coeff[7]+(15.0*qr[5]+15.0*ql[5]+30.0*qc[5]-15.58845726811989*qr[1]+15.58845726811989*ql[1])*coeff[6]+8.660254037844386*coeff[4]*qr[5]-8.660254037844386*coeff[4]*ql[5]+(8.660254037844386*qr[2]-8.660254037844386*ql[2]-9.0*qr[0]-9.0*ql[0]+18.0*qc[0])*coeff[5]+((-9.0*qr[1])-9.0*ql[1]+18.0*qc[1])*coeff[4])*Jfac; 
  out[2] += -0.03125*((12.12435565298214*qr[5]-12.12435565298214*ql[5]-15.0*qr[1]-15.0*ql[1]+30.0*qc[1])*coeff[7]+(12.12435565298214*qr[2]-12.12435565298214*ql[2]-15.0*qr[0]-15.0*ql[0]+30.0*qc[0])*coeff[6]+7.0*coeff[5]*qr[5]+7.0*coeff[5]*ql[5]+46.0*coeff[5]*qc[5]+(8.660254037844386*ql[1]-8.660254037844386*qr[1])*coeff[5]+(7.0*qr[2]+7.0*ql[2]+46.0*qc[2]-8.660254037844386*qr[0]+8.660254037844386*ql[0])*coeff[4])*Jfac; 
  out[3] += -0.03125*((15.0*coeff[7]+8.660254037844386*coeff[5])*qr[11]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[11]+30.0*coeff[7]*qc[11]+(15.0*coeff[6]+8.660254037844386*coeff[4])*qr[7]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[7]+30.0*coeff[6]*qc[7]+(15.58845726811989*ql[6]-15.58845726811989*qr[6])*coeff[7]-9.0*coeff[5]*qr[6]-9.0*coeff[5]*ql[6]+18.0*coeff[5]*qc[6]+(15.58845726811989*ql[3]-15.58845726811989*qr[3])*coeff[6]+((-9.0*qr[3])-9.0*ql[3]+18.0*qc[3])*coeff[4])*Jfac; 
  out[4] += -0.03125*((15.0*coeff[7]+8.660254037844386*coeff[5])*qr[12]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[12]+30.0*coeff[7]*qc[12]+(15.0*coeff[6]+8.660254037844386*coeff[4])*qr[9]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[9]+30.0*coeff[6]*qc[9]+((-15.58845726811989*coeff[7])-9.0*coeff[5])*qr[8]+(15.58845726811989*coeff[7]-9.0*coeff[5])*ql[8]+18.0*coeff[5]*qc[8]+(15.58845726811989*ql[4]-15.58845726811989*qr[4])*coeff[6]-9.0*coeff[4]*qr[4]-9.0*coeff[4]*ql[4]+18.0*coeff[4]*qc[4])*Jfac; 
  out[5] += -0.03125*((12.12435565298214*qr[2]-12.12435565298214*ql[2]-15.0*qr[0]-15.0*ql[0]+30.0*qc[0])*coeff[7]+(12.12435565298214*qr[5]-12.12435565298214*ql[5]-15.0*qr[1]-15.0*ql[1]+30.0*qc[1])*coeff[6]+7.0*coeff[4]*qr[5]+7.0*coeff[4]*ql[5]+46.0*coeff[4]*qc[5]+(7.0*qr[2]+7.0*ql[2]+46.0*qc[2]-8.660254037844386*qr[0]+8.660254037844386*ql[0])*coeff[5]+(8.660254037844386*ql[1]-8.660254037844386*qr[1])*coeff[4])*Jfac; 
  out[6] += -0.03125*((15.0*coeff[6]+8.660254037844386*coeff[4])*qr[11]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[11]+30.0*coeff[6]*qc[11]+(15.0*coeff[7]+8.660254037844386*coeff[5])*qr[7]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[7]+30.0*coeff[7]*qc[7]+(15.58845726811989*ql[3]-15.58845726811989*qr[3])*coeff[7]+((-15.58845726811989*coeff[6])-9.0*coeff[4])*qr[6]+(15.58845726811989*coeff[6]-9.0*coeff[4])*ql[6]+18.0*coeff[4]*qc[6]+((-9.0*qr[3])-9.0*ql[3]+18.0*qc[3])*coeff[5])*Jfac; 
  out[7] += -0.03125*((12.12435565298214*coeff[7]+7.0*coeff[5])*qr[11]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[11]+46.0*coeff[5]*qc[11]+(12.12435565298214*coeff[6]+7.0*coeff[4])*qr[7]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[7]+46.0*coeff[4]*qc[7]+((-15.0*qr[6])-15.0*ql[6]+30.0*qc[6])*coeff[7]-8.660254037844386*coeff[5]*qr[6]+8.660254037844386*coeff[5]*ql[6]+((-15.0*qr[3])-15.0*ql[3]+30.0*qc[3])*coeff[6]+(8.660254037844386*ql[3]-8.660254037844386*qr[3])*coeff[4])*Jfac; 
  out[8] += -0.03125*((15.0*coeff[6]+8.660254037844386*coeff[4])*qr[12]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[12]+30.0*coeff[6]*qc[12]+(15.0*coeff[7]+8.660254037844386*coeff[5])*qr[9]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[9]+30.0*coeff[7]*qc[9]+((-15.58845726811989*coeff[6])-9.0*coeff[4])*qr[8]+(15.58845726811989*coeff[6]-9.0*coeff[4])*ql[8]+18.0*coeff[4]*qc[8]+(15.58845726811989*ql[4]-15.58845726811989*qr[4])*coeff[7]+((-9.0*qr[4])-9.0*ql[4]+18.0*qc[4])*coeff[5])*Jfac; 
  out[9] += -0.03125*((12.12435565298214*coeff[7]+7.0*coeff[5])*qr[12]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[12]+46.0*coeff[5]*qc[12]+(12.12435565298214*coeff[6]+7.0*coeff[4])*qr[9]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[9]+46.0*coeff[4]*qc[9]+((-15.0*coeff[7])-8.660254037844386*coeff[5])*qr[8]+(8.660254037844386*coeff[5]-15.0*coeff[7])*ql[8]+30.0*coeff[7]*qc[8]+((-15.0*qr[4])-15.0*ql[4]+30.0*qc[4])*coeff[6]-8.660254037844386*coeff[4]*qr[4]+8.660254037844386*coeff[4]*ql[4])*Jfac; 
  out[10] += -0.03125*((15.0*coeff[7]+8.660254037844386*coeff[5])*qr[15]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[15]+30.0*coeff[7]*qc[15]+(15.0*coeff[6]+8.660254037844386*coeff[4])*qr[14]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[14]+30.0*coeff[6]*qc[14]+((-15.58845726811989*coeff[7])-9.0*coeff[5])*qr[13]+(15.58845726811989*coeff[7]-9.0*coeff[5])*ql[13]+18.0*coeff[5]*qc[13]+((-15.58845726811989*coeff[6])-9.0*coeff[4])*qr[10]+(15.58845726811989*coeff[6]-9.0*coeff[4])*ql[10]+18.0*coeff[4]*qc[10])*Jfac; 
  out[11] += -0.03125*((12.12435565298214*coeff[6]+7.0*coeff[4])*qr[11]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[11]+46.0*coeff[4]*qc[11]+(12.12435565298214*coeff[7]+7.0*coeff[5])*qr[7]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[7]+46.0*coeff[5]*qc[7]+((-15.0*qr[3])-15.0*ql[3]+30.0*qc[3])*coeff[7]+((-15.0*coeff[6])-8.660254037844386*coeff[4])*qr[6]+(8.660254037844386*coeff[4]-15.0*coeff[6])*ql[6]+30.0*coeff[6]*qc[6]+(8.660254037844386*ql[3]-8.660254037844386*qr[3])*coeff[5])*Jfac; 
  out[12] += -0.03125*((12.12435565298214*coeff[6]+7.0*coeff[4])*qr[12]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[12]+46.0*coeff[4]*qc[12]+(12.12435565298214*coeff[7]+7.0*coeff[5])*qr[9]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[9]+46.0*coeff[5]*qc[9]+((-15.0*coeff[6])-8.660254037844386*coeff[4])*qr[8]+(8.660254037844386*coeff[4]-15.0*coeff[6])*ql[8]+30.0*coeff[6]*qc[8]+((-15.0*qr[4])-15.0*ql[4]+30.0*qc[4])*coeff[7]+(8.660254037844386*ql[4]-8.660254037844386*qr[4])*coeff[5])*Jfac; 
  out[13] += -0.03125*((15.0*coeff[6]+8.660254037844386*coeff[4])*qr[15]+(15.0*coeff[6]-8.660254037844386*coeff[4])*ql[15]+30.0*coeff[6]*qc[15]+(15.0*coeff[7]+8.660254037844386*coeff[5])*qr[14]+(15.0*coeff[7]-8.660254037844386*coeff[5])*ql[14]+30.0*coeff[7]*qc[14]+((-15.58845726811989*coeff[6])-9.0*coeff[4])*qr[13]+(15.58845726811989*coeff[6]-9.0*coeff[4])*ql[13]+18.0*coeff[4]*qc[13]+((-15.58845726811989*coeff[7])-9.0*coeff[5])*qr[10]+(15.58845726811989*coeff[7]-9.0*coeff[5])*ql[10]+18.0*coeff[5]*qc[10])*Jfac; 
  out[14] += -0.03125*((12.12435565298214*coeff[7]+7.0*coeff[5])*qr[15]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[15]+46.0*coeff[5]*qc[15]+(12.12435565298214*coeff[6]+7.0*coeff[4])*qr[14]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[14]+46.0*coeff[4]*qc[14]+((-15.0*coeff[7])-8.660254037844386*coeff[5])*qr[13]+(8.660254037844386*coeff[5]-15.0*coeff[7])*ql[13]+30.0*coeff[7]*qc[13]+((-15.0*coeff[6])-8.660254037844386*coeff[4])*qr[10]+(8.660254037844386*coeff[4]-15.0*coeff[6])*ql[10]+30.0*coeff[6]*qc[10])*Jfac; 
  out[15] += -0.03125*((12.12435565298214*coeff[6]+7.0*coeff[4])*qr[15]+(7.0*coeff[4]-12.12435565298214*coeff[6])*ql[15]+46.0*coeff[4]*qc[15]+(12.12435565298214*coeff[7]+7.0*coeff[5])*qr[14]+(7.0*coeff[5]-12.12435565298214*coeff[7])*ql[14]+46.0*coeff[5]*qc[14]+((-15.0*coeff[6])-8.660254037844386*coeff[4])*qr[13]+(8.660254037844386*coeff[4]-15.0*coeff[6])*ql[13]+30.0*coeff[6]*qc[13]+((-15.0*coeff[7])-8.660254037844386*coeff[5])*qr[10]+(8.660254037844386*coeff[5]-15.0*coeff[7])*ql[10]+30.0*coeff[7]*qc[10])*Jfac; 
  out[16] += -0.00625*((75.0*coeff[7]+43.30127018922193*coeff[5])*qr[20]+(75.0*coeff[7]-43.30127018922193*coeff[5])*ql[20]+150.0*coeff[7]*qc[20]+(75.00000000000001*coeff[6]+43.30127018922195*coeff[4])*qr[18]+(75.00000000000001*coeff[6]-43.30127018922195*coeff[4])*ql[18]+150.0*coeff[6]*qc[18]+((-77.94228634059948*coeff[7])-45.0*coeff[5])*qr[17]+(77.94228634059948*coeff[7]-45.0*coeff[5])*ql[17]+90.0*coeff[5]*qc[17]+((-77.94228634059945*coeff[6])-45.0*coeff[4])*qr[16]+(77.94228634059945*coeff[6]-45.0*coeff[4])*ql[16]+90.0*coeff[4]*qc[16])*Jfac; 
  out[17] += -0.00625*((75.00000000000001*coeff[6]+43.30127018922195*coeff[4])*qr[20]+(75.00000000000001*coeff[6]-43.30127018922195*coeff[4])*ql[20]+150.0*coeff[6]*qc[20]+(75.0*coeff[7]+43.30127018922193*coeff[5])*qr[18]+(75.0*coeff[7]-43.30127018922193*coeff[5])*ql[18]+150.0*coeff[7]*qc[18]+((-77.94228634059945*coeff[6])-45.0*coeff[4])*qr[17]+(77.94228634059945*coeff[6]-45.0*coeff[4])*ql[17]+90.0*coeff[4]*qc[17]+((-77.94228634059948*coeff[7])-45.0*coeff[5])*qr[16]+(77.94228634059948*coeff[7]-45.0*coeff[5])*ql[16]+90.0*coeff[5]*qc[16])*Jfac; 
  out[18] += -0.002083333333333333*((181.8653347947321*coeff[7]+105.0*coeff[5])*qr[20]+(105.0*coeff[5]-181.8653347947321*coeff[7])*ql[20]+690.0*coeff[5]*qc[20]+(181.8653347947321*coeff[6]+105.0*coeff[4])*qr[18]+(105.0*coeff[4]-181.8653347947321*coeff[6])*ql[18]+690.0*coeff[4]*qc[18]+((-225.0*coeff[7])-129.9038105676658*coeff[5])*qr[17]+(129.9038105676658*coeff[5]-225.0*coeff[7])*ql[17]+450.0*coeff[7]*qc[17]+((-225.0*coeff[6])-129.9038105676658*coeff[4])*qr[16]+(129.9038105676658*coeff[4]-225.0*coeff[6])*ql[16]+450.0000000000001*coeff[6]*qc[16])*Jfac; 
  out[19] += -0.00625*((75.0*coeff[7]+43.30127018922193*coeff[5])*qr[23]+(75.0*coeff[7]-43.30127018922193*coeff[5])*ql[23]+150.0*coeff[7]*qc[23]+(75.00000000000001*coeff[6]+43.30127018922195*coeff[4])*qr[22]+(75.00000000000001*coeff[6]-43.30127018922195*coeff[4])*ql[22]+150.0*coeff[6]*qc[22]+((-77.94228634059948*coeff[7])-45.0*coeff[5])*qr[21]+(77.94228634059948*coeff[7]-45.0*coeff[5])*ql[21]+90.0*coeff[5]*qc[21]+((-77.94228634059945*coeff[6])-45.0*coeff[4])*qr[19]+(77.94228634059945*coeff[6]-45.0*coeff[4])*ql[19]+90.0*coeff[4]*qc[19])*Jfac; 
  out[20] += -0.002083333333333333*((181.8653347947321*coeff[6]+105.0*coeff[4])*qr[20]+(105.0*coeff[4]-181.8653347947321*coeff[6])*ql[20]+690.0*coeff[4]*qc[20]+(181.8653347947321*coeff[7]+105.0*coeff[5])*qr[18]+(105.0*coeff[5]-181.8653347947321*coeff[7])*ql[18]+690.0*coeff[5]*qc[18]+((-225.0*coeff[6])-129.9038105676658*coeff[4])*qr[17]+(129.9038105676658*coeff[4]-225.0*coeff[6])*ql[17]+450.0000000000001*coeff[6]*qc[17]+((-225.0*coeff[7])-129.9038105676658*coeff[5])*qr[16]+(129.9038105676658*coeff[5]-225.0*coeff[7])*ql[16]+450.0*coeff[7]*qc[16])*Jfac; 
  out[21] += -0.00625*((75.00000000000001*coeff[6]+43.30127018922195*coeff[4])*qr[23]+(75.00000000000001*coeff[6]-43.30127018922195*coeff[4])*ql[23]+150.0*coeff[6]*qc[23]+(75.0*coeff[7]+43.30127018922193*coeff[5])*qr[22]+(75.0*coeff[7]-43.30127018922193*coeff[5])*ql[22]+150.0*coeff[7]*qc[22]+((-77.94228634059945*coeff[6])-45.0*coeff[4])*qr[21]+(77.94228634059945*coeff[6]-45.0*coeff[4])*ql[21]+90.0*coeff[4]*qc[21]+((-77.94228634059948*coeff[7])-45.0*coeff[5])*qr[19]+(77.94228634059948*coeff[7]-45.0*coeff[5])*ql[19]+90.0*coeff[5]*qc[19])*Jfac; 
  out[22] += -0.002083333333333333*((181.8653347947321*coeff[7]+105.0*coeff[5])*qr[23]+(105.0*coeff[5]-181.8653347947321*coeff[7])*ql[23]+690.0*coeff[5]*qc[23]+(181.8653347947321*coeff[6]+105.0*coeff[4])*qr[22]+(105.0*coeff[4]-181.8653347947321*coeff[6])*ql[22]+690.0*coeff[4]*qc[22]+((-225.0*coeff[7])-129.9038105676658*coeff[5])*qr[21]+(129.9038105676658*coeff[5]-225.0*coeff[7])*ql[21]+450.0*coeff[7]*qc[21]+((-225.0*coeff[6])-129.9038105676658*coeff[4])*qr[19]+(129.9038105676658*coeff[4]-225.0*coeff[6])*ql[19]+450.0000000000001*coeff[6]*qc[19])*Jfac; 
  out[23] += -0.002083333333333333*((181.8653347947321*coeff[6]+105.0*coeff[4])*qr[23]+(105.0*coeff[4]-181.8653347947321*coeff[6])*ql[23]+690.0*coeff[4]*qc[23]+(181.8653347947321*coeff[7]+105.0*coeff[5])*qr[22]+(105.0*coeff[5]-181.8653347947321*coeff[7])*ql[22]+690.0*coeff[5]*qc[22]+((-225.0*coeff[6])-129.9038105676658*coeff[4])*qr[21]+(129.9038105676658*coeff[4]-225.0*coeff[6])*ql[21]+450.0000000000001*coeff[6]*qc[21]+((-225.0*coeff[7])-129.9038105676658*coeff[5])*qr[19]+(129.9038105676658*coeff[5]-225.0*coeff[7])*ql[19]+450.0*coeff[7]*qc[19])*Jfac; 

  return 0.;

}
