#include <gkyl_dg_diffusion_vlasov_kernels.h>

GKYL_CU_DH double dg_diffusion_vlasov_order2_surfx_1x1v_ser_p2_constcoeff(const double *w, const double *dx, const double *coeff, const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{
  // w[NDIM]: Cell-center coordinate.
  // dxv[NDIM]: Cell length.
  // coeff: Diffusion coefficient.
  // ql: Input field in the left cell.
  // qc: Input field in the center cell.
  // qr: Input field in the right cell.
  // out: Incremented output.

  const double Jfac = pow(2./dx[0],2.);

  out[0] += 0.0125*(53.66563145999496*coeff[0]*qr[4]+53.66563145999496*coeff[0]*ql[4]-107.3312629199899*coeff[0]*qc[4]-95.26279441628824*coeff[0]*qr[1]+95.26279441628824*coeff[0]*ql[1]+75.0*coeff[0]*qr[0]+75.0*coeff[0]*ql[0]-150.0*coeff[0]*qc[0])*Jfac; 
  out[1] += 0.003125*(236.2519841186524*coeff[0]*qr[4]-236.2519841186524*coeff[0]*ql[4]-465.0*coeff[0]*qr[1]-465.0*coeff[0]*ql[1]-1710.0*coeff[0]*qc[1]+381.051177665153*coeff[0]*qr[0]-381.051177665153*coeff[0]*ql[0])*Jfac; 
  out[2] += 0.0125*(53.66563145999495*coeff[0]*qr[6]+53.66563145999495*coeff[0]*ql[6]-107.3312629199899*coeff[0]*qc[6]-95.26279441628824*coeff[0]*qr[3]+95.26279441628824*coeff[0]*ql[3]+75.0*coeff[0]*qr[2]+75.0*coeff[0]*ql[2]-150.0*coeff[0]*qc[2])*Jfac; 
  out[3] += 0.003125*(236.2519841186524*coeff[0]*qr[6]-236.2519841186524*coeff[0]*ql[6]-465.0*coeff[0]*qr[3]-465.0*coeff[0]*ql[3]-1710.0*coeff[0]*qc[3]+381.051177665153*coeff[0]*qr[2]-381.051177665153*coeff[0]*ql[2])*Jfac; 
  out[4] += -0.015625*(9.0*coeff[0]*qr[4]+9.0*coeff[0]*ql[4]+402.0*coeff[0]*qc[4]+19.36491673103709*coeff[0]*qr[1]-19.36491673103709*coeff[0]*ql[1]-26.83281572999748*coeff[0]*qr[0]-26.83281572999748*coeff[0]*ql[0]+53.66563145999496*coeff[0]*qc[0])*Jfac; 
  out[5] += -0.0125*(95.26279441628826*coeff[0]*qr[7]-95.26279441628826*coeff[0]*ql[7]-75.0*coeff[0]*qr[5]-75.0*coeff[0]*ql[5]+150.0*coeff[0]*qc[5])*Jfac; 
  out[6] += -0.015625*(9.0*coeff[0]*qr[6]+9.0*coeff[0]*ql[6]+402.0*coeff[0]*qc[6]+19.36491673103708*coeff[0]*qr[3]-19.36491673103708*coeff[0]*ql[3]-26.83281572999747*coeff[0]*qr[2]-26.83281572999747*coeff[0]*ql[2]+53.66563145999495*coeff[0]*qc[2])*Jfac; 
  out[7] += -0.003125*(465.0*coeff[0]*qr[7]+465.0*coeff[0]*ql[7]+1710.0*coeff[0]*qc[7]-381.051177665153*coeff[0]*qr[5]+381.051177665153*coeff[0]*ql[5])*Jfac; 

  return 0.;

}

GKYL_CU_DH double dg_diffusion_vlasov_order2_surfx_1x1v_ser_p2_varcoeff(const double *w, const double *dx, const double *coeff, const double *ql, const double *qc, const double *qr, double* GKYL_RESTRICT out) 
{
  // w[NDIM]: Cell-center coordinate.
  // dxv[NDIM]: Cell length.
  // coeff: Diffusion coefficient.
  // ql: Input field in the left cell.
  // qc: Input field in the center cell.
  // qr: Input field in the right cell.
  // out: Incremented output.

  const double Jfac = pow(2./dx[0],2.);

  out[0] += 0.00625*((169.7056274847715*coeff[2]+131.4534138012399*coeff[1]+75.89466384404115*coeff[0])*qr[4]+(169.7056274847715*coeff[2]-131.4534138012399*coeff[1]+75.89466384404115*coeff[0])*ql[4]+((-339.411254969543*coeff[2])-151.7893276880823*coeff[0])*qc[4]+((-301.2474066278414*qr[1])+301.2474066278414*ql[1]+237.1708245126285*qr[0]+237.1708245126285*ql[0]-474.3416490252571*qc[0])*coeff[2]+((-233.3452377915607*coeff[1])-134.7219358530748*coeff[0])*qr[1]+(134.7219358530748*coeff[0]-233.3452377915607*coeff[1])*ql[1]-466.6904755831215*coeff[1]*qc[1]+(183.7117307087383*qr[0]-183.7117307087383*ql[0])*coeff[1]+106.0660171779821*coeff[0]*qr[0]+106.0660171779821*coeff[0]*ql[0]-212.1320343559643*coeff[0]*qc[0])*Jfac; 
  out[1] += 0.0015625*((747.0943715488693*coeff[2]+578.6968118108135*coeff[1]+334.1107600781513*coeff[0])*qr[4]+((-747.0943715488693*coeff[2])+578.6968118108135*coeff[1]-334.1107600781513*coeff[0])*ql[4]-2485.550240892347*coeff[1]*qc[4]+((-1470.459111978297*qr[1])-1470.459111978297*ql[1]-2371.708245126285*qc[1]+1204.989626511366*qr[0]-1204.989626511366*ql[0])*coeff[2]+((-1139.012730394178*coeff[1])-657.6093065034893*coeff[0])*qr[1]+(1139.012730394178*coeff[1]-657.6093065034893*coeff[0])*ql[1]-2418.305191657993*coeff[0]*qc[1]+(933.3809511662431*qr[0]+933.3809511662431*ql[0]-1866.761902332486*qc[0])*coeff[1]+538.8877434122994*coeff[0]*qr[0]-538.8877434122994*coeff[0]*ql[0])*Jfac; 
  out[2] += 0.00625*((169.7056274847715*coeff[2]+131.4534138012399*coeff[1]+75.89466384404115*coeff[0])*qr[6]+(169.7056274847715*coeff[2]-131.4534138012399*coeff[1]+75.89466384404115*coeff[0])*ql[6]+((-339.411254969543*coeff[2])-151.7893276880823*coeff[0])*qc[6]+((-301.2474066278414*coeff[2])-233.3452377915607*coeff[1]-134.7219358530748*coeff[0])*qr[3]+(301.2474066278414*coeff[2]-233.3452377915607*coeff[1]+134.7219358530748*coeff[0])*ql[3]-466.6904755831215*coeff[1]*qc[3]+(237.1708245126285*coeff[2]+183.7117307087383*coeff[1]+106.0660171779821*coeff[0])*qr[2]+(237.1708245126285*coeff[2]-183.7117307087383*coeff[1]+106.0660171779821*coeff[0])*ql[2]+((-474.3416490252571*coeff[2])-212.1320343559643*coeff[0])*qc[2])*Jfac; 
  out[3] += 0.0015625*((747.0943715488694*coeff[2]+578.6968118108134*coeff[1]+334.1107600781513*coeff[0])*qr[6]+((-747.0943715488694*coeff[2])+578.6968118108134*coeff[1]-334.1107600781513*coeff[0])*ql[6]-2485.550240892347*coeff[1]*qc[6]+((-1470.459111978297*coeff[2])-1139.012730394178*coeff[1]-657.6093065034893*coeff[0])*qr[3]+((-1470.459111978297*coeff[2])+1139.012730394178*coeff[1]-657.6093065034893*coeff[0])*ql[3]+((-2371.708245126285*coeff[2])-2418.305191657993*coeff[0])*qc[3]+(1204.989626511366*coeff[2]+933.3809511662431*coeff[1]+538.8877434122994*coeff[0])*qr[2]+((-1204.989626511366*coeff[2])+933.3809511662431*coeff[1]-538.8877434122994*coeff[0])*ql[2]-1866.761902332486*coeff[1]*qc[2])*Jfac; 
  out[4] += -0.0078125*((28.46049894151542*coeff[2]+22.0454076850486*coeff[1]+12.72792206135786*coeff[0])*qr[4]+(28.46049894151542*coeff[2]-22.0454076850486*coeff[1]+12.72792206135786*coeff[0])*ql[4]+(568.5138520739844*coeff[0]-550.2363128692981*coeff[2])*qc[4]+(61.23724356957945*qr[1]-61.23724356957945*ql[1]-84.85281374238573*qr[0]-84.85281374238573*ql[0]+169.7056274847715*qc[0])*coeff[2]+(47.43416490252571*coeff[1]+27.38612787525831*coeff[0])*qr[1]+(47.43416490252571*coeff[1]-27.38612787525831*coeff[0])*ql[1]+360.4996532591953*coeff[1]*qc[1]+(65.72670690061996*ql[0]-65.72670690061996*qr[0])*coeff[1]-37.94733192202057*coeff[0]*qr[0]-37.94733192202057*coeff[0]*ql[0]+75.89466384404115*coeff[0]*qc[0])*Jfac; 
  out[5] += -0.00625*((301.2474066278414*coeff[2]+233.3452377915607*coeff[1]+134.7219358530748*coeff[0])*qr[7]+((-301.2474066278414*coeff[2])+233.3452377915607*coeff[1]-134.7219358530748*coeff[0])*ql[7]+466.6904755831214*coeff[1]*qc[7]+((-237.1708245126285*coeff[2])-183.7117307087383*coeff[1]-106.0660171779821*coeff[0])*qr[5]+((-237.1708245126285*coeff[2])+183.7117307087383*coeff[1]-106.0660171779821*coeff[0])*ql[5]+(474.3416490252571*coeff[2]+212.1320343559643*coeff[0])*qc[5])*Jfac; 
  out[6] += -0.0078125*((28.46049894151542*coeff[2]+22.0454076850486*coeff[1]+12.72792206135786*coeff[0])*qr[6]+(28.46049894151542*coeff[2]-22.0454076850486*coeff[1]+12.72792206135786*coeff[0])*ql[6]+(568.5138520739844*coeff[0]-550.2363128692981*coeff[2])*qc[6]+(61.23724356957948*coeff[2]+47.43416490252569*coeff[1]+27.38612787525831*coeff[0])*qr[3]+((-61.23724356957948*coeff[2])+47.43416490252569*coeff[1]-27.38612787525831*coeff[0])*ql[3]+360.4996532591954*coeff[1]*qc[3]+((-84.85281374238573*coeff[2])-65.72670690061996*coeff[1]-37.94733192202057*coeff[0])*qr[2]+((-84.85281374238573*coeff[2])+65.72670690061996*coeff[1]-37.94733192202057*coeff[0])*ql[2]+(169.7056274847715*coeff[2]+75.89466384404115*coeff[0])*qc[2])*Jfac; 
  out[7] += -0.0015625*((1470.459111978297*coeff[2]+1139.012730394178*coeff[1]+657.6093065034893*coeff[0])*qr[7]+(1470.459111978297*coeff[2]-1139.012730394178*coeff[1]+657.6093065034893*coeff[0])*ql[7]+(2371.708245126285*coeff[2]+2418.305191657993*coeff[0])*qc[7]+((-1204.989626511366*coeff[2])-933.3809511662431*coeff[1]-538.8877434122994*coeff[0])*qr[5]+(1204.989626511366*coeff[2]-933.3809511662431*coeff[1]+538.8877434122994*coeff[0])*ql[5]+1866.761902332487*coeff[1]*qc[5])*Jfac; 

  return 0.;

}
