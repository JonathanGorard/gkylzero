#include <gkyl_skin_surf_from_ghost_kernels.h> 

GKYL_CU_DH void skin_surf_from_ghost_lower_2x_ser_p1(const double *fghost, double *fskin) 
{ 
  // fghost: field in the ghost cell.
  // fskin: field in the skin cell.

  double fghostSurf[2];
  fghostSurf[0] = 1.224744871391589*fghost[2]+0.7071067811865475*fghost[0]; 
  fghostSurf[1] = 1.224744871391589*fghost[3]+0.7071067811865475*fghost[1]; 

  fskin[0] = 0.8660254037844386*fskin[2]+0.5*fskin[0]+0.7071067811865475*fghostSurf[0]; 
  fskin[1] = 0.8660254037844386*fskin[3]+0.5*fskin[1]+0.7071067811865475*fghostSurf[1]; 
  fskin[2] = 0.5*fskin[2]+0.2886751345948129*fskin[0]-0.408248290463863*fghostSurf[0]; 
  fskin[3] = 0.5*fskin[3]+0.2886751345948129*fskin[1]-0.408248290463863*fghostSurf[1]; 

}

GKYL_CU_DH void skin_surf_from_ghost_upper_2x_ser_p1(const double *fghost, double *fskin) 
{ 
  // fghost: field in the ghost cell.
  // fskin: field in the skin cell.

  double fghostSurf[2];
  fghostSurf[0] = 0.7071067811865475*fghost[0]-1.224744871391589*fghost[2]; 
  fghostSurf[1] = 0.7071067811865475*fghost[1]-1.224744871391589*fghost[3]; 

  fskin[0] = -(0.8660254037844386*fskin[2])+0.5*fskin[0]+0.7071067811865475*fghostSurf[0]; 
  fskin[1] = -(0.8660254037844386*fskin[3])+0.5*fskin[1]+0.7071067811865475*fghostSurf[1]; 
  fskin[2] = 0.5*fskin[2]-0.2886751345948129*fskin[0]+0.408248290463863*fghostSurf[0]; 
  fskin[3] = 0.5*fskin[3]-0.2886751345948129*fskin[1]+0.408248290463863*fghostSurf[1]; 

}
