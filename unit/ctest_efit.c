#include <stdio.h>
#include <math.h>
#include <string.h>
#include <acutest.h>
#include <gkyl_array.h>
#include <gkyl_array_rio.h>
#include <gkyl_array_ops.h>
#include <gkyl_eval_on_nodes.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_util.h>
#include <gkyl_basis.h>

#include <gkyl_efit.h>


void test_1(){
  char* filepath = "./efit_data/asdex.geqdsk";
  int rzpoly_order = 2;
  int fluxpoly_order = 1;
  struct gkyl_efit* efit = gkyl_efit_new(filepath,rzpoly_order, fluxpoly_order, false);

  printf( "rdim=%g zdim=%g rcentr=%g rleft=%g zmid=%g  rmaxis=%g zmaxis=%g simag=%1.16e sibry=%1.16e bcentr=%g  current=%g simag=%g rmaxis=%g   zmaxis=%g sibry=%g \n", efit->rdim, efit->zdim, efit->rcentr, efit->rleft, efit->zmid, efit->rmaxis, efit->zmaxis, efit->simag, efit->sibry, efit->bcentr, efit-> current, efit->simag, efit->rmaxis, efit-> zmaxis, efit->sibry);
  gkyl_grid_sub_array_write(efit->rzgrid, efit->rzlocal, efit->psizr, "asdex_psi.gkyl");
  gkyl_grid_sub_array_write(efit->rzgrid, efit->rzlocal, efit->psibyrzr, "asdex_psibyr.gkyl");
  gkyl_grid_sub_array_write(efit->rzgrid, efit->rzlocal, efit->psibyr2zr, "asdex_psibyr2.gkyl");
  gkyl_grid_sub_array_write(efit->fluxgrid, efit->fluxlocal, efit->fpolflux, "asdex_fpol.gkyl");
  gkyl_grid_sub_array_write(efit->fluxgrid, efit->fluxlocal, efit->qflux, "asdex_q.gkyl");

  gkyl_efit_release(efit);

}

TEST_LIST = {
  { "test_1", test_1},
  { NULL, NULL },
};