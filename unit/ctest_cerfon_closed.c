#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


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

#include <gkyl_calc_bmag.h>
#include <gkyl_calc_bmag_kernels.h>
#include <gkyl_geo_gyrokinetic.h>


#include <gkyl_calc_metric.h>
#include <gkyl_calc_metric_kernels.h>
#include <gkyl_calc_derived_geo.h>
#include <gkyl_calc_derived_geo_kernels.h>
#include <gkyl_geo_gyrokinetic.h>


// Helper Functions

struct mapc2p_ctx{
   struct gkyl_geo_gyrokinetic* app;
   struct gkyl_geo_gyrokinetic_geo_inp* ginp;
};


// Cerfon equilibrium
struct cerfon_ctx {
  double R0, psi_prefactor, B0;
};

static inline double sq(double x) { return x*x; }
static inline double cub(double x) { return x*x*x; }
static inline double qad(double x) { return x*x*x*x; }
static inline double pen(double x) { return x*x*x*x*x; }
static inline double hex(double x) { return x*x*x*x*x*x; }



static inline void
comp_to_phys(int ndim, const double *eta,
  const double * GKYL_RESTRICT dx, const double * GKYL_RESTRICT xc,
  double* GKYL_RESTRICT xout)
{
  for (int d=0; d<ndim; ++d) xout[d] = 0.5*dx[d]*eta[d]+xc[d];
}




void
psi(double t, const double *xn, double *fout, void *ctx)
{
  struct cerfon_ctx *s = ctx;
  double R0 = s->R0, psi_prefactor = s->psi_prefactor;
  double R = xn[0], Z = xn[1];
  double x = R/R0, y = Z/R0;

  fout[0] = psi_prefactor*(0.00373804283369699*hex(x)*log(x) - 0.00574955335438162*hex(x) - 0.0448565140043639*qad(x)*sq(y)*log(x) + 0.0503044260840946*qad(x)*sq(y) + 0.017623348727471*qad(x)*log(x) + 0.0956643504553683*qad(x) + 0.0299043426695759*sq(x)*qad(y)*log(x) - 0.0160920841654771*sq(x)*qad(y) - 0.0704933949098842*sq(x)*sq(y)*log(x) + 0.0644725519961135*sq(x)*sq(y) - 7.00898484784405e-5*sq(x)*log(x) - 0.303766642191745*sq(x) - 0.00199362284463839*hex(y) + 0.0117488991516474*qad(y) + 7.00898484784405e-5*sq(y) + 0.0145368720253975);
}



void
psibyr(double t, const double *xn, double *fout, void *ctx)
{
  double R = xn[0], Z = xn[1];
  psi(t,xn,fout,ctx);
  fout[0] = fout[0]/R;
}

void
psibyr2(double t, const double *xn, double *fout, void *ctx)
{
  double R = xn[0], Z = xn[1];
  psi(t,xn,fout,ctx);
  fout[0] = fout[0]/R/R;
}

void
bphi_func(double t, const double *xn, double *fout, void *ctx)
{
  struct cerfon_ctx *s = ctx;
  double B0 = s->B0, R0 = s->R0;
  double R = xn[0];
  fout[0] = B0*R0/R;
}


void
test_1()
{
  clock_t start, end;
  double cpu_time_used;
  start = clock();


  struct cerfon_ctx sctx = {  .R0 = 2.5, .psi_prefactor = 1.0, .B0 = 0.55 };

  
  // create RZ grid
  double lower[] = { 0.01, -6.0 }, upper[] = { 6.0, 6.0 };
  int cells[] = { 64, 128 };

  struct gkyl_rect_grid rzgrid;
  gkyl_rect_grid_init(&rzgrid, 2, lower, upper, cells);

  // RZ ranges
  struct gkyl_range rzlocal, rzlocal_ext;
  int nghost[GKYL_MAX_CDIM] = { 1, 1 };
  gkyl_create_grid_ranges(&rzgrid, nghost, &rzlocal_ext, &rzlocal);

  // RZ basis function
  int rzpoly_order = 2;
  struct gkyl_basis rzbasis;
  gkyl_cart_modal_serendip(&rzbasis, 2, rzpoly_order);

  // allocate psiRZ array, initialize and write it to file
  struct gkyl_array *psiRZ = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);
  
  gkyl_eval_on_nodes *eon = gkyl_eval_on_nodes_new(&rzgrid, &rzbasis, 1, &psi, &sctx);
  gkyl_eval_on_nodes_advance(eon, 0.0, &rzlocal, psiRZ);
  gkyl_eval_on_nodes_release(eon);

  //gkyl_grid_sub_array_write(&rzgrid, &rzlocal, psiRZ, "test_bmag_psi.gkyl");


  gkyl_geo_gyrokinetic *geo = gkyl_geo_gyrokinetic_new(&(struct gkyl_geo_gyrokinetic_inp) {
      // psiRZ and related inputs
      .rzgrid = &rzgrid,
      .rzbasis = &rzbasis,
      .psiRZ = psiRZ,
      .rzlocal = &rzlocal,
      .B0 = sctx.B0,
      .R0 = sctx.R0,

      .quad_param = {  .eps = 1e-12 }
    }
  );

  // compute outboard SOL geometry
  int npsi = 32, ntheta = 32;
  double psi_min = 0.1, psi_max = 1.2;
  double dpsi = (psi_max-psi_min)/npsi;
  double dtheta = M_PI/ntheta;
  psi_min += dpsi;

  psi_min = 0.03636363636363636; // This gives ghost node on psisep for 32 cells
  //psi_min = 0.07058823529411765; // This gives ghost node on psisep for 16 cells
  psi_min = -0.2; //arbitrary
  printf("psimin = %g\n", psi_min);
  psi_max = -0.15;
  
  // Computational grid: theta X psi X alpha (only 2D for now)
  //double clower[] = { psi_min, -0.3, -2.9 };
  //double cupper[] = {psi_max, 0.3, 2.9 };

  double clower[] = { psi_min, -0.01, -3.14 };
  double cupper[] = {psi_max, 0.01, 3.14 };
  int ccells[] = { 2, 1, 16 };



  struct gkyl_rect_grid cgrid;
  gkyl_rect_grid_init(&cgrid, 3, clower, cupper, ccells);

  printf("CGRID INFO:\n cgrid.lower = %g,%g,%g\n cgrid.upper = %g,%g,%g\n cgrid.dx= %g,%g,%g\n", cgrid.lower[0],cgrid.lower[1], cgrid.lower[2],cgrid.upper[0],cgrid.upper[1], cgrid.upper[2], cgrid.dx[0], cgrid.dx[1], cgrid.dx[2]);

  struct gkyl_range clocal, clocal_ext, conversion_range;
  int cnghost[GKYL_MAX_CDIM] = { 1, 1, 1 };
  gkyl_create_grid_ranges(&cgrid, cnghost, &clocal_ext, &clocal);

  // BCs, 0 is periodic, 1 is nonperiodic
  int bcs[3] = {1,0,1};

  // calcgeom will go into the ghost y cells based on bc. If bc[1]=1 we use ghosts.
  // Need to pass appropriate conversion to modal range depending on the bcs
  if(bcs[1]==1){
    int sublower[3] = {clocal.lower[0], clocal_ext.lower[1], clocal.lower[2]};
    int subupper[3] = {clocal.upper[0], clocal_ext.upper[1], clocal.upper[2]};
    gkyl_sub_range_init(&conversion_range, &clocal_ext, sublower, subupper);
  }
  else{
    conversion_range = clocal;
  }


  printf("clocal ext in y lower = %d, %d, %d\n", conversion_range.lower[0], conversion_range.lower[1], conversion_range.lower[2]);
  printf("clocal ext in y upper = %d, %d, %d\n", conversion_range.upper[0], conversion_range.upper[1], conversion_range.upper[2]);

  int cpoly_order = 1;
  struct gkyl_basis cbasis;
  gkyl_cart_modal_serendip(&cbasis, 3, cpoly_order);


  struct gkyl_geo_gyrokinetic_geo_inp ginp = {
    .cgrid = &cgrid,
    .bcs = bcs,
    .cbasis = &cbasis,
    .ftype = GKYL_CORE,
    .rclose = upper[0],
    .rright = upper[0],
    .rleft = lower[0],
    .zmin = -4.0,
    .zmax = 4.0,
  
    .write_node_coord_array = true,
    .node_file_nm = "cerfonclosed_nodes.gkyl"
  }; 




  //Do Ammar's calcgeom

  struct gkyl_array *mapc2p_arr = gkyl_array_new(GKYL_DOUBLE, 3*cbasis.num_basis, clocal_ext.volume);
  gkyl_geo_gyrokinetic_calcgeom(geo, &ginp, mapc2p_arr, &conversion_range);

  printf("writing mapc2p file from calcgeom\n");
  gkyl_grid_sub_array_write(&cgrid, &clocal, mapc2p_arr, "cerfonclosed_mapc2pfile.gkyl");
  printf("wrote mapc2p file\n");

  //make psi
  gkyl_eval_on_nodes *eval_psi = gkyl_eval_on_nodes_new(&rzgrid, &rzbasis, 1, psi, &sctx);
  struct gkyl_array* psidg = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);
  gkyl_eval_on_nodes_advance(eval_psi, 0.0, &rzlocal_ext, psidg); //on ghosts with ext_range
  gkyl_grid_sub_array_write(&rzgrid, &rzlocal, psidg, "cerfonclosed_psi.gkyl");

  gkyl_eval_on_nodes *eval_psibyr = gkyl_eval_on_nodes_new(&rzgrid, &rzbasis, 1, psibyr, &sctx);
  struct gkyl_array* psibyrdg = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);
  gkyl_eval_on_nodes_advance(eval_psibyr, 0.0, &rzlocal_ext, psibyrdg); //on ghosts with ext_range
                                                                       //
  gkyl_eval_on_nodes *eval_psibyr2 = gkyl_eval_on_nodes_new(&rzgrid, &rzbasis, 1, psibyr2, &sctx);
  struct gkyl_array* psibyr2dg = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);
  gkyl_eval_on_nodes_advance(eval_psibyr2, 0.0, &rzlocal_ext, psibyr2dg); //on ghosts with ext_range
  printf("made the psi arays\n");

  gkyl_eval_on_nodes *eval_bphi= gkyl_eval_on_nodes_new(&rzgrid, &rzbasis, 1, bphi_func, &sctx);
  struct gkyl_array* bphidg = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);
  gkyl_eval_on_nodes_advance(eval_bphi, 0.0, &rzlocal_ext, bphidg); //on ghosts with ext_range
  printf("made the Bphi array\n");

  //make bmag
  struct gkyl_array* bmagdg = gkyl_array_new(GKYL_DOUBLE, rzbasis.num_basis, rzlocal_ext.volume);


  struct gkyl_array* bmagFld= gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);

  printf("calculating bmag \n");
  gkyl_calc_bmag *calculator = gkyl_calc_bmag_new(&cbasis, &rzbasis, &cgrid, &rzgrid, geo, &ginp, false);
  printf("allocated bmag calculator\n");

  gkyl_calc_bmag_advance(calculator, &clocal, &clocal_ext, &rzlocal, &rzlocal_ext, psidg, psibyrdg, psibyr2dg, bphidg, bmagFld, mapc2p_arr);
  printf("advanced bmag calculator\n");



  char fileNm[1024];
  do{
    printf("writing the comp bmag file \n");
    const char *fmt = "%s_compbmag.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
    gkyl_grid_sub_array_write(&cgrid, &clocal, bmagFld, fileNm);
  } while (0);

  do{
    printf("writing the bphi file \n");
    const char *fmt = "%s_bphi.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
    gkyl_grid_sub_array_write(&rzgrid, &rzlocal, bphidg, fileNm);
  } while (0);

  printf("calculating metrics \n");
  struct gkyl_array *gFld = gkyl_array_new(GKYL_DOUBLE, 6*cbasis.num_basis, clocal_ext.volume);

  
    
  printf("creating\n");
  gkyl_calc_metric *mcalculator = gkyl_calc_metric_new(&cbasis, &cgrid, bcs, false);
  printf("advancing\n");
  //gkyl_calc_metric_advance( mcalculator, &clocal, mapc2p_arr, gFld);
  printf("DZC = %g, %g, %g\n", geo->dzc[0],  geo->dzc[1],  geo->dzc[2]);
  gkyl_calc_metric_advance(mcalculator, geo->nrange, geo->mc2p_nodal_fd, geo->dzc, gFld, &conversion_range);

  do{
    printf("writing the gij file \n");
    const char *fmt = "%s_g_ij.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
    gkyl_grid_sub_array_write(&cgrid, &clocal, gFld, fileNm);
  } while (0);

  // done doing metrics
  printf("calculating jacobian, upper metrics, etc\n");
  //Now calculate the jacobian and upper metrics
  //struct gkyl_array *bmagFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *jFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *jinvFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *grFld = gkyl_array_new(GKYL_DOUBLE, 6*cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *biFld = gkyl_array_new(GKYL_DOUBLE, 3*cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *cmagFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *jtotFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *jtotinvFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *bmaginvFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *bmaginvsqFld = gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *gxxJ= gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *gxyJ= gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  struct gkyl_array *gyyJ= gkyl_array_new(GKYL_DOUBLE, cbasis.num_basis, clocal_ext.volume);
  printf("created geo fields\n");
  gkyl_calc_derived_geo *jcalculator = gkyl_calc_derived_geo_new(&cbasis, &cgrid, false);
  printf("made the j calculator \n");
  gkyl_calc_derived_geo_advance( jcalculator, &clocal, gFld, bmagFld, jFld, jinvFld, grFld, biFld, cmagFld, jtotFld, jtotinvFld, bmaginvFld, bmaginvsqFld, gxxJ, gxyJ, gyyJ);
  

  do{
    printf("writing the  second comp bmag file \n");
    const char *fmt = "%s_compbmag2.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
    gkyl_grid_sub_array_write(&cgrid, &clocal, bmagFld, fileNm);
  } while (0);

    do{
      printf("writing the j file \n");
      const char *fmt = "%s_j.gkyl";
      snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
      gkyl_grid_sub_array_write(&cgrid, &clocal, jFld, fileNm);
    } while (0);

    do{
      printf("writing the jtot file \n");
      const char *fmt = "%s_jtot.gkyl";
      snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
      gkyl_grid_sub_array_write(&cgrid, &clocal, jtotFld, fileNm);
    } while (0);

    do{
      printf("writing the cmag file \n");
      const char *fmt = "%s_cmag.gkyl";
      snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
      gkyl_grid_sub_array_write(&cgrid, &clocal, cmagFld, fileNm);
    } while (0);
 
  
    do{
      printf("writing the g^ij file \n");
      const char *fmt = "%s_gij.gkyl";
      snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
      gkyl_grid_sub_array_write(&cgrid, &clocal, grFld, fileNm);
    } while (0);


    do{
      printf("writing the b_i file \n");
      const char *fmt = "%s_bi.gkyl";
      snprintf(fileNm, sizeof fileNm, fmt, "cerfonclosed");
      gkyl_grid_sub_array_write(&cgrid, &clocal, biFld, fileNm);
    } while (0);








  gkyl_array_release(mapc2p_arr);
  gkyl_array_release(psidg);
  gkyl_array_release(psibyrdg);
  gkyl_array_release(psibyr2dg);
  gkyl_array_release(bmagFld);
  gkyl_array_release(bmagdg);
  gkyl_array_release(cmagFld);
  gkyl_array_release(jFld);
  gkyl_array_release(jinvFld);
  gkyl_array_release(biFld);

  gkyl_array_release(jtotFld);
  gkyl_array_release(jtotinvFld);
  gkyl_array_release(bmaginvFld);
  gkyl_array_release(bmaginvsqFld);
  gkyl_array_release(gxxJ);
  gkyl_array_release(gxyJ);
  gkyl_array_release(gyyJ);
  gkyl_geo_gyrokinetic_release(geo);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("total time = %g\n", cpu_time_used);
}

TEST_LIST = {
  { "test_1", test_1},
  { NULL, NULL },
};