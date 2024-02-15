#include <acutest.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_dg_bin_ops.h>
#include <gkyl_correct_maxwellian_gyrokinetic.h>
#include <gkyl_gk_geometry.h>
#include <gkyl_gk_geometry_mapc2p.h>
#include <gkyl_mom_calc.h>
#include <gkyl_mom_gyrokinetic.h>
#include <gkyl_proj_maxwellian_on_basis.h>
#include <gkyl_proj_on_basis.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_util.h>
#include <math.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_app_priv.h>

void
mapc2p_3x(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx)
{
  xp[0] = xc[0]; xp[1] = xc[1]; xp[2] = xc[2];
}


void
bmag_func_3x(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0], y = xc[1], z = xc[2];
  fout[0] = 0.5;
}

void eval_M0(double t, const double *xn, double *restrict fout, void *ctx)
{
  double x = xn[0];
  fout[0] = 1.0e19*(1.0 + 0.5*cos(x));
}
void eval_vtsq(double t, const double *xn, double* restrict fout, void *ctx)
{
  double x = xn[0];
  double vtsq = (10.0*1.602e-19/9.1e-31)*exp(-x*x/(M_PI));
  fout[0] = vtsq;
}
void eval_M1(double t, const double *xn, double *restrict fout, void *ctx)
{
  double x = xn[0];
  double M0[1];
  eval_M0(t, xn, M0, ctx);
  double n = M0[0];

  double vtsq[1];
  eval_vtsq(t, xn, vtsq, ctx);
  double vt = sqrt(vtsq[0]); 

  fout[0] = n*vt;
}
void eval_M2(double t, const double *xn, double* restrict fout, void *ctx)
{ 
  double x = xn[0];
  double M0[1];
  eval_M0(t, xn, M0, ctx);
  double n = M0[0];

  double vtsq[1];
  eval_vtsq(t, xn, vtsq, ctx);

  double M1[1];
  eval_M1(t, xn, M1, ctx);
  fout[0] = M0[0]*vtsq[0] + M1[0]*M1[0]/M0[0];
}

void test_1x1v(int poly_order, bool use_gpu)
{
  double mass = 9.1e-31;
  double err_max = 1e-14, iter_max = 50;
  double vt = sqrt(10.0*1.602e-19/9.1e-31); // reference temperature
  double lower[] = {-M_PI, -5.0*vt}, upper[] = {M_PI, 5.0*vt};
  int cells[] = {4, 16};
  int vdim = 1, cdim = 1;
  int ndim = cdim + vdim;
  double confLower[] = {lower[0]}, confUpper[] = {upper[0]};
  int confCells[] = {cells[0]};
  
  // Grids
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);
  struct gkyl_rect_grid confGrid;
  gkyl_rect_grid_init(&confGrid, cdim, confLower, confUpper, confCells);

  // Basis functions
  struct gkyl_basis basis, confBasis;
  if (poly_order==1)
    gkyl_cart_modal_gkhybrid(&basis, cdim, vdim);
  else
    gkyl_cart_modal_serendip(&basis, ndim, poly_order);
  gkyl_cart_modal_serendip(&confBasis, cdim, poly_order);

  // Configuration space range
  int confGhost[] = {1};
  struct gkyl_range confLocal, confLocal_ext;
  gkyl_create_grid_ranges(&confGrid, confGhost, &confLocal_ext, &confLocal);

  // Phase space range
  int ghost[] = {confGhost[0], 0};
  struct gkyl_range local, local_ext;
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);

  // Initialize geometry
  struct gkyl_gyrokinetic_geometry geometry_input = {
      .geometry_id = GKYL_MAPC2P,
      .world = {0.0, 0.0},
      .mapc2p = mapc2p_3x, // mapping of computational to physical space
      .c2p_ctx = 0,
      .bmag_func = bmag_func_3x, // magnetic field magnitude
      .bmag_ctx =0 
  };
  struct gkyl_rect_grid geo_grid;
  struct gkyl_range geo_local;
  struct gkyl_range geo_local_ext;
  struct gkyl_basis geo_basis;
  geo_grid = augment_grid(confGrid, geometry_input);
  gkyl_create_grid_ranges(&geo_grid, confGhost, &geo_local_ext, &geo_local);
  gkyl_cart_modal_serendip(&geo_basis, 3, poly_order);
  struct gk_geometry* gk_geom_3d;
  gk_geom_3d = gkyl_gk_geometry_mapc2p_new(&geo_grid, &geo_local, &geo_local_ext, &geo_basis, 
      geometry_input.mapc2p, geometry_input.c2p_ctx, geometry_input.bmag_func,  geometry_input.bmag_ctx, false);
  // deflate geometry if necessary
  struct gk_geometry *gk_geom = gkyl_gk_geometry_deflate(gk_geom_3d, &confGrid, &confLocal, &confLocal_ext, 
      &confBasis, false);
  gkyl_gk_geometry_release(gk_geom_3d);

  // Create correct moment arrays
  struct gkyl_array *m0_in_ho, *m1_in_ho, *m2_in_ho;
  m0_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  m1_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  m2_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  gkyl_proj_on_basis *proj_m0 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M0, NULL);
  gkyl_proj_on_basis *proj_m1 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M1, NULL);
  gkyl_proj_on_basis *proj_m2 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M2, NULL);
  gkyl_proj_on_basis_advance(proj_m0, 0.0, &confLocal, m0_in_ho);
  gkyl_proj_on_basis_advance(proj_m1, 0.0, &confLocal, m1_in_ho);
  gkyl_proj_on_basis_advance(proj_m2, 0.0, &confLocal, m2_in_ho);
  struct gkyl_array *m0_in, *m1_in, *m2_in;
  if (use_gpu) {
    m0_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    m1_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    m2_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    gkyl_array_copy(m0_in, m0_in_ho);
    gkyl_array_copy(m1_in, m1_in_ho);
    gkyl_array_copy(m2_in, m2_in_ho);
  } else {
    m0_in = m0_in_ho;
    m1_in = m1_in_ho;
    m2_in = m2_in_ho;
  }

  // Project the Maxwellian on basis
  // (1) proj_maxwellian expects the moments as a single array
  struct gkyl_array *moms_in = mkarr(use_gpu, 3*confBasis.num_basis, confLocal_ext.volume);
  gkyl_array_set_offset(moms_in, 1., m0_in, 0*confBasis.num_basis);
  gkyl_array_set_offset(moms_in, 1., m1_in, 1*confBasis.num_basis);
  gkyl_array_set_offset(moms_in, 1., m2_in, 2*confBasis.num_basis);
  // (2) create distribution function array
  gkyl_proj_maxwellian_on_basis *proj_maxwellian = gkyl_proj_maxwellian_on_basis_new(&grid, &confBasis, &basis, poly_order+1, use_gpu);
  struct gkyl_array *fM = mkarr(use_gpu, basis.num_basis, local_ext.volume);
  gkyl_proj_gkmaxwellian_on_basis_lab_mom(proj_maxwellian, &local, &confLocal, moms_in, gk_geom->bmag, gk_geom->jacobtot, mass, fM);
  // (3) copy from device to host
  struct gkyl_array *fM_ho;
  if (use_gpu) {
    fM_ho = mkarr(basis.num_basis, local_ext.volume, false);
    gkyl_array_copy(fM_ho, fM);
  } else {
    fM_ho = fM;
  }
  // Optionally write out the inital distribution function and desired initial moments  
  // char fname_fM_ic[1024];
  // sprintf(fname_fM_ic, "ctest_correct_maxwellian_%dx%dv_p%d.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&grid, &local, fM_ho, fname_fM_ic);
  // char moms_ic[1024];
  // sprintf(moms_ic, "ctest_correct_maxwellian_moms_%dx%dv_p%d.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&confGrid, &confLocal, moms_in, moms_ic);
  // Create a Maxwellian with corrected moments
  struct gkyl_correct_maxwellian_gyrokinetic_inp inp = {
    .phase_grid = &grid,
    .conf_grid = &confGrid,
    .phase_basis = &basis,
    .conf_basis = &confBasis,

    .conf_local = &confLocal,
    .conf_local_ext = &confLocal_ext,
    .mass = mass, 
    .gk_geom = gk_geom,
    .max_iter = 50, 
    .eps_err = 1.0e-14, 
    .use_gpu = use_gpu
  };
  gkyl_correct_maxwellian_gyrokinetic *corr_max = gkyl_correct_maxwellian_gyrokinetic_new(&inp);
  gkyl_correct_maxwellian_gyrokinetic_advance(corr_max, fM, moms_in, &confLocal, &local);
  gkyl_correct_maxwellian_gyrokinetic_release(corr_max);
  if (use_gpu) {
    gkyl_array_copy(fM_ho, fM);
  } else {
    fM_ho = fM;
  }
  // Calculate the corrected moments
  // (1) create the calculators
  struct gkyl_mom_type *MOMS_t = gkyl_mom_gyrokinetic_new(&confBasis, &basis, &confLocal, mass, gk_geom, "ThreeMoments", use_gpu);
  gkyl_mom_calc *momsCalc = gkyl_mom_calc_new(&grid, MOMS_t, use_gpu);
  // (2) calculate the moments and copy from host to device
  struct gkyl_array *moms_corr = mkarr(use_gpu, 3*confBasis.num_basis, confLocal_ext.volume);
  struct gkyl_array *moms_corr_ho;
  if (use_gpu) { 
    gkyl_mom_calc_advance_cu(momsCalc, &local, &confLocal, fM, moms_corr); 
    moms_corr_ho = mkarr(3*confBasis.num_basis, confLocal_ext.volume, false);
    gkyl_array_copy(moms_corr_ho, moms_corr); 
  } else {
    gkyl_mom_calc_advance(momsCalc, &local, &confLocal, fM, moms_corr);
    moms_corr_ho = moms_corr;
  }
  // Optionally write out the corrected distribution function and moments  
  // char fname_fM_corr[1024];
  // sprintf(fname_fM_corr, "ctest_correct_maxwellian_%dx%dv_p%d_corr.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&grid, &local, fM_ho, fname_fM_corr);
  // char moms_corr_ic[1024];
  // sprintf(moms_corr_ic, "ctest_correct_maxwellian_moms_%dx%dv_p%d_corr.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&confGrid, &confLocal,moms_corr_ho, moms_corr_ic);

  // (4) compare the correct moments with the input moments
  for (int k=0; k<cells[0]; k++) {
    int idx[] = {k+1};
    long linidx = gkyl_range_idx(&confLocal, idx);
    const double *m0in = gkyl_array_cfetch(m0_in_ho, linidx);
    const double *m1in = gkyl_array_cfetch(m1_in_ho, linidx);
    const double *m2in = gkyl_array_cfetch(m2_in_ho, linidx);
    const double *momsCorr = gkyl_array_cfetch(moms_corr_ho, linidx);
    for (int m=0; m<confBasis.num_basis; m++) {
      TEST_CHECK( gkyl_compare(m0in[m], momsCorr[m+0*confBasis.num_basis], 1e-12) );
      TEST_CHECK( gkyl_compare(m1in[m], momsCorr[m+1*confBasis.num_basis], 1e-12) );
      TEST_CHECK( gkyl_compare(m2in[m], momsCorr[m+2*confBasis.num_basis], 1e-12) );
      TEST_MSG("Expected: %.13e, \t%.13e, \t%.13e, \tin cell (%d)", m0in[m], m1in[m], m2in[m], idx[0]);
      TEST_MSG("Produced: %.13e, \t%.13e, \t%.13e", momsCorr[m+0*confBasis.num_basis], momsCorr[m+1*confBasis.num_basis], momsCorr[m+2*confBasis.num_basis]);
    }
  }
 
  // Release memory for moment data object
  gkyl_gk_geometry_release(gk_geom);  
  gkyl_array_release(m0_in);
  gkyl_array_release(m1_in);
  gkyl_array_release(m2_in);
  gkyl_array_release(moms_in);
  gkyl_array_release(moms_corr);
  gkyl_array_release(fM);
  if (use_gpu) {
    gkyl_array_release(m0_in_ho);
    gkyl_array_release(m1_in_ho);
    gkyl_array_release(m2_in_ho);
    gkyl_array_release(moms_corr_ho);
    gkyl_array_release(fM_ho);
  }
  gkyl_proj_on_basis_release(proj_m0);
  gkyl_proj_on_basis_release(proj_m1);
  gkyl_proj_on_basis_release(proj_m2);
  gkyl_proj_maxwellian_on_basis_release(proj_maxwellian);
  gkyl_mom_calc_release(momsCalc);
  gkyl_mom_type_release(MOMS_t);
}

void test_1x2v(int poly_order, bool use_gpu)
{
  double mass = 1.0;
  double err_max = 1e-14, iter_max = 50;
  double vt = sqrt(10.0*1.602e-19/9.1e-31); // reference temperature
  double lower[] = {-M_PI, -5.0*vt, 0.0}, upper[] = {M_PI, 5.0*vt, 5.0*vt*vt};
  int cells[] = {4, 16, 16};
  int vdim = 2, cdim = 1;
  int ndim = cdim + vdim;

  double confLower[] = {lower[0]}, confUpper[] = {upper[0]};
  int confCells[] = {cells[0]};

  // Grids
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);
  struct gkyl_rect_grid confGrid;
  gkyl_rect_grid_init(&confGrid, cdim, confLower, confUpper, confCells);

  // Basis functions
  struct gkyl_basis basis, confBasis;
  if (poly_order==1)
    gkyl_cart_modal_gkhybrid(&basis, cdim, vdim);
  else
    gkyl_cart_modal_serendip(&basis, ndim, poly_order);
  gkyl_cart_modal_serendip(&confBasis, cdim, poly_order);

  // Configuration space range
  int confGhost[] = {1};
  struct gkyl_range confLocal, confLocal_ext;
  gkyl_create_grid_ranges(&confGrid, confGhost, &confLocal_ext, &confLocal);

  // Phase space range
  int ghost[] = {confGhost[0], 0, 0};
  struct gkyl_range local, local_ext;
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);

  // Initialize geometry
  struct gkyl_gyrokinetic_geometry geometry_input = {
      .geometry_id = GKYL_MAPC2P,
      .world = {0.0, 0.0},
      .mapc2p = mapc2p_3x, // mapping of computational to physical space
      .c2p_ctx = 0,
      .bmag_func = bmag_func_3x, // magnetic field magnitude
      .bmag_ctx =0 
  };
  struct gkyl_rect_grid geo_grid;
  struct gkyl_range geo_local;
  struct gkyl_range geo_local_ext;
  struct gkyl_basis geo_basis;
  geo_grid = augment_grid(confGrid, geometry_input);
  gkyl_create_grid_ranges(&geo_grid, confGhost, &geo_local_ext, &geo_local);
  gkyl_cart_modal_serendip(&geo_basis, 3, poly_order);
  struct gk_geometry* gk_geom_3d;
  gk_geom_3d = gkyl_gk_geometry_mapc2p_new(&geo_grid, &geo_local, &geo_local_ext, &geo_basis, 
      geometry_input.mapc2p, geometry_input.c2p_ctx, geometry_input.bmag_func,  geometry_input.bmag_ctx, false);
  // deflate geometry if necessary
  struct gk_geometry *gk_geom = gkyl_gk_geometry_deflate(gk_geom_3d, &confGrid, &confLocal, &confLocal_ext, 
      &confBasis, false);
  gkyl_gk_geometry_release(gk_geom_3d);

  // Create correct moment arrays
  struct gkyl_array *m0_in_ho, *m1_in_ho, *m2_in_ho;
  m0_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  m1_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  m2_in_ho = mkarr(false, confBasis.num_basis, confLocal_ext.volume);
  gkyl_proj_on_basis *proj_m0 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M0, NULL);
  gkyl_proj_on_basis *proj_m1 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M1, NULL);
  gkyl_proj_on_basis *proj_m2 = gkyl_proj_on_basis_new(&confGrid, &confBasis, poly_order+1, 1, eval_M2, NULL);
  gkyl_proj_on_basis_advance(proj_m0, 0.0, &confLocal, m0_in_ho);
  gkyl_proj_on_basis_advance(proj_m1, 0.0, &confLocal, m1_in_ho);
  gkyl_proj_on_basis_advance(proj_m2, 0.0, &confLocal, m2_in_ho);
  struct gkyl_array *m0_in, *m1_in, *m2_in;
  if (use_gpu) {
    m0_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    m1_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    m2_in = mkarr(use_gpu, confBasis.num_basis, confLocal_ext.volume);
    gkyl_array_copy(m0_in, m0_in_ho);
    gkyl_array_copy(m1_in, m1_in_ho);
    gkyl_array_copy(m2_in, m2_in_ho);
  } else {
    m0_in = m0_in_ho;
    m1_in = m1_in_ho;
    m2_in = m2_in_ho;
  }

  // Project the Maxwellian on basis
  // (1) proj_maxwellian expects the moments as a single array
  struct gkyl_array *moms_in = mkarr(use_gpu, 3*confBasis.num_basis, confLocal_ext.volume);
  gkyl_array_set_offset(moms_in, 1., m0_in, 0*confBasis.num_basis);
  gkyl_array_set_offset(moms_in, 1., m1_in, 1*confBasis.num_basis);
  gkyl_array_set_offset(moms_in, 1., m2_in, 2*confBasis.num_basis);
  // (2) create distribution function array
  gkyl_proj_maxwellian_on_basis *proj_maxwellian = gkyl_proj_maxwellian_on_basis_new(&grid, &confBasis, &basis, poly_order+1, use_gpu);
  struct gkyl_array *fM = mkarr(use_gpu, basis.num_basis, local_ext.volume);
  gkyl_proj_gkmaxwellian_on_basis_lab_mom(proj_maxwellian, &local, &confLocal, moms_in, gk_geom->bmag, gk_geom->jacobtot, mass, fM);
  // (3) copy from device to host
  struct gkyl_array *fM_ho;
  if (use_gpu) {
    fM_ho = mkarr(basis.num_basis, local_ext.volume, false);
    gkyl_array_copy(fM_ho, fM);
  } else {
    fM_ho = fM;
  }
  // Optionally write out the initial Maxwellian and initial desired moments
  // char fname_fM_ic[1024];
  // sprintf(fname_fM_ic, "ctest_correct_maxwellian_%dx%dv_p%d.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&grid, &local, fM_ho, fname_fM_ic);
  // char moms_ic[1024];
  // sprintf(moms_ic, "ctest_correct_maxwellian_moms_%dx%dv_p%d.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&confGrid, &confLocal, moms_in, moms_ic);

  // Create a Maxwellian with corrected moments
  struct gkyl_correct_maxwellian_gyrokinetic_inp inp = {
    .phase_grid = &grid,
    .conf_grid = &confGrid,
    .phase_basis = &basis,
    .conf_basis = &confBasis,

    .conf_local = &confLocal,
    .conf_local_ext = &confLocal_ext,
    .mass = mass, 
    .gk_geom = gk_geom,
    .max_iter = 50, 
    .eps_err = 1.0e-14, 
    .use_gpu = use_gpu
  };
  gkyl_correct_maxwellian_gyrokinetic *corr_max = gkyl_correct_maxwellian_gyrokinetic_new(&inp);
  gkyl_correct_maxwellian_gyrokinetic_advance(corr_max, fM, moms_in,&confLocal, &local);
  gkyl_correct_maxwellian_gyrokinetic_release(corr_max);
  if (use_gpu) {
    gkyl_array_copy(fM_ho, fM);
  } else {
    fM_ho = fM;
  }
  // Calculate the corrected moments
  // (1) create the calculators
  struct gkyl_mom_type *MOMS_t = gkyl_mom_gyrokinetic_new(&confBasis, &basis, &confLocal, mass, gk_geom, "ThreeMoments", use_gpu);
  gkyl_mom_calc *momsCalc = gkyl_mom_calc_new(&grid, MOMS_t, use_gpu);
  // (2) calculate the moments and copy from host to device
  struct gkyl_array *moms_corr = mkarr(use_gpu, 3*confBasis.num_basis, confLocal_ext.volume);
  struct gkyl_array *moms_corr_ho;
  if (use_gpu) { 
    gkyl_mom_calc_advance_cu(momsCalc, &local, &confLocal, fM, moms_corr); 
    moms_corr_ho = mkarr(3*confBasis.num_basis, confLocal_ext.volume, false);
    gkyl_array_copy(moms_corr_ho, moms_corr); 
  } else {
    gkyl_mom_calc_advance(momsCalc, &local, &confLocal, fM, moms_corr);
    moms_corr_ho = moms_corr;
  }
  // Optionally write out the corrected distribution function and moments  
  // char fname_fM_corr[1024];
  // sprintf(fname_fM_corr, "ctest_correct_maxwellian_%dx%dv_p%d_corr.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&grid, &local, fM_ho, fname_fM_corr);
  // char moms_corr_ic[1024];
  // sprintf(moms_corr_ic, "ctest_correct_maxwellian_moms_%dx%dv_p%d_corr.gkyl", cdim, vdim, poly_order);
  // gkyl_grid_sub_array_write(&confGrid, &confLocal,moms_corr_ho, moms_corr_ic);

  // (4) compare the correct moments with the input moments
  for (int k=0; k<cells[0]; k++) {
    int idx[] = {k+1};
    long linidx = gkyl_range_idx(&confLocal, idx);
    const double *m0in = gkyl_array_cfetch(m0_in_ho, linidx);
    const double *m1in = gkyl_array_cfetch(m1_in_ho, linidx);
    const double *m2in = gkyl_array_cfetch(m2_in_ho, linidx);
    const double *momsCorr = gkyl_array_cfetch(moms_corr_ho, linidx);
    for (int m=0; m<confBasis.num_basis; m++) {
      TEST_CHECK( gkyl_compare(m0in[m], momsCorr[m+0*confBasis.num_basis], 1e-12) );
      TEST_CHECK( gkyl_compare(m1in[m], momsCorr[m+1*confBasis.num_basis], 1e-12) );
      TEST_CHECK( gkyl_compare(m2in[m], momsCorr[m+2*confBasis.num_basis], 1e-12) );
      TEST_MSG("Expected: %.13e, \t%.13e, \t%.13e, \tin cell (%d)", m0in[m], m1in[m], m2in[m], idx[0]);
      TEST_MSG("Produced: %.13e, \t%.13e, \t%.13e", momsCorr[m+0*confBasis.num_basis], momsCorr[m+1*confBasis.num_basis], momsCorr[m+2*confBasis.num_basis]);
    }
  }

  // Release memory for moment data object
  gkyl_gk_geometry_release(gk_geom);  
  gkyl_array_release(m0_in);
  gkyl_array_release(m1_in);
  gkyl_array_release(m2_in);
  gkyl_array_release(moms_in);
  gkyl_array_release(moms_corr);
  gkyl_array_release(fM);
  if (use_gpu) {
    gkyl_array_release(m0_in_ho);
    gkyl_array_release(m1_in_ho);
    gkyl_array_release(m2_in_ho);
    gkyl_array_release(moms_corr_ho);
    gkyl_array_release(fM_ho);
  }
  gkyl_proj_on_basis_release(proj_m0);
  gkyl_proj_on_basis_release(proj_m1);
  gkyl_proj_on_basis_release(proj_m2);
  gkyl_proj_maxwellian_on_basis_release(proj_maxwellian);
  gkyl_mom_calc_release(momsCalc);
  gkyl_mom_type_release(MOMS_t);
}

// Run the test
void test_1x1v_p1() {test_1x1v(1, false);}
void test_1x1v_p2() {test_1x1v(2, false);}
void test_1x2v_p1() {test_1x2v(1, false);}
void test_1x2v_p2() {test_1x2v(2, false);}

#ifdef GKYL_HAVE_CUDA
void test_1x1v_p1_gpu() {test_1x1v(1, true);}
void test_1x1v_p2_gpu() {test_1x1v(2, true);}
void test_1x2v_p1_gpu() {test_1x2v(1, true);}
void test_1x2v_p2_gpu() {test_1x2v(2, true);}
#endif

TEST_LIST = {
  {"test_1x1v_p1", test_1x1v_p1},
  {"test_1x1v_p2", test_1x1v_p2},
  {"test_1x2v_p1", test_1x2v_p1},
  {"test_1x2v_p2", test_1x2v_p2},
#ifdef GKYL_HAVE_CUDA
  {"test_1x1v_p1_gpu", test_1x1v_p1_gpu},
  {"test_1x1v_p2_gpu", test_1x1v_p2_gpu},
  {"test_1x2v_p1_gpu", test_1x2v_p1_gpu},
  {"test_1x2v_p2_gpu", test_1x2v_p2_gpu},
#endif
  {NULL, NULL},
};
