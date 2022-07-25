// Test updater that imposes basic BCs.
//
#include <acutest.h>

#include <gkyl_alloc.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_basis.h>
#include <gkyl_bc_basic.h>
#include <gkyl_proj_on_basis.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_rect_grid.h>

void evalFunc_1x1v(double t, const double *xn, double *restrict fout,
                   void *ctx) {
  double x = xn[0], vx = xn[1];
  fout[0] = (x) * (vx - 0.5) * (vx - 0.5);
}

void evalFunc_1x2v(double t, const double *xn, double *restrict fout,
                   void *ctx) {
  double x = xn[0], vx = xn[1], vy = xn[2];
  fout[0] = (x * x) * (vx - 0.5) * (vy - 0.5);
}

void evalFunc_2x2v(double t, const double *xn, double *restrict fout,
                   void *ctx) {
  double x = xn[0], y = xn[1];
  double vx = xn[2], vy = xn[3];
  fout[0] = x * y * (vx - 1) * (vy - 2);
}

void evalFunc_3x2v(double t, const double *xn, double *restrict fout,
                   void *ctx) {
  double x = xn[0], y = xn[1], z = xn[2];
  double vx = xn[3], vy = xn[4];
  fout[0] = (x - 1) * y * (z + 1) * (vx - 1) * (vy - 2);
}

// allocate array (filled with zeros)
static struct gkyl_array *mkarr(long nc, long size) {
  struct gkyl_array *a = gkyl_array_new(GKYL_DOUBLE, nc, size);
  return a;
}

// allocate cu_dev array
static struct gkyl_array *mkarr_cu(long nc, long size) {
  struct gkyl_array *a = gkyl_array_cu_dev_new(GKYL_DOUBLE, nc, size);
  return a;
}

struct skin_ghost_ranges {
  struct gkyl_range lower_skin[GKYL_MAX_DIM];
  struct gkyl_range lower_ghost[GKYL_MAX_DIM];

  struct gkyl_range upper_skin[GKYL_MAX_DIM];
  struct gkyl_range upper_ghost[GKYL_MAX_DIM];
};

// Create ghost and skin sub-ranges given a parent range
static void skin_ghost_ranges_init(struct skin_ghost_ranges *sgr,
                                   const struct gkyl_range *parent,
                                   const int *ghost) {
  int ndim = parent->ndim;
  for (int d = 0; d < ndim; ++d) {
    gkyl_skin_ghost_ranges(&sgr->lower_skin[d], &sgr->lower_ghost[d], d,
                           GKYL_LOWER_EDGE, parent, ghost);
    gkyl_skin_ghost_ranges(&sgr->upper_skin[d], &sgr->upper_ghost[d], d,
                           GKYL_UPPER_EDGE, parent, ghost);
  }
}

void test_bc(int cdim, int vdim, int poly_order, char *boundary_type, bool useGPU) {
  int ndim = cdim + vdim;
  double lower[ndim], upper[ndim];
  int cells[ndim];
  for (int ix = 0; ix < ndim; ix++) {
    lower[ix] = -2.0;
    upper[ix] = 2.0;
    cells[ix] = ix < cdim ? 4 : 2;
  }
  double confLower[cdim], confUpper[cdim];
  int confCells[cdim];
  for (int ix = 0; ix < cdim; ix++) {
    confLower[ix] = lower[ix];
    confUpper[ix] = upper[ix];
    confCells[ix] = cells[ix];
  }

  // grids
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);
  struct gkyl_rect_grid confGrid;
  gkyl_rect_grid_init(&confGrid, cdim, confLower, confUpper, confCells);

  // basis functions
  struct gkyl_basis basis, confBasis;
  gkyl_cart_modal_serendip(&basis, ndim, poly_order);
  gkyl_cart_modal_serendip(&confBasis, cdim, poly_order);

  int confGhost[cdim];
  for (int d = 0; d < cdim; d++) {
    confGhost[d] = 1;
  }
  struct gkyl_range confLocal,
      confLocal_ext; // local, local-ext conf-space ranges
  gkyl_create_grid_ranges(&confGrid, confGhost, &confLocal_ext, &confLocal);
  struct skin_ghost_ranges confSkin_ghost; // conf-space skin/ghost
  skin_ghost_ranges_init(&confSkin_ghost, &confLocal_ext, confGhost);

  int ghost[ndim];
  for (int d = 0; d < cdim; d++) {
    ghost[d] = confGhost[d];
  }
  for (int d = cdim; d < ndim; d++) {
    ghost[d] = 0;
  }

  struct gkyl_range local, local_ext; // local, local-ext phase-space ranges
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);
  struct skin_ghost_ranges skin_ghost; // phase-space skin/ghost
  skin_ghost_ranges_init(&skin_ghost, &local_ext, ghost);

  // projection updater for dist-function
  gkyl_proj_on_basis *projDistf;
  if (cdim == 1 && vdim == 1) {
    projDistf = gkyl_proj_on_basis_new(&grid, &basis, poly_order + 1, 1,
                                       evalFunc_1x1v, NULL);
  } else if (cdim == 1 && vdim == 2) {
    projDistf = gkyl_proj_on_basis_new(&grid, &basis, poly_order + 1, 1,
                                       evalFunc_1x2v, NULL);
  } else if (cdim == 2 && vdim == 2) {
    projDistf = gkyl_proj_on_basis_new(&grid, &basis, poly_order + 1, 1,
                                       evalFunc_2x2v, NULL);
  } else if (cdim == 3 && vdim == 2) {
    projDistf = gkyl_proj_on_basis_new(&grid, &basis, poly_order + 1, 1,
                                       evalFunc_3x2v, NULL);
  }

  // create distribution function array
  struct gkyl_array *distf;
  distf = mkarr(basis.num_basis, local_ext.volume);

  // project distribution function on basis
  gkyl_proj_on_basis_advance(projDistf, 0.0, &local_ext, distf);


  // Create the boundary condition
  // GPU notes:
  // Change the false in dg_vlasov_new to true to trip the flag inside
  // bc_create(). Put copy_buffer() fxns on GPU too. Need distf on gpu. 
  for (int bc_dir = 0; bc_dir < cdim; bc_dir++) {

    // Determine the size of the BC buffer
    long buff_sz = 0;
    for (int d = 0; d < cdim; ++d) {
      long vol = skin_ghost.lower_skin[d].volume;
      buff_sz = buff_sz > vol ? buff_sz : vol;
    }
    struct gkyl_array *bc_buffer;
    bc_buffer = mkarr(basis.num_basis, buff_sz);

    // Apply BC to the lower ghost cells
    struct gkyl_bc_basic *bclo;
    if (strcmp(boundary_type, "reflect") == 0) {
      bclo = gkyl_bc_basic_new(bc_dir, GKYL_LOWER_EDGE, &local_ext, ghost, GKYL_BC_REFLECT, &basis, cdim, useGPU);
    } else if (strcmp(boundary_type, "absorb") == 0) {
      bclo = gkyl_bc_basic_new(bc_dir, GKYL_LOWER_EDGE, &local_ext, ghost, GKYL_BC_ABSORB, &basis, cdim, useGPU);
    }

    gkyl_bc_basic_advance(bclo, bc_buffer, distf);
    gkyl_bc_basic_release(bclo);

    // Apply BC to the upper ghost cells
    struct gkyl_bc_basic *bcup;
    if (strcmp(boundary_type, "reflect") == 0) {
      bcup = gkyl_bc_basic_new(bc_dir, GKYL_UPPER_EDGE, &local_ext, ghost, GKYL_BC_REFLECT, &basis, cdim, useGPU);
    } else if (strcmp(boundary_type, "absorb") == 0) {
      bcup = gkyl_bc_basic_new(bc_dir, GKYL_UPPER_EDGE, &local_ext, ghost, GKYL_BC_ABSORB, &basis, cdim, useGPU);
    }

    gkyl_bc_basic_advance(bcup, bc_buffer, distf);
    gkyl_bc_basic_release(bcup);

    gkyl_array_release(bc_buffer);
  }

  // Check lower ghost cells after applying BC
  struct gkyl_range_iter iter, iter_skin;
  for (int d = 0; d < cdim; d++) {
    gkyl_range_iter_init(&iter, &skin_ghost.lower_ghost[d]);
    while (gkyl_range_iter_next(&iter)) {
      // Find the index and value of f at the ghost and adjacent skin cells
      iter_skin = iter;
      iter_skin.idx[d] = iter.idx[d] + 1;
      int linidx_ghost = gkyl_range_idx(skin_ghost.lower_ghost, iter.idx);
      int linidx_skin  = gkyl_range_idx(skin_ghost.lower_skin,   iter_skin.idx);
      const double *val_ghost = gkyl_array_cfetch(distf, linidx_ghost);
      const double *val_skin  = gkyl_array_cfetch(distf,  linidx_skin);

      if (strcmp(boundary_type, "reflect") == 0) {
        // Flip the skin value to manually apply reflect BC to skin cell
        double val_correct[basis.num_basis];
        basis.flip_odd_sign(d,        val_skin,    val_correct);
        basis.flip_odd_sign(d + cdim, val_correct, val_correct);

        // Check values
        // printf("\n\nCell %i and %i where d=%i:", linidx_ghost,
        // linidx_skin,d);
        for (int i = 0; i < basis.num_basis; i++) {
          TEST_CHECK(gkyl_compare(val_ghost[i], val_correct[i], 1e-12));
          // printf("   %10.4f  %10.4f",val_ghost[i],val_correct[i]);
        }
      } else if (strcmp(boundary_type, "absorb") == 0) {
        for (int i = 0; i < basis.num_basis; i++) {
          TEST_CHECK(gkyl_compare(val_ghost[i], 0, 1e-12));
        }
      }
    }

    // Check upper ghost cells after applying BC
    gkyl_range_iter_init(&iter, &skin_ghost.upper_ghost[d]);
    while (gkyl_range_iter_next(&iter)) {
      // Find the index and value of f at the ghost and adjacent skin cells
      iter_skin = iter;
      iter_skin.idx[d] = iter.idx[d] - 1;
      int linidx_ghost = gkyl_range_idx(skin_ghost.upper_ghost, iter.idx);
      int linidx_skin  = gkyl_range_idx(skin_ghost.upper_skin,  iter_skin.idx);
      const double *val_ghost = gkyl_array_cfetch(distf, linidx_ghost);
      const double *val_skin  = gkyl_array_cfetch(distf, linidx_skin);

      if (strcmp(boundary_type, "reflect") == 0) {
        // Flip the skin value to manually apply reflect BC to skin cell
        double val_correct[basis.num_basis];
        basis.flip_odd_sign(d,        val_skin,    val_correct);
        basis.flip_odd_sign(d + cdim, val_correct, val_correct);

        // Check values
        for (int i = 0; i < basis.num_basis; i++) {
          TEST_CHECK(gkyl_compare(val_ghost[i], val_correct[i], 1e-12));
        }
      } else if (strcmp(boundary_type, "absorb") == 0) {
        for (int i = 0; i < basis.num_basis; i++) {
          TEST_CHECK(gkyl_compare(val_ghost[i], 0, 1e-12));
        }
      }
    }
  }

  // release memory for moment data object
  gkyl_proj_on_basis_release(projDistf);
  gkyl_array_release(distf);
}

void test_bc_reflect_1x1v_p1() { test_bc(1, 1, 1, "reflect",  false); }
void test_bc_reflect_1x2v_p1() { test_bc(1, 2, 1, "reflect",  false); }
void test_bc_reflect_2x2v_p1() { test_bc(2, 2, 1, "reflect",  false); }
void test_bc_reflect_3x2v_p1() { test_bc(2, 2, 1, "reflect",  false); }
void test_bc_reflect_1x1v_p2() { test_bc(1, 1, 2, "reflect",  false); }
void test_bc_reflect_1x2v_p2() { test_bc(1, 2, 2, "reflect",  false); }
void test_bc_reflect_2x2v_p2() { test_bc(2, 2, 2, "reflect",  false); }
void test_bc_reflect_3x2v_p2() { test_bc(2, 2, 2, "reflect",  false); }
void test_bc_absorb_1x1v_p1() { test_bc(1, 1, 1, "absorb",false); }
void test_bc_absorb_1x2v_p1() { test_bc(1, 2, 1, "absorb",false); }
void test_bc_absorb_2x2v_p1() { test_bc(2, 2, 1, "absorb",false); }
void test_bc_absorb_3x2v_p1() { test_bc(2, 2, 1, "absorb",false); }
void test_bc_absorb_1x1v_p2() { test_bc(1, 1, 2, "absorb",false); }
void test_bc_absorb_1x2v_p2() { test_bc(1, 2, 2, "absorb",false); }
void test_bc_absorb_2x2v_p2() { test_bc(2, 2, 2, "absorb",false); }
void test_bc_absorb_3x2v_p2() { test_bc(2, 2, 2, "absorb",false); }

void test_bc_reflect_1x1v_p1_gpu(){ test_bc(1, 1, 1, "reflect",  true); }

TEST_LIST = {
    {"test_bc_reflect_1x1v_p1", test_bc_reflect_1x1v_p1},
    {"test_bc_reflect_1x2v_p1", test_bc_reflect_1x2v_p1},
    {"test_bc_reflect_2x2v_p1", test_bc_reflect_2x2v_p1},
    {"test_bc_reflect_3x2v_p1", test_bc_reflect_3x2v_p1},
    {"test_bc_reflect_1x1v_p2", test_bc_reflect_1x1v_p2},
    {"test_bc_reflect_1x2v_p2", test_bc_reflect_1x2v_p2},
    {"test_bc_reflect_2x2v_p2", test_bc_reflect_2x2v_p2},
    {"test_bc_reflect_3x2v_p2", test_bc_reflect_3x2v_p2},
    {"test_bc_absorb_1x1v_p1", test_bc_absorb_1x1v_p1},
    {"test_bc_absorb_1x2v_p1", test_bc_absorb_1x2v_p1},
    {"test_bc_absorb_2x2v_p1", test_bc_absorb_2x2v_p1},
    {"test_bc_absorb_3x2v_p1", test_bc_absorb_3x2v_p1},
    {"test_bc_absorb_1x1v_p2", test_bc_absorb_1x1v_p2},
    {"test_bc_absorb_1x2v_p2", test_bc_absorb_1x2v_p2},
    {"test_bc_absorb_2x2v_p2", test_bc_absorb_2x2v_p2},
    {"test_bc_absorb_3x2v_p2", test_bc_absorb_3x2v_p2},
    {"test_bc_reflect_1x1v_p1_gpu", test_bc_reflect_1x1v_p1_gpu},
    {NULL, NULL},
};