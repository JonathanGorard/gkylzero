// Test integration of a gkyl_array over a range.
//

#include <acutest.h>
#include <gkyl_rect_grid.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_proj_on_basis.h>
#include <gkyl_array_integrate.h>
#include <gkyl_array_average.h>
#include <math.h>
#include <assert.h>
#include <gkyl_dg_bin_ops.h>

// allocate array (filled with zeros)
static struct gkyl_array*
mkarr(long nc, long size, bool use_gpu)
{
  struct gkyl_array *a = use_gpu? gkyl_array_cu_dev_new(GKYL_DOUBLE, nc, size)
                                : gkyl_array_new(GKYL_DOUBLE, nc, size);
  return a;
}
//----------------- TEST 1x ------------------
// function to average
void evalFunc_1x(double t, const double *xn, double* restrict fout, void *ctx)
{
  double x = xn[0];
  double lower[] = {-4.0}, upper[] = {6.0}; // Has to match the test below.
  double Lx = upper[0]-lower[0];
  double k_x = 2.*M_PI/Lx;
  double phi = 0.5;

  fout[0] = x*sin(k_x*x + phi);
}
// to weight the integral
void evalWeight_1x(double t, const double *xn, double* restrict fout, void *ctx)
{
  double x = xn[0];
  fout[0] = 1+x*x;
}

double solution_1x() { 
  // Solution from a trapz integration with Python (see code at the end)
  return -0.4189328208844751;
  }

void test_1x(int poly_order, bool use_gpu)
{
  printf("\n");

  //------------- 1. Define grid and basis ----------------
  // Define grid parameters
  double lower[] = {-4.0}, upper[] = {6.0};
  int cells[] = {32};
  int ndim = sizeof(lower) / sizeof(lower[0]);
  // Initialize the grid
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);
  // Initialize the polynomial basis
  struct gkyl_basis basis;
  gkyl_cart_modal_serendip(&basis, ndim, poly_order);
  // Create ranges (local and extended) to include ghost cells
  int ghost[] = {1};
  struct gkyl_range local, local_ext;
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);

  //------------- 2. Project the target function ----------------
  // Create a projection updater for the target function
  gkyl_proj_on_basis *projf = gkyl_proj_on_basis_new(
      &grid, &basis, poly_order + 1, 1, evalFunc_1x, NULL);
  // Create an array to store the projected function
  struct gkyl_array *fx_c = mkarr(basis.num_basis, local_ext.volume, use_gpu);
  // Project the target function onto the basis
  gkyl_proj_on_basis_advance(projf, 0.0, &local, fx_c);
  gkyl_proj_on_basis_release(projf);

  //------------- 3. Project the weight function ----------------
  // Create a projection updater for the weight function
  gkyl_proj_on_basis *proj_weight = gkyl_proj_on_basis_new(
      &grid, &basis, poly_order + 1, 1, evalWeight_1x, NULL);
  // Create an array to store the projected weight function
  struct gkyl_array *wx_c = mkarr(basis.num_basis, local_ext.volume, use_gpu);
  // Project the weight function onto the basis
  gkyl_proj_on_basis_advance(proj_weight, 0.0, &local_ext, wx_c);
  gkyl_proj_on_basis_release(proj_weight);

  //------------- 4. Compute weighted average ----------------
  // Define the reduced range and basis for averaging
  struct gkyl_range red_local;
  gkyl_range_init(&red_local, 1, &local.lower[0], &local.lower[0]);
  struct gkyl_basis red_basis;
  gkyl_cart_modal_serendip(&red_basis, 1, poly_order);
  // Create an array to store the average result
  struct gkyl_array *avgf_c = mkarr(red_basis.num_basis, red_local.volume, use_gpu);
  // Create and run the array average updater
    struct gkyl_array_average_inp inp_avg_full = {
    .grid = &grid,
    .tot_basis = basis,
    .sub_basis = red_basis,
    .tot_rng = &local,
    .tot_rng_ext = &local_ext,
    .sub_rng = &red_local,
    .weights = wx_c,
    .op = GKYL_ARRAY_AVERAGE_OP,
    .use_gpu = use_gpu
  };
  struct gkyl_array_average *avg_full = gkyl_array_average_new(&inp_avg_full);

  /*
    Run the updater to:
    1. Integrate the weight function.
    2. Compute the weighted integral of the target function.
    3. Normalize the result by dividing (2) by (1).
  */
  gkyl_array_average_advance(avg_full, fx_c, avgf_c);
  gkyl_array_average_release(avg_full);

  //------------- 5. Fetch and transfer results ----------------
  // Retrieve the computed average from the device (if applicable)
  const double *avg_c0 = gkyl_array_cfetch(avgf_c, 0);
  double *avg_c0_ho = gkyl_malloc(sizeof(double));
  if (use_gpu)
    gkyl_cu_memcpy(avg_c0_ho, avg_c0, sizeof(double), GKYL_CU_MEMCPY_D2H);
  else
    memcpy(avg_c0_ho, avg_c0, sizeof(double));

  //------------- 6. Check results ----------------
  /*
    Compare the computed result with the reference solution obtained from Python.
    Precision depends on the number of cells used in the grid.
  */
  double result = avg_c0_ho[0]*0.5*sqrt(2);
  printf("Result: %g, solution: %g\n",result,solution_1x());
  if (cells[0] < 8) {
    TEST_CHECK(gkyl_compare(result, solution_1x(), 1e-2));
  } else if (cells[0] < 16) {
    TEST_CHECK(gkyl_compare(result, solution_1x(), 1e-3));
  } else if (cells[0] < 32) {
    TEST_CHECK(gkyl_compare(result, solution_1x(), 1e-4));
  } else if (cells[0] < 64) {
    TEST_CHECK(gkyl_compare(result, solution_1x(), 1e-5));
  }
  printf("Relative error: %e\n", fabs(result - solution_1x()) / fabs(solution_1x()));

  //------------- 7. Clean up ----------------
  gkyl_array_release(avgf_c);
  gkyl_array_release(fx_c);
  gkyl_array_release(wx_c);
}

//----------------- TEST 2x ------------------
// function to average
void evalFunc_2x(double t, const double *xn, double* restrict fout, void *ctx)
{
  double x = xn[0];
  double y = xn[1];
  double lower[] = {-4., -3.}, upper[] = {6., 5.};
  double Lx = upper[0]-lower[0];
  double Ly = upper[1]-lower[1];
  double k_x = 2.*M_PI/Lx;
  double k_y = 2.*M_PI/Ly;
  double phi = 0.5;

  fout[0] = 1 + sin(k_x*x + k_y*y);
  fout[0] = x * y * sin(1.5*k_x*x + 0.75*k_y*y + phi) * cos(1.42*k_y*y);
}
// to weight the integral
void evalWeight_2x(double t, const double *xn, double* restrict fout, void *ctx)
{
  double x = xn[0];
  double y = xn[1];
  // fout[0] = 10;
  fout[0] = 1 + x*x + y*y;
}

double solution_2x(){
  // return 1;
  // Solution from a trapz integration with Python (see code at the end)
  return -0.6715118302909872;
}

void test_2x_1step(int poly_order, bool use_gpu)
{
  // Define grid and basis
  double lower[] = {-4.0, -3.0}, upper[] = {6.0, 5.0};
  int cells[] = {2, 4};
  int ndim = sizeof(lower) / sizeof(lower[0]);

  // Initialize the grid
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);

  // Initialize the polynomial basis
  struct gkyl_basis basis;
  gkyl_cart_modal_serendip(&basis, ndim, poly_order);

  // Create ranges (local and extended, no ghost cells in this case)
  int ghost[] = {1, 1};
  struct gkyl_range local, local_ext;
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);

  // Create an array to store the projected target function
  struct gkyl_array *fxy_c = mkarr(basis.num_basis, local_ext.volume, use_gpu);

  // Project the target function
  gkyl_proj_on_basis *projf = gkyl_proj_on_basis_new(
      &grid, &basis, poly_order+1, 1, evalFunc_2x, NULL);
  gkyl_proj_on_basis_advance(projf, 0.0, &local, fxy_c);
  gkyl_proj_on_basis_release(projf);

  // Define the reduced range and basis for averaging
  struct gkyl_range red_local;
  int lower_red[] = {local.lower[0]};
  int upper_red[] = {local.lower[0]-1};
  gkyl_range_init(&red_local, 1, lower_red, upper_red);

  // Create an array to store the averaged result
  struct gkyl_array *avgf_c = mkarr(1, 1, use_gpu);

  // Create and run the array average updater
  struct gkyl_array_average_inp inp_avg_xy = {
    .grid = &grid,
    .tot_basis = basis,
    .sub_basis = basis, // Not used.
    .tot_rng = &local,
    .tot_rng_ext = &local_ext,
    .sub_rng = &red_local,
    .op = GKYL_ARRAY_AVERAGE_OP,
    .use_gpu = use_gpu
  };
  struct gkyl_array_average *avg_xy = gkyl_array_average_new(&inp_avg_xy);
  gkyl_array_average_advance(avg_xy, fxy_c, avgf_c);
  gkyl_array_average_release(avg_xy);

  // Compare the computed result with the average computed with another updater.
  double *avgf_ref = use_gpu? gkyl_cu_malloc(sizeof(double)) : gkyl_malloc(sizeof(double));
  struct gkyl_array_integrate* arr_integ = gkyl_array_integrate_new(&grid, &basis, 1, GKYL_ARRAY_INTEGRATE_OP_NONE, use_gpu);
  gkyl_array_integrate_advance(arr_integ, fxy_c, 1.0, fxy_c, &local, avgf_ref);
  gkyl_array_integrate_release(arr_integ);
  double *avgf_ref_ho = gkyl_malloc(sizeof(double));
  if (use_gpu)
    gkyl_cu_memcpy(avgf_ref_ho, avgf_ref, sizeof(double), GKYL_CU_MEMCPY_D2H);
  else
    memcpy(avgf_ref_ho, avgf_ref, sizeof(double));
  double grid_vol = 1;
  for (int d=0; d<ndim; d++) grid_vol *= grid.upper[d] - grid.lower[d];
  avgf_ref_ho[0] = avgf_ref_ho[0]/grid_vol;

  double avgf = ((double*) avgf_c->data)[0];
  printf("\tGot: %g | Expected: %g\n",avgf,avgf_ref_ho[0]);
  printf("\tRelative error: %e\n", fabs(avgf - avgf_ref_ho[0]) / fabs(avgf_ref_ho[0]));

  gkyl_free(avgf_ref_ho);
  if (use_gpu)
    gkyl_cu_free(avgf_ref);
  else
    gkyl_free(avgf_ref);
  gkyl_array_release(avgf_c);
  gkyl_array_release(fxy_c);
}


void test_2x_2steps(int poly_order, bool use_gpu)
{
  double lower[] = {-4.0, -3.0}, upper[] = {6.0, 5.0};
  int cells[] = {2, 4};
  int ndim = sizeof(lower) / sizeof(lower[0]);

  // Initialize the grid
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, ndim, lower, upper, cells);

  // Initialize the polynomial basis
  struct gkyl_basis basis;
  gkyl_cart_modal_serendip(&basis, ndim, poly_order);

  // Create ranges (local and extended)
  int ghost[] = {1, 1};
  struct gkyl_range local, local_ext;
  gkyl_create_grid_ranges(&grid, ghost, &local_ext, &local);

  // Create an array to store the target function.
  struct gkyl_array *fxy_c = mkarr(basis.num_basis, local_ext.volume, use_gpu);

  // Project the target function
  gkyl_proj_on_basis *projf = gkyl_proj_on_basis_new(
      &grid, &basis, poly_order+1, 1, evalFunc_2x, NULL);
  gkyl_proj_on_basis_advance(projf, 0.0, &local_ext, fxy_c);
  gkyl_proj_on_basis_release(projf);

  // Average over x only
  // Define the reduced grid, range and basis for averaging along x only
  struct gkyl_rect_grid grid_y;
  gkyl_rect_grid_init(&grid_y, 1, &grid.lower[1], &grid.upper[1], &cells[1]);
  struct gkyl_range local_y, local_y_ext;
  gkyl_create_grid_ranges(&grid_y, &ghost[1], &local_y_ext, &local_y);
  struct gkyl_basis basis_y;
  gkyl_cart_modal_serendip(&basis_y, 1, poly_order);

  // Create an array to store the averaged result
  struct gkyl_array *fy_c = mkarr(basis_y.num_basis, local_y_ext.volume, use_gpu);

  // Create and run the array average updater to average on x only
  struct gkyl_array_average_inp inp_avg_x = {
    .grid = &grid,
    .tot_basis = basis,
    .sub_basis = basis_y,
    .tot_rng = &local,
    .tot_rng_ext = &local_ext,
    .sub_rng = &local_y,
    .weights = NULL,
    .op = GKYL_ARRAY_AVERAGE_OP_Y,
    .use_gpu = use_gpu
  };
  struct gkyl_array_average *avg_x = gkyl_array_average_new(&inp_avg_x);
  gkyl_array_average_advance(avg_x, fxy_c, fy_c);
  gkyl_array_average_release(avg_x);

  // Average over y now
  // Define the reduced range and basis for scalar result
  struct gkyl_range red_local;
  int lower_red[] = {local.lower[0]};
  int upper_red[] = {local.lower[0]-1};
  gkyl_range_init(&red_local, 1, lower_red, upper_red);

  // Create an array to store the averaged result
  struct gkyl_array *avgf_c = mkarr(1, 1, use_gpu);

  // Create and run the array average updater to average on x only
  struct gkyl_array_average_inp inp_int_y = {
    .grid = &grid_y,
    .tot_basis = basis_y,
    .sub_basis = basis_y, // Not used.
    .tot_rng = &local_y,
    .tot_rng_ext = &local_y_ext,
    .sub_rng = &red_local,
    .weights = NULL,
    .op = GKYL_ARRAY_AVERAGE_OP,
    .use_gpu = use_gpu
  };
  struct gkyl_array_average *int_y = gkyl_array_average_new(&inp_int_y);
  gkyl_array_average_advance(int_y, fy_c, avgf_c);
  gkyl_array_average_release(int_y);

  // Compare the computed result with the average computed with another updater.
  double *avgf_ref = use_gpu? gkyl_cu_malloc(sizeof(double)) : gkyl_malloc(sizeof(double));
  struct gkyl_array_integrate* arr_integ = gkyl_array_integrate_new(&grid, &basis, 1, GKYL_ARRAY_INTEGRATE_OP_NONE, use_gpu);
  gkyl_array_integrate_advance(arr_integ, fxy_c, 1.0, fxy_c, &local, avgf_ref);
  gkyl_array_integrate_release(arr_integ);
  double *avgf_ref_ho = gkyl_malloc(sizeof(double));
  if (use_gpu)
    gkyl_cu_memcpy(avgf_ref_ho, avgf_ref, sizeof(double), GKYL_CU_MEMCPY_D2H);
  else
    memcpy(avgf_ref_ho, avgf_ref, sizeof(double));
  double grid_vol = 1;
  for (int d=0; d<ndim; d++) grid_vol *= grid.upper[d] - grid.lower[d];
  avgf_ref_ho[0] = avgf_ref_ho[0]/grid_vol;

  double avgf = ((double*) avgf_c->data)[0];
  printf("\tGot: %g | Expected: %g\n",avgf,avgf_ref_ho[0]);
  printf("\tRelative error: %e\n", fabs(avgf - avgf_ref_ho[0]) / fabs(avgf_ref_ho[0]));

  gkyl_free(avgf_ref_ho);
  if (use_gpu)
    gkyl_cu_free(avgf_ref);
  else
    gkyl_free(avgf_ref);
  gkyl_array_release(fxy_c);
  gkyl_array_release(fy_c);
  gkyl_array_release(avgf_c);
}

void test_1x_cpu()
{
  // p=1
  test_1x(1, false);

  // p=2
  // test_1x(2, false);

}

void test_2x_cpu_1step()
{
  // p=1
  test_2x_1step(1, false);

  // p=2
  // test_2x(2, false);
}

void test_2x_cpu_2steps()
{
  // p=1
  test_2x_2steps(1, false);

  // p=2
  // test_2x_2steps(2, false);
}

#ifdef GKYL_HAVE_CUDA
void test_1x_gpu()
{
  // p=1
  test_1x(1, true);

  // p=2
  // test_1x(2, true);
}

void test_2x_gpu()
{
  // p=1
  test_2x_nc1_op(1, true);
  
  // p=2
  // test_2x_nc1_op(2, true);
}
#endif

TEST_LIST = {
  { "test_1x_cpu", test_1x_cpu },
  { "test_2x_cpu_1step", test_2x_cpu_1step },
  { "test_2x_cpu_2steps", test_2x_cpu_2steps },
#ifdef GKYL_HAVE_CUDA
  { "test_1x_gpu", test_1x_gpu },
  { "test_2x_gpu", test_2x_gpu },
#endif
  { NULL, NULL },
};


//-------- PYTHON CODE FOR THE SOLUTION OF TEST_1X
/*
import numpy as np
from scipy.integrate import quad

# Define the function to integrate
def f(x, k_x, phi):
    return x * np.sin(k_x * x + phi)

def w(x):
    return 1+x**2

def fw(x, k_x, phi):
    return w(x)*x * np.sin(k_x * x + phi)


# Parameters
a, b = -4, 6
k_x = 2.0 * np.pi / (b-a)  # Wave number
phi = 0.5  # Phase

# Perform numerical integration
int_fw, efw = quad(fw, a, b, args=(k_x, phi))
int_w,  ew = quad(w, a, b)

# Result
normalized_result = int_fw / int_w
total_error = efw + ew

print("Normalized Result:", normalized_result)
print("Total Error:", total_error)
*/
/* OUTPUT
Normalized Result: -0.4189328208844751
Total Error: 3.983333298112789e-12
*/

//-------- PYTHON CODE FOR THE SOLUTION OF TEST_2X
/*
import numpy as np

# Define the function to integrate
def f(x, y, k_x, k_y, phi):
    return x * y * np.sin(1.5 * k_x * x + 0.75*k_y * y + phi) * np.cos(1.42*k_y*y)

def w(x, y):
    return 1 + x**2 + y**2

def fw(x, y, k_x, k_y, phi):
    return w(x, y) * f(x, y, k_x, k_y, phi)

# Parameters
ax, bx = -4, 6  # Bounds for x
ay, by = -3, 5  # Bounds for y
k_x = 2.0 * np.pi / (bx - ax)  # Wave number for x
k_y = 2.0 * np.pi / (by - ay)  # Wave number for y
phi = 0.5  # Phase

# Create grid points
nx, ny = 1024, 1024  # Number of grid points in x and y
x = np.linspace(ax, bx, nx); y = np.linspace(ay, by, ny)
dx = (bx - ax) / (nx - 1); dy = (by - ay) / (ny - 1)
# Create 2D grids
X, Y = np.meshgrid(x, y, indexing="ij")

# Evaluate the functions on the grid
f_values = f(X, Y, k_x, k_y, phi)
w_values = w(X, Y)
fw_values = fw(X, Y, k_x, k_y, phi)

# Perform integration using the trapezoidal rule
int_fw = np.trapz(np.trapz(fw_values, x=y, axis=1), x=x)
int_w = np.trapz(np.trapz(w_values, x=y, axis=1), x=x)

# Normalize the result
normalized_result = int_fw / int_w

# Output
print("Normalized Result:", normalized_result)
*/
/* OUTPUT
Normalized Result: -0.6715118302909872
*/
