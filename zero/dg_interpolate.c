#include <gkyl_dg_interpolate.h>
#include <gkyl_dg_interpolate_priv.h>
#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>

struct gkyl_dg_interpolate*
gkyl_dg_interpolate_new(int cdim, const struct gkyl_basis *pbasis,
  const struct gkyl_rect_grid *grid_do, const struct gkyl_rect_grid *grid_tar,
  const struct gkyl_range *range_do, const struct gkyl_range *range_tar,
  const int *nghost, bool pre_alloc_fields, bool use_gpu)
{
  // Allocate space for new updater.
  struct gkyl_dg_interpolate *up = gkyl_malloc(sizeof(*up));

  up->use_gpu = use_gpu;
  up->ndim = pbasis->ndim;
  up->grid_do = *grid_do;
  up->grid_tar = *grid_tar;

  // Perform some basic checks:
  assert(grid_do->ndim == grid_tar->ndim);
  for (int d=0; d<up->ndim; d++) {
    assert(fabs(grid_do->lower[d] - grid_tar->lower[d]) < 1e-14);
    assert(fabs(grid_do->upper[d] - grid_tar->upper[d]) < 1e-14);

    const int num_prime_facs_max = 32;
    int prime_factors[num_prime_facs_max];
    int num_prime_facs_do = dg_interp_prime_factors(grid_do->cells[d], prime_factors, num_prime_facs_max);
    for (int k=0; k<num_prime_facs_do; k++)
      assert(prime_factors[k] == 2 || prime_factors[k] == 3); // Only (2^a)*(3^b) grids supported.
 
    int num_prime_facs_tar = dg_interp_prime_factors(grid_tar->cells[d], prime_factors, num_prime_facs_max);
    for (int k=0; k<num_prime_facs_tar; k++)
      assert(prime_factors[k] == 2 || prime_factors[k] == 3); // Only (2^a)*(3^b) grids supported.
  }

  // Make a list of directions to be coarsened/refined.
  up->num_interp_dirs = 0;
  for (int d=0; d<up->ndim; d++) {
    if (grid_do->cells[d] != grid_tar->cells[d]) {
      up->interp_dirs[up->num_interp_dirs] = d;
      up->num_interp_dirs++;
    }
  }

  // Create a series of grids, each one the same as the previous one except
  // that it has the number of target cells along one new direction.
  up->grids = gkyl_malloc((up->num_interp_dirs+1) * sizeof(struct gkyl_rect_grid));
  double lower_new[up->ndim], upper_new[up->ndim];
  for (int d=0; d<up->ndim; d++) {
    lower_new[d] = grid_do->lower[d];
    upper_new[d] = grid_do->upper[d];
  }
  memcpy(&up->grids[0], grid_do, sizeof(struct gkyl_rect_grid));
  for (int k=1; k<up->num_interp_dirs; k++) {
    int cells_new[up->ndim];
    for (int d=0; d<up->ndim; d++)
      cells_new[d] = up->grids[k-1].cells[d];
    cells_new[up->interp_dirs[k-1]] = grid_tar->cells[up->interp_dirs[k-1]];
    gkyl_rect_grid_init(&up->grids[k], up->ndim, lower_new, upper_new, cells_new);
  }
  memcpy(&up->grids[up->num_interp_dirs], grid_tar, sizeof(struct gkyl_rect_grid));

  // Create a donor and target range for each interpolation.
  up->ranges = gkyl_malloc((up->num_interp_dirs+1) * sizeof(struct gkyl_range));
  struct gkyl_range *ranges_ext = gkyl_malloc((up->num_interp_dirs+1) * sizeof(struct gkyl_range));
  memcpy(&up->ranges[0], range_do, sizeof(struct gkyl_range));
  for (int k=1; k<up->num_interp_dirs; k++) {
    gkyl_create_grid_ranges(&up->grids[k], nghost, &ranges_ext[k], &up->ranges[k]);
  }
  memcpy(&up->ranges[up->num_interp_dirs], range_tar, sizeof(struct gkyl_range));

  // Create an interpolation updater for each interpolation.
  up->interp_ops = gkyl_malloc(up->num_interp_dirs * sizeof(struct gkyl_dg_interpolate*));
  if (up->num_interp_dirs == 1) {
    up->interp_ops[0] = up;
  }
  else {
    for (int k=0; k<up->num_interp_dirs; k++)
      up->interp_ops[k] = gkyl_dg_interpolate_new(cdim, pbasis, &up->grids[k], &up->grids[k+1],
        &up->ranges[k], &up->ranges[k+1], nghost, false, use_gpu);
  }

//  if (pre_alloc_fields) {
    // Pre-allocate fields for intermediate grids.
    up->fields = gkyl_malloc((up->num_interp_dirs+1) * sizeof(struct gkyl_array *));
    for (int k=1; k<up->num_interp_dirs; k++) {
      up->fields[k] = gkyl_array_new(GKYL_DOUBLE, pbasis->num_basis, ranges_ext[k].volume);
    }
//  }
  gkyl_free(ranges_ext);

  if (up->num_interp_dirs > 1)
    return up; // Only allocate the remaining objects if doing 1D interpolation.

  // Identify direction to be coarsened/refined:
  for (int d=0; d<up->ndim; d++) {
    if (grid_do->cells[d] != grid_tar->cells[d]) {
      up->dir = d;
      break;
    }
  }

  // Choose kernels that translates the DG coefficients.
  up->interp = dg_interp_choose_gk_interp_kernel(*pbasis, up->dir);

  // Ratio of cell-lengths in each direction.
  up->dxRat = grid_do->dx[up->dir]/grid_tar->dx[up->dir];

  // Map from grid to stencil index in each direction.
  if (up->dxRat > 1)
    up->grid2stencil = dg_interp_index_stencil_map_refine;
  else
    up->grid2stencil = dg_interp_index_stencil_map_coarsen;
  

  // Interior (away from boundaries) stencil size.
  int intStencilSize = -1;
  if (up->dxRat > 1) {
    // Mesh refinement.
    // Brute force search. Start with the size of the boundary stencil.
    int maxSize = floor(up->dxRat) + ceil( up->dxRat-floor(up->dxRat) );
    for (int i=2; i<grid_do->cells[up->dir]; i++) {
       double decimalL = 1-((i-1)*up->dxRat-floor((i-1)*up->dxRat));
       double decimalU = 1-(ceil(i*up->dxRat)-i*up->dxRat);
       int currSize = floor(up->dxRat-decimalL-decimalU) + ceil(decimalL) + ceil(decimalU);
       maxSize = GKYL_MAX2(maxSize, currSize);
    }
    intStencilSize = maxSize;
  }
  else {
    // Mesh coarsening, or no change in resolution (dxRat=1).
    intStencilSize = 1+ceil(1*(1/up->dxRat-floor(1/up->dxRat)));
  }

  // This updater will loop through the donor grid. At each cell it will give
  // the interpolation kernel the target grid cells, one at a time, and the
  // kernel will add the corresponding contributions to the target field DG
  // coefficients in that each target grid cell.
  // There are 3 stencils or regions. For example, in 2D the regions are
  //   .-----.-----.-----.
  //   |     |     |     |
  //   |  1  |  0  |  2  |
  //   |     |     |     |
  //   .-----.-----.-----.
  // For each region we will make a list of the number of target
  // cells to fetch contributions from in each direction.
  up->offset_upper = gkyl_malloc(3 * sizeof(int));
  int sc = 0; // Current number of stencils.
  // Interior stencil (region 0).
  up->offset_upper[0] = intStencilSize;
  sc++;
  // Other stencils.
  for (int mp=-1; mp<2; mp += 2) {
    up->offset_upper[sc] = floor(up->dxRat) + ceil(up->dxRat - floor(up->dxRat));
    sc++;
  }

#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu) {
    // Allocate a device copy of the updater.
    gkyl_dg_interpolate_new_cu(up, *pbasis);
  }
#endif

  return up;
}

static void
dg_interpolate_advance_1x(gkyl_dg_interpolate* up,
  const struct gkyl_range *range_do, const struct gkyl_range *range_tar,
  const struct gkyl_array *GKYL_RESTRICT fdo, struct gkyl_array *GKYL_RESTRICT ftar)
{

#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu) {
    gkyl_dg_interpolate_advance_1x_cu(up, range_do, range_tar, fdo, ftar);
    return;
  }
#endif

  gkyl_array_clear_range(ftar, 0.0, range_tar);

  int idx_tar[GKYL_MAX_DIM] = {-1};
  int idx_tar_lo;
  double xc_do[GKYL_MAX_DIM];
  double xc_tar[GKYL_MAX_DIM];

  // Loop over the donor range.
  struct gkyl_range_iter iter_do;
  gkyl_range_iter_init(&iter_do, range_do);
  while (gkyl_range_iter_next(&iter_do)) {

    int *idx_do = iter_do.idx;

    gkyl_rect_grid_cell_center(&up->grid_do, idx_do, xc_do);

    long linidx_do = gkyl_range_idx(range_do, idx_do);
    const double *fdo_c = gkyl_array_cfetch(fdo, linidx_do);

    // Compute the index of the lower target cell this cell contributes to.
    double eveOI = up->dxRat*(idx_do[up->dir]-1);
    idx_tar_lo = ceil(eveOI)+((int) ceil(eveOI-floor(eveOI))+1) % 2;

    // Get the index to the stencil for this donor cell.
    int idx_sten = up->grid2stencil(idx_do[up->dir], up->grid_do.cells[up->dir], up->dxRat);
    idx_sten -= 1;

    for (int d=0; d<up->ndim; d++)
      idx_tar[d] = idx_do[d];

    // Loop over the target-grid cells this donor cell contributes to.
    for (int off=0; off<up->offset_upper[idx_sten]; off++) {

      idx_tar[up->dir] = idx_tar_lo + off;

      gkyl_rect_grid_cell_center(&up->grid_tar, idx_tar, xc_tar);

      long linidx_tar = gkyl_range_idx(range_tar, idx_tar);
      double *ftar_c = gkyl_array_fetch(ftar, linidx_tar);

      up->interp(xc_do, xc_tar, up->grid_do.dx, up->grid_tar.dx, fdo_c, ftar_c);
      
    }

  }
}

void
gkyl_dg_interpolate_advance(gkyl_dg_interpolate* up,
  struct gkyl_array *fdo, struct gkyl_array *ftar)
{
#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu) {
    gkyl_dg_interpolate_advance_cu(up, range_do, range_tar, fdo, ftar);
    return;
  }
#endif

  up->fields[0] = fdo;
  up->fields[up->num_interp_dirs] = ftar;

  // Loop over interpolating dimensions and do each interpolation separately.
  for (int k=0; k<up->num_interp_dirs; k++) {
    dg_interpolate_advance_1x(up->interp_ops[k], &up->ranges[k], &up->ranges[k+1], 
      up->fields[k], up->fields[k+1]);
  }

}

void
gkyl_dg_interpolate_release(gkyl_dg_interpolate* up)
{
  // Release memory associated with this updater.

  if (up->num_interp_dirs == 1) {
    if (!up->use_gpu)
      gkyl_free(up->offset_upper);
#ifdef GKYL_HAVE_CUDA
    if (up->use_gpu) {
      gkyl_cu_free(up->offset_upper);
      if (GKYL_IS_CU_ALLOC(up->flags))
        gkyl_cu_free(up->on_dev);
    }
#endif
  }

  gkyl_free(up->grids);
  if (up->num_interp_dirs > 1) {
    for (int k=0; k<up->num_interp_dirs; k++)
      gkyl_dg_interpolate_release(up->interp_ops[k]);
  }
  gkyl_free(up->interp_ops);
  gkyl_free(up->ranges);
  for (int k=1; k<up->num_interp_dirs; k++)
    gkyl_array_release(up->fields[k]);
  gkyl_free(up->fields);

  gkyl_free(up);
}
