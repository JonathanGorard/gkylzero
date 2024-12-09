#include <gkyl_alloc.h>
#include <gkyl_alloc_flags_priv.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_average_priv.h>
#include <gkyl_array_average.h>
#include <gkyl_dg_bin_ops.h>

#include <assert.h>

struct gkyl_array_average*
gkyl_array_average_new(const struct gkyl_array_average_inp *inp)
{
  // works for p = 1 only
  assert(inp->tot_basis.poly_order == 1); 

  // Allocate space for new updater.
  struct gkyl_array_average *up = gkyl_malloc(sizeof(struct gkyl_array_average));

  up->use_gpu = inp->use_gpu;
  up->ndim = inp->tot_basis.ndim;
  up->tot_basis = inp->tot_basis;
  up->sub_basis = inp->sub_basis;

  // copy the total and sub ranges on the updater 
  up->tot_rng = *inp->tot_rng;
  up->tot_rng_ext = *inp->tot_rng_ext;
  up->sub_rng = *inp->sub_rng;

  up->avg_dirs_vol = 1.0;

  // Set up the array of all dimensions that are conserved after the average (=0 for removed)
  // according to the operation input variable
  for (unsigned d=0; d < up->ndim; ++d) up->issub_dim[d] = 0;
  switch (inp->op) {
    case GKYL_ARRAY_AVERAGE_OP: // Full integration
      assert(inp->tot_basis.ndim >= 1); // Ensure at least 1 dimension exists
      for (unsigned d=0; d < up->ndim; ++d) 
        up->avg_dirs_vol *= inp->grid->upper[d] - inp->grid->lower[d]; 
      break;
    case GKYL_ARRAY_AVERAGE_OP_X: // integration all except x
      assert(inp->tot_basis.ndim >= 1); // Ensure at least 1 dimension exists
      up->issub_dim[0] = 1; // here the first dimension is conserved
      up->avg_dirs_vol *= inp->grid->upper[1] - inp->grid->lower[1]; 
      break;
    case GKYL_ARRAY_AVERAGE_OP_Y: // integration all except y
      assert(inp->tot_basis.ndim >= 2); // Ensure at least 2 dimensions for Y
      up->issub_dim[1] = 1;
      up->avg_dirs_vol *= inp->grid->upper[0] - inp->grid->lower[0]; 
      break;
    default:
      assert(false && "Invalid operation in switch(op)");
      break;
  }

  // Compute the cell sub-dimensional volume
  up->subvol = 1.0;
  for (unsigned d=0; d < up->ndim; ++d)
    if (up->issub_dim[d] == 0) up->subvol *= inp->grid->dx[d]/2.0;

  // Choose the kernel that performs the desired operation within the integral.
  gkyl_array_average_choose_kernel(up, &up->tot_basis, inp->op);

#ifdef GKYL_HAVE_CUDA
  if (use_gpu)
    return gkyl_array_average_cu_dev_new(up, grid, up->tot_basis, inp->op);
#endif

  return up;
}

void gkyl_array_average_advance(gkyl_array_average *up, 
  const struct gkyl_array * fin, struct gkyl_array *avgout)
{

#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu) {
    gkyl_array_average_advance_cu(up, tot_rng, sub_rng, fin, out);
    return;
  }
#endif

  gkyl_array_clear(avgout, 0.0);

  if (up->sub_rng.volume > 0) {
    // Loop through reduced range.
    struct gkyl_range_iter sub_iter;
    gkyl_range_iter_init(&sub_iter, &up->sub_rng);
    while (gkyl_range_iter_next(&sub_iter)) {
      long sub_lidx = gkyl_range_idx(&up->sub_rng, sub_iter.idx);
      double *avg_i = gkyl_array_fetch(avgout, sub_lidx);

      // Loop through complementary range, sub + cmp = full.
      struct gkyl_range cmp_rng;
      struct gkyl_range_iter cmp_iter;
      int parent_idx[GKYL_MAX_DIM];
      parent_idx[1] = sub_iter.idx[0];
      parent_idx[0] = sub_iter.idx[0];
      gkyl_range_deflate(&cmp_rng, &up->tot_rng, up->issub_dim, parent_idx);
      gkyl_range_iter_no_split_init(&cmp_iter, &cmp_rng);

      while (gkyl_range_iter_next(&cmp_iter)) {
        long cmp_lidx = gkyl_range_idx(&cmp_rng, cmp_iter.idx);
        const double *fin_i = gkyl_array_cfetch(fin, cmp_lidx);
        up->kernel(up->subvol, NULL, fin_i, avg_i);
      }
    }

    // Divide by the domain volume.
    gkyl_array_scale_range(avgout, 1.0/up->avg_dirs_vol, &up->sub_rng);
  } 
  else {
    // This is the case if we are asking for a full integration
    struct gkyl_range_iter tot_iter;
    // this is the complementary range, sub + cmp = full
    // We now loop on the range of the entire array
    gkyl_range_iter_init(&tot_iter, &up->tot_rng);
    while (gkyl_range_iter_next(&tot_iter)) {
      long tot_lidx = gkyl_range_idx(&up->tot_rng, tot_iter.idx);
      const double *fin_i = gkyl_array_cfetch(fin, tot_lidx);
      double *avg_i = gkyl_array_fetch(avgout, 0);
      up->kernel(up->subvol, NULL, fin_i, avg_i);
    }

    // Divide by the domain volume.
    gkyl_array_scale(avgout, 1.0/up->avg_dirs_vol);
  }

}

void gkyl_array_average_release(gkyl_array_average *up)
{
  // Release memory associated with this updater.
#ifdef GKYL_HAVE_CUDA
  if (up->use_gpu)
    gkyl_cu_free(up->on_dev);
#endif
  gkyl_free(up);
}
