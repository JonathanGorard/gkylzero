#include <acutest.h>
#include <gkyl_null_comm.h>

void
test_1d()
{
  struct gkyl_range range;
  gkyl_range_init(&range, 1, (int[]) { 1 }, (int[]) { 100 });

  int cuts[] = { 1 };
  struct gkyl_rect_decomp *decomp =
    gkyl_rect_decomp_new_from_cuts(range.ndim, cuts, &range);

  struct gkyl_comm *comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp
    }
  );

  int rank;
  gkyl_comm_get_rank(comm, &rank);
  TEST_CHECK( rank == 0 );

  int sz;
  gkyl_comm_get_size(comm, &sz);
  TEST_CHECK( sz == 1 );  

  double out[3], inp[3] = { 2.0, 4.0, 8.0 };
  gkyl_comm_all_reduce(comm, GKYL_DOUBLE, GKYL_MIN, 3, inp, out);

  for (int i=0; i<3; ++i)
    TEST_CHECK( out[i] == inp[i] );

  int nghost[] = { 1 };
  struct gkyl_range local, local_ext;
  gkyl_create_ranges(&decomp->ranges[rank], nghost, &local_ext, &local);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, range.ndim, local_ext.volume);
  gkyl_array_clear(arr, 200005);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &local);
  while (gkyl_range_iter_next(&iter)) {
    long idx = gkyl_range_idx(&local, iter.idx);
    double  *f = gkyl_array_fetch(arr, idx);
    f[0] = iter.idx[0];
  }

  int per_dirs[] = { 0 };
  gkyl_comm_array_per_sync(comm, &local, &local_ext, 1, per_dirs, arr );

  int idx[GKYL_MAX_DIM] = { 0 };
  
  for (int d=0; d<local.ndim; ++d) {
    int ncell = gkyl_range_shape(&local, d);

    gkyl_range_iter_init(&iter, &local_ext);
    while (gkyl_range_iter_next(&iter)) {

      if (!gkyl_range_contains_idx(&local, iter.idx)) {
        long lidx = gkyl_range_idx(&local_ext, iter.idx);
        
        for (int n=0; n<local.ndim; ++n)
          idx[n] = iter.idx[n];
        if (idx[d] > local.upper[d])
          idx[d] = idx[d] - ncell;
        else
          idx[d] = idx[d] + ncell;

        const double  *f = gkyl_array_cfetch(arr, lidx);
        TEST_CHECK( idx[0] == f[0] );
      }
    }
  }

  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_array_release(arr);
}

void
test_2d()
{
  struct gkyl_range range;
  gkyl_range_init(&range, 2, (int[]) { 1, 1 }, (int[]) { 4, 4 });

  int cuts[] = { 1, 1 };
  struct gkyl_rect_decomp *decomp =
    gkyl_rect_decomp_new_from_cuts(range.ndim, cuts, &range);

  struct gkyl_comm *comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp
    }
  );

  int rank;
  gkyl_comm_get_rank(comm, &rank);

  int nghost[] = { 1, 1 };
  struct gkyl_range local, local_ext;
  gkyl_create_ranges(&decomp->ranges[rank], nghost, &local_ext, &local);

  struct gkyl_range local_x[2], local_ext_x[2];
  gkyl_create_ranges(&decomp->ranges[rank], (int[]) { nghost[0], 0 },
    &local_ext_x[0], &local_x[0]);
  
  gkyl_create_ranges(&decomp->ranges[rank], (int[]) { 0, nghost[1] },
    &local_ext_x[1], &local_x[1]);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, range.ndim, local_ext.volume);
  gkyl_array_clear(arr, 200005);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &local);
  while (gkyl_range_iter_next(&iter)) {
    long idx = gkyl_range_idx(&local, iter.idx);
    double  *f = gkyl_array_fetch(arr, idx);

    for (int d=0; d<local.ndim; ++d)
      f[d] = iter.idx[d];
  }

  int per_dirs[] = { 0, 1 };
  gkyl_comm_array_per_sync(comm, &local, &local_ext, 2, per_dirs, arr );

  int idx[GKYL_MAX_DIM] = { 0 };
  int count = 0;
  
  for (int d=0; d<local.ndim; ++d) {
    int ncell = gkyl_range_shape(&local, d);

    gkyl_range_iter_init(&iter, &local_ext_x[d]);
    while (gkyl_range_iter_next(&iter)) {

      if (!gkyl_range_contains_idx(&local, iter.idx)) {
        long lidx = gkyl_range_idx(&local_ext, iter.idx);
        
        for (int n=0; n<local.ndim; ++n)
          idx[n] = iter.idx[n];
        if (idx[d] > local.upper[d])
          idx[d] = idx[d] - ncell;
        else
          idx[d] = idx[d] + ncell;

        const double  *f = gkyl_array_cfetch(arr, lidx);

//        printf("%d: idx(%d,%d) : (%d,%d) == (%g, %g)\n",
//          count++, iter.idx[0], iter.idx[1],
//          idx[0], idx[1],
//          f[0], f[1]);
        
        /* for (int n=0; n<local.ndim; ++n) */
        /*   TEST_CHECK( idx[n] == f[n] ); */
      }
    }
  }

  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_array_release(arr);
}

void
test_io_2d()
{
  int cells[] = { 32, 32 };
  struct gkyl_range range;
  gkyl_range_init_from_shape(&range, 2, cells);

  int cuts[] = { 1, 1 };
  struct gkyl_rect_decomp *decomp =
    gkyl_rect_decomp_new_from_cuts(range.ndim, cuts, &range);

  struct gkyl_comm *comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp
    }
  );

  double lower[] = {0.0, 0.5}, upper[] = {1.0, 2.5};
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, 2, lower, upper, cells);

  int rank;
  gkyl_comm_get_rank(comm, &rank);

  int nghost[] = { 1, 1 };
  struct gkyl_range local, local_ext;
  gkyl_create_ranges(&decomp->ranges[rank], nghost, &local_ext, &local);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, range.ndim, local_ext.volume);
  gkyl_array_clear(arr, 1.5);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &local);
  while (gkyl_range_iter_next(&iter)) {
    long idx = gkyl_range_idx(&local, iter.idx);
    double  *f = gkyl_array_fetch(arr, idx);

    double xc[GKYL_MAX_DIM] = { 0.0 };
    gkyl_rect_grid_cell_center(&grid, iter.idx, xc);
    f[0] = sin(2*M_PI*xc[0])*sin(2*M_PI*xc[1]);
    f[1] = cos(2*M_PI*xc[0])*sin(2*M_PI*xc[1]);
  }  

  int status = gkyl_comm_array_write(comm, &grid, &local, arr, "ctest_null_comm_io_2d.gkyl");

  struct gkyl_array *arr_rw = gkyl_array_new(GKYL_DOUBLE, range.ndim, local_ext.volume);  
  status =
    gkyl_comm_array_read(comm, &grid, &local, arr_rw, "ctest_null_comm_io_2d.gkyl");
  TEST_CHECK( status == 0 );

  gkyl_range_iter_init(&iter, &local);
  while (gkyl_range_iter_next(&iter)) {
    long idx = gkyl_range_idx(&local, iter.idx);
    const double *f = gkyl_array_cfetch(arr, idx);
    const double *frw = gkyl_array_cfetch(arr_rw, idx);

    TEST_CHECK( gkyl_compare_double(f[0], frw[0], 1e-15) );
    TEST_CHECK( gkyl_compare_double(f[1], frw[1], 1e-15) );
  }
  
  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_array_release(arr);
  gkyl_array_release(arr_rw);  
}

TEST_LIST = {
  { "test_1d", test_1d },
  { "test_2d", test_2d },
  { "test_io_2d", test_io_2d },
  { NULL, NULL },
};
