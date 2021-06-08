#include <acutest.h>

#include <gkyl_alloc.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_rio.h>
#include <gkyl_range.h>

void test_range_0()
{
  struct gkyl_range range;
  gkyl_range_init(&range, 0, NULL, NULL);

  TEST_CHECK( range.ndim == 0 );
  TEST_CHECK( range.volume == 1 );

  TEST_CHECK( gkyl_range_is_sub_range(&range) == 0 );
}

void test_range_1()
{
  int lower[] = {1, 1}, upper[] = {10, 20};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  TEST_CHECK( range.ndim == 2);
  TEST_CHECK( range.volume == 200);

  for (unsigned i=0; i<2; ++i) {
    TEST_CHECK( range.lower[i] == lower[i]);
    TEST_CHECK( range.upper[i] == upper[i]);
  }
}

void test_range_shape()
{
  int shape[] = {25, 50};
  struct gkyl_range range;
  gkyl_range_init_from_shape(&range, 2, shape);

  TEST_CHECK( range.ndim == 2);
  TEST_CHECK( range.volume == 25*50);

  for (unsigned i=0; i<2; ++i) {
    TEST_CHECK( range.lower[i] == 0);
    TEST_CHECK( range.upper[i] == shape[i]-1);
  }
}

void test_sub_range()
{
  int lower[] = {1, 1}, upper[] = {10, 20};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int sublower[] = {2, 2}, subupper[] = { 5, 10 };
  struct gkyl_range subrange;
  gkyl_sub_range_init(&subrange, &range, sublower, subupper);

  TEST_CHECK( subrange.volume == 4*9 );
  TEST_CHECK( gkyl_range_is_sub_range(&subrange) == 1 );

  for (unsigned d=0; d<2; ++d) {
    TEST_CHECK( subrange.lower[d] == sublower[d] );
    TEST_CHECK( subrange.upper[d] == subupper[d] );
  }

  for (int i=subrange.lower[0]; i<=subrange.upper[0]; ++i)
    for (int j=subrange.lower[1]; j<=subrange.upper[1]; ++j)
      TEST_CHECK( gkyl_ridx(subrange, i, j) == gkyl_ridx(range, i, j) );

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &subrange);

  int invIdx[2];
  while (gkyl_range_iter_next(&iter)) {
    int lidx = gkyl_ridx(subrange, iter.idx[0], iter.idx[1]);
    gkyl_range_inv_idx(&subrange, lidx, invIdx);

    TEST_CHECK( invIdx[0] == iter.idx[0] );
    TEST_CHECK( invIdx[1] == iter.idx[1] );
  }
  
}

void test_sub_range_inv_idx()
{
  int lower[] = {1, 1}, upper[] = {10, 20};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int sublower[] = {2, 2}, subupper[] = { 5, 10 };
  struct gkyl_range subrange;
  gkyl_sub_range_init(&subrange, &range, sublower, subupper);

  // create a range spanning sublower and subupper
  struct gkyl_range range2;
  gkyl_range_init(&range2, 2, sublower, subupper);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &subrange);  

  int idx[2];
  for (int i=0; i<range2.volume; ++i) {

    gkyl_range_iter_next(&iter); // bump iterator
    
    // compute inverse index into subrange
    gkyl_sub_range_inv_idx(&subrange, i, idx);

    TEST_CHECK(idx[0] == iter.idx[0]);
    TEST_CHECK(idx[1] == iter.idx[1]);
  }
}

void test_shorten()
{
  int lower[] = {1, 1, 1}, upper[] = {20, 30, 20};
  struct gkyl_range range, shortr, shortr2;

  gkyl_range_init(&range, 3, lower, upper);
  gkyl_range_shorten(&shortr, &range, 1, 1);
  gkyl_range_shorten(&shortr2, &range, 1, 3);

  TEST_CHECK( gkyl_range_is_sub_range(&shortr) == 1 );

  // shortr
  TEST_CHECK( shortr.lower[0] == range.lower[0] );
  TEST_CHECK( shortr.upper[0] == range.upper[0] );

  TEST_CHECK( shortr.lower[1] == range.lower[1] );
  TEST_CHECK( shortr.upper[1] == range.lower[1] ); // shortened range

  TEST_CHECK( shortr.lower[2] == range.lower[2] );
  TEST_CHECK( shortr.upper[2] == range.upper[2] );

  TEST_CHECK( shortr.volume == 20*20 );

  // shortr2
  TEST_CHECK( shortr2.lower[0] == range.lower[0] );
  TEST_CHECK( shortr2.upper[0] == range.upper[0] );

  TEST_CHECK( shortr2.lower[1] == range.lower[1] );
  TEST_CHECK( shortr2.upper[1] == range.lower[1]+3-1 ); // shortened range

  TEST_CHECK( shortr2.lower[2] == range.lower[2] );
  TEST_CHECK( shortr2.upper[2] == range.upper[2] );

  TEST_CHECK( shortr2.volume == 20*20*3 );
}

void test_skin()
{
  int lower[] = {1, 1}, upper[] = {10, 10};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  // Test skin cell ranges
  struct gkyl_range lowerSkinRange1;
  gkyl_range_lower_skin(&lowerSkinRange1, &range, 0, 1);
  TEST_CHECK( gkyl_range_is_sub_range(&lowerSkinRange1) == 1 );
  
  TEST_CHECK( lowerSkinRange1.volume == 10 );
  TEST_CHECK( lowerSkinRange1.lower[0] == 1 );
  TEST_CHECK( lowerSkinRange1.upper[0] == 1 );
  TEST_CHECK( lowerSkinRange1.lower[1] == 1 );
  TEST_CHECK( lowerSkinRange1.upper[1] == 10 );

  struct gkyl_range lowerSkinRange2;
  gkyl_range_lower_skin(&lowerSkinRange2, &range, 1, 1);
  
  TEST_CHECK( lowerSkinRange2.volume == 10 );
  TEST_CHECK( lowerSkinRange2.lower[0] == 1 );
  TEST_CHECK( lowerSkinRange2.upper[0] == 10 );
  TEST_CHECK( lowerSkinRange2.lower[1] == 1 );
  TEST_CHECK( lowerSkinRange2.upper[1] == 1 );

  struct gkyl_range upperSkinRange1;
  gkyl_range_upper_skin(&upperSkinRange1, &range, 0, 1);
  
  TEST_CHECK( upperSkinRange1.volume == 10 );
  TEST_CHECK( upperSkinRange1.lower[0] == 10 );
  TEST_CHECK( upperSkinRange1.upper[0] == 10 );
  TEST_CHECK( upperSkinRange1.lower[1] == 1 );
  TEST_CHECK( upperSkinRange1.upper[1] == 10 );

  struct gkyl_range upperSkinRange2;
  gkyl_range_upper_skin(&upperSkinRange2, &range, 1, 1);
  
  TEST_CHECK( upperSkinRange2.volume == 10 );
  TEST_CHECK( upperSkinRange2.lower[0] == 1 );
  TEST_CHECK( upperSkinRange2.upper[0] == 10 );
  TEST_CHECK( upperSkinRange2.lower[1] == 10 );
  TEST_CHECK( upperSkinRange2.upper[1] == 10 );
}

void test_range_index_1d()
{
  int lower[] = {-3}, upper[] = {17};
  struct gkyl_range range;
  gkyl_range_init(&range, 1, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    TEST_CHECK( count++ == gkyl_ridx(range, i) );

  TEST_CHECK(count == range.volume);
}

void test_range_index_2d()
{
  int lower[2] = {1, 1}, upper[2] = {10, 20};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      TEST_CHECK( count++ == gkyl_ridx(range, i, j) );

  TEST_CHECK(count == range.volume);
}

void test_range_index_3d()
{
  int lower[] = {1, 1, 2}, upper[] = {10, 20, 29};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k)
        TEST_CHECK( count++ == gkyl_ridx(range, i, j, k) );

  TEST_CHECK(count == range.volume);
}

void test_range_index_4d()
{
  int lower[] = {1, 1, 2, -1}, upper[] = {10, 20, 29, 10};
  struct gkyl_range range;
  gkyl_range_init(&range, 4, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k)
        for (int l=range.lower[3]; l<=range.upper[3]; ++l)
          TEST_CHECK( count++ == gkyl_ridx(range, i, j, k, l) );

  TEST_CHECK(count == range.volume);
}

void test_range_index_5d()
{
  int lower[] = {1, 1, 2, 0, 1}, upper[] = {4, 5, 8, 4, 3};
  struct gkyl_range range;
  gkyl_range_init(&range, 5, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k)
        for (int l=range.lower[3]; l<=range.upper[3]; ++l)
          for (int m=range.lower[4]; m<=range.upper[4]; ++m)
            TEST_CHECK( count++ == gkyl_ridx(range, i, j, k, l, m) );

  TEST_CHECK(count == range.volume);
}

void test_range_index_6d()
{
  int lower[] = {1, 1, 2, 0, 1, -5}, upper[] = {4, 5, 8, 4, 3, -1};
  struct gkyl_range range;
  gkyl_range_init(&range, 6, lower, upper);

  long count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k)
        for (int l=range.lower[3]; l<=range.upper[3]; ++l)
          for (int m=range.lower[4]; m<=range.upper[4]; ++m)
            for (int n=range.lower[5]; n<=range.upper[5]; ++n)
              TEST_CHECK( count++ == gkyl_ridx(range, i, j, k, l, m, n) );

  TEST_CHECK(count == range.volume);
}

void test_range_idx()
{
  int lower[] = {1, 1, 2}, upper[] = {10, 20, 29};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);

  long count = 0; int idx[3];
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k) {
        idx[0] = i; idx[1] = j; idx[2] = k;
        TEST_CHECK( count++ == gkyl_range_idx(&range, idx) );
      }
  TEST_CHECK(count == range.volume);

  count = 0;
  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k) {
        idx[0] = i; idx[1] = j; idx[2] = k;
        TEST_CHECK( count++ == gkyl_ridxn(range, idx) );
      }
  TEST_CHECK(count == range.volume);    
}

void test_range_offset()
{
  int lower[] = {1, 1, 2}, upper[] = {10, 20, 29};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);

  // offset of top-cell
  int offIdx[] = {0, 0, 1};
  int offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == 1);

  // offset of bottom-cell
  offIdx[0] = 0; offIdx[1] = 0; offIdx[2] = -1;
  offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == -1);

  // offset of right-cell
  offIdx[0] = 0; offIdx[1] = 1; offIdx[2] = 0;
  offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == 28);

  // offset of left-cell
  offIdx[0] = 0; offIdx[1] = -1; offIdx[2] = 0;
  offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == -28);  

  // offset of right-cell
  offIdx[0] = 1; offIdx[1] = 0; offIdx[2] = 0;
  offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == 28*20);

  // offset of left-cell
  offIdx[0] = -1; offIdx[1] = 0; offIdx[2] = 0;
  offset = gkyl_range_offset(&range, offIdx);
  TEST_CHECK(offset == -28*20);
}

void test_range_iter_2d()
{
  int lower[2] = {1, 1}, upper[2] = {2, 3};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);
  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &range);

  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j) {
      TEST_CHECK( 1 == gkyl_range_iter_next(&iter) );
      TEST_CHECK( iter.idx[0] == i );
      TEST_CHECK( iter.idx[1] == j );
    }
}

void test_range_iter_3d()
{
  int lower[3] = {1, 1, -10}, upper[3] = {2, 5, -5};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);
  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &range);

  for (int i=range.lower[0]; i<=range.upper[0]; ++i)
    for (int j=range.lower[1]; j<=range.upper[1]; ++j)
      for (int k=range.lower[2]; k<=range.upper[2]; ++k) {
        TEST_CHECK( 1 == gkyl_range_iter_next(&iter) );
        TEST_CHECK( iter.idx[0] == i );
        TEST_CHECK( iter.idx[1] == j );
        TEST_CHECK( iter.idx[2] == k );
      }
}

void test_range_inv_idx()
{
  int lower[] = {1, 1, 1}, upper[] = {5, 5, 4};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &range);

  int idx[3], linIdx = 0;
  while (gkyl_range_iter_next(&iter)) {
    gkyl_range_inv_idx(&range, linIdx++, idx);
    for (int d=0; d<range.ndim; ++d)
      TEST_CHECK( iter.idx[d] == idx[d] );
  }
}

void test_huge_range()
{
  int lower[] = {1, 1, 1, 1, 1, 1}, upper[] = {64, 64, 64, 64, 64, 64};
  struct gkyl_range range;
  gkyl_range_init(&range, 6, lower, upper);

  long vol = 1L*64*64*64*64*64*64;

  TEST_CHECK( vol == range.volume );

  long lidx = gkyl_ridxn(range, lower);
  long uidx = gkyl_ridxn(range, upper);
  TEST_CHECK( (uidx-lidx+1) == range.volume );

  int idx[6];
  gkyl_range_inv_idx(&range, uidx, idx);
  for (unsigned d=0; d<6; ++d)
    TEST_CHECK( idx[d] == upper[d] );
}

void test_range_deflate()
{
  int lower[] = {1, 1, 1}, upper[] = {10, 20, 30};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);

  int remDir[] = {0, 0, 1}, locDir[] = {0, 0, lower[2]};
  struct gkyl_range defr;
  gkyl_range_deflate(&defr, &range, remDir, locDir);

  TEST_CHECK( gkyl_range_is_sub_range(&defr) );

  TEST_CHECK( defr.ndim == 2 );
  TEST_CHECK( defr.volume == 200 );
  
  TEST_CHECK( defr.lower[0] == lower[0] );
  TEST_CHECK( defr.upper[0] == upper[0] );
  
  TEST_CHECK( defr.lower[1] == lower[1] );
  TEST_CHECK( defr.upper[1] == upper[1] );

  int idx[3]; idx[2] = lower[2];

  // loop over deflated region
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &defr);
  while (gkyl_range_iter_next(&iter)) {
    idx[0] = iter.idx[0]; idx[1] = iter.idx[1];
    TEST_CHECK(
      gkyl_range_idx(&defr, iter.idx) == gkyl_range_idx(&range, idx)
    );
  }

  // remove two directions
  remDir[0] = 0; remDir[1] = 1; remDir[2] = 1;
  locDir[0] = 0; locDir[1] = upper[1]; locDir[2] = lower[2];
  gkyl_range_deflate(&defr, &range, remDir, locDir);

  TEST_CHECK( defr.ndim == 1 );
  TEST_CHECK( defr.volume == 10 );
  
  TEST_CHECK( defr.lower[0] == lower[0] );
  TEST_CHECK( defr.upper[0] == upper[0] );
  
  idx[1] = upper[1]; idx[2] = lower[2];
  // loop over deflated region
  gkyl_range_iter_init(&iter, &defr);
  while (gkyl_range_iter_next(&iter)) {
    idx[0] = iter.idx[0];
    TEST_CHECK(
      gkyl_range_idx(&defr, iter.idx) == gkyl_range_idx(&range, idx)
    );
  }

  // remove all three directions
  remDir[0] = 1; remDir[1] = 1; remDir[2] = 1;
  locDir[0] = 5; locDir[1] = upper[1]; locDir[2] = lower[2];
  gkyl_range_deflate(&defr, &range, remDir, locDir);

  TEST_CHECK( defr.ndim == 0 );
  TEST_CHECK( defr.volume == 1 );
  
  idx[0] = 5; idx[1] = upper[1]; idx[2] = lower[2];
  // loop over deflated region
  gkyl_range_iter_init(&iter, &defr);
  while (gkyl_range_iter_next(&iter)) {
    TEST_CHECK(
      gkyl_range_idx(&defr, iter.idx) == gkyl_range_idx(&range, idx)
    );
  }
}

void test_range_skip_iter()
{
  int lower[] = {0, 0, 0}, upper[] = {5, 9, 17};
  struct gkyl_range range;
  gkyl_range_init(&range, 3, lower, upper);

  // skip iter for full range 
  struct gkyl_range_skip_iter skip;
  gkyl_range_skip_iter_init(&skip, &range);

  TEST_CHECK( skip.delta == range.volume );
  TEST_CHECK( skip.range.ndim == 0 );  
  TEST_CHECK( skip.range.volume == 1 );
  TEST_CHECK( gkyl_range_idx(&skip.range, NULL) == 0 );

  // ---
  int lowerSub[] = {1, 1, 1}, upperSub[] = {4, 8, 16};
  struct gkyl_range localRange;
  gkyl_sub_range_init(&localRange, &range, lowerSub, upperSub);

  // skip iter for local range 
  gkyl_range_skip_iter_init(&skip, &localRange);

  TEST_CHECK( skip.delta == 16 );
  TEST_CHECK( skip.delta*skip.range.volume == localRange.volume );
  TEST_CHECK( skip.range.ndim == 2 );

  long count = 0;
  // loops are an outer while loop, with an inner for loop
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &skip.range);
  while (gkyl_range_iter_next(&iter)) {
    long start = gkyl_range_idx(&skip.range, iter.idx);
    for (long i=start; i<start+skip.delta; ++i)
      count += 1;
  }
  TEST_CHECK( count == localRange.volume );
}

void test_range_skip_iter_2()
{
  int lower[] = {0, 0, 0, 1, 1, 1}, upper[] = {17, 17, 17, 16, 16, 16};
  struct gkyl_range range;
  gkyl_range_init(&range, 6, lower, upper);

  int lowerSub[] = {1, 1, 1, 1, 1, 1}, upperSub[] = {16, 16, 16, 16, 16, 16};
  struct gkyl_range localRange;
  gkyl_sub_range_init(&localRange, &range, lowerSub, upperSub);

  // skip iter for local range
  struct gkyl_range_skip_iter skip;  
  gkyl_range_skip_iter_init(&skip, &localRange);

  TEST_CHECK( skip.delta*skip.range.volume == 16*16*16*16*16*16 );
}

void test_range_split_1()
{
  int lower[] = { 1 }, upper[] = { 10 };
  struct gkyl_range range;
  gkyl_range_init(&range, 1, lower, upper);

  int idx[range.ndim];
  struct gkyl_range_iter iter;

  int nsplits = 4, curr_start = 0, tot = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&range, nsplits, tid);

    gkyl_range_iter_init(&iter, &range);
    gkyl_range_inv_idx(&range, curr_start, idx);

    for (int d=0; d<range.ndim; ++d)
      TEST_CHECK( idx[d] == iter.idx[d] );
    curr_start += iter.bumps_left;
    tot += iter.bumps_left;
  }
  TEST_CHECK( tot == range.volume );
}

void test_range_split_2()
{
  int lower[] = { 1, 2 }, upper[] = { 10, 25 };
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int idx[range.ndim];
  struct gkyl_range_iter iter;  

  int nsplits = 4, curr_start = 0, tot = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&range, nsplits, tid);
    
    gkyl_range_iter_init(&iter, &range);
    gkyl_range_inv_idx(&range, curr_start, idx);
    
    for (int d=0; d<range.ndim; ++d)
      TEST_CHECK( idx[d] == iter.idx[d] );
    curr_start += iter.bumps_left;
    tot += iter.bumps_left;
  }
  TEST_CHECK( tot == range.volume );

  gkyl_range_set_split(&range, nsplits, 0);
  
  tot = 0;
  gkyl_range_iter_no_split_init(&iter, &range);
  while (gkyl_range_iter_next(&iter)) {
    tot += 1;
  }
  TEST_CHECK( tot == range.volume );
}

void test_range_split_3()
{
  int lower[] = { 1, 2 }, upper[] = { 10, 25 };
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int idx[range.ndim];
  struct gkyl_range_iter iter;  

  int nsplits = 4, curr_start = 0, tot = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    struct gkyl_range sr = gkyl_range_split(&range, nsplits, tid);

    gkyl_range_iter_init(&iter, &sr);
    gkyl_range_inv_idx(&sr, curr_start, idx);
    
    for (int d=0; d<sr.ndim; ++d)
      TEST_CHECK( idx[d] == iter.idx[d] );
    curr_start += iter.bumps_left;
    tot += iter.bumps_left;
  }
  TEST_CHECK( tot == range.volume );

  tot = 0;
  gkyl_range_iter_no_split_init(&iter, &range);
  while (gkyl_range_iter_next(&iter)) {
    tot += 1;
  }
  TEST_CHECK( tot == range.volume );
}

void test_sub_range_split()
{
  int lower[] = {1, 1}, upper[] = {10, 20};
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int sublower[] = {2, 2}, subupper[] = { 5, 10 };
  struct gkyl_range subrange;
  gkyl_sub_range_init(&subrange, &range, sublower, subupper);
  TEST_CHECK( gkyl_range_is_sub_range(&subrange) == 1 );

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &subrange);

  long th_start = gkyl_range_idx(&subrange, iter.idx);
  TEST_CHECK( th_start == gkyl_range_idx(&subrange, subrange.lower) );
  TEST_CHECK( iter.bumps_left == subrange.volume );

  int start_idx[] = {
    gkyl_ridx(subrange, 2, 2),
    gkyl_ridx(subrange, 3, 2),
    gkyl_ridx(subrange, 4, 2),
    gkyl_ridx(subrange, 5, 2),
  };

  int nsplits = 4, tot = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&subrange, nsplits, tid);
    gkyl_range_iter_init(&iter, &subrange);
    
    long th_start = gkyl_range_idx(&subrange, iter.idx);
    TEST_CHECK( th_start == start_idx[tid] );
    tot += iter.bumps_left;
  }
  TEST_CHECK( tot == subrange.volume );
}

void test_range_split_iter_1()
{
  int lower[] = { 1 }, upper[] = { 13 };
  struct gkyl_range range;
  gkyl_range_init(&range, 1, lower, upper);

  int nsplits = 16;
  long tot_vol = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&range, nsplits, tid);

    long vol = 0;
    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);
    while (gkyl_range_iter_next(&iter))
      vol += 1;
    tot_vol += vol;
  }
  TEST_CHECK( tot_vol == range.volume );
}

void test_range_split_iter_2()
{
  int lower[] = { 1, 1 }, upper[] = { 10, 25 };
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  int nsplits = 4;
  long tot_vol = 0;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&range, nsplits, tid);

    long vol = 0;
    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);
    while (gkyl_range_iter_next(&iter))
      vol += 1;
    tot_vol += vol;
  }
  TEST_CHECK( tot_vol == range.volume );
}

void test_range_split_iter_3()
{
  int lower[] = { 1, 1 }, upper[] = { 10, 25 };
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, 1, range.volume);
  gkyl_array_clear(arr, 0.0);

  int nsplits = 4;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&range, nsplits, tid);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);
    
    while (gkyl_range_iter_next(&iter)) {
      long lidx = gkyl_range_idx(&range, iter.idx);
      double *d = gkyl_array_fetch(arr, lidx);
      d[0] += 1.0;
    }
  }

  // reset to default
  gkyl_range_set_split(&range, 1, 0);
  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &range);
  while (gkyl_range_iter_next(&iter)) {
    long lidx = gkyl_range_idx(&range, iter.idx);
    double *d = gkyl_array_fetch(arr, lidx);

    TEST_CHECK( d[0] == 1.0 );
  }

  gkyl_array_release(arr);
}

void test_sub_range_split_iter()
{
  int lower[] = { 1, 1 }, upper[] = { 20, 25 };
  struct gkyl_range range;
  gkyl_range_init(&range, 2, lower, upper);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, 1, range.volume);
  gkyl_array_clear(arr, 0.0);

  int sublower[] = {2, 2}, subupper[] = { 15, 14 };
  struct gkyl_range subrange;
  gkyl_sub_range_init(&subrange, &range, sublower, subupper);  

  int nsplits = 2;
  for (int tid=0; tid<nsplits; ++tid) {
    gkyl_range_set_split(&subrange, nsplits, tid);
    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &subrange);
    
    while (gkyl_range_iter_next(&iter)) {
      long lidx = gkyl_range_idx(&subrange, iter.idx);
      double *d = gkyl_array_fetch(arr, lidx);
      d[0] += 1.0;
    }
  }

  // reset to default
  gkyl_range_set_split(&subrange, 1, 0);
  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &subrange);
  while (gkyl_range_iter_next(&iter)) {
    long lidx = gkyl_range_idx(&subrange, iter.idx);
    double *d = gkyl_array_fetch(arr, lidx);

    TEST_CHECK( d[0] == 1.0 );
  }

  gkyl_array_release(arr);
}

void test_nested_iter()
{
  int shape[] = {2, 2, 4, 8};
  
  struct gkyl_range phase_range, conf_range, vel_range;
  gkyl_range_init_from_shape(&phase_range, 4, shape);
  gkyl_range_init_from_shape(&conf_range, 2, shape);

  struct gkyl_array *carr = gkyl_array_new(GKYL_DOUBLE, 1, conf_range.volume);
  gkyl_array_clear(carr, 0.0);

  struct gkyl_range_iter conf_iter, vel_iter;

  int rem_dir[] = { 1, 1, 0, 0};
  
  gkyl_range_iter_init(&conf_iter, &conf_range);
  while (gkyl_range_iter_next(&conf_iter)) {

    gkyl_range_deflate(&vel_range, &phase_range, rem_dir, conf_iter.idx);
    gkyl_range_iter_init(&vel_iter, &vel_range);

    double *carrdat = gkyl_array_fetch(carr, gkyl_range_idx(&conf_range, conf_iter.idx));
    while (gkyl_range_iter_next(&vel_iter))
      carrdat[0] += 1;
  }

  gkyl_range_iter_init(&conf_iter, &conf_range);
  while (gkyl_range_iter_next(&conf_iter)) {
    double *carrdat = gkyl_array_fetch(carr, gkyl_range_idx(&conf_range, conf_iter.idx));
    TEST_CHECK( carrdat[0] == 32.0 );
  }

  gkyl_array_release(carr);
}

void test_intersect()
{
  struct gkyl_range r1, r2, r3, r4, inter;

  gkyl_range_init(&r1, 2, (int[]) { 1, 1 }, (int[]) { 20, 20 } );
  gkyl_range_init(&r2, 2, (int[]) { 6, 6 }, (int[]) { 10, 10 } );
  gkyl_range_init(&r3, 2, (int[]) { 6, 6 }, (int[]) { 32, 40 } );
  gkyl_range_init(&r4, 2, (int[]) { 32, 40 }, (int[]) { 64, 80 } );

  // r1 inter r2
  TEST_CHECK( 1 == gkyl_range_intersect(&inter, &r1, &r2) );

  TEST_CHECK ( 25 == inter.volume );
  TEST_CHECK ( 6 == inter.lower[0] );
  TEST_CHECK ( 6 == inter.lower[1] );
  TEST_CHECK ( 10 == inter.upper[0] );
  TEST_CHECK ( 10 == inter.upper[1] );

  // r1 inter r3
  TEST_CHECK( 1 == gkyl_range_intersect(&inter, &r1, &r3) );

  TEST_CHECK ( 15*15 == inter.volume );
  TEST_CHECK ( 6 == inter.lower[0] );
  TEST_CHECK ( 6 == inter.lower[1] );
  TEST_CHECK ( 20 == inter.upper[0] );
  TEST_CHECK ( 20 == inter.upper[1] );

  // r1 inter r4
  TEST_CHECK( 0 == gkyl_range_intersect(&inter, &r1, &r4) );
  
}

// CUDA specific tests
#ifdef GKYL_HAVE_CUDA

/* Function signatures of kernel calls */
int cu_range_test(const struct gkyl_range rng);

void test_cu_range()
{
  int shape[] = {25, 50};
  struct gkyl_range range;
  gkyl_range_init_from_shape(&range, 2, shape);

  // copy range to device
  //struct gkyl_range *cu_range = gkyl_range_clone_on_cu_dev(&range);

  // call kernel
  int nfail = cu_range_test(range);
  TEST_CHECK( nfail == 0 );

  //gkyl_cu_free(cu_range);
}

#endif

TEST_LIST = {
  { "range_0", test_range_0 },
  { "range_1", test_range_1 },
  { "range_shape",  test_range_shape },
  { "sub_range",  test_sub_range },
  { "sub_range_inv_idx",  test_sub_range_inv_idx },
  { "shorten", test_shorten },
  { "skin", test_skin },
  { "range_index_1d", test_range_index_1d },
  { "range_index_2d", test_range_index_2d },
  { "range_index_3d", test_range_index_3d },
  { "range_index_4d", test_range_index_4d },
  { "range_index_5d", test_range_index_5d },
  { "range_index_6d", test_range_index_6d },
  { "range_index_idx", test_range_idx },
  { "range_offset", test_range_offset },
  { "range_inv_idx", test_range_inv_idx },
  { "huge_range", test_huge_range },
  { "range_deflate", test_range_deflate },
  { "range_skip_iter", test_range_skip_iter },
  { "range_skip_iter_2", test_range_skip_iter_2 },
  { "range_split_1", test_range_split_1 },
  { "range_split_2", test_range_split_2 },
  { "range_split_3", test_range_split_3 },
  { "sub_range_split", test_sub_range_split },
  { "range_split_iter_1", test_range_split_iter_1 },  
  { "range_split_iter_2", test_range_split_iter_2 },
  { "range_split_iter_3", test_range_split_iter_3 },
  { "sub_range_split_iter", test_sub_range_split_iter },
  { "nested_iter", test_nested_iter },
  { "intersect", test_intersect },
#ifdef GKYL_HAVE_CUDA
  { "cu_range", test_cu_range },
#endif  
  { NULL, NULL },
};
