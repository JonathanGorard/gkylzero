#pragma once

#include <gkyl_array.h>
#include <gkyl_range.h>
#include <gkyl_rect_grid.h>

struct gkyl_array_header_info;

/**
 * Write out grid and array data header data to file. Note that only
 * HEADER is written and NOT the array data itself.
 *
 * @param grid Grid object to write
 * @param hrd Header data.
 * @return Status flag: 0 if write succeeded, 'errno' otherwise
 */
int gkyl_grid_sub_array_header_write_fp(const struct gkyl_rect_grid *grid,
  const struct gkyl_array_header_info *hdr, FILE *fp);

/**
 * Read grid and array data header data from file. Note that only
 * HEADER is read and NOT the array data itself.
 * 
 * If the array header contains meta-data then the 'meta' char * is
 * allocated in the @a hdr struct.
 *
 * YOU MUST FREE 'hdr->meta' using gkyl_free or else this will result
 * in a memory leak!
 *
 * @param grid Grid object to read
 * @param hrd On output, Header data.
 * @return Status flag: 0 if read succeeded, 'errno' otherwise
 */
int gkyl_grid_sub_array_header_read_fp(struct gkyl_rect_grid *grid,
  struct gkyl_array_header_info *hdr, FILE *fp);

/**
 * Release memory for header.
 *
 * @param hdr Header memory to release
 */
void gkyl_grid_sub_array_header_release(struct gkyl_array_header_info *hdr);