#include <gkyl_multib_conn.h>
#include <gkyl_gyrokinetic_multib.h>
#include <gkyl_gyrokinetic_multib_priv.h>
#include <gkyl_multib_comm_conn.h>

// Maximum number of blocks
#define GKYL_MAX_BLOCKS 12



/**
 * Count number of distinct elements in an array of ints
 * .
 * @param a input array of ints
 * @param n length of a
 * return number of unique elements in a
 */
static int count_distinct(int a[], int n)
{
   int i, j, count = 1;
   for (i = 1; i < n; i++) { // Check if a[i] is a new element
     for (j = 0; j < i; j++) {
       if (a[i] == a[j])    // Check if a[i] has already been found 
          break;            // Break if it is a duplicate
     }
     if (i == j)
       count++;     //increment the number of distinct elements
   }
   return count;
}

/**
 * Populate an output array with the distinct elements in an array of ints
 * .
 * @param a input array of ints
 * @param n length of input array
 * @param unique_array on output contains the unique elements in a
 * return number of unique elements in a
 */
static int get_unique(int *a, int n, int *unique_array) {
   unique_array[0] = a[0]; // The first element of a is the first unique element
   int i, j, count = 1;
   for (i = 1; i < n; i++) { // Check if a[i] is a new element
     for (j = 0; j < i; j++) {
       if (a[i] == a[j])    // Check if a[i] has already been found 
          break;            // Break if it is a duplicate
     }
     if (i == j) {
       count++;     //increment the number of distinct elements
       unique_array[i] = a[i];
     }
   }
   return count;
}


/** Insert an element at the beginning of an array of ints
 * @param arr of length n+1 (padded with one dummy value at the end)
 * @param n number of values in arr before insertion
 * @param new val value to insert
*/
static void
insert_below(int* arr, int n, int new_val)
{
  int temp_arr[GKYL_MAX_BLOCKS] = {-1};
  for (int i = 0; i<n; i++) {
    temp_arr[i+1] = arr[i];
  }
  temp_arr[0] = new_val;
  n+=1;
  for (int i = 0; i<n; i++) {
    arr[i] = temp_arr[i];
  }

}

/** Insert an element at the end of an array of ints
 * @param arr of length n+1 (padded with one dummy value at the end)
 * @param n number of values in arr before insertion
 * @param new val value to insert
*/
static void
insert_above(int* arr, int n, int new_val)
{
  arr[n] = new_val;
}

//// This function should be called in a loop over num blocks local
//void
//set_crossz_idxs(struct gkyl_gyrokinetic_multib_app *mba, int myidx, int* crossz_blocks, int* num_blocks){
//  struct gkyl_block_topo *btopo = mba->btopo;
//  struct gkyl_block_connections *conn = btopo->conn;
//  int dir = 1;
//
//  int bidx = myidx;
//  crossz_blocks[0] = bidx;
//  *num_blocks = 1;
//
//  while(true) {
//    if (conn[bidx].connections[dir][0].edge == GKYL_BLOCK_EDGE_PHYSICAL) {
//      break;
//    }
//    else if (conn[bidx].connections[dir][0].edge == GKYL_BLOCK_EDGE_UPPER_POSITIVE) { 
//      insert_below(crossz_blocks, num_blocks, conn[bidx].connections[dir][0].bid);
//      bidx = conn[bidx].connections[dir][0].bid;
//    }
//  }
//
//  bidx = myidx;
//  while(true) {
//    if (conn[bidx].connections[dir][1].edge == GKYL_BLOCK_EDGE_PHYSICAL) {
//      break;
//    }
//    else if (conn[bidx].connections[dir][1].edge == GKYL_BLOCK_EDGE_LOWER_POSITIVE) { 
//      insert_above(crossz_blocks, num_blocks, conn[bidx].connections[dir][1].bid);
//      bidx = conn[bidx].connections[dir][1].bid;
//    }
//  }
//
//}





/**
 *  Get the block indices of neighbors (adjacent blocks) in a direction.
 * 
 * @param mbapp multiblock app object
 * @param bidx block index
 * @param dir
 * @param neighbor_idxs on output indices of neighboring blocks
 * return number of neighbors
 */
int get_neighbors(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int dir, int *neighbor_idxs)
{
  struct gkyl_block_connections conn = mbapp->block_topo->conn[bidx];
  int neighbor_num = 0;
  for (int e = 0; e < 2; e++) {
    if (conn.connections[dir][e].edge != GKYL_PHYSICAL) {
      neighbor_idxs[neighbor_num] = conn.connections[dir][e].bid;
      neighbor_num += 1;
    }
  }
  return neighbor_num;
}

/**
 *  Get the number of neighbors (adjacent blocks) in a direction.
 * 
 * @param mbapp multiblock app object
 * @param bidx block index
 * @param direction
 * return number of neighbors
 */
int get_num_neighbors(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int dir)
{
  int neighbor_idxs[100] = {-1};
  int neighbor_num = get_neighbors(mbapp, bidx, dir, neighbor_idxs);
  return neighbor_num;
}




/**
 *  Get the indices of connected blocks in a direction.
 * 
 * @param mbapp multiblock app object
 * @param bidx block index
 * @param direction
 * @param block_list ordered indices of connected blocks including self
 * return number of connected blocks 
 */
int get_connected(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int dir, int* block_list)
{
  struct gkyl_block_connections conn;
  block_list[0] = bidx;
  int num_blocks = 1;
  int curr_bidx = bidx;
  while(true) {
    conn = mbapp->block_topo->conn[curr_bidx];
    if (conn.connections[dir][0].edge == GKYL_PHYSICAL) {
      break;
    }
    else if (conn.connections[dir][0].edge == GKYL_UPPER_POSITIVE) { 
      if (conn.connections[dir][0].bid == bidx) return num_blocks;
      insert_below(block_list, num_blocks, conn.connections[dir][0].bid);
      curr_bidx = conn.connections[dir][0].bid;
      num_blocks+=1;
    }
  }

  curr_bidx = bidx;
  while(true) {
    conn = mbapp->block_topo->conn[curr_bidx];
    if (conn.connections[dir][1].edge == GKYL_PHYSICAL) {
      break;
    }
    else if (conn.connections[dir][1].edge == GKYL_LOWER_POSITIVE) { 
      if (conn.connections[dir][1].bid == bidx) return num_blocks;
      insert_above(block_list, num_blocks, conn.connections[dir][1].bid);
      curr_bidx = conn.connections[dir][1].bid;
      num_blocks+=1;
    }
  }
  
  return num_blocks;
}

/**
 *  Get the number of connected blocks in a direction.
 * 
 * @param mbapp multiblock app object
 * @param bidx block index
 * @param direction
 * return number of connected blocks 
 */
int get_num_connected(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int dir)
{
  int block_list[100] = {-1};
  int num_blocks = get_connected(mbapp, bidx, dir, block_list);
  return num_blocks;
}



/**
 * Check if a block corner is an interior corner
 * @param mbapp multib app object
 * @param bidx block index
 * @param edges list of edges (0 for lower, 1 for upper)
*/
int check_corner(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int* edges)
{
  struct gkyl_block_topo *btopo =  mbapp->block_topo;
  struct gkyl_block_connections conn = mbapp->block_topo->conn[bidx];
  int ndim = btopo->ndim;
  int interior = 1; // true
  for (int i = 0; i < ndim; i++) {
    if(conn.connections[i][edges[i]].edge == GKYL_PHYSICAL)  {
      interior = 0;
      break;
    }
  }
  return interior;
}

/**
 * Get the list of blocks that touch a specific corner of a block
 * @param mbapp multib app object
 * @param bidx block index
 * @param edges list of edges of length ndim (0 for lower, 1 for upper)
 * @param block list on output a list of block indices that touch the corner
*/
int get_corner_connected(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int* edges, int* block_list)
{
  struct gkyl_block_topo *btopo =  mbapp->block_topo;
  int ndim = btopo->ndim;
  int num_corner_connected = 0;
  struct gkyl_block_connections conn = mbapp->block_topo->conn[bidx];
  int interior = check_corner(mbapp, bidx, edges);
  if (interior == 0) return num_corner_connected;
  num_corner_connected+=1;
  block_list[0] = bidx;
  int next_dir = 0;
  int next_edges[ndim];
  for (int i = 0; i < ndim; i++) next_edges[i] = edges[i];

  while(true) {
    int next_bidx = conn.connections[next_dir][next_edges[next_dir]].bid;
    if (next_bidx == bidx) break; // back at original
    block_list[num_corner_connected] = next_bidx;
    num_corner_connected+=1;
    next_edges[next_dir] = !next_edges[next_dir]; // 0 ->1 or 1 ->0
    next_dir = !next_dir;                                         
    interior = check_corner(mbapp, next_bidx, next_edges);
    if(interior == 0) break; // no more corners
    conn = mbapp->block_topo->conn[next_bidx];
  }

  return num_corner_connected;
}

/**
 * Get the number of blocks that touch a specific corner of a block
 * @param mbapp multib app object
 * @param bidx block index
 * @param edges list of edges of length ndim (0 for lower, 1 for upper)
 * return number of blocks touching this corner
*/
int get_num_corner_connected(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int* edges)
{
  int block_list[100] = {-1};
  int num_corner_connected = get_corner_connected(mbapp, bidx, edges, block_list);
  return num_corner_connected;
}


/** 
 * @param mbapp multib app object
 * @param bidx block index
 * @param dir direction in which to find neighbors or connected blocks
 * @param conn_id type of connection : GKYL_CONN_NEIGHBOR, _ALL, or _CORNER
 */
void gkyl_multib_conn_get_connection(struct gkyl_gyrokinetic_multib_app *mbapp, int bidx, int dir, enum gkyl_conn_id conn_id)
{
  struct gkyl_block_topo *btopo =  mbapp->block_topo;
  struct gkyl_block_connections conn = btopo->conn[bidx];

  if (conn_id == GKYL_CONN_NEIGHBOR) {
      int num_neighbors = get_num_neighbors(mbapp, bidx, dir);
      int neighbors[num_neighbors];
      get_neighbors(mbapp, bidx, dir, neighbors);
      printf("dir %d neighbors for block %d : ", dir, bidx);
      for( int i = 0; i <num_neighbors; i++) printf(" %d", neighbors[i]);
      printf("\n");
  }
  else if (conn_id == GKYL_CONN_ALL) {
      int num_connected = get_num_connected(mbapp, bidx, dir);
      int connected_bidxs[num_connected];
      get_connected(mbapp, bidx, dir, connected_bidxs);
      printf("dir %d connected for block %d : ", dir, bidx);
      for( int i = 0; i <num_connected; i++) printf(" %d", connected_bidxs[i]);
      printf("\n");
  }
  else if (conn_id == GKYL_CONN_CORNER) {
      for( int e0 = 0; e0 < 2; e0++) {
        for( int e1 = 0; e1 < 2; e1++) {
          int edges[2] = {e0,e1};
          int num_connected = get_num_corner_connected(mbapp, bidx, edges);
          printf("corner num connected at corner (%d,%d) = %d\n", e0, e1, num_connected);
          int connected_bidxs[num_connected];
          get_corner_connected(mbapp, bidx, edges, connected_bidxs);
          printf("corner connected for block %d at corner (%d,%d): ", bidx, e0,e1);
          for( int i = 0; i <num_connected; i++) printf(" %d", connected_bidxs[i]);
          printf("\n");
        }
      }
  }

}
