#include <assert.h>
#include <string.h>
#include <gkyl_util.h>
#include <gkyl_alloc.h>

#include "cart_modal_serendip_priv.h"

void
gkyl_cart_modal_serendip(struct gkyl_basis *basis, int ndim, int poly_order)
{
  assert(ndim>0 && ndim<=6);
  assert(ev_list[ndim].ev[poly_order]);
  
  basis->ndim = ndim;
  basis->poly_order = poly_order;
  basis->num_basis = num_basis_list[ndim].count[poly_order];
  strcpy(basis->id, "serendipity");
  basis->b_type = GKYL_BASIS_MODAL_SERENDIPITY;

  // function pointers
  basis->eval = ev_list[ndim].ev[poly_order];
  basis->eval_expand = eve_list[ndim].ev[poly_order];
  basis->eval_grad_expand = eveg_list[ndim].ev[poly_order];
  basis->flip_odd_sign = fos_list[ndim].fs[poly_order];
  basis->flip_even_sign = fes_list[ndim].fs[poly_order];
  basis->node_list = nl_list[ndim].nl[poly_order];
  basis->nodal_to_modal = n2m_list[ndim].n2m[poly_order];
}

struct gkyl_basis *
gkyl_cart_modal_serendip_new(int ndim, int poly_order)
{
  struct gkyl_basis *basis = gkyl_malloc(sizeof(struct gkyl_basis));
  gkyl_cart_modal_serendip(basis, ndim, poly_order);
  return basis;
}

#ifndef GKYL_HAVE_CUDA
void
gkyl_cart_modal_serendip_cu_dev(struct gkyl_basis *basis, int ndim, int poly_order)
{
  assert(false);
}

struct gkyl_basis *
gkyl_cart_modal_serendip_cu_dev_new(int ndim, int poly_order)
{
  assert(false);
}
#endif
