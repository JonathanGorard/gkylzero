#include <assert.h>
#include <gkyl_gyrokinetic_priv.h>

void 
gk_neut_species_source_init(struct gkyl_gyrokinetic_app *app, struct gk_neut_species *s, 
  struct gk_source *src)
{
  int vdim = app->vdim+1;
  // we need to ensure source has same shape as distribution function
  src->source = mkarr(app->use_gpu, app->neut_basis.num_basis, s->local_ext.volume);
  src->source_id = s->source_id;
  src->source_host = src->source;
  if (app->use_gpu) {
    src->source_host = mkarr(false, app->neut_basis.num_basis, s->local_ext.volume);
  }

  src->write_source = s->info.source.write_source; // optional flag to write out source

  src->num_sources = s->info.source.num_sources;
  for (int k=0; k<s->info.source.num_sources; k++) {
    gk_neut_species_projection_init(app, s, s->info.source.projection[k], &src->proj_source[k]);
  }

  // Allocate data and updaters for diagnostic moments.
  src->num_diag_moments = s->info.num_diag_moments;
  s->src.moms = gkyl_malloc(sizeof(struct gk_species_moment[src->num_diag_moments]));
  for (int m=0; m<src->num_diag_moments; ++m) {
    gk_neut_species_moment_init(app, s, &s->src.moms[m], s->info.diag_moments[m]);
  }

  // Allocate data and updaters for integrated moments.
  gk_neut_species_moment_init(app, s, &s->src.integ_moms, "Integrated");
  if (app->use_gpu) {
    s->src.red_integ_diag = gkyl_cu_malloc(sizeof(double[vdim+2]));
    s->src.red_integ_diag_global = gkyl_cu_malloc(sizeof(double[vdim+2]));
  } 
  else {
    s->src.red_integ_diag = gkyl_malloc(sizeof(double[vdim+2]));
    s->src.red_integ_diag_global = gkyl_malloc(sizeof(double[vdim+2]));
  }
  // allocate dynamic-vector to store all-reduced integrated moments 
  s->src.integ_diag = gkyl_dynvec_new(GKYL_DOUBLE, vdim+2);
  s->src.is_first_integ_write_call = true;
}

void
gk_neut_species_source_calc(gkyl_gyrokinetic_app *app, const struct gk_neut_species *s, 
  struct gk_source *src, double tm)
{
  struct gkyl_array *source_tmp = mkarr(app->use_gpu, app->neut_basis.num_basis, s->local_ext.volume);
  for (int k=0; k<s->info.source.num_sources; k++) {
    gk_neut_species_projection_calc(app, s, &src->proj_source[k], source_tmp, tm);
    gkyl_array_accumulate(src->source, 1., source_tmp);
  }
  gkyl_array_release(source_tmp);
}

// Compute rhs of the source
void
gk_neut_species_source_rhs(gkyl_gyrokinetic_app *app, const struct gk_neut_species *species,
  struct gk_source *src, const struct gkyl_array *fin, struct gkyl_array *rhs)
{
  gkyl_array_accumulate(rhs, 1.0, src->source);
}

void
gk_neut_species_source_release(const struct gkyl_gyrokinetic_app *app, const struct gk_source *src)
{
  gkyl_array_release(src->source);
  if (app->use_gpu) {
    gkyl_array_release(src->source_host);
  }
  for (int k=0; k<src->num_sources; k++) {
    gk_neut_species_projection_release(app, &src->proj_source[k]);
  }

  // Release moment data.
  for (int i=0; i<src->num_diag_moments; ++i) {
    gk_neut_species_moment_release(app, &src->moms[i]);
  }
  gkyl_free(src->moms);
  gk_neut_species_moment_release(app, &src->integ_moms); 
  if (app->use_gpu) {
    gkyl_cu_free(src->red_integ_diag);
    gkyl_cu_free(src->red_integ_diag_global);
  }
  else {
    gkyl_free(src->red_integ_diag);
    gkyl_free(src->red_integ_diag_global);
  }  
  gkyl_dynvec_release(src->integ_diag);
}