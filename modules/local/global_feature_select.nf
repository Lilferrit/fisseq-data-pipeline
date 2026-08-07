nextflow.enable.dsl = 2

// GLOBAL_FEATURE_SELECT: wraps python -m fisseq_data_pipeline.globalfeatureselect.
// Runs once per active global channel. Unlike the BATCHWISE feature-selection
// chain (FINALIZE_FEATURE_SELECT et al., which parallelize genuinely
// expensive cell-level bootstrap work), this process needs no Nextflow-level
// fan-out: it reads the channel's member batches' already-published BATCHWISE
// aggregates/blocklists directly off pipeline_dir (same "glob published
// output" idiom ANOVA_NORMALIZED/OVWT_GLOBAL already use -- see AGENTS.md),
// looping over batch_stems in Python. params.feature_select_types is passed
// through so that glob is filtered to the currently-configured feature
// types -- otherwise stale per-feature-type files left behind by a prior
// run with a larger feature_select_types (publishDir mode: 'copy' never
// deletes them) would silently leak into the global aggregate. No path()
// file inputs, so there is no same-named-file staging collision to design
// around. The channel identifier is named "chan" below -- "channel" is a
// reserved Nextflow binding (lowercase alias for the Channel class), see
// AGENTS.md.
process GLOBAL_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(batch_stems), val(pipeline_dir), val(publish_subdir), val(min_batches_ok), val(feature_select_types), \
          val(run_pca), val(pca_n_components), val(run_umap), val(umap_n_components), val(umap_n_neighbors), \
          val(umap_metric), val(umap_min_dist), val(umap_random_state)

    output:
    // pca_components.parquet only exists when run_pca=true
    tuple val(chan), path("aggregate.parquet"), path("blocklist.parquet"), path("pca_components.parquet", optional: true)

    script:
    def stemsArg = "[" + batch_stems.join(',') + "]"
    def minArg = (min_batches_ok == null) ? "" : "min_batches_ok=${min_batches_ok}"
    def typesArg = "[" + feature_select_types.join(',') + "]"
    """
    echo "Starting GLOBAL_FEATURE_SELECT for ${chan}"
    python -m fisseq_data_pipeline.globalfeatureselect \\
        output_dir=. \\
        pipeline_dir=${pipeline_dir} \\
        "batch_stems=${stemsArg}" \\
        "feature_select_types=${typesArg}" \\
        ${minArg} \\
        run_pca=${run_pca} \\
        pca_n_components=${pca_n_components} \\
        run_umap=${run_umap} \\
        umap_n_components=${umap_n_components} \\
        umap_n_neighbors=${umap_n_neighbors} \\
        umap_metric=${umap_metric} \\
        umap_min_dist=${umap_min_dist} \\
        umap_random_state=${umap_random_state}
    """
}
