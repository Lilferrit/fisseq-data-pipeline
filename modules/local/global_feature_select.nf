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
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(batch_stems), val(pipeline_dir), val(publish_subdir)

    output:
    // pca_components.parquet only exists when run_pca=true -- must be its
    // own output statement, not another element of the tuple below: see the
    // matching comment in finalize_feature_select.nf for why (per-element
    // `optional: true` inside a multi-element tuple output isn't honored on
    // this Nextflow version and would silently drop the whole tuple).
    tuple val(chan), path("aggregate.parquet"), path("blocklist.parquet")
    path("pca_components.parquet", optional: true)

    when:
    task.ext.when == null || task.ext.when

    script:
    def stemsArg = "[" + batch_stems.join(',') + "]"
    def minArg = (params.global_feature_select_min_batches_ok == null) ? "" : "min_batches_ok=${params.global_feature_select_min_batches_ok}"
    def typesArg = "[" + params.feature_select_types.join(',') + "]"
    """
    echo "Starting GLOBAL_FEATURE_SELECT for ${chan}"
    python -m fisseq_data_pipeline.globalfeatureselect \\
        output_dir=. \\
        pipeline_dir=${pipeline_dir} \\
        "batch_stems=${stemsArg}" \\
        "feature_select_types=${typesArg}" \\
        ${minArg} \\
        label_column=${params.filter_label_column} \\
        run_pca=${params.run_pca} \\
        pca_n_components=${params.pca_n_components} \\
        run_umap=${params.run_umap} \\
        umap_n_components=${params.umap_n_components} \\
        umap_n_neighbors=${params.umap_n_neighbors} \\
        umap_metric=${params.umap_metric} \\
        umap_min_dist=${params.umap_min_dist} \\
        random_seed=${params.random_seed}
    """
}
