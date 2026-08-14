nextflow.enable.dsl = 2

// GLOBAL_FEATURE_SELECT: wraps python -m fisseq_data_pipeline.globalfeatureselect.
// Runs once per active global channel. Unlike the BATCHWISE feature-selection
// chain (FINALIZE_FEATURE_SELECT et al., which parallelize genuinely
// expensive cell-level bootstrap work), this process needs no Nextflow-level
// fan-out: it reuses the channel's member batches' own BATCHWISE aggregates/
// blocklists, already produced as real Nextflow outputs earlier in this same
// run (AGGREGATE_FEATURE_TYPE_BATCHWISE/COMBINE_BLOCKLISTS_BATCHWISE), passed
// in as real `path` inputs (agg_files/blocklist_files) rather than re-derived
// via a `val(pipeline_dir)` + `val(batch_stems)` Python-side glob -- the old
// approach broke -resume cache invalidation, since a task hash built from
// scalar batch_stems/pipeline_dir strings can't detect when the underlying
// files' *contents* change without the batch membership itself changing (see
// AGENTS.md gotcha 6).
//
// Every batch names its aggregate files "<feature_type>.parquet" and its
// blocklist "blocklist.parquet", so staging many batches' files into one
// task collides on basename -- resolved via `stageAs` with a single `*`,
// which Nextflow auto-numbers by list order ("agg_input_1.parquet",
// "agg_input_2.parquet", ...), paired with the parallel val(agg_batch_stems)/
// val(bl_batch_stems) lists (built from the same source list in
// workflows/fisseq.nf, so ordering matches) that tell the Python side which
// staged file belongs to which batch. feature-type identity is no longer
// threaded through at all: join_feature_type_files (utils/featuretypes.py)
// joins by file content/schema, not filename, so it never needed it -- and
// since Nextflow now only ever supplies the exact currently-configured
// feature_select_types (there is no way for a stale on-disk file to reach
// this process), the old "filter out stale/unconfigured feature-type files"
// logic is gone too, not just relocated.
//
// The channel identifier is named "chan" below -- "channel" is a reserved
// Nextflow binding (lowercase alias for the Channel class), see AGENTS.md.
process GLOBAL_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(agg_batch_stems), path(agg_files, stageAs: "agg_input_*.parquet"), \
          val(bl_batch_stems), path(blocklist_files, stageAs: "bl_input_*.parquet"), \
          val(publish_subdir), val(min_batches_ok), \
          val(run_pca), val(pca_n_components), val(run_umap), val(umap_n_components), val(umap_n_neighbors), \
          val(umap_metric), val(umap_min_dist), val(umap_random_state)

    output:
    // pca_components.parquet only exists when run_pca=true -- must be its
    // own output statement, not another element of the tuple below: see the
    // matching comment in finalize_feature_select.nf for why (per-element
    // `optional: true` inside a multi-element tuple output isn't honored on
    // this Nextflow version and would silently drop the whole tuple).
    tuple val(chan), path("aggregate.parquet"), path("blocklist.parquet")
    path("pca_components.parquet", optional: true)
    // aggregate_{feature_type}.parquet: one file per aggregate feature type
    // present in the cross-batch median aggregate
    // (globalfeatureselect.classify_features_by_type), always attempted --
    // no config toggle. Glob "aggregate_*.parquet" cannot collide with
    // "aggregate.parquet" above (no underscore follows "aggregate" there).
    // Own output statement for the same tuple/optional reason as
    // pca_components.parquet above. Marked optional as defense-in-depth for
    // the degenerate case where no column in the cross-batch median
    // aggregate matches a known aggregator suffix -- unreachable in normal
    // pipeline operation but possible for a direct/non-Nextflow caller.
    path("aggregate_*.parquet", optional: true)

    script:
    def aggStemsArg = "[" + agg_batch_stems.collect { s -> "'${s}'" }.join(',') + "]"
    def blStemsArg = "[" + bl_batch_stems.collect { s -> "'${s}'" }.join(',') + "]"
    def minArg = (min_batches_ok == null) ? "" : "min_batches_ok=${min_batches_ok}"
    """
    echo "Starting GLOBAL_FEATURE_SELECT for ${chan}"
    python -m fisseq_data_pipeline.globalfeatureselect \\
        output_dir=. \\
        "agg_batch_stems=${aggStemsArg}" \\
        n_agg_files=${agg_files.size()} \\
        "bl_batch_stems=${blStemsArg}" \\
        n_blocklist_files=${blocklist_files.size()} \\
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
