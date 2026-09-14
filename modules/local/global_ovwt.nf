nextflow.enable.dsl = 2

// GLOBAL_OVWT: wraps python -m fisseq_data_pipeline.globalovwt. Runs once per
// active global channel, over that channel's member experiments only.
//
// Two steps, not one: per experiment, z-score auroc_pooled/auroc_median_barcode
// against that experiment's own synonymous variants, THEN take the
// cross-experiment median of the z-scored values -- not a direct median of raw
// AUROC. Raw AUROC is not comparable across experiments (different cell counts
// and batch effects shift where a genuinely-neutral variant's score sits), so
// each experiment is re-centered against its own synonymous population before
// pooling.
//
// This replaces OVWT_GLOBAL, which fit a single classifier on pooled cells from
// every batch. Aggregating per-experiment scores keeps each experiment's own
// wildtype/synonymous baseline intact instead of blending them.
//
// stageAs: "res_input_*.parquet" avoids every experiment's identically-named
// results.parquet colliding when collected into this one task. Nextflow
// substitutes the `*` with an empty string when exactly one file is staged
// (res_input_.parquet) and 1-indexes only from two files up --
// fisseq_data_pipeline.utils.nextflow_staging.reconstruct_staged_paths encodes
// that rule, and batch_stems arrives in the same order groupTuple emitted the
// paths in.
//
// The channel identifier is named "chan" below -- "channel" is a reserved
// Nextflow binding (lowercase alias for the Channel class), see AGENTS.md.
process GLOBAL_OVWT {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/${chan}/ovwt_distinguishability" }, mode: 'copy'

    input:
    tuple val(chan), path(results_parquets, stageAs: "res_input_*.parquet"), val(batch_stems)

    output:
    tuple val(chan), path("global_scores.parquet"), emit: global_scores

    when:
    task.ext.when == null || task.ext.when

    script:
    def stemsArg = "[" + batch_stems.join(',') + "]"
    """
    echo "Starting GLOBAL_OVWT for ${chan}"
    python -m fisseq_data_pipeline.globalovwt \\
        output_dir=. \\
        "batch_stems=${stemsArg}" \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
