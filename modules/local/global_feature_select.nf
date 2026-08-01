nextflow.enable.dsl = 2

// GLOBAL_FEATURE_SELECT: wraps python -m fisseq_data_pipeline.globalfeatureselect.
// Runs once per active global group. Unlike the BATCHWISE feature-selection
// chain (FINALIZE_FEATURE_SELECT et al., which parallelize genuinely
// expensive cell-level bootstrap work), this process needs no Nextflow-level
// fan-out: it reads the group's member batches' already-published BATCHWISE
// aggregates/blocklists directly off pipeline_dir (same "glob published
// output" idiom ANOVA_NORMALIZED/OVWT_GLOBAL already use -- see AGENTS.md),
// looping over batch_stems in Python. No path() file inputs, so there is no
// same-named-file staging collision to design around.
process GLOBAL_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(group), val(batch_stems), val(pipeline_dir), val(publish_subdir), val(min_batches_ok)

    output:
    tuple val(group), path("aggregate.parquet"), path("blocklist.parquet")

    script:
    def stemsArg = "[" + batch_stems.join(',') + "]"
    def minArg = (min_batches_ok == null) ? "" : "min_batches_ok=${min_batches_ok}"
    """
    echo "Starting GLOBAL_FEATURE_SELECT for ${group}"
    python -m fisseq_data_pipeline.globalfeatureselect \\
        output_dir=. \\
        pipeline_dir=${pipeline_dir} \\
        "batch_stems=${stemsArg}" \\
        ${minArg}
    """
}
