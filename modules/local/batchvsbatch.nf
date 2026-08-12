nextflow.enable.dsl = 2

// BATCHVSBATCH: wraps python -m fisseq_data_pipeline.batchvsbatch. Parameterized over which cells
// files, use_parent_name, publish subdirectory, and block_list_file to use,
// so workflows/fisseq.nf invokes this process twice via
// `include { BATCHVSBATCH as X }` aliasing: once pre-normalization
// (QC-filtered cells, unfiltered -- block_list_file=null) and once
// post-normalization (filtered against the ANOVA_BLOCKLIST output). Each
// alias is additionally invoked once per active global channel (see
// workflows/fisseq.nf), so publish_subdir carries the full path (including
// "global/<channel>/batchvsbatch/...") rather than just "pre"/"post".
// block_list_file is a val (not a staged path) so a Groovy null can be
// passed through directly for the unfiltered (pre) call -- see
// ovwt_batchwise.nf for the same convention. cells_files is a real `path`
// input (that channel's STAGE_CHANNEL_CELLS output, collected via
// workflows/fisseq.nf's perChannelSignal) rather than a directory-glob
// `val` -- see anova.nf's comment for why. pipeline_dir is NOT used in the
// script below (a pre-existing, unrelated leftover -- flagged for a
// separate follow-up cleanup, not removed here since it's out of this
// change's scope).
process BATCHVSBATCH {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(pipeline_dir), path(cells_files), val(use_parent_name), val(publish_subdir), val(block_list_file)

    output:
    path("results.parquet")

    script:
    """
    echo "Starting BATCHVSBATCH for ${publish_subdir}"
    python -m fisseq_data_pipeline.batchvsbatch \\
        output_dir=. \\
        "input_file=./*.parquet" \\
        use_parent_name=${use_parent_name} \\
        min_cells=${params.batchvsbatch_min_cells} \\
        min_batches=${params.batchvsbatch_min_batches} \\
        block_list_file=${block_list_file}
    """
}
