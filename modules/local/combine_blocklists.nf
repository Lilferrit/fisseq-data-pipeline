nextflow.enable.dsl = 2

// COMBINE_BLOCKLISTS: wraps fisseq-combine-blocklists. Feature-selection
// stage 3 — concatenates every feature type's BLOCKLIST output (for one
// batch or global) into a single combined blocklist, consumed by
// FINALIZE_FEATURE_SELECT.
process COMBINE_BLOCKLISTS {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), path(blocklist_files), val(publish_subdir)

    output:
    tuple val(batch_key), path("blocklist.parquet")

    script:
    """
    echo "Starting COMBINE_BLOCKLISTS for ${batch_key}"
    fisseq-combine-blocklists \\
        output_dir=. \\
        "blocklist_files=*.parquet"
    """
}
