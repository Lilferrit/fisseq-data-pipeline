nextflow.enable.dsl = 2

// ANOVA_BLOCKLIST: wraps python -m fisseq_data_pipeline.anovablocklist. Consumes ANOVA_NORMALIZED's
// anova.parquet and marks each feature ok/blocked based on whether its
// ANOVA p-value indicates a statistically significant batch effect
// (p_value < params.anova_blocklist_pvalue_threshold => blocked). Runs once
// per active global channel (see workflows/fisseq.nf), scoped to only that
// channel's ANOVA_NORMALIZED output. Publishes anova_blocklist.parquet under
// publish_subdir (e.g. "global/<channel>/anova_blocklist"). Note: the
// channel identifier is named "chan" below, not "channel" -- "channel" is a
// reserved Nextflow binding (lowercase alias for the Channel class) and
// silently resolves wrong if reused as a variable name -- see AGENTS.md.
process ANOVA_BLOCKLIST {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), path(anova_file), val(publish_subdir)

    output:
    tuple val(chan), path("anova_blocklist.parquet")

    script:
    """
    echo "Starting ANOVA_BLOCKLIST for ${publish_subdir}"
    python -m fisseq_data_pipeline.anovablocklist \\
        output_dir=. \\
        anova_file=${anova_file} \\
        pvalue_threshold=${params.anova_blocklist_pvalue_threshold}
    """
}
