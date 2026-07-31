nextflow.enable.dsl = 2

// ANOVA_BLOCKLIST: wraps python -m fisseq_data_pipeline.anovablocklist. Consumes ANOVA_NORMALIZED's
// anova.parquet and marks each feature ok/blocked based on whether its
// ANOVA p-value indicates a statistically significant batch effect
// (p_value < params.anova_blocklist_pvalue_threshold => blocked). Always
// runs -- not gated by any global group or params.run_feature_selection --
// since OVWT_BATCHWISE_FEATURE_FILTERED needs it unconditionally. Publishes
// anova_blocklist.parquet under anova_blocklist/.
process ANOVA_BLOCKLIST {
    errorStrategy 'ignore'
    publishDir "${params.pipeline_dir}/anova_blocklist", mode: 'copy'

    input:
    path(anova_file)

    output:
    path("anova_blocklist.parquet")

    script:
    """
    echo "Starting ANOVA_BLOCKLIST"
    python -m fisseq_data_pipeline.anovablocklist \\
        output_dir=. \\
        anova_file=${anova_file} \\
        pvalue_threshold=${params.anova_blocklist_pvalue_threshold}
    """
}
