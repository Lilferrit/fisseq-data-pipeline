nextflow.enable.dsl = 2

process FEATURE_SELECT_GLOBAL {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/feature_select_global", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("global.parquet")
    path("feature_correlations.parquet")

    script:
    // TODO: global box plot visualization
    // TODO: global UMAP visualization
    """
    echo "Starting FEATURE_SELECT_GLOBAL for global"
    fisseq-feature-select \\
        output_dir=. \\
        output_root=global \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        aggregator=${params.aggregator} \\
        minimum_correlation=${params.minimum_correlation} \\
        compute_impact_score=true
    mv global.feature_correlations.parquet feature_correlations.parquet
    mv global.*.parquet global.parquet
    """
}
