nextflow.enable.dsl = 2

// output_dir=./out_select avoids the staged input (${batch_stem}.parquet) being
// overwritten by the identically-named tool output when Nextflow uses symlink staging.
process FEATURE_SELECT_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet)

    output:
    tuple val(batch_stem), \
          path("${batch_stem}.parquet"), \
          path("feature_correlations.parquet")

    script:
    // TODO: box plot visualization
    // TODO: replicate-replicate correlation plot
    // TODO: UMAP visualization
    // TODO: overlapping variants plot
    """
    echo "Starting FEATURE_SELECT_BATCHWISE for ${batch_stem}"
    mkdir -p out_select
    fisseq-feature-select \\
        output_dir=./out_select \\
        input_file=${normalized_parquet} \\
        aggregator=${params.aggregator} \\
        minimum_correlation=${params.minimum_correlation} \\
        compute_impact_score=true
    mv out_select/${batch_stem}.parquet ./
    mv out_select/feature_correlations.parquet ./
    """
}
