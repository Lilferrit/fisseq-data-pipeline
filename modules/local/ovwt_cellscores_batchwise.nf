nextflow.enable.dsl = 2

process OVWT_CELLSCORES_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/ovwt_cellscores_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), path(models_pkl)

    output:
    tuple val(batch_stem), path("cell_scores.parquet")

    script:
    """
    echo "Starting OVWT_CELLSCORES_BATCHWISE for ${batch_stem}"
    fisseq-ovwt-cell-scores \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        models_path=${models_pkl}
    """
}
