nextflow.enable.dsl = 2

process BATCH_CORRECT_TRANSFORM {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/batch_correction/cells", mode: 'copy'

    input:
    tuple val(batch_stem), path(filtered_cells), path(stats_vb), path(centroids)

    output:
    tuple val(batch_stem), path("${batch_stem}.parquet"), emit: corrected

    script:
    """
    echo "Starting BATCH_CORRECT_TRANSFORM for ${batch_stem}"
    fisseq-batch-correct-transform \\
        output_dir=. \\
        output_root=${batch_stem} \\
        input_file=${filtered_cells} \\
        batch=${batch_stem} \\
        stats_file=${stats_vb} \\
        centroids_file=${centroids} \\
        wt_label=WT
    mv ${batch_stem}.filtered_cells.parquet ${batch_stem}.parquet
    """
}
