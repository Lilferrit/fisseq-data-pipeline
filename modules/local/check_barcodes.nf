nextflow.enable.dsl = 2

// CHECK_BARCODES: wraps fisseq-check-barcodes. Runs once per batch on that
// batch's single-cell scores (OVWT_CELLSCORES_BATCHWISE output), grouping by
// variant and running a pairwise Tukey HSD across each variant's barcodes
// (using each cell's score against its own trained one-vs-wildtype model as
// the response variable) to flag barcodes whose cells score anomalously
// relative to the variant's other barcodes. Variants with only one distinct
// barcode are skipped (nothing to compare). Gated by params.run_check_barcodes,
// which also forces params.single_cell_scores on -- see workflows/fisseq.nf.
process CHECK_BARCODES {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.input_dir}/check_barcodes/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(scores_file)

    output:
    tuple val(batch_stem), path("results.parquet")

    script:
    """
    echo "Starting CHECK_BARCODES for ${batch_stem}"
    fisseq-check-barcodes \\
        output_dir=. \\
        input_file=${scores_file} \\
        min_cells=${params.barcode_check_min_cells} \\
        alpha=${params.barcode_check_alpha}
    """
}
