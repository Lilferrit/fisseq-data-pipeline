nextflow.enable.dsl = 2

// BARCODE_BLOCKLIST: wraps python -m fisseq_data_pipeline.barcodeblocklist. Runs once per batch on
// that batch's CHECK_BARCODES results.parquet, aggregating each barcode's
// p_adj values (pooled across both the `barcode` and `comparison_barcode`
// columns) via median, and marking barcode_ok = median_p_adj >=
// params.barcode_blocklist_pvalue_threshold. Unlike ANOVA_BLOCKLIST (global,
// single-shot), this is per-batch, matching CHECK_BARCODES' cadence. Gated
// implicitly by params.run_barcode_filtered_ovwt (which forces
// run_check_barcodes on) -- see workflows/fisseq.nf.
process BARCODE_BLOCKLIST {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/barcode_blocklist/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(results_file), val(barcode_blocklist_pvalue_threshold)

    output:
    tuple val(batch_stem), path("barcode_blocklist.parquet")

    script:
    """
    echo "Starting BARCODE_BLOCKLIST for ${batch_stem}"
    python -m fisseq_data_pipeline.barcodeblocklist \\
        output_dir=. \\
        check_barcodes_file=${results_file} \\
        pvalue_threshold=${barcode_blocklist_pvalue_threshold}
    """
}
