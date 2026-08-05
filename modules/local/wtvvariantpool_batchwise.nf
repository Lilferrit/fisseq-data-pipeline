nextflow.enable.dsl = 2

// WTVVARIANTPOOL_BATCHWISE: wraps python -m fisseq_data_pipeline.wtvvariantpool. Runs once per
// batch on that batch's normalized cells. Builds a pool of non-wildtype cells
// whose meta_aa_changes (via classify_variant) falls in
// wtvvariantpool_variant_classes (default ["Synonymous"]), then trains one
// binary XGBoost classifier per wildtype barcode (after
// wtvvariantpool_min_cells_per_barcode filtering) distinguishing that
// barcode's cells from the pooled variant cells. The pool is optionally
// downsampled via wtvvariantpool_downsample_variant_pool (null/false =
// disabled, true = match the largest surviving wildtype barcode's cell
// count, an int = exact target count). Gated per batch by
// params.run_wtvvariantpool, which defaults to FALSE (unlike run_wtvwt's
// default true) -- see workflows/fisseq.nf's batchGates(). Publishes
// results.parquet and models.pkl under wtvvariantpool_batchwise/<batch_stem>/.
process WTVVARIANTPOOL_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/wtvvariantpool_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), val(wtvvariantpool_min_cells_per_barcode), \
          val(wtvvariantpool_variant_classes), val(wtvvariantpool_downsample_variant_pool)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl")

    script:
    // wtvvariantpool_variant_classes is always a non-empty List<String> --
    // rendered the same way qc_downsample_classes/qc_variant_downsample_classes
    // are in qc_filter.nf.
    def variantClassesArg =
        "'[${wtvvariantpool_variant_classes.collect { c -> "\"${c}\"" }.join(',')}]'"
    """
    echo "Starting WTVVARIANTPOOL_BATCHWISE for ${batch_stem}"
    python -m fisseq_data_pipeline.wtvvariantpool \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells_per_barcode=${wtvvariantpool_min_cells_per_barcode} \\
        variant_classes=${variantClassesArg} \\
        downsample_variant_pool=${wtvvariantpool_downsample_variant_pool}
    """
}
