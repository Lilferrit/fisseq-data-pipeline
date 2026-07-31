nextflow.enable.dsl = 2

// QC_FILTER: wraps python -m fisseq_data_pipeline.qcfilter. Runs once per batch parquet in input/,
// applying edit-distance, barcode-count, and variant-barcode-count filters.
// Before those QC thresholds run, if params.qc_n_variants is set,
// qc_variant_downsample_classes is restricted to at most that many distinct
// variants (qc_variant_downsample_mode: "top" by cell count, or "random").
// After QC thresholding, if params.qc_downsample_amounts is set,
// filtered_cells.parquet is also augmented with reproducibly-downsampled
// pseudo-variant rows (one distinctly-tagged group per amount) built only
// from cells that already passed those filters; barcode_counts.parquet and
// variants_per_barcode.parquet reflect the post-variant-selection,
// pre-pseudo-variant population either way.
// Publishes filtered_cells.parquet, barcode_counts.parquet, and
// variants_per_barcode.parquet under qc_filter/<batch_stem>/.
process QC_FILTER {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/qc_filter/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(input_file), val(barcode_count_threshold), \
          val(variant_barcode_count_threshold), val(edit_distance_threshold), \
          val(qc_n_variants), val(qc_variant_downsample_classes), val(qc_variant_downsample_mode), \
          val(qc_downsample_amounts), val(qc_downsample_classes), val(qc_downsample_seed)

    output:
    tuple val(batch_stem), \
          path("filtered_cells.parquet"), \
          path("barcode_counts.parquet"), \
          path("variants_per_barcode.parquet"), \
          emit: qc_outputs

    script:
    // qc_downsample_amounts: null -> Hydra null-passthrough; a bare scalar
    // (CLI or nextflow.config) interpolates unquoted like the other numeric
    // thresholds; a genuine Groovy List (nextflow.config literal or batch
    // YAML override) becomes a single-quoted Hydra bracket-list override,
    // extending the 'cell_files=[...]' pattern below so the shell doesn't
    // glob-expand it.
    def amountsArg = (qc_downsample_amounts == null)
        ? 'null'
        : (qc_downsample_amounts instanceof List)
            ? "'[${qc_downsample_amounts.join(',')}]'"
            : "${qc_downsample_amounts}"
    // qc_downsample_classes/qc_variant_downsample_classes are always
    // non-empty List<String>; each element needs its own quoting (values
    // like "Single Missense" contain a space) inside the bracket-list.
    def downsampleClassesArg = "'[${qc_downsample_classes.collect { "\"${it}\"" }.join(',')}]'"
    def variantClassesArg = "'[${qc_variant_downsample_classes.collect { "\"${it}\"" }.join(',')}]'"
    """
    echo "Starting QC_FILTER for ${batch_stem}"
    python -m fisseq_data_pipeline.qcfilter \\
        output_dir=. \\
        'cell_files=[${input_file}]' \\
        bc_threshold=${barcode_count_threshold} \\
        variant_bc_threshold=${variant_barcode_count_threshold} \\
        edit_distance_threshold=${edit_distance_threshold} \\
        n_variants=${qc_n_variants} \\
        variant_downsample_classes=${variantClassesArg} \\
        variant_downsample_mode=${qc_variant_downsample_mode} \\
        downsample_amounts=${amountsArg} \\
        downsample_classes=${downsampleClassesArg} \\
        downsample_seed=${qc_downsample_seed}
    """
}
