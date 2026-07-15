nextflow.enable.dsl = 2

// QC_FILTER: wraps fisseq-qc-filter. Runs once per batch parquet in input/,
// applying edit-distance, barcode-count, and variant-barcode-count filters.
// Publishes filtered_cells.parquet, barcode_counts.parquet, and
// variants_per_barcode.parquet under qc_filter/<batch_stem>/.
process QC_FILTER {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/qc_filter/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(input_file)

    output:
    tuple val(batch_stem), \
          path("filtered_cells.parquet"), \
          path("barcode_counts.parquet"), \
          path("variants_per_barcode.parquet"), \
          emit: qc_outputs

    script:
    """
    echo "Starting QC_FILTER for ${batch_stem}"
    fisseq-qc-filter \\
        output_dir=. \\
        'cell_files=[${input_file}]' \\
        bc_threshold=${params.bc_threshold} \\
        variant_bc_threshold=${params.variant_bc_threshold} \\
        edit_distance_threshold=${params.edit_distance_threshold}
    """
}
