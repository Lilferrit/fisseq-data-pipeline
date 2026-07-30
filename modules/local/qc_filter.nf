nextflow.enable.dsl = 2

// QC_FILTER: wraps python -m fisseq_data_pipeline.qcfilter. Runs once per batch parquet in input/,
// applying edit-distance, barcode-count, and variant-barcode-count filters.
// When params.qc_downsample_fraction is set, filtered_cells.parquet is also
// augmented with reproducibly-downsampled pseudo-variant rows built only
// from cells that already passed those filters; barcode_counts.parquet and
// variants_per_barcode.parquet reflect pre-downsampling counts either way.
// Publishes filtered_cells.parquet, barcode_counts.parquet, and
// variants_per_barcode.parquet under qc_filter/<batch_stem>/.
process QC_FILTER {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/qc_filter/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(input_file), val(barcode_count_threshold), \
          val(variant_barcode_count_threshold), val(edit_distance_threshold), \
          val(qc_downsample_fraction), val(qc_downsample_seed)

    output:
    tuple val(batch_stem), \
          path("filtered_cells.parquet"), \
          path("barcode_counts.parquet"), \
          path("variants_per_barcode.parquet"), \
          emit: qc_outputs

    script:
    """
    echo "Starting QC_FILTER for ${batch_stem}"
    python -m fisseq_data_pipeline.qcfilter \\
        output_dir=. \\
        'cell_files=[${input_file}]' \\
        bc_threshold=${barcode_count_threshold} \\
        variant_bc_threshold=${variant_barcode_count_threshold} \\
        edit_distance_threshold=${edit_distance_threshold} \\
        downsample_fraction=${qc_downsample_fraction} \\
        downsample_seed=${qc_downsample_seed}
    """
}
