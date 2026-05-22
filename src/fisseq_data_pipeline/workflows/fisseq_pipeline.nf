#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * FISSEQ Data Pipeline — Nextflow DSL2
 *
 * DAG:
 *
 *   input/*.parquet
 *        │
 *        ▼
 *   QC_FILTER  (per batch) ──────────────────────────────────────────────┐
 *        │                                                                │ barcode_counts
 *        ▼                                                                │
 *   NORMALIZE  (per batch)                                                │
 *        │                                                                │
 *        ├──► PERMANOVA_WT            (global, waits for all NORMALIZE)  │
 *        ├──► PERMANOVA_SYN           (global, waits for all NORMALIZE)  │
 *        ├──► OVWT_BATCHWISE          (per batch)                        │
 *        ├──► OVWT_GLOBAL             (global, waits for all NORMALIZE)  │
 *        ├──► FEATURE_SELECT_BATCHWISE (per batch) ◄─────────────────────┘
 *        └──► FEATURE_SELECT_GLOBAL   (global, waits for all NORMALIZE)
 *
 * Output layout:
 *   {input_dir}/
 *     qc_filter/{batch_stem}/          filtered_cells, barcode_counts, summary TSV
 *     normalization/cells/             {batch_stem}.parquet
 *     normalization/normalizers/       {batch_stem}.normalizer.parquet
 *     permanova/wildtype/              permanova.parquet
 *     permanova/synonymous/            permanova.parquet
 *     ovwt_batchwise/{batch_stem}/     results.csv (enriched), models.pkl
 *     ovwt_global/                     results.csv, models.pkl
 *     feature_select_batchwise/{batch_stem}/  {batch_stem}.parquet, feature_correlations, enriched
 *     feature_select_global/           global.parquet, feature_correlations, redundancy-filtered
 */

// ── Parameters ───────────────────────────────────────────────────────────────

params.input_dir               = null
params.bc_threshold            = 10
params.variant_bc_threshold    = 4
params.edit_distance_threshold = 1
params.minimum_correlation     = 0.5
params.permanova_n_bootstraps  = 200
params.permanova_sample_size   = 1000
params.ovwt_min_cells          = 250
params.downsample_wt           = 5000
params.aggregator              = "multi"

// ── Processes ────────────────────────────────────────────────────────────────

process QC_FILTER {
    errorStrategy 'terminate'
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

// NORMALIZE: output_root namespaces outputs so all batches can share normalization/.
// cells go to normalization/cells/ and normalizers go to normalization/normalizers/
// so that the permanova/ovwt glob "normalization/cells/*.parquet" only hits cell data.
process NORMALIZE {
    errorStrategy 'terminate'
    publishDir "${params.input_dir}/normalization", mode: 'copy', saveAs: { fname ->
        fname.endsWith('.normalizer.parquet') ? "normalizers/${fname}" : "cells/${fname}"
    }

    input:
    tuple val(batch_stem), path(filtered_cells)

    output:
    tuple val(batch_stem), path("${batch_stem}.parquet"), emit: normalized
    path("${batch_stem}.normalizer.parquet"),              emit: normalizer

    script:
    """
    echo "Starting NORMALIZE for ${batch_stem}"
    fisseq-normalize \\
        output_dir=. \\
        output_root=${batch_stem} \\
        input_file=${filtered_cells} \\
        save_normalizer=true
    mv ${batch_stem}.filtered_cells.parquet ${batch_stem}.parquet
    """
}

process PERMANOVA_WT {
    errorStrategy 'terminate'
    publishDir "${params.input_dir}/permanova/wildtype", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA_WT for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        variant_class_filter=WT \\
        n_bootstraps=${params.permanova_n_bootstraps} \\
        sample_size=${params.permanova_sample_size}
    """
}

process PERMANOVA_SYN {
    errorStrategy 'terminate'
    publishDir "${params.input_dir}/permanova/synonymous", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA_SYN for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        variant_class_filter=Synonymous \\
        n_bootstraps=${params.permanova_n_bootstraps} \\
        sample_size=${params.permanova_sample_size}
    """
}

process OVWT_BATCHWISE {
    errorStrategy 'terminate'
    publishDir { "${params.input_dir}/ovwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl")

    script:
    // TODO: add per-batch OvWT visualization
    """
    echo "Starting OVWT_BATCHWISE for ${batch_stem}"
    fisseq-ovwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.downsample_wt}
    """
}

process OVWT_GLOBAL {
    errorStrategy 'terminate'
    publishDir "${params.input_dir}/ovwt_global", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("results.parquet")
    path("models.pkl")

    script:
    // TODO: add global OvWT visualization
    """
    echo "Starting OVWT_GLOBAL for global"
    fisseq-ovwt \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.downsample_wt}
    """
}

// output_dir=./out_select avoids the staged input (${batch_stem}.parquet) being
// overwritten by the identically-named tool output when Nextflow uses symlink staging.
process FEATURE_SELECT_BATCHWISE {
    errorStrategy 'terminate'
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

process FEATURE_SELECT_GLOBAL {
    errorStrategy 'terminate'
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

// ── Workflow ─────────────────────────────────────────────────────────────────

workflow {
    // Validate required parameters (must be inside workflow in DSL2)
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq_pipeline.nf --input_dir /path/to/data"
    }
    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory()) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }
    def inputParquets = inputSubdir.listFiles()?.findAll { it.name.endsWith('.parquet') } ?: []
    if (inputParquets.size() == 0) {
        error "ERROR: No .parquet files found in ${params.input_dir}/input"
    }

    // Source channel: one tuple per batch parquet in input/
    input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [ f.baseName, f ] }

    // Step 1: QC filter (per batch)
    qc_ch = QC_FILTER(input_ch).qc_outputs

    // Step 2: Normalization (per batch)
    // qc_ch carries: (batch_stem, filtered_cells, barcode_counts, variants_per_barcode)
    norm_input_ch = qc_ch.map { stem, fc, bc, vpb -> [ stem, fc ] }
    NORMALIZE(norm_input_ch)
    norm_ch = NORMALIZE.out.normalized  // tuple(batch_stem, normalized_parquet)

    // Collect all batch stems as a single-element signal for global steps
    all_stems_signal = norm_ch.map { stem, p -> stem }.collect()

    // Resolve input_dir to absolute path so global process scripts can glob published outputs.
    // Relative paths (e.g. ".") break inside Nextflow work directories.
    // .map here preserves the "wait for all batches" dependency while emitting just the path.
    def input_dir_abs = file(params.input_dir).toAbsolutePath().toString()
    global_signal = all_stems_signal.map { _stems -> input_dir_abs }

    // Step 3: PERMANOVA — global, two variant-class sub-runs
    PERMANOVA_WT(global_signal)
    PERMANOVA_SYN(global_signal)

    // Step 4: OvWT — batchwise
    OVWT_BATCHWISE(norm_ch)

    // Step 5: OvWT — global
    OVWT_GLOBAL(global_signal)

    // Step 6: Feature selection — batchwise
    FEATURE_SELECT_BATCHWISE(norm_ch)

    // Step 7: Feature selection — global
    FEATURE_SELECT_GLOBAL(global_signal)
}
