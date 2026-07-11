nextflow.enable.dsl = 2

include { QC_FILTER                 } from '../modules/local/qc_filter'
include { NORMALIZE                 } from '../modules/local/normalize'
include { BATCHVSBATCH_PRE          } from '../modules/local/batchvsbatch_pre'
include { BATCHVSBATCH_POST         } from '../modules/local/batchvsbatch_post'
include { OVWT_BATCHWISE            } from '../modules/local/ovwt_batchwise'
include { OVWT_GLOBAL               } from '../modules/local/ovwt_global'
include { FEATURE_SELECT_BATCHWISE  } from '../modules/local/feature_select_batchwise'
include { FEATURE_SELECT_GLOBAL     } from '../modules/local/feature_select_global'
include { PERMANOVA                 } from '../modules/local/permanova'
include { BATCH_CORRECT_FIT         } from '../modules/local/batch_correct_fit'
include { BATCH_CORRECT_TRANSFORM   } from '../modules/local/batch_correct_transform'
include { PERMANOVA_BATCH_CORRECTED } from '../modules/local/permanova_batch_corrected'

workflow FisseqPipeline {
    // Validate required parameters (must be inside workflow in DSL2)
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq.nf --input_dir /path/to/data"
    }
    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory()) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }
    def inputParquets = inputSubdir.listFiles()?.findAll { it.name.endsWith('.parquet') } ?: []
    if (inputParquets.size() == 0) {
        error "ERROR: No .parquet files found in ${params.input_dir}/input"
    }

    // Resolve input_dir to absolute path so global process scripts can glob published outputs.
    // Relative paths (e.g. ".") break inside Nextflow work directories.
    def input_dir_abs = file(params.input_dir).toAbsolutePath().toString()

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

    // Single-element signal that fires once all QC_FILTER batches are done.
    // .map preserves the "wait for all batches" dependency while emitting just the path.
    qc_signal = qc_ch.map { stem, fc, bc, vpb -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Single-element signal that fires once all NORMALIZE batches are done.
    global_signal = norm_ch.map { stem, p -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Step 3: Batch-vs-batch — pre batch correction (QC-filtered cells, before normalization)
    BATCHVSBATCH_PRE(qc_signal)

    // Step 4: Batch-vs-batch — post batch correction (normalized cells)
    BATCHVSBATCH_POST(global_signal)

    // Step 5: OvWT — batchwise
    OVWT_BATCHWISE(norm_ch)

    // Step 6: OvWT — global
    OVWT_GLOBAL(global_signal)

    // Step 7: Feature selection — batchwise
    FEATURE_SELECT_BATCHWISE(norm_ch)

    // Step 8: Feature selection — global
    FEATURE_SELECT_GLOBAL(global_signal)

    // Step 9: PERMANOVA — batch-effect assessment (normalized cells)
    PERMANOVA(global_signal)

    // New branch: qc_filtering -> batch_correction -> permanova (independent of normalize)
    // Step 1: fit centroid batch correction across all batches (global, waits for all QC_FILTER)
    fit_out = BATCH_CORRECT_FIT(qc_signal).fit_outputs  // tuple(stats_vb, centroids), single emission

    // Step 2: apply batch correction (per batch); .combine() broadcasts the single
    // fit_out pair onto every per-batch tuple from qc_ch.
    bc_transform_input_ch = qc_ch
        .map { stem, fc, bc, vpb -> [ stem, fc ] }
        .combine(fit_out)

    BATCH_CORRECT_TRANSFORM(bc_transform_input_ch)
    bc_ch = BATCH_CORRECT_TRANSFORM.out.corrected  // tuple(batch_stem, corrected_parquet)

    // Single-element signal that fires once all BATCH_CORRECT_TRANSFORM batches are done.
    bc_signal = bc_ch.map { stem, p -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Step 3: PERMANOVA on batch-corrected cells
    PERMANOVA_BATCH_CORRECTED(bc_signal)
}
