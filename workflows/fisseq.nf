nextflow.enable.dsl = 2

include { QC_FILTER                } from '../modules/local/qc_filter'
include { NORMALIZE                } from '../modules/local/normalize'
include { PERMANOVA_WT             } from '../modules/local/permanova_wt'
include { PERMANOVA_SYN            } from '../modules/local/permanova_syn'
include { OVWT_BATCHWISE           } from '../modules/local/ovwt_batchwise'
include { OVWT_GLOBAL              } from '../modules/local/ovwt_global'
include { FEATURE_SELECT_BATCHWISE } from '../modules/local/feature_select_batchwise'
include { FEATURE_SELECT_GLOBAL    } from '../modules/local/feature_select_global'

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
