nextflow.enable.dsl = 2

// OvwtPipeline: lighter alternative to FisseqPipeline, selected via
// `--workflow ovwt`. Wires QC_FILTER -> OVWT_BATCHWISE ->
// OVWT_CELLSCORES_BATCHWISE only — no normalization, batch correction, or
// feature selection.
include { INPUT                     } from '../modules/local/input'
include { QC_FILTER                 } from '../modules/local/qc_filter'
include { OVWT_BATCHWISE            } from '../modules/local/ovwt_batchwise'
include { OVWT_CELLSCORES_BATCHWISE } from '../modules/local/ovwt_cellscores_batchwise'

workflow OvwtPipeline {
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq.nf -entry OvwtPipeline --input_dir /path/to/data"
    }

    // See workflows/fisseq.nf for the full rationale behind this block
    // (relaxed validation + dedup against config-derived files).
    def config_files = []
    if (params.config_dir != null) {
        def configSubdir = file(params.config_dir)
        if (!configSubdir.isDirectory()) {
            error "ERROR: ${params.config_dir} does not exist or is not a directory"
        }
        config_files = configSubdir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
        if (config_files.size() == 0) {
            error "ERROR: No .yaml files found in ${params.config_dir}"
        }
    }
    def config_names = config_files.collect { it.baseName } as Set

    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory() && params.config_dir == null) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }

    glob_input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [f.baseName, f] }
        .filter { name, f -> !(name in config_names) }

    if (params.config_dir != null) {
        config_ch = Channel.fromList(config_files).map { f -> [f.baseName, f] }
        generated_ch = INPUT(config_ch)
        input_ch = glob_input_ch.mix(generated_ch)
    } else {
        input_ch = glob_input_ch
    }

    // Step 1: QC filter (per batch)
    qc_ch = QC_FILTER(input_ch).qc_outputs

    // Step 2: Batchwise OvWT — trains models and saves split index files
    ovwt_input_ch = qc_ch.map { stem, fc, _bc, _vpb -> [stem, fc] }
    OVWT_BATCHWISE(ovwt_input_ch)

    // Step 3: Score test-set cells via the saved index (auto-detected by load_input)
    cellscores_input_ch = OVWT_BATCHWISE.out
        .map { stem, _res, mdl, test_idx -> [stem, test_idx, mdl] }
    OVWT_CELLSCORES_BATCHWISE(cellscores_input_ch)
}
