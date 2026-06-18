nextflow.enable.dsl = 2

include { QC_FILTER                 } from '../modules/local/qc_filter'
include { OVWT_BATCHWISE            } from '../modules/local/ovwt_batchwise'
include { OVWT_CELLSCORES_BATCHWISE } from '../modules/local/ovwt_cellscores_batchwise'

workflow OvwtPipeline {
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq.nf -entry OvwtPipeline --input_dir /path/to/data"
    }
    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory()) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }

    input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [f.baseName, f] }

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
