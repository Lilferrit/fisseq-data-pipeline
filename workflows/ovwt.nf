nextflow.enable.dsl = 2

// OvwtPipeline: lighter alternative to FisseqPipeline, selected via
// `--pipeline_mode ovwt`. Wires QC_FILTER -> OVWT_BATCHWISE ->
// OVWT_CELLSCORES_BATCHWISE (scoring the params.single_cell_scores_split
// split; always runs here, unlike FisseqPipeline where it's optional) ->
// CHECK_BARCODES (optional, gated by params.run_check_barcodes) — no
// normalization, batch correction, or feature selection.
include { INPUT                     } from '../modules/local/input'
include { QC_FILTER                 } from '../modules/local/qc_filter'
include { OVWT_BATCHWISE            } from '../modules/local/ovwt_batchwise'
include { OVWT_CELLSCORES_BATCHWISE } from '../modules/local/ovwt_cellscores_batchwise'
include { CHECK_BARCODES            } from '../modules/local/check_barcodes'

workflow OvwtPipeline {
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq.nf -entry OvwtPipeline --input_dir /path/to/data"
    }

    // See workflows/fisseq.nf for the full rationale behind this block
    // (relaxed validation + dedup against config-derived files).
    def config_files = []
    if (params.yaml_config_dir != null) {
        def configSubdir = file(params.yaml_config_dir)
        if (!configSubdir.isDirectory()) {
            error "ERROR: ${params.yaml_config_dir} does not exist or is not a directory"
        }
        config_files = configSubdir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
        if (config_files.size() == 0) {
            error "ERROR: No .yaml files found in ${params.yaml_config_dir}"
        }
    }
    def config_names = config_files.collect { it.baseName } as Set

    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory() && params.yaml_config_dir == null) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }

    glob_input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [f.baseName, f] }
        .filter { name, f -> !(name in config_names) }

    if (params.yaml_config_dir != null) {
        config_ch = Channel.fromList(config_files).map { f -> [f.baseName, f] }
        generated_ch = INPUT(config_ch)
        input_ch = glob_input_ch.mix(generated_ch)
    } else {
        input_ch = glob_input_ch
    }

    // Step 1: QC filter (per batch)
    qc_ch = QC_FILTER(input_ch).qc_outputs

    // Step 2: Batchwise OvWT — trains models and saves split index files.
    // OVWT_BATCHWISE is shared with FisseqPipeline, which parameterizes it
    // over block_list_file/publish_subdir (see modules/local/ovwt_batchwise.nf);
    // OvwtPipeline doesn't run ANOVA at all, so it always passes block_list_file=null
    // (unfiltered) and preserves the original ovwt_batchwise/ output path.
    ovwt_input_ch = qc_ch.map { stem, fc, _bc, _vpb -> tuple(stem, fc, null, "ovwt_batchwise") }
    OVWT_BATCHWISE(ovwt_input_ch)

    // Step 3: Score the params.single_cell_scores_split split's cells via
    // the saved index (auto-detected by load_input).
    score_source = params.single_cell_scores_split
    if (!(score_source in ["test", "train"])) {
        error "ERROR: --single_cell_scores_split must be 'test' or 'train', got '${score_source}'"
    }
    cellscores_input_ch = OVWT_BATCHWISE.out
        .map { stem, _res, mdl, test_idx, train_idx ->
            [stem, (score_source == "test") ? test_idx : train_idx, mdl]
        }
    OVWT_CELLSCORES_BATCHWISE(cellscores_input_ch)

    // Step 4: per-batch barcode-outlier check (optional, gated on
    // params.run_check_barcodes).
    run_check_barcodes = params.run_check_barcodes.toString().toBoolean()
    if (run_check_barcodes) {
        CHECK_BARCODES(OVWT_CELLSCORES_BATCHWISE.out)
    }
}
