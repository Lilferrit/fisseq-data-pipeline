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

    // Pipeline-wide defaults for every batch-overridable key -- see
    // workflows/fisseq.nf's identical block for the full rationale.
    // OvwtPipeline doesn't structurally consume every key here (e.g. the
    // feature-selection ones), but BatchParams.resolve() expects a complete
    // defaults map for the full OVERRIDABLE_KEYS set so a batch YAML
    // setting one of those keys is still validated/logged consistently
    // (it just has no process to wire into in this pipeline_mode).
    def batchParamDefaults = [
        barcode_count_threshold           : params.barcode_count_threshold,
        variant_barcode_count_threshold   : params.variant_barcode_count_threshold,
        edit_distance_threshold           : params.edit_distance_threshold,
        qc_downsample_fraction            : params.qc_downsample_fraction,
        qc_downsample_seed                : params.qc_downsample_seed,
        barcode_blocklist_pvalue_threshold: params.barcode_blocklist_pvalue_threshold,
        ovwt_min_cells                    : params.ovwt_min_cells,
        ovwt_downsample_wt                : params.ovwt_downsample_wt,
        max_cells_per_barcode_wt          : params.max_cells_per_barcode_wt,
        max_cells_per_barcode_variant     : params.max_cells_per_barcode_variant,
        feature_select_downsample_wt      : params.feature_select_downsample_wt,
        feature_select_min_correlation    : params.feature_select_min_correlation,
        barcode_check_min_cells           : params.barcode_check_min_cells,
        barcode_check_alpha               : params.barcode_check_alpha,
        single_cell_scores_split          : params.single_cell_scores_split,
        run_ovwt                          : params.run_ovwt.toString().toBoolean(),
        run_feature_filtered_ovwt         : params.run_feature_filtered_ovwt.toString().toBoolean(),
        run_single_cell_scores            : params.run_single_cell_scores.toString().toBoolean(),
        run_check_barcodes                : params.run_check_barcodes.toString().toBoolean(),
        run_barcode_filtered_ovwt         : params.run_barcode_filtered_ovwt.toString().toBoolean(),
        run_feature_selection             : params.run_feature_selection.toString().toBoolean(),
        top_n_missense                    : params.top_n_missense,
        convert_first                     : params.convert_first.toString().toBoolean(),
        temp_dir                          : params.temp_dir,
        feature_allowlist_file            : params.feature_allowlist_file,
        feature_blocklist_file            : params.feature_blocklist_file,
    ]
    if (!(batchParamDefaults.single_cell_scores_split in ["test", "train"])) {
        error "ERROR: --single_cell_scores_split must be 'test' or 'train', got '${batchParamDefaults.single_cell_scores_split}'"
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

    // Resolve every batch YAML's overrides once -- see workflows/fisseq.nf
    // and lib/BatchParams.groovy for the full rationale.
    def resolvedBatchConfigs = [:].withDefault { batchParamDefaults }
    config_files.each { f ->
        def stem = f.baseName
        def yamlMap = (new org.yaml.snakeyaml.Yaml().load(f.text) ?: [:]) as Map
        def resolution
        try {
            resolution = BatchParams.resolve(stem, batchParamDefaults, yamlMap)
        } catch (IllegalArgumentException | IllegalStateException e) {
            error "ERROR: ${e.message}"
        }
        resolution.overrides.each { o ->
            log.info "Batch '${o.batch}': overriding ${o.key} (default=${o.defaultValue}) -> ${o.overrideValue}"
        }
        if (!(resolution.resolved.single_cell_scores_split in ["test", "train"])) {
            error "ERROR: batch '${stem}': single_cell_scores_split must be 'test' or 'train', got '${resolution.resolved.single_cell_scores_split}'"
        }
        resolvedBatchConfigs[stem] = resolution.resolved
    }

    def inputSubdir = file("${params.input_dir}/input")
    if (!inputSubdir.isDirectory() && params.yaml_config_dir == null) {
        error "ERROR: ${params.input_dir}/input does not exist or is not a directory"
    }

    glob_input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [f.baseName, f] }
        .filter { name, f -> !(name in config_names) }

    if (params.yaml_config_dir != null) {
        config_ch = Channel.fromList(config_files).map { f ->
            def stem = f.baseName
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, cfg.input_paths, cfg.top_n_missense, cfg.convert_first,
                  cfg.temp_dir, cfg.feature_allowlist_file, cfg.feature_blocklist_file)
        }
        generated_ch = INPUT(config_ch)
        input_ch = glob_input_ch.mix(generated_ch)
    } else {
        input_ch = glob_input_ch
    }

    // Step 1: QC filter (per batch)
    qc_input_ch = input_ch.map { stem, f ->
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, f, cfg.barcode_count_threshold, cfg.variant_barcode_count_threshold,
              cfg.edit_distance_threshold, cfg.qc_downsample_fraction, cfg.qc_downsample_seed)
    }
    qc_ch = QC_FILTER(qc_input_ch).qc_outputs

    // Step 2: Batchwise OvWT — trains models and saves split index files.
    // OVWT_BATCHWISE is shared with FisseqPipeline, which parameterizes it
    // over feature_block_list_file/barcode_block_list_file/publish_subdir
    // (see modules/local/ovwt_batchwise.nf); OvwtPipeline doesn't run
    // ANOVA_BLOCKLIST or BARCODE_BLOCKLIST at all, so it always passes both
    // block-list vals as null (unfiltered) and preserves the original
    // ovwt_batchwise/ output path.
    ovwt_input_ch = qc_ch.map { stem, fc, _bc, _vpb ->
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, fc, null, null, "ovwt_batchwise", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
              cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
    }
    OVWT_BATCHWISE(ovwt_input_ch)

    // Step 3: Score each batch's resolved single_cell_scores_split's cells
    // via the saved index (auto-detected by load_input).
    cellscores_input_ch = OVWT_BATCHWISE.out
        .map { stem, _res, mdl, test_idx, train_idx ->
            def split = resolvedBatchConfigs[stem].single_cell_scores_split
            [stem, (split == "test") ? test_idx : train_idx, mdl]
        }
    OVWT_CELLSCORES_BATCHWISE(cellscores_input_ch)

    // Step 4: per-batch barcode-outlier check, per-batch gated on that
    // batch's resolved run_check_barcodes -- no "implies"/"requires" chain
    // here (unlike FisseqPipeline's batchGates()), since OVWT_CELLSCORES_BATCHWISE
    // always runs unconditionally in this pipeline.
    check_barcodes_input_ch = OVWT_CELLSCORES_BATCHWISE.out
        .filter { stem, scores -> resolvedBatchConfigs[stem].run_check_barcodes.toString().toBoolean() }
        .map { stem, scores ->
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, scores, cfg.barcode_check_min_cells, cfg.barcode_check_alpha)
        }
    CHECK_BARCODES(check_barcodes_input_ch)
}
