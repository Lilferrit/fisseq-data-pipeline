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
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required.\n  Usage: nextflow run fisseq.nf -entry OvwtPipeline --pipeline_dir /path/to/data"
    }

    // Pipeline-wide defaults for every batch-overridable key -- see
    // workflows/fisseq.nf's identical block for the full rationale.
    // OvwtPipeline doesn't structurally consume every key here (e.g. the
    // feature-selection ones), but BatchParams.resolve() expects a complete
    // defaults map for the full OVERRIDABLE_KEYS set so a batch YAML
    // setting one of those keys is still validated/logged consistently
    // (it just has no process to wire into in this pipeline_mode). Likewise,
    // global_channel is validated/normalized by BatchParams.resolve()
    // independently of this map (same as input_paths) -- OvwtPipeline has no
    // global processes to gate, so it's accepted but never read here.
    def batchParamDefaults = [
        barcode_count_threshold           : params.barcode_count_threshold,
        variant_barcode_count_threshold   : params.variant_barcode_count_threshold,
        edit_distance_threshold           : params.edit_distance_threshold,
        qc_n_variants                     : params.qc_n_variants,
        qc_variant_downsample_classes     : params.qc_variant_downsample_classes,
        qc_variant_downsample_mode        : params.qc_variant_downsample_mode,
        qc_downsample_amounts             : params.qc_downsample_amounts,
        qc_downsample_classes             : params.qc_downsample_classes,
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
        run_single_cell_scores            : params.run_single_cell_scores.toString().toBoolean(),
        run_check_barcodes                : params.run_check_barcodes.toString().toBoolean(),
        run_barcode_filtered_ovwt         : params.run_barcode_filtered_ovwt.toString().toBoolean(),
        run_feature_selection             : params.run_feature_selection.toString().toBoolean(),
        feature_allowlist_file            : params.feature_allowlist_file,
        feature_blocklist_file            : params.feature_blocklist_file,
        csv_schema_scan_rows              : params.csv_schema_scan_rows,
    ]
    if (!(batchParamDefaults.single_cell_scores_split in ["test", "train"])) {
        error "ERROR: --single_cell_scores_split must be 'test' or 'train', got '${batchParamDefaults.single_cell_scores_split}'"
    }

    // Mandatory YAML configs -- see workflows/fisseq.nf for the full
    // rationale (identical block).
    def configsDir = file("${params.pipeline_dir}/configs")
    if (!configsDir.isDirectory()) {
        error "ERROR: ${params.pipeline_dir}/configs does not exist or is not a directory"
    }
    def config_files = configsDir.listFiles()?.findAll { f -> f.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // Resolve every batch YAML's overrides once -- see workflows/fisseq.nf
    // and lib/BatchParams.groovy for the full rationale.
    def resolvedBatchConfigs = [:]
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

    config_ch = channel.fromList(config_files).map { f ->
        def stem = f.baseName
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, cfg.input_paths, cfg.feature_allowlist_file, cfg.feature_blocklist_file,
              cfg.csv_schema_scan_rows)
    }
    input_ch = INPUT(config_ch)

    // Step 1: QC filter (per batch)
    qc_input_ch = input_ch.map { stem, f ->
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, f, cfg.barcode_count_threshold, cfg.variant_barcode_count_threshold,
              cfg.edit_distance_threshold, cfg.qc_n_variants, cfg.qc_variant_downsample_classes,
              cfg.qc_variant_downsample_mode, cfg.qc_downsample_amounts, cfg.qc_downsample_classes,
              cfg.qc_downsample_seed)
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
        .filter { stem, _scores -> resolvedBatchConfigs[stem].run_check_barcodes.toString().toBoolean() }
        .map { stem, scores ->
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, scores, cfg.barcode_check_min_cells, cfg.barcode_check_alpha)
        }
    CHECK_BARCODES(check_barcodes_input_ch)
}
