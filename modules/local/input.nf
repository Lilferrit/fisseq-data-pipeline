nextflow.enable.dsl = 2

// INPUT: wraps python -m fisseq_data_pipeline.input. Runs once per mandatory
// YAML config file in <pipeline_dir>/configs/, producing one input/-ready
// cell-level Parquet file by loading and merging a batch's input_paths.
// Every batch must have a config file -- there is no pre-staged-parquet
// mode. Variant-class/count-based restriction (formerly done here) now
// happens in QC_FILTER itself (see modules/local/qc_filter.nf's
// qc_n_variants), so it applies uniformly to every batch.
//
// Takes the resolved per-batch scalars (see lib/BatchParams.groovy and
// workflows/fisseq.nf's/ovwt.nf's resolvedBatchConfigs) as individual val()
// inputs rather than the raw hand-authored YAML file, so Nextflow hashes
// only these scalars for -resume caching -- a non-semantic edit to the
// original YAML, or a change to a key this pipeline doesn't consume, no
// longer busts this task's cache. The script rebuilds a minimal YAML from
// these scalars (input_paths, feature_allowlist_file, feature_blocklist_file,
// csv_schema_scan_rows) since python -m fisseq_data_pipeline.input's
// config_path still expects a file (input_paths is a list, so it can't
// cleanly become a single Hydra CLI override).
process INPUT {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.pipeline_dir}/input" }, mode: 'copy'

    input:
    tuple val(name), val(input_paths), val(feature_allowlist_file), val(feature_blocklist_file), val(csv_schema_scan_rows)

    output:
    tuple val(name), path("${name}.parquet")

    script:
    def inputPathsYaml = input_paths.collect { p -> "'${p}'" }.join(', ')
    def allowlistYaml = (feature_allowlist_file == null) ? 'null' : "'${feature_allowlist_file}'"
    def blocklistYaml = (feature_blocklist_file == null) ? 'null' : "'${feature_blocklist_file}'"
    def schemaScanRowsYaml = (csv_schema_scan_rows == null) ? 'null' : "${csv_schema_scan_rows}"
    """
    echo "Starting INPUT for ${name}"
    cat > resolved_config.yaml <<'EOF'
input_paths: [${inputPathsYaml}]
feature_allowlist_file: ${allowlistYaml}
feature_blocklist_file: ${blocklistYaml}
csv_schema_scan_rows: ${schemaScanRowsYaml}
EOF
    python -m fisseq_data_pipeline.input \\
        output_dir=. \\
        output_root=${name} \\
        config_path=resolved_config.yaml
    mv ${name}.output.parquet ${name}.parquet
    """
}
