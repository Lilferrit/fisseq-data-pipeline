nextflow.enable.dsl = 2

// INPUT: wraps python -m fisseq_data_pipeline.input. Runs once per YAML config file in
// params.yaml_config_dir (optional upstream stage), producing one input/-ready
// cell-level Parquet file from a variant-selection spec instead of a
// pre-staged raw batch file. Publishes into the same input/ directory
// pre-staged batches live in, so QC_FILTER treats both origins identically.
//
// Takes the resolved per-batch scalars (see lib/BatchParams.groovy and
// workflows/fisseq.nf's/ovwt.nf's resolvedBatchConfigs) as individual val()
// inputs rather than the raw hand-authored YAML file, so Nextflow hashes
// only these scalars for -resume caching -- a non-semantic edit to the
// original YAML, or a change to a key this pipeline doesn't consume, no
// longer busts this task's cache. The script rebuilds a minimal YAML from
// these scalars since python -m fisseq_data_pipeline.input's config_path
// still expects a file (input_paths is a list, so it can't cleanly become a
// single Hydra CLI override).
process INPUT {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/input" }, mode: 'copy'

    input:
    tuple val(name), val(input_paths), val(top_n_missense), val(convert_first), \
          val(temp_dir), val(feature_allowlist_file), val(feature_blocklist_file)

    output:
    tuple val(name), path("${name}.parquet")

    script:
    def inputPathsYaml = input_paths.collect { "'${it}'" }.join(', ')
    def topNMissenseYaml = (top_n_missense == null) ? 'null' : "${top_n_missense}"
    def tempDirYaml = (temp_dir == null) ? 'null' : "'${temp_dir}'"
    def allowlistYaml = (feature_allowlist_file == null) ? 'null' : "'${feature_allowlist_file}'"
    def blocklistYaml = (feature_blocklist_file == null) ? 'null' : "'${feature_blocklist_file}'"
    """
    echo "Starting INPUT for ${name}"
    cat > resolved_config.yaml <<'EOF'
input_paths: [${inputPathsYaml}]
top_n_missense: ${topNMissenseYaml}
convert_first: ${convert_first}
temp_dir: ${tempDirYaml}
feature_allowlist_file: ${allowlistYaml}
feature_blocklist_file: ${blocklistYaml}
EOF
    python -m fisseq_data_pipeline.input \\
        output_dir=. \\
        output_root=${name} \\
        config_path=resolved_config.yaml
    mv ${name}.output.parquet ${name}.parquet
    """
}
