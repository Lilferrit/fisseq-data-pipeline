nextflow.enable.dsl = 2

// ANOVA: wraps python -m fisseq_data_pipeline.anova. Parameterized over which cells glob and
// publish subdirectory to use, so workflows/fisseq.nf invokes this process
// twice via `include { ANOVA as X }` aliasing: once against normalized
// cells, once against batch-corrected cells. Runs once per active global
// channel (see workflows/fisseq.nf), scoped to only that channel's member
// batches -- the channel identifier ("chan"; "channel" itself is a reserved
// Nextflow binding -- see AGENTS.md) is carried through the output tuple so
// downstream consumers (ANOVA_BLOCKLIST) can join back to the right channel.
process ANOVA {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(cells_glob), val(publish_subdir)

    output:
    tuple val(chan), path("anova.parquet")

    script:
    """
    echo "Starting ANOVA for ${publish_subdir}"
    python -m fisseq_data_pipeline.anova \\
        output_dir=. \\
        "input_file=${cells_glob}"
    """
}
