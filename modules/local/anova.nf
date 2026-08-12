nextflow.enable.dsl = 2

// ANOVA: wraps python -m fisseq_data_pipeline.anova. Parameterized over which cells files and
// publish subdirectory to use, so workflows/fisseq.nf invokes this process
// twice via `include { ANOVA as X }` aliasing: once against normalized
// cells, once against batch-corrected cells. Runs once per active global
// channel (see workflows/fisseq.nf), scoped to only that channel's member
// batches -- the channel identifier ("chan"; "channel" itself is a reserved
// Nextflow binding -- see AGENTS.md) is carried through the output tuple so
// downstream consumers (ANOVA_BLOCKLIST) can join back to the right channel.
// cells_files is a real `path` input (that channel's STAGE_CHANNEL_CELLS/
// BATCH_CORRECT_TRANSFORM outputs, collected via workflows/fisseq.nf's
// perChannelSignal) rather than a directory-glob `val` -- this makes the
// actual file set part of Nextflow's task hash, so -resume correctly
// reruns when a file is added/removed/edited, not just when the glob string
// itself changes. python -m fisseq_data_pipeline.anova's input_file= still accepts a glob, so the
// script below globs the staged files locally to preserve identical CLI
// behavior.
process ANOVA {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), path(cells_files), val(publish_subdir)

    output:
    tuple val(chan), path("anova.parquet")

    script:
    """
    echo "Starting ANOVA for ${publish_subdir}"
    python -m fisseq_data_pipeline.anova \\
        output_dir=. \\
        "input_file=./*.parquet"
    """
}
