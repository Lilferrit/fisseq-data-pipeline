"""Helpers for reading files a Nextflow process staged under a `stageAs` pattern.

A process that collects one same-named file per experiment (e.g. every
experiment's ``results.parquet``) must rename them on staging or they collide in
the task work directory. The convention here, matching
``modules/local/global_ovwt.nf``, is ``stageAs: "<prefix>_*.parquet"`` paired
with a ``val(batch_stems)`` list in the same order.
"""


def reconstruct_staged_paths(n: int, prefix: str) -> list[str]:
    """
    Reconstruct the filenames Nextflow produces for a ``stageAs`` glob pattern.

    Nextflow substitutes the ``*`` in ``stageAs: "<prefix>_*.parquet"`` with an
    **empty string** when exactly one file is staged, and only 1-indexes it from
    two files up. So a single-experiment channel yields ``<prefix>_.parquet``
    (no digit), not ``<prefix>_1.parquet`` -- a real edge case, since a global
    channel with one member experiment is perfectly legal.

    Parameters
    ----------
    n : int
        Number of staged files, i.e. the length of the accompanying
        ``batch_stems`` list.
    prefix : str
        The ``stageAs`` prefix, without the trailing ``_*`` or extension.

    Returns
    -------
    list[str]
        Staged filenames, in the order Nextflow emitted them -- which is the
        order ``batch_stems`` is in.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return [f"{prefix}_.parquet"]
    return [f"{prefix}_{i}.parquet" for i in range(1, n + 1)]
