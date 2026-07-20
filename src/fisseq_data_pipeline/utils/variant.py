"""Variant label string classification into biological categories.

Defines :func:`classify_variant`, which parses a variant label (e.g. ``"A123G"``)
into one of ``Frameshift``, ``3nt Deletion``, ``Nonsense``, ``WT``, ``Synonymous``,
``Single Missense``, or ``Other``, and :func:`strip_variant_tag`, which removes an
optional ``:<tag>`` metadata suffix (e.g. the ``:downsampled-half`` pseudo-variant
tag produced by ``input.py``) from a variant label.
"""

import re


def classify_variant(v: str) -> str:
    """
    Classify a variant label string into a biological category.

    Assumes ``v`` has already been tag-stripped (see :func:`strip_variant_tag`)
    — a raw tagged label such as ``"M1K:downsampled-half"`` is not itself
    parsed by this function's category rules and would not reliably classify
    the same as its untagged base.

    Parameters
    ----------
    v : str
        Variant label string (e.g. ``"A123G"``, ``"A123fs"``, ``"WT"``).

    Returns
    -------
    str
        One of: ``"Frameshift"``, ``"3nt Deletion"``, ``"Nonsense"``,
        ``"WT"``, ``"Synonymous"``, ``"Single Missense"``, or ``"Other"``.
    """
    if "fs" in v:
        return "Frameshift"
    if v.endswith("-"):
        parts = v.split("|")
        n = len(parts)
        if n == 1:
            return "3nt Deletion"
        if n == 2 and int(parts[0][1:-1]) == int(parts[1][1:-1]) - 1:
            return "3nt Deletion"
        return "Other"
    if "X" in v or "*" in v:
        return "Nonsense"
    if "WT" in v:
        return "WT"
    m = re.match(r"([A-Z])(\d+)([A-Z])", v)
    if m is None:
        return "Other"
    return "Synonymous" if m.group(1) == m.group(3) else "Single Missense"


def strip_variant_tag(v: str) -> str:
    """
    Strip a trailing ``:<tag>`` suffix from a variant label, if present.

    Parameters
    ----------
    v : str
        Variant label, optionally suffixed with ``:<tag>`` (e.g.
        ``"M1K:downsampled-half"``).

    Returns
    -------
    str
        The label with any ``:<tag>`` suffix removed. Returns ``v`` unchanged
        if no ``:`` is present.
    """
    return v.split(":", 1)[0]
