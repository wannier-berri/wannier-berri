"""Focused parser for the NNKP data consumed by WannierBerri."""

import re

import numpy as np


_BLOCK_PATTERN = re.compile(
    r"^[ \t]*begin[ \t]+(?P<name>\w+)[ \t]*\r?\n"
    r"(?P<contents>.*?)"
    r"^[ \t]*end[ \t]+(?P=name)[ \t]*(?:\r?\n|$)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _parse_blocks(string):
    return {
        match.group("name").lower(): match.group("contents")
        for match in _BLOCK_PATTERN.finditer(string)
    }


def _parse_counted_array(string, columns, dtype=float, name="block"):
    lines = [line.strip() for line in string.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{name} block is empty")

    count = int(lines[0])
    contents = " ".join(lines[1:]).replace("D", "E").replace("d", "e")
    values = np.fromstring(contents, sep=" ", dtype=dtype)
    expected_size = count * columns
    if values.size != expected_size:
        raise ValueError(
            f"{name} block declares {count} entries with {columns} values each, "
            f"but contains {values.size} values"
        )
    return values.reshape((count, columns))


def parse_projections(string):
    projections = _parse_counted_array(string, columns=13, name="projections")
    return [
        {
            "center": projection[:3],
            "l": int(projection[3]),
            "mr": int(projection[4]),
            "r": int(projection[5]),
            "z-axis": projection[6:9],
            "x-axis": projection[9:12],
            "zona": projection[12],
        }
        for projection in projections
    ]


def parse_spinor_projections(string):
    projections = _parse_counted_array(string, columns=17, name="spinor_projections")
    return [
        {
            "center": projection[:3],
            "l": int(projection[3]),
            "mr": int(projection[4]),
            "r": int(projection[5]),
            "z-axis": projection[6:9],
            "x-axis": projection[9:12],
            "zona": projection[12],
            "spin": int(projection[13]),
            "spin-axis": projection[14:],
        }
        for projection in projections
    ]


def parse_nnkp(string, parse_optional=True):
    """Parse the NNKP data used by WannierBerri.

    The reciprocal lattice is deliberately not parsed. It is reconstructed
    from the direct lattice so that both bases have one numerical source.
    Optional projection blocks can be skipped by consumers that do not use
    them, such as :class:`BKVectors`.
    """
    blocks = _parse_blocks(string)
    required = {"real_lattice", "kpoints", "nnkpts"}
    missing = required.difference(blocks)
    if missing:
        raise ValueError(f"NNKP file is missing required blocks: {sorted(missing)}")

    real_lattice_string = blocks["real_lattice"].replace("D", "E").replace("d", "e")
    real_lattice = np.fromstring(real_lattice_string, sep=" ")
    if real_lattice.size != 9:
        raise ValueError(f"real_lattice block should contain 9 values, got {real_lattice.size}")

    kpoints = _parse_counted_array(blocks["kpoints"], columns=3, name="kpoints")
    nnkpt_lines = [line.strip() for line in blocks["nnkpts"].splitlines() if line.strip()]
    if not nnkpt_lines:
        raise ValueError("nnkpts block is empty")
    neighbours_per_kpoint = int(nnkpt_lines[0])
    nnkpts_values = np.fromstring(" ".join(nnkpt_lines[1:]), sep=" ", dtype=int)
    expected_size = len(kpoints) * neighbours_per_kpoint * 5
    if nnkpts_values.size != expected_size:
        raise ValueError(
            f"nnkpts block declares {neighbours_per_kpoint} neighbours for {len(kpoints)} k-points, "
            f"but contains {nnkpts_values.size} values"
        )

    result = {
        "real_lattice": real_lattice.reshape((3, 3)),
        "kpoints": kpoints,
        "nnkpts": nnkpts_values.reshape((-1, 5)),
    }
    if parse_optional:
        if "projections" in blocks:
            result["projections"] = parse_projections(blocks["projections"])
        if "spinor_projections" in blocks:
            result["spinor_projections"] = parse_spinor_projections(blocks["spinor_projections"])
    return result
