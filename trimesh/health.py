"""
health.py
---------

Inspect triangle meshes for common production issues and optionally repair
the issues which can be repaired safely.

The module deliberately loads files with ``process=False`` so duplicate
vertices, degenerate faces, and inconsistent winding are still present when
they are inspected.
"""

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import graph, grouping, triangles, util
from .base import Trimesh
from .exchange.load import available_formats, load_scene

__all__ = [
    "check_file",
    "check_mesh",
    "check_paths",
    "fix_file",
    "fix_mesh",
    "format_report",
    "main",
]


_CHECK_TITLES = {
    "non_manifold_edges": "Non-manifold edges",
    "degenerate_faces": "Degenerate faces",
    "duplicate_vertices": "Duplicate vertices",
    "uv_overlap": "UV overlap",
    "texture_paths": "Texture paths",
    "normals": "Normal orientation",
    "bounds_scale": "Bounding-box scale",
}

_TEXTURE_COMMANDS = {
    "map_ka",
    "map_kd",
    "map_ke",
    "map_ks",
    "map_ns",
    "map_d",
    "map_bump",
    "bump",
    "disp",
    "decal",
    "refl",
    "norm",
}


def _check(
    name,
    status="pass",
    count=0,
    message="",
    suggestion="No action required.",
    fixable=False,
    details=None,
):
    """Build a consistently-shaped check result."""
    severity = {"pass": "none", "skipped": "none"}.get(status, status)
    return {
        "name": name,
        "title": _CHECK_TITLES[name],
        "status": status,
        "severity": severity,
        "count": int(count),
        "message": message,
        "suggestion": suggestion,
        "fixable": bool(fixable),
        "details": {} if details is None else details,
    }


def _scale(mesh):
    """Return a positive characteristic scale for the mesh."""
    if len(mesh.vertices) == 0:
        return 1.0
    extents = np.asarray(mesh.extents, dtype=np.float64)
    finite = extents[np.isfinite(extents)]
    if len(finite) == 0:
        return 1.0
    result = float(finite.max())
    return result if result > 0.0 else 1.0


def _area_tolerance(mesh):
    """Area below this value is considered numerically degenerate."""
    scale = _scale(mesh)
    return max(scale * scale * 1e-14, np.finfo(np.float64).tiny)


def _degenerate_mask(mesh):
    """Return a mask selecting faces with repeated vertices or zero area."""
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)

    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    crosses = triangles.cross(tri)
    area = np.linalg.norm(crosses, axis=1) * 0.5
    return repeated | ~np.isfinite(area) | (area <= _area_tolerance(mesh))


def _edge_usage(mesh):
    """Return sorted unique edges, edge counts, and a face-to-edge inverse."""
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    if len(edges) == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    unique, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    return unique, counts, inverse


def _check_non_manifold(mesh):
    edges, counts, _ = _edge_usage(mesh)
    bad = np.flatnonzero(counts > 2)
    if len(bad) == 0:
        return _check(
            "non_manifold_edges",
            message="Every edge has at most two incident faces.",
        )

    sample = [
        {"vertices": edges[index].tolist(), "faces": int(counts[index])}
        for index in bad[:10]
    ]
    return _check(
        "non_manifold_edges",
        status="error",
        count=len(bad),
        message=f"{len(bad)} edge(s) have more than two incident faces.",
        suggestion="Remove duplicate/extra incident faces or split the junction. `--fix` removes extra incident faces and may leave a hole.",
        fixable=True,
        details={
            "edge_count": len(edges),
            "ratio": float(len(bad) / len(edges)),
            "sample": sample,
        },
    )


def _check_degenerate(mesh):
    mask = _degenerate_mask(mesh)
    count = int(mask.sum())
    if count == 0:
        return _check(
            "degenerate_faces",
            message="No zero-area or repeated-index faces.",
        )

    indices = np.flatnonzero(mask)
    return _check(
        "degenerate_faces",
        status="error",
        count=count,
        message=f"{count} face(s) are zero-area or reference a repeated vertex.",
        suggestion="Delete degenerate faces and rebuild the affected surface.",
        fixable=True,
        details={
            "ratio": float(count / max(len(mesh.faces), 1)),
            "sample_faces": indices[:10].tolist(),
        },
    )


def _duplicate_mask_and_groups(mesh):
    """Find duplicate vertices while preserving intentional UV seams."""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return np.zeros(0, dtype=bool), [], 0

    columns = [vertices]
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None and np.shape(uv) == (len(vertices), 2):
        columns.append(np.asarray(uv))
    # Only include explicit imported normals, not generated ones.
    if "vertex_normals" in mesh._cache.cache:
        normals = mesh._cache["vertex_normals"]
        if np.shape(normals) == vertices.shape:
            columns.append(np.asarray(normals))

    hashed = grouping.hashable_rows(np.column_stack(columns))
    _unique, inverse, counts = np.unique(hashed, return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(counts > 1)
    if len(duplicate_groups) == 0:
        return np.zeros(len(vertices), dtype=bool), [], 0

    mask = counts[inverse] > 1
    groups = []
    for group in duplicate_groups[:10]:
        members = np.flatnonzero(inverse == group)
        groups.append(
            {
                "vertices": members[:10].tolist(),
                "position": vertices[members[0]].tolist(),
                "count": len(members),
            }
        )
    return mask, groups, len(duplicate_groups)


def _check_duplicates(mesh):
    mask, groups, group_count = _duplicate_mask_and_groups(mesh)
    count = int(mask.sum())
    if count == 0:
        return _check(
            "duplicate_vertices",
            message="No duplicate vertices with matching UV/normal attributes.",
        )

    return _check(
        "duplicate_vertices",
        status="warning",
        count=count,
        message=f"{count} vertices duplicate another vertex's full attributes.",
        suggestion="Merge duplicate vertices while preserving UV seams.",
        fixable=True,
        details={"groups": groups, "group_count": group_count},
    )


def _cross2(a, b):
    """2D cross product which does not rely on deprecated numpy behavior."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_in_triangle(points, tri, epsilon):
    """Test each of three points against each of three triangle edges."""
    first = tri
    second = np.roll(tri, -1, axis=1)
    edges = second - first
    relative = points[:, :, None, :] - first[:, None, :, :]
    side = _cross2(edges[:, None, :, :], relative)
    positive = np.all(side >= -epsilon, axis=-1)
    negative = np.all(side <= epsilon, axis=-1)
    return positive | negative


def _triangle_intersection_area(first, second, epsilon):
    """
    Compute triangle-triangle intersection areas.

    Parameters
    ----------
    first, second : (n, 3, 2) float
      Triangle pairs.
    epsilon : float
      Relative tolerance for coincident edges and points.

    Returns
    -------
    area : (n,) float
      Intersection area for each pair.
    """
    count = len(first)
    if count == 0:
        return np.zeros(0, dtype=np.float64)

    next_index = np.roll(np.arange(3), -1)
    p = first[:, :, None, :]
    r = first[:, next_index, None, :] - p
    q = second[:, None, :, :]
    s = second[:, None, next_index, :] - q
    denominator = _cross2(r, s)
    relative = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(relative, s) / denominator
        u = _cross2(relative, r) / denominator

    scale = max(float(np.ptp(np.concatenate((first, second)), axis=1).max()), 1e-12)
    denominator_tol = scale * scale * 1e-12
    line_tol = 1e-8
    hit = (
        (np.abs(denominator) > denominator_tol)
        & (t >= -line_tol)
        & (t <= 1.0 + line_tol)
        & (u >= -line_tol)
        & (u <= 1.0 + line_tol)
    )
    with np.errstate(invalid="ignore"):
        intersections = p + r * t[..., None]

    first_inside = _point_in_triangle(first, second, epsilon)
    second_inside = _point_in_triangle(second, first, epsilon)
    points = np.concatenate((first, second, intersections.reshape((count, 9, 2))), axis=1)
    valid = np.concatenate(
        (
            first_inside,
            second_inside,
            hit.reshape((count, 9)),
        ),
        axis=1,
    )
    points = np.where(valid[..., None], points, 0.0)
    valid_count = valid.sum(axis=1)

    safe_valid = valid.copy()
    safe_valid[valid_count < 3] = False
    weights = safe_valid[..., None]
    safe_count = np.maximum(valid_count, 1)[:, None]
    centroid = (points * weights).sum(axis=1) / safe_count
    angle = np.arctan2(
        points[..., 1] - centroid[:, None, 1],
        points[..., 0] - centroid[:, None, 0],
    )
    angle[~valid] = 2.0 * np.pi
    order = np.argsort(angle, axis=1)
    points = np.take_along_axis(points, order[..., None], axis=1)
    valid = np.take_along_axis(valid, order, axis=1)

    # Fill trailing slots with the first vertex. These duplicate points add
    # zero area while keeping the shoelace calculation vectorized.
    first_point = points[:, :1, :]
    points = np.where(valid[..., None], points, first_point)
    following = np.roll(points, -1, axis=1)
    shoelace = _cross2(points, following).sum(axis=1)
    result = np.abs(shoelace) * 0.5
    result[valid_count < 3] = 0.0
    return result


def _uv_candidate_pairs(triangles_uv, area_tolerance, limit=2_000_000):
    """Build nearby triangle pairs without comparing every pair globally."""
    mins = triangles_uv.min(axis=1)
    maxs = triangles_uv.max(axis=1)
    span = maxs.max(axis=0) - mins.min(axis=0)
    cell = max(float(span.max()) / 64.0, np.finfo(np.float64).eps)
    origin = mins.min(axis=0)
    bins = defaultdict(list)

    for index in np.flatnonzero(
        np.linalg.norm(triangles_uv[:, 1] - triangles_uv[:, 0], axis=1)
        * np.linalg.norm(triangles_uv[:, 2] - triangles_uv[:, 0], axis=1)
        > area_tolerance
    ):
        low = np.floor((mins[index] - origin) / cell).astype(np.int64)
        high = np.floor((maxs[index] - origin) / cell).astype(np.int64)
        for x in range(int(low[0]), int(high[0]) + 1):
            for y in range(int(low[1]), int(high[1]) + 1):
                bins[(x, y)].append(int(index))

    encoded = []
    encoded_count = 0
    truncated = False
    face_count = len(triangles_uv)
    for members in bins.values():
        if len(members) < 2:
            continue
        members = np.asarray(members, dtype=np.int64)
        rows, cols = np.triu_indices(len(members), k=1)
        pair = members[rows] * face_count + members[cols]
        encoded.append(pair)
        encoded_count += len(pair)
        if encoded_count > limit:
            truncated = True
            break

    if len(encoded) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), truncated

    unique = np.unique(np.concatenate(encoded))[:limit]
    return unique // face_count, unique % face_count, truncated


def _uv_overlap(mesh, threshold):
    """Estimate the fraction of UV area participating in overlaps."""
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(mesh.faces) == 0:
        return _check(
            "uv_overlap",
            status="skipped",
            message="Mesh has no UV coordinates.",
        )
    if np.shape(uv) != (len(mesh.vertices), 2):
        return _check(
            "uv_overlap",
            status="warning",
            count=1,
            message="UV coordinates are malformed or do not match the vertex count.",
            suggestion="Re-import or re-unwrap the mesh.",
            details={"shape": list(np.shape(uv))},
        )

    values = np.asarray(uv, dtype=np.float64)
    uv_triangles = values[np.asarray(mesh.faces)]
    if not np.isfinite(uv_triangles).all():
        return _check(
            "uv_overlap",
            status="warning",
            count=1,
            message="UV coordinates contain NaN or infinite values.",
            suggestion="Repair invalid UV coordinates and re-unwrap the mesh.",
            details={"shape": list(values.shape)},
        )

    uv_area = (
        np.abs(
            _cross2(
                uv_triangles[:, 1] - uv_triangles[:, 0],
                uv_triangles[:, 2] - uv_triangles[:, 0],
            )
        )
        * 0.5
    )
    total_area = float(uv_area.sum())
    if total_area <= _area_tolerance(mesh):
        return _check(
            "uv_overlap",
            status="warning",
            count=1,
            message="All UV triangles have zero area.",
            suggestion="Re-unwrap the mesh so UV triangles have non-zero area.",
            fixable=True,
            details={"total_uv_area": total_area},
        )

    first, second, truncated = _uv_candidate_pairs(uv_triangles, _area_tolerance(mesh))
    pair_area = np.zeros(len(first), dtype=np.float64)
    epsilon = max(float(np.ptp(values, axis=0).max()) * 1e-10, 1e-12)
    chunk_size = 200_000
    for start in range(0, len(first), chunk_size):
        stop = start + chunk_size
        pair_area[start:stop] = _triangle_intersection_area(
            uv_triangles[first[start:stop]],
            uv_triangles[second[start:stop]],
            epsilon,
        )

    tolerance = _area_tolerance(mesh)
    real = pair_area > tolerance
    face_overlap = np.zeros(len(uv_triangles), dtype=np.float64)
    face_overlap += np.bincount(
        first[real], weights=pair_area[real], minlength=len(face_overlap)
    )
    face_overlap += np.bincount(
        second[real], weights=pair_area[real], minlength=len(face_overlap)
    )
    covered = np.minimum(face_overlap, uv_area)
    overlap_area = float(covered.sum())
    ratio = min(max(overlap_area / total_area, 0.0), 1.0)
    overlapping_faces = int((covered > tolerance).sum())

    if ratio <= threshold:
        return _check(
            "uv_overlap",
            message="UV overlap is below the configured threshold.",
            details={
                "overlap_rate": ratio,
                "threshold": threshold,
                "overlapping_faces": overlapping_faces,
                "total_uv_area": total_area,
                "candidate_pairs_truncated": truncated,
            },
        )

    return _check(
        "uv_overlap",
        status="warning",
        count=overlapping_faces,
        message=f"Approximately {ratio * 100.0:.3f}% of UV area overlaps.",
        suggestion="Re-unwrap the mesh and re-bake textures. `--fix` creates a non-overlapping per-face atlas which changes texture mapping.",
        fixable=True,
        details={
            "overlap_rate": ratio,
            "threshold": threshold,
            "overlapping_faces": overlapping_faces,
            "overlapping_pairs": int(real.sum()),
            "total_uv_area": total_area,
            "candidate_pairs_truncated": truncated,
        },
    )


def _local_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def _texture_records_gltf(path):
    try:
        with open(path, "rb") as file_obj:
            raw = file_obj.read()
        if raw[:4] == b"glTF":
            offset = 20
            length, kind = np.frombuffer(raw, dtype="<u4", count=2, offset=12)
            if kind != 0x4E4F534A:
                return []
            document = json.loads(raw[offset : offset + int(length)].decode("utf-8"))
        else:
            document = json.loads(raw.decode("utf-8"))
    except BaseException:
        return []

    records = []
    for image in document.get("images", []):
        uri = image.get("uri")
        if not isinstance(uri, str):
            continue
        if uri.startswith("data:"):
            records.append(
                {
                    "reference": "<embedded>",
                    "kind": "texture",
                    "exists": True,
                    "embedded": True,
                }
            )
            continue
        if urllib.parse.urlparse(uri).scheme in ("http", "https"):
            records.append(
                {
                    "reference": uri,
                    "kind": "texture",
                    "exists": None,
                    "remote": True,
                }
            )
            continue
        reference = urllib.parse.unquote(uri)
        resolved = os.path.abspath(os.path.join(os.path.dirname(path), reference))
        records.append(
            {
                "reference": reference,
                "kind": "texture",
                "resolved": resolved,
                "exists": os.path.isfile(resolved),
            }
        )
    return records


def _texture_records_dae(path):
    try:
        root = ElementTree.parse(path).getroot()
    except BaseException:
        return []
    records = []
    for element in root.iter():
        if _local_name(element.tag) != "init_from" or not element.text:
            continue
        reference = element.text.strip()
        if not reference or reference.startswith("data:"):
            continue
        resolved = os.path.abspath(os.path.join(os.path.dirname(path), reference))
        records.append(
            {
                "reference": reference,
                "kind": "texture",
                "resolved": resolved,
                "exists": os.path.isfile(resolved),
            }
        )
    return records


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _mtl_texture_reference(line, command):
    """Extract an MTL texture filename, preserving unquoted spaces."""
    rest = line[len(command) :].strip()
    if not rest:
        return None
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    if not tokens:
        return None
    if not tokens[0].startswith("-"):
        return " ".join(tokens)

    index = 0
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index].lower()
        index += 1
        if option in ("-o", "-s", "-t"):
            consumed = 0
            while consumed < 3 and index < len(tokens) and _is_number(tokens[index]):
                index += 1
                consumed += 1
        elif option == "-mm":
            index += min(2, len(tokens) - index)
        else:
            index += min(1, len(tokens) - index)
    return " ".join(tokens[index:]) if index < len(tokens) else None


def _texture_records_obj(path):
    records = []
    try:
        with open(path, encoding="utf-8", errors="replace") as file_obj:
            lines = file_obj.readlines()
    except OSError:
        return records

    material_files = []
    for line in lines:
        stripped = line.strip()
        if not stripped.lower().startswith("mtllib "):
            continue
        raw = stripped[7:].strip()
        try:
            names = shlex.split(raw)
        except ValueError:
            names = raw.split()
        whole_path = os.path.abspath(os.path.join(os.path.dirname(path), raw))
        if len(names) > 1 and os.path.isfile(whole_path):
            names = [raw]
        for name in names:
            resolved = os.path.abspath(os.path.join(os.path.dirname(path), name))
            material_files.append((name, resolved))
            records.append(
                {
                    "reference": name,
                    "kind": "material",
                    "resolved": resolved,
                    "exists": os.path.isfile(resolved),
                }
            )

    for name, material_path in material_files:
        if not os.path.isfile(material_path):
            continue
        try:
            with open(material_path, encoding="utf-8", errors="replace") as file_obj:
                material_lines = file_obj.readlines()
        except OSError:
            continue
        for line in material_lines:
            stripped = line.strip()
            command = stripped.split(maxsplit=1)[0].lower() if stripped else ""
            if command not in _TEXTURE_COMMANDS:
                continue
            reference = _mtl_texture_reference(stripped, command)
            if reference is None:
                continue
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(path), reference)),
                os.path.abspath(os.path.join(os.path.dirname(material_path), reference)),
            ]
            resolved = next(
                (candidate for candidate in candidates if os.path.isfile(candidate)),
                candidates[0],
            )
            records.append(
                {
                    "reference": reference,
                    "kind": "texture",
                    "resolved": resolved,
                    "exists": os.path.isfile(resolved),
                    "material": name,
                }
            )
    return records


def texture_references(path):
    """
    Return external material and texture references for a model file.

    Supported reference scanners currently cover OBJ/MTL, glTF/GLB, and DAE.
    Archives and formats without external texture references return an empty
    list.
    """
    extension = str(path).lower()
    if extension.endswith((".obj",)):
        return _texture_records_obj(path)
    if extension.endswith((".gltf", ".glb")):
        return _texture_records_gltf(path)
    if extension.endswith((".dae",)):
        return _texture_records_dae(path)
    if extension.endswith(".zip"):
        records = []
        try:
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith((".obj", ".gltf", ".dae")):
                        # Internal archive references are resolved by trimesh's
                        # ZipResolver and are considered embedded.
                        records.append(
                            {
                                "reference": name,
                                "kind": "archive",
                                "exists": True,
                                "embedded": True,
                            }
                        )
        except BaseException:
            pass
        return records
    return []


def _check_textures(records):
    if len(records) == 0:
        return _check(
            "texture_paths",
            message="No external texture or material references.",
        )

    missing = [record for record in records if record.get("exists") is False]
    remote = [record for record in records if record.get("remote")]
    if len(missing) == 0:
        return _check(
            "texture_paths",
            message="All external texture and material references exist.",
            details={
                "reference_count": len(records),
                "remote_unverified": len(remote),
            },
        )

    return _check(
        "texture_paths",
        status="error",
        count=len(missing),
        message=f"{len(missing)} external texture/material reference(s) are missing.",
        suggestion="Restore the missing files or remove the broken material reference.",
        details={
            "missing": missing[:20],
            "reference_count": len(records),
            "remote_unverified": len(remote),
        },
    )


def _face_components(mesh):
    """Return face-index arrays for each face-connected component."""
    count = len(mesh.faces)
    if count == 0:
        return []
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return [np.asarray([index], dtype=np.int64) for index in range(count)]

    parent = np.arange(count, dtype=np.int64)

    def find(index):
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            following = parent[index]
            parent[index] = root
            index = following
        return root

    for first, second in adjacency:
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    order = np.argsort(roots)
    sorted_roots = roots[order]
    boundaries = np.flatnonzero(np.diff(sorted_roots)) + 1
    return np.split(order, boundaries)


def _check_normals(mesh):
    if len(mesh.faces) == 0:
        return _check(
            "normals",
            status="skipped",
            message="Mesh has no faces.",
        )

    inconsistent = False
    try:
        inconsistent = not bool(mesh.is_winding_consistent)
    except BaseException:
        inconsistent = True

    inverted = []
    components = _face_components(mesh)
    closed = 0
    volume_tolerance = max(_scale(mesh) ** 3 * 1e-12, np.finfo(np.float64).tiny)
    try:
        tri = np.asarray(mesh.triangles)
        crosses = np.asarray(mesh.triangles_cross)
        face_edges = np.asarray(mesh.edges).reshape((-1, 6))
        for group in components:
            if len(group) < 4:
                continue
            is_tight, is_wound = graph.is_watertight(face_edges[group].reshape((-1, 2)))
            if not (is_tight and is_wound):
                continue
            closed += 1
            volume = triangles.mass_properties(
                tri[group], crosses=crosses[group], skip_inertia=True
            ).volume
            if volume < -volume_tolerance:
                inverted.append({"face_count": len(group), "volume": float(volume)})
    except BaseException:
        # Malformed topology should not prevent other checks from running.
        pass

    count = len(inverted) + int(inconsistent)
    if count == 0:
        message = "Closed components face outward."
        if closed == 0:
            message = "No closed components; open-surface orientation was not asserted."
        return _check("normals", message=message, details={"closed_components": closed})

    return _check(
        "normals",
        status="error",
        count=count,
        message=f"{len(inverted)} closed component(s) have inward-facing normals; winding is inconsistent: {str(inconsistent).lower()}.",
        suggestion="Flip inverted components and make adjacent face winding consistent.",
        fixable=True,
        details={
            "inverted_components": inverted[:10],
            "inconsistent_winding": inconsistent,
            "closed_components": closed,
        },
    )


def _check_bounds(mesh, min_size, max_size, max_aspect_ratio):
    if len(mesh.vertices) == 0:
        return _check(
            "bounds_scale",
            status="error",
            count=1,
            message="Mesh has no vertices.",
            suggestion="Restore or regenerate the mesh data.",
            details={"reason": "empty"},
        )

    extents = np.asarray(mesh.extents, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.isfinite(extents).all():
        return _check(
            "bounds_scale",
            status="error",
            count=1,
            message="Bounding box contains non-finite values.",
            suggestion="Remove NaN/Inf vertices and check the source transform.",
            details={"extents": extents.tolist(), "reason": "non_finite"},
        )

    largest = float(extents.max())
    positive = extents[extents > 0.0]
    aspect = float(largest / positive.min()) if len(positive) else None
    reasons = []
    if largest <= 0.0:
        reasons.append("zero_size")
    if largest < min_size:
        reasons.append("below_min_size")
    if largest > max_size:
        reasons.append("above_max_size")
    if aspect is not None and aspect > max_aspect_ratio:
        reasons.append("extreme_aspect_ratio")

    details = {
        "extents": extents.tolist(),
        "bounds": bounds.tolist(),
        "largest_extent": largest,
        "aspect_ratio": aspect,
        "thresholds": {
            "min_size": min_size,
            "max_size": max_size,
            "max_aspect_ratio": max_aspect_ratio,
        },
    }
    if len(reasons) == 0:
        return _check(
            "bounds_scale",
            message="Bounding-box dimensions are within configured limits.",
            details=details,
        )

    details["reasons"] = reasons
    return _check(
        "bounds_scale",
        status="warning",
        count=1,
        message="Bounding-box scale looks unusual.",
        suggestion="Verify scene units and the export/import transform. This check does not auto-scale models because the intended unit is unknown.",
        details=details,
    )


def check_mesh(
    mesh,
    texture_records=None,
    uv_overlap_threshold=1e-4,
    min_size=1e-6,
    max_size=1e6,
    max_aspect_ratio=1e4,
):
    """
    Inspect one :class:`trimesh.Trimesh` and return a JSON-safe report.

    Parameters
    ----------
    mesh : Trimesh
      Mesh to inspect.
    texture_records : list[dict] | None
      Output from :func:`texture_references`. If omitted, texture paths are
      reported as skipped for a directly-supplied mesh.
    uv_overlap_threshold : float
      UV overlap fraction above which the check is a warning.
    min_size, max_size : float
      Absolute bounding-box size limits.
    max_aspect_ratio : float
      Maximum largest/smallest non-zero bounding-box ratio.

    Returns
    -------
    report : dict
      JSON-safe per-mesh report.
    """
    checks = {
        "non_manifold_edges": _check_non_manifold(mesh),
        "degenerate_faces": _check_degenerate(mesh),
        "duplicate_vertices": _check_duplicates(mesh),
        "uv_overlap": _uv_overlap(mesh, uv_overlap_threshold),
        "texture_paths": (
            _check(
                "texture_paths",
                status="skipped",
                message="No source file was supplied for texture lookup.",
            )
            if texture_records is None
            else _check_textures(texture_records)
        ),
        "normals": _check_normals(mesh),
        "bounds_scale": _check_bounds(mesh, min_size, max_size, max_aspect_ratio),
    }
    return _summarize(checks)


def _summarize(checks):
    errors = sum(check["severity"] == "error" for check in checks.values())
    warnings = sum(check["severity"] == "warning" for check in checks.values())
    status = "error" if errors else ("warning" if warnings else "pass")
    return {
        "status": status,
        "summary": {"errors": errors, "warnings": warnings},
        "checks": checks,
    }


def _merge_check_results(results):
    """Merge per-geometry checks into one file-level result."""
    merged = {}
    for name in _CHECK_TITLES:
        values = [result["checks"][name] for result in results]
        statuses = [value["status"] for value in values]
        if "error" in statuses:
            status = "error"
        elif "warning" in statuses:
            status = "warning"
        elif "pass" in statuses:
            status = "pass"
        else:
            status = "skipped"
        count = sum(value["count"] for value in values)
        first = next(
            (value for value in values if value["status"] == status),
            values[0] if values else None,
        )
        if first is None:
            continue
        details = dict(first["details"])
        details["geometry_checks"] = [
            {
                "name": geometry_name,
                "status": value["status"],
                "count": value["count"],
                "details": value["details"],
            }
            for geometry_name, value in zip(
                [result["geometry"]["name"] for result in results], values
            )
            if value["status"] in ("warning", "error", "skipped")
        ]
        merged[name] = {
            "name": name,
            "title": _CHECK_TITLES[name],
            "status": status,
            "severity": {"pass": "none", "skipped": "none"}.get(status, status),
            "count": int(count),
            "message": first["message"],
            "suggestion": first["suggestion"],
            "fixable": any(value["fixable"] for value in values),
            "details": details,
        }
    return merged


def _load(path):
    return load_scene(path, process=False, maintain_order=True)


def check_file(
    path,
    uv_overlap_threshold=1e-4,
    min_size=1e-6,
    max_size=1e6,
    max_aspect_ratio=1e4,
):
    """
    Inspect a model file and return a JSON-safe report.

    Triangle geometry is loaded without automatic processing so defects are
    observed rather than silently repaired by the loader.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    report = {
        "path": path,
        "status": "error",
        "summary": {"errors": 1, "warnings": 0},
        "geometry": {},
        "checks": {},
    }
    try:
        scene = _load(path)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    records = texture_references(path)
    geometry_reports = []
    vertex_count = 0
    face_count = 0
    for name, geometry in scene.geometry.items():
        if not isinstance(geometry, Trimesh):
            continue
        current = check_mesh(
            geometry,
            texture_records=None,
            uv_overlap_threshold=uv_overlap_threshold,
            min_size=min_size,
            max_size=max_size,
            max_aspect_ratio=max_aspect_ratio,
        )
        current["geometry"] = {
            "name": str(name),
            "vertices": len(geometry.vertices),
            "faces": len(geometry.faces),
            "watertight": bool(geometry.is_watertight),
            "winding_consistent": bool(geometry.is_winding_consistent),
            "extents": np.asarray(geometry.extents, dtype=np.float64).tolist(),
        }
        geometry_reports.append(current)
        vertex_count += len(geometry.vertices)
        face_count += len(geometry.faces)

    if len(geometry_reports) == 0:
        report["error"] = "No triangle mesh geometry found."
        return report

    checks = _merge_check_results(geometry_reports)
    checks["texture_paths"] = _check_textures(records)
    summary = _summarize(checks)
    report.update(summary)
    report["geometry"] = {
        "count": len(geometry_reports),
        "vertices": int(vertex_count),
        "faces": int(face_count),
        "names": [item["geometry"]["name"] for item in geometry_reports],
    }
    return report


def _remove_duplicate_faces(mesh):
    if len(mesh.faces) == 0:
        return 0
    canonical = np.sort(np.asarray(mesh.faces), axis=1)
    _, first, _inverse, _counts = np.unique(
        canonical, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    if len(first) == len(canonical):
        return 0
    keep = np.zeros(len(canonical), dtype=bool)
    keep[first] = True
    removed = int((~keep).sum())
    mesh.update_faces(keep)
    return removed


def _remove_non_manifold_faces(mesh):
    """Remove extra faces until every edge has at most two incident faces."""
    removed = 0
    while True:
        _edges, counts, inverse = _edge_usage(mesh)
        bad = np.flatnonzero(counts > 2)
        if len(bad) == 0:
            return removed

        order = np.argsort(inverse)
        sorted_inverse = inverse[order]
        boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(order)]))
        keep = np.ones(len(mesh.faces), dtype=bool)
        record_faces = np.repeat(np.arange(len(mesh.faces)), 3)[order]
        changed = False
        for edge_index in bad:
            incident = record_faces[starts[edge_index] : stops[edge_index]]
            incident = np.unique(incident[keep[incident]])
            if len(incident) > 2:
                keep[incident[2:]] = False
                changed = True
        if not changed:
            return removed
        count = int((~keep).sum())
        removed += count
        mesh.update_faces(keep)


def _edge_direction(triangle, edge):
    """Return 1 if a triangle traverses an edge as written, -1 if reversed."""
    for index in range(3):
        first = int(triangle[index])
        second = int(triangle[(index + 1) % 3])
        if first == int(edge[0]) and second == int(edge[1]):
            return 1
        if first == int(edge[1]) and second == int(edge[0]):
            return -1
    return 0


def _fix_winding(mesh):
    """Make connected face patches use consistent winding without networkx."""
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return 0
    shared_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    faces = np.asarray(mesh.faces).copy()
    neighbors = [[] for _ in range(len(faces))]
    for index, (first, second) in enumerate(adjacency):
        edge = shared_edges[index]
        relation = _edge_direction(faces[first], edge) == _edge_direction(
            faces[second], edge
        )
        neighbors[int(first)].append((int(second), relation))
        neighbors[int(second)].append((int(first), relation))

    flipped = np.zeros(len(faces), dtype=bool)
    visited = np.zeros(len(faces), dtype=bool)
    for start in range(len(faces)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor, relation in neighbors[current]:
                expected = bool(flipped[current]) ^ bool(relation)
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                flipped[neighbor] = expected
                stack.append(neighbor)

    count = int(flipped.sum())
    if count:
        faces[flipped] = np.fliplr(faces[flipped])
        mesh.faces = faces
    return count


def _fix_inverted_components(mesh):
    """Flip closed components whose signed volume is negative."""
    components = _face_components(mesh)
    faces = np.asarray(mesh.faces).copy()
    tri = np.asarray(mesh.triangles)
    crosses = np.asarray(mesh.triangles_cross)
    face_edges = np.asarray(mesh.edges).reshape((-1, 6))
    volume_tolerance = max(_scale(mesh) ** 3 * 1e-12, np.finfo(np.float64).tiny)
    flip = np.zeros(len(faces), dtype=bool)
    for component in components:
        if len(component) < 4:
            continue
        is_tight, is_wound = graph.is_watertight(face_edges[component].reshape((-1, 2)))
        if not (is_tight and is_wound):
            continue
        volume = triangles.mass_properties(
            tri[component], crosses=crosses[component], skip_inertia=True
        ).volume
        if volume < -volume_tolerance:
            flip[component] = True

    count = int(flip.sum())
    if count:
        faces[flip] = np.fliplr(faces[flip])
        mesh.faces = faces
    return count


def _per_face_uv(mesh):
    """Build a deterministic, non-overlapping UV triangle per face."""
    count = len(mesh.faces)
    result = mesh.copy()
    result.unmerge_vertices()
    columns = math.ceil(math.sqrt(max(count, 1)))
    cell = 1.0 / columns
    inset = cell * 0.08
    uv = np.zeros((count * 3, 2), dtype=np.float64)
    for index in range(count):
        row, column = divmod(index, columns)
        x0 = column * cell + inset
        y0 = row * cell + inset
        x1 = (column + 1) * cell - inset
        y1 = (row + 1) * cell - inset
        uv[index * 3 : index * 3 + 3] = [[x0, y0], [x1, y0], [x0, y1]]
    result.visual.uv = uv
    return result


def fix_mesh(mesh, uv_overlap_threshold=1e-4):
    """
    Repair automatically-fixable problems in a mesh.

    This mutates ``mesh`` when possible and returns a possibly-new mesh plus
    a list of actions. UV repair changes texture mapping and requires the
    texture to be re-baked.
    """
    actions = []

    before_vertices = len(mesh.vertices)
    mesh.merge_vertices()
    merged = before_vertices - len(mesh.vertices)
    if merged:
        actions.append(
            {
                "check": "duplicate_vertices",
                "action": "merged duplicate vertices",
                "count": int(merged),
            }
        )

    degenerate = _degenerate_mask(mesh)
    if degenerate.any():
        count = int(degenerate.sum())
        mesh.update_faces(~degenerate)
        actions.append(
            {
                "check": "degenerate_faces",
                "action": "removed degenerate faces",
                "count": count,
            }
        )

    duplicate_faces = _remove_duplicate_faces(mesh)
    if duplicate_faces:
        actions.append(
            {
                "check": "non_manifold_edges",
                "action": "removed duplicate faces",
                "count": duplicate_faces,
            }
        )

    non_manifold = _remove_non_manifold_faces(mesh)
    if non_manifold:
        actions.append(
            {
                "check": "non_manifold_edges",
                "action": "removed extra faces at non-manifold edges",
                "count": non_manifold,
            }
        )

    report = check_mesh(mesh, texture_records=[])
    if report["checks"]["normals"]["status"] == "error":
        winding_count = _fix_winding(mesh)
        inversion_count = _fix_inverted_components(mesh)
        actions.append(
            {
                "check": "normals",
                "action": "made winding consistent and flipped inward components",
                "count": winding_count + inversion_count,
            }
        )

    uv_check = _uv_overlap(mesh, uv_overlap_threshold)
    if uv_check["status"] == "warning" and len(mesh.faces) > 0:
        mesh = _per_face_uv(mesh)
        actions.append(
            {
                "check": "uv_overlap",
                "action": "rebuilt UVs as a non-overlapping per-face atlas",
                "count": uv_check["count"],
            }
        )
    return mesh, actions


def _export_scene(scene, path):
    """Export through a temporary main file and replace the destination."""
    extension = util.split_extension(path, special=["tar.gz", "tar.bz2"])
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=".trimesh-health-", suffix="." + extension, dir=directory
    )
    os.close(handle)
    try:
        scene.export(temporary, file_type=extension)
        if os.path.isfile(path):
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def fix_file(
    path,
    output=None,
    backup=False,
    uv_overlap_threshold=1e-4,
    min_size=1e-6,
    max_size=1e6,
    max_aspect_ratio=1e4,
):
    """
    Repair a model file and return a post-fix report.

    The repaired main file is written to ``output`` when supplied, otherwise
    in place. The report contains a ``fix`` object with attempted actions.
    """
    source = os.path.abspath(os.path.expanduser(str(path)))
    destination = (
        source if output is None else os.path.abspath(os.path.expanduser(str(output)))
    )
    before = check_file(
        source,
        uv_overlap_threshold=uv_overlap_threshold,
        min_size=min_size,
        max_size=max_size,
        max_aspect_ratio=max_aspect_ratio,
    )
    result = dict(before)
    result["fix"] = {"attempted": False, "actions": [], "error": None}
    if "error" in before:
        return result

    try:
        scene = _load(source)
        actions = []
        for name, geometry in list(scene.geometry.items()):
            if isinstance(geometry, Trimesh):
                fixed, current_actions = fix_mesh(
                    geometry, uv_overlap_threshold=uv_overlap_threshold
                )
                scene.geometry[name] = fixed
                actions.extend(current_actions)
        result["fix"]["attempted"] = True
        result["fix"]["actions"] = actions

        if len(actions) == 0:
            if destination != source:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)
        else:
            if backup and destination == source:
                shutil.copy2(source, source + ".bak")
            _export_scene(scene, destination)
    except BaseException as error:
        result["fix"]["error"] = f"{type(error).__name__}: {error}"
        return result

    after = check_file(
        destination,
        uv_overlap_threshold=uv_overlap_threshold,
        min_size=min_size,
        max_size=max_size,
        max_aspect_ratio=max_aspect_ratio,
    )
    after["fix"] = result["fix"]
    if destination != source:
        after["source"] = source
    return after


def _discover(paths, recursive=True):
    formats = {str(item).lower() for item in available_formats()}
    formats.update({"obj", "glb", "gltf", "dae", "3mf", "ply", "stl", "off"})
    found = []
    for value in paths:
        path = os.path.abspath(os.path.expanduser(str(value)))
        if os.path.isfile(path):
            found.append(path)
        elif os.path.isdir(path):
            if recursive:
                walker = (
                    os.path.join(root, name)
                    for root, _, names in os.walk(path)
                    for name in names
                )
            else:
                walker = (entry.path for entry in os.scandir(path) if entry.is_file())
            for candidate in walker:
                extension = util.split_extension(
                    candidate, special=["tar.gz", "tar.bz2"]
                ).lower()
                if extension in formats:
                    found.append(os.path.abspath(candidate))
    return sorted(set(found))


class _Progress:
    """Small terminal progress display which writes only to stderr."""

    def __init__(self, total, enabled=True):
        self.total = total
        self.enabled = enabled and total > 1
        self.complete = 0
        self.started = time.time()
        self.tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.next_report = 0

    def update(self, label=""):
        if not self.enabled:
            return
        self.complete += 1
        percent = 100.0 * self.complete / self.total
        if self.tty:
            sys.stderr.write(
                f"\r[{self.complete}/{self.total}] {percent:5.1f}% {label[:60]:<60}"
            )
            sys.stderr.flush()
            if self.complete == self.total:
                sys.stderr.write("\n")
        elif self.complete >= self.next_report or self.complete == self.total:
            elapsed = time.time() - self.started
            sys.stderr.write(
                f"[{self.complete}/{self.total}] {percent:5.1f}% ({elapsed:.1f}s) {label}\n"
            )
            sys.stderr.flush()
            self.next_report = self.complete + max(1, self.total // 100)


def check_paths(
    paths,
    jobs=1,
    recursive=True,
    progress=True,
    uv_overlap_threshold=1e-4,
    min_size=1e-6,
    max_size=1e6,
    max_aspect_ratio=1e4,
):
    """
    Check files or directories concurrently.

    Returns a batch report containing a ``files`` list. Results preserve input
    order even when ``jobs > 1``.
    """
    files = _discover(paths, recursive=recursive)
    progress_display = _Progress(len(files), enabled=progress)

    def worker(path):
        return check_file(
            path,
            uv_overlap_threshold=uv_overlap_threshold,
            min_size=min_size,
            max_size=max_size,
            max_aspect_ratio=max_aspect_ratio,
        )

    reports = []
    if jobs <= 1 or len(files) <= 1:
        for path in files:
            reports.append(worker(path))
            progress_display.update(os.path.basename(path))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for path, report in zip(files, executor.map(worker, files)):
                reports.append(report)
                progress_display.update(os.path.basename(path))
    return _batch_report(reports, files)


def _batch_report(reports, files):
    errors = sum(report.get("summary", {}).get("errors", 0) for report in reports)
    warnings = sum(report.get("summary", {}).get("warnings", 0) for report in reports)
    passed = sum(report.get("status") == "pass" for report in reports)
    failed_files = sum(report.get("status") == "error" for report in reports)
    return {
        "schema_version": 1,
        "inputs": [os.path.abspath(os.path.expanduser(str(path))) for path in files],
        "summary": {
            "files": len(reports),
            "passed": int(passed),
            "errors": int(errors),
            "warnings": int(warnings),
            "failed_files": int(failed_files),
        },
        "files": reports,
    }


def _format_check(check, indent="  "):
    severity = check["severity"].upper()
    if check["status"] == "pass":
        marker = "OK"
    elif check["status"] == "skipped":
        marker = "SKIP"
    else:
        marker = severity
    lines = [
        f"{indent}[{marker}] {check['title']}: {check['message']}",
    ]
    if check["status"] in ("warning", "error"):
        lines.append(f"{indent}  fix: {check['suggestion']}")
    return lines


def format_report(report):
    """Render a batch or single-file report as human-readable text."""
    lines = ["trimesh health report", ""]
    files = report.get("files") if "files" in report else [report]
    for file_report in files:
        lines.append(f"{file_report['path']}")
        status = file_report.get("status", "error").upper()
        lines.append(f"  status: {status}")
        if "error" in file_report:
            lines.append(f"  error: {file_report['error']}")
            lines.append("")
            continue
        geometry = file_report.get("geometry", {})
        lines.append(
            f"  geometry: {geometry.get('count', 0)} mesh(es), {geometry.get('vertices', 0)} vertices, {geometry.get('faces', 0)} faces"
        )
        for check in file_report.get("checks", {}).values():
            lines.extend(_format_check(check))
        fix = file_report.get("fix")
        if fix and fix.get("actions"):
            lines.append("  fixes:")
            for action in fix["actions"]:
                lines.append(
                    f"    - {action['check']}: {action['action']} ({action['count']})"
                )
        if fix and fix.get("error"):
            lines.append(f"  fix error: {fix['error']}")
        lines.append("")

    summary = report.get("summary", {})
    lines.append(
        f"summary: {summary.get('files', len(files))} file(s), {summary.get('errors', 0)} error(s), {summary.get('warnings', 0)} warning(s), {summary.get('passed', 0)} passed"
    )
    return "\n".join(lines) + "\n"


def _json_write(report, destination):
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if destination == "-":
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return
    destination = os.path.abspath(os.path.expanduser(destination))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as file_obj:
        file_obj.write(text)
        file_obj.write("\n")


def _parser():
    parser = argparse.ArgumentParser(
        prog="trimesh-health",
        description="Inspect triangle model files and optionally repair them.",
    )
    parser.add_argument("inputs", nargs="+", help="Model file(s) or directories.")
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of concurrent workers."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not descend into subdirectories.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the JSON report to PATH. Use '-' for stdout.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Repair fixable issues and re-check the repaired file.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Write repaired files under DIR instead of in place.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .bak copy before an in-place repair.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable batch progress output on stderr.",
    )
    parser.add_argument(
        "--uv-threshold",
        type=float,
        default=1e-4,
        help="UV overlap fraction treated as a warning (default: 0.0001).",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=1e-6,
        help="Lower bounding-box size warning threshold (default: 1e-6).",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=1e6,
        help="Upper bounding-box size warning threshold (default: 1e6).",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1e4,
        help="Bounding-box aspect-ratio warning threshold (default: 1e4).",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "warning", "error"),
        default="error",
        help="Exit nonzero at this severity (default: error).",
    )
    return parser


def _repair_target(path, input_roots, output_dir):
    if output_dir is None:
        return None
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    relative = os.path.basename(path)
    for root in input_roots:
        if os.path.isdir(root):
            try:
                relative = os.path.relpath(path, root)
                break
            except ValueError:
                continue
        elif os.path.realpath(path) == os.path.realpath(root):
            relative = os.path.basename(path)
            break
    return os.path.join(output_dir, relative)


def main(argv=None):
    """Command-line entry point for ``trimesh-health`` and ``trimesh check``."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be greater than zero")

    files = _discover(args.inputs, recursive=not args.no_recursive)
    if len(files) == 0:
        sys.stderr.write("No supported model files found.\n")
        return 1

    input_roots = [
        os.path.abspath(os.path.expanduser(str(value))) for value in args.inputs
    ]
    targets = {path: _repair_target(path, input_roots, args.output_dir) for path in files}
    progress_display = _Progress(len(files), enabled=not args.no_progress)

    def worker(path):
        if args.fix:
            return fix_file(
                path,
                output=targets[path],
                backup=args.backup,
                uv_overlap_threshold=args.uv_threshold,
                min_size=args.min_size,
                max_size=args.max_size,
                max_aspect_ratio=args.max_aspect_ratio,
            )
        return check_file(
            path,
            uv_overlap_threshold=args.uv_threshold,
            min_size=args.min_size,
            max_size=args.max_size,
            max_aspect_ratio=args.max_aspect_ratio,
        )

    reports = []
    if args.jobs <= 1 or len(files) <= 1:
        for path in files:
            reports.append(worker(path))
            progress_display.update(os.path.basename(path))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for path, result in zip(files, executor.map(worker, files)):
                reports.append(result)
                progress_display.update(os.path.basename(path))

    batch = _batch_report(reports, files)
    text = format_report(batch)
    if args.json == "-":
        _json_write(batch, "-")
        sys.stderr.write(text)
    else:
        sys.stdout.write(text)
        if args.json is not None:
            _json_write(batch, args.json)

    summary = batch["summary"]
    if args.fail_on == "warning":
        return int(summary["errors"] > 0 or summary["warnings"] > 0)
    if args.fail_on == "error":
        return int(summary["errors"] > 0 or summary["failed_files"] > 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
