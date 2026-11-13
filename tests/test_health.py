import base64
import json
import os

import pytest

from trimesh import health
from trimesh.__main__ import main as trimesh_main

try:
    from . import generic as g
except BaseException:
    import generic as g


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(text)
    return str(path)


def _write_obj(path, body):
    return _write(
        path,
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 3 0 0",
                "v 4 0 0",
                "v 3 1 0",
                body,
                "",
            ]
        ),
    )


def test_non_manifold_detect_and_fix(tmp_path):
    path = tmp_path / "nonmanifold.obj"
    _write_obj(
        path,
        "\n".join(
            [
                "v 0 -1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 2 1 4",
                "f 2 1 5",
            ]
        ),
    )

    before = health.check_file(path)
    check = before["checks"]["non_manifold_edges"]
    assert check["severity"] == "error"
    assert check["count"] == 1

    after = health.fix_file(path)
    assert after["fix"]["actions"]
    assert after["checks"]["non_manifold_edges"]["status"] == "pass"
    assert after["checks"]["non_manifold_edges"]["count"] == 0


def test_degenerate_and_duplicate_detect_and_fix(tmp_path):
    path = tmp_path / "degenerate.obj"
    _write_obj(
        path,
        "\n".join(
            [
                "v 1 0 0",
                "v 2 0 0",
                "f 1 2 3",
                "f 1 4 5",
                "f 1 2 1",
            ]
        ),
    )

    before = health.check_file(path)
    assert before["checks"]["duplicate_vertices"]["count"] >= 2
    assert before["checks"]["degenerate_faces"]["count"] >= 1

    after = health.fix_file(path)
    assert after["checks"]["duplicate_vertices"]["status"] == "pass"
    assert after["checks"]["degenerate_faces"]["status"] == "pass"


def test_uv_overlap_detect_and_fix(tmp_path):
    path = tmp_path / "overlap.obj"
    _write(
        path,
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 3 0 0",
                "v 4 0 0",
                "v 3 1 0",
                "vt 0 0",
                "vt 0.5 0",
                "vt 0 0.5",
                "f 1/1 2/2 3/3",
                "f 4/1 5/2 6/3",
                "",
            ]
        ),
    )

    before = health.check_file(path)
    check = before["checks"]["uv_overlap"]
    assert check["status"] == "warning"
    assert check["details"]["overlap_rate"] == pytest.approx(1.0)

    after = health.fix_file(path)
    assert after["checks"]["uv_overlap"]["status"] == "pass"
    assert after["checks"]["uv_overlap"]["details"]["overlap_rate"] < 1e-4


@pytest.mark.parametrize("inconsistent", [False, True])
def test_normals_detect_and_fix(tmp_path, inconsistent):
    path = tmp_path / "normals.ply"
    mesh = g.trimesh.creation.box()
    if inconsistent:
        mesh.faces[0] = mesh.faces[0][::-1]
    else:
        mesh.invert()
    mesh.export(path)

    before = health.check_file(path)
    check = before["checks"]["normals"]
    assert check["severity"] == "error"

    after = health.fix_file(path)
    assert after["checks"]["normals"]["status"] == "pass"
    assert after["checks"]["normals"]["count"] == 0


def test_texture_paths(tmp_path):
    missing_obj = tmp_path / "missing.obj"
    _write(
        missing_obj,
        "\n".join(
            [
                "mtllib missing.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "",
            ]
        ),
    )
    _write(tmp_path / "missing.mtl", "newmtl broken\nmap_Kd nowhere.png\n")

    report = health.check_file(missing_obj)
    check = report["checks"]["texture_paths"]
    assert check["severity"] == "error"
    assert check["count"] == 1
    assert check["fixable"] is False
    assert check["details"]["missing"][0]["reference"] == "nowhere.png"

    good_obj = tmp_path / "good.obj"
    _write(
        good_obj,
        "\n".join(
            [
                "mtllib good.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "",
            ]
        ),
    )
    _write(tmp_path / "good.mtl", "newmtl ok\nmap_Kd good texture.png\n")
    (tmp_path / "good texture.png").write_bytes(PNG_1X1)
    assert health.check_file(good_obj)["checks"]["texture_paths"]["status"] == "pass"


def test_bounds_scale_warning(tmp_path):
    path = tmp_path / "tiny.ply"
    g.trimesh.creation.box(extents=[1e-8, 1e-8, 1e-8]).export(path)

    check = health.check_file(path)["checks"]["bounds_scale"]
    assert check["status"] == "warning"
    assert "below_min_size" in check["details"]["reasons"]
    assert check["fixable"] is False


def test_batch_cli_json(tmp_path, capsys):
    directory = tmp_path / "models"
    directory.mkdir()
    g.trimesh.creation.box().export(directory / "good.ply")
    good = g.trimesh.creation.box()
    good.invert()
    good.export(directory / "inverted.ply")
    _write(directory / "broken.obj", "v 0 0 0\nv 1 0 0\nf 1 2 1\n")

    report_path = tmp_path / "report.json"
    code = health.main(
        [
            str(directory),
            "--jobs",
            "3",
            "--json",
            str(report_path),
            "--no-progress",
            "--fail-on",
            "never",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "trimesh health report" in output
    with open(report_path, encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    assert report["summary"]["files"] == 3
    assert len(report["files"]) == 3
    assert any(item["checks"]["normals"]["status"] == "error" for item in report["files"])
    assert any(item["status"] == "pass" for item in report["files"])


def test_trimesh_check_entry(tmp_path, capsys):
    path = tmp_path / "good.ply"
    g.trimesh.creation.box().export(path)
    code = trimesh_main(["health", str(path), "--no-progress", "--fail-on", "never"])
    assert code == 0
    assert "status: PASS" in capsys.readouterr().out


def test_report_is_json_serializable():
    report = health.check_mesh(g.trimesh.creation.box())
    assert json.loads(json.dumps(report))["status"] == "pass"
    assert set(report["checks"]) == {
        "non_manifold_edges",
        "degenerate_faces",
        "duplicate_vertices",
        "uv_overlap",
        "texture_paths",
        "normals",
        "bounds_scale",
    }


def test_entry_point_declared():
    pyproject = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject, encoding="utf-8") as file_obj:
        text = file_obj.read()
    assert 'trimesh-health = "trimesh.health:main"' in text
