"""Tests for the dataset download utilities.

Most tests run against a local HTTP server so the suite is offline-friendly;
the live quantms test is skipped unless ``MSNETLOADER_LIVE_TESTS=1``.
"""

import functools
import http.server
import logging
import os
import re
import threading
from pathlib import Path

import pytest

import msnetloader.download as download_mod
from msnetloader import download_dataset, list_remote_files

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ACCESSIONS = "PXD999999"


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files, supporting HEAD and byte-range GET requests."""

    served_ranges = []

    def log_message(self, *args):  # keep test output clean
        pass

    def do_HEAD(self):
        path = Path(self.translate_path(self.path))
        if path.is_file():
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            super().do_GET()
            return
        if not path.is_file():
            self.send_error(404)
            return

        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header:
            type(self).served_ranges.append(range_header)
            match = re.match(r"bytes=(\d+)-(\d*)$", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) + 1 if match.group(2) else len(data)
                chunk = data[start:end]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(chunk)
                return

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def remote_server(tmp_path):
    """Local HTTP server mimicking the quantms dataset layout."""
    root = tmp_path / "remote"
    nested = root / "quantms" / "datasets" / ACCESSIONS / "abc123456789"
    nested.mkdir(parents=True)

    files = {
        "PXD999999-MSNet.parquet": b"x" * (2 * 1024 * 1024 + 123),
        "PXD999999.dataset.parquet": b"PAR1" + b"dataset-metadata" * 10,
        "PXD999999.ontology.parquet": b"ontology-annotations",
        "PXD999999.provenance.parquet": b"provenance-metadata",
        "PXD999999.run.parquet": b"run-metadata",
        "PXD999999.sample.parquet": b"sample-metadata",
        "provenance.json": b'{"project": "PXD999999"}',
    }
    for name, content in files.items():
        (nested / name).write_bytes(content)

    handler = functools.partial(_RangeHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{server.server_port}",
            "base": f"http://127.0.0.1:{server.server_port}/quantms/datasets/{ACCESSIONS}/abc123456789",
            "files": files,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_all_files(remote_server, tmp_path):
    paths = download_dataset(
        ACCESSIONS, data_dir=tmp_path, files="all", base_url=remote_server["base"], progress=False
    )

    assert len(paths) == len(remote_server["files"])
    for name, content in remote_server["files"].items():
        assert (tmp_path / name).read_bytes() == content


def test_download_selected_files(remote_server, tmp_path):
    paths = download_dataset(
        ACCESSIONS, data_dir=tmp_path, files=["dataset", "run"],
        base_url=remote_server["base"], progress=False,
    )

    assert sorted(p.name for p in paths) == ["PXD999999.dataset.parquet", "PXD999999.run.parquet"]
    assert (tmp_path / "PXD999999.dataset.parquet").read_bytes() == remote_server["files"]["PXD999999.dataset.parquet"]
    assert (tmp_path / "PXD999999-MSNet.parquet").exists() is False


def test_download_from_full_url(remote_server, tmp_path):
    """A full dataset URL must be accepted and used as the base URL."""
    paths = download_dataset(
        remote_server["base"] + "/", data_dir=tmp_path, files=["provenance_json"], progress=False
    )

    assert paths[0].name == "provenance.json"
    assert paths[0].read_bytes() == remote_server["files"]["provenance.json"]


def test_download_skips_existing(remote_server, tmp_path):
    kwargs = dict(data_dir=tmp_path, files=["dataset"], base_url=remote_server["base"], progress=False)
    download_dataset(ACCESSIONS, **kwargs)

    target = tmp_path / "PXD999999.dataset.parquet"
    mtime = target.stat().st_mtime_ns

    paths = download_dataset(ACCESSIONS, **kwargs)
    assert paths[0] == target
    assert target.stat().st_mtime_ns == mtime


def test_download_force_replaces(remote_server, tmp_path):
    kwargs = dict(data_dir=tmp_path, files=["dataset"], base_url=remote_server["base"], progress=False)
    download_dataset(ACCESSIONS, **kwargs)

    target = tmp_path / "PXD999999.dataset.parquet"
    target.write_bytes(b"corrupted")

    download_dataset(ACCESSIONS, force=True, **kwargs)
    assert target.read_bytes() == remote_server["files"]["PXD999999.dataset.parquet"]


def test_download_resumes_from_part_file(remote_server, tmp_path):
    content = remote_server["files"]["PXD999999-MSNet.parquet"]
    target = tmp_path / "PXD999999-MSNet.parquet"
    part = target.with_suffix(target.suffix + ".part")
    part.write_bytes(content[:100])

    _RangeHandler.served_ranges.clear()
    download_dataset(ACCESSIONS, data_dir=tmp_path, files=["msnet"], base_url=remote_server["base"], progress=False)

    assert target.read_bytes() == content
    assert not part.exists()
    assert _RangeHandler.served_ranges == ["bytes=100-"]


def test_list_remote_files(remote_server):
    names = list_remote_files(ACCESSIONS, base_url=remote_server["base"])
    assert set(names) == set(remote_server["files"])


def test_browse_source_resolution(remote_server, tmp_path, monkeypatch):
    """source='browse' must discover the version subdirectory from the listing."""
    monkeypatch.setattr(download_mod, "BROWSE_BASE", f"{remote_server['url']}/quantms/datasets")

    download_dataset(
        ACCESSIONS, data_dir=tmp_path, files=["dataset"], source="browse", progress=False
    )
    assert (tmp_path / "PXD999999.dataset.parquet").read_bytes() == remote_server["files"]["PXD999999.dataset.parquet"]


def test_unknown_file_type_raises(remote_server, tmp_path):
    with pytest.raises(ValueError):
        download_dataset(ACCESSIONS, data_dir=tmp_path, files=["nonsense"], base_url=remote_server["base"])


def test_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        download_dataset(ACCESSIONS, data_dir=tmp_path, files=["dataset"], source="s3")


@pytest.mark.skipif(
    not os.environ.get("MSNETLOADER_LIVE_TESTS"),
    reason="live network test; set MSNETLOADER_LIVE_TESTS=1 to run",
)
def test_download_from_quantms_ftp(tmp_path):
    """Smoke test against the real quantms FTP mirror (small metadata file)."""
    paths = download_dataset("PXD021013", data_dir=tmp_path, files=["dataset"], progress=False)
    assert len(paths) == 1
    assert paths[0].stat().st_size > 0
    assert paths[0].read_bytes()[:4] == b"PAR1"
