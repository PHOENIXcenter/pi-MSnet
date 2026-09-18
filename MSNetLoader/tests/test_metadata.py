"""Tests for metadata-driven dataset selection and batch download."""

import functools
import http.server
import json
import logging
import os
import threading
from pathlib import Path

import pytest

import msnetloader.download as download_mod
from msnetloader import (
    download_by_metadata,
    download_datasets,
    fetch_collection_metadata,
    search_datasets,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

METADATA = {
    "PXD000001": {
        "accession": "PXD000001",
        "title": "PXD000001",
        "enzyme": "Trypsin",
        "instrument": "Orbitrap Fusion Lumos",
        "species": "Homo sapiens",
        "superkingdom": "Eukaryote",
        "fragment_method": "HCD@32.00",
        "label": "label free",
        "acquisition_method": "DDA",
        "psm_count": 1000,
        "runs_total": 10,
    },
    "PXD000003": {
        "accession": "PXD000003",
        "title": "PXD000003",
        "enzyme": "unspecific cleavage",
        "instrument": "Q Exactive HF-X",
        "species": "Mus musculus",
        "superkingdom": "Eukaryote",
        "fragment_method": "HCD@27.00",
        "label": "label free",
        "acquisition_method": "DDA",
        "psm_count": 500,
        "runs_total": 5,
    },
}
# PXD000002-Trypsin deliberately has no portal metadata (variant split).

FOLDERS = ["PXD000001", "PXD000002-Trypsin", "PXD000003"]


class _MetadataHandler(http.server.SimpleHTTPRequestHandler):
    request_count = 0

    def log_message(self, *args):  # keep test output clean
        pass

    def _count(self):
        type(self).request_count += 1

    def do_HEAD(self):
        self._count()
        path = Path(self.translate_path(self.path))
        if path.is_file():
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        self._count()
        if self.path.rstrip("/") == "/ftp/msnet":
            rows = "".join(
                f'<tr><td><a href="{folder}/">{folder}/</a></td></tr>' for folder in FOLDERS
            )
            html = (
                "<html><head><title>Index of msnet</title></head><body>"
                f"<h1>Index of msnet</h1><table>{rows}</table></body></html>"
            )
            data = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()


@pytest.fixture
def metadata_server(tmp_path, monkeypatch):
    """Local server mimicking the quantms FTP listing + portal metadata JSON."""
    root = tmp_path / "remote"
    (root / "portal" / "data" / "collections" / "msnet" / "datasets").mkdir(parents=True)

    for folder in FOLDERS:
        dataset_dir = root / "ftp" / "msnet" / folder
        dataset_dir.mkdir(parents=True)
        (dataset_dir / f"{folder}-MSNet.parquet").write_bytes(f"msnet-{folder}".encode())
        (dataset_dir / f"{folder}.dataset.parquet").write_bytes(f"dataset-{folder}".encode())

    for accession, metadata in METADATA.items():
        target = root / "portal" / "data" / "collections" / "msnet" / "datasets" / f"{accession}.json"
        target.write_text(json.dumps(metadata), encoding="utf-8")

    handler = functools.partial(_MetadataHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(download_mod, "DEFAULT_FTP_BASE", f"{base}/ftp/msnet")
    monkeypatch.setattr(download_mod, "PORTAL_BASE", f"{base}/portal")
    _MetadataHandler.request_count = 0
    try:
        yield {"base": base}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_collection_metadata(metadata_server, tmp_path):
    datasets = fetch_collection_metadata(cache_dir=tmp_path / "cache", max_age=0)

    by_accession = {item["accession"]: item for item in datasets}
    assert set(by_accession) == set(FOLDERS)

    full = by_accession["PXD000001"]
    assert full["enzyme"] == "Trypsin"
    assert full["instrument"] == "Orbitrap Fusion Lumos"
    assert full["psm_count"] == 1000

    # variant folder without portal metadata degrades to a minimal entry
    assert by_accession["PXD000002-Trypsin"] == {"accession": "PXD000002-Trypsin"}

    assert (tmp_path / "cache" / "msnet_metadata.json").exists()


def test_fetch_collection_metadata_cache(metadata_server, tmp_path):
    cache_dir = tmp_path / "cache"

    fetch_collection_metadata(cache_dir=cache_dir, max_age=86400)
    after_first = _MetadataHandler.request_count

    # fresh cache: no new requests
    fetch_collection_metadata(cache_dir=cache_dir, max_age=86400)
    assert _MetadataHandler.request_count == after_first

    # max_age=0 forces a refetch
    fetch_collection_metadata(cache_dir=cache_dir, max_age=0)
    assert _MetadataHandler.request_count > after_first


def test_search_datasets_text_filters(metadata_server, tmp_path):
    kwargs = dict(cache_dir=tmp_path / "cache", max_age=0)

    matches = search_datasets(enzyme="trypsin", **kwargs)
    assert [m["accession"] for m in matches] == ["PXD000001"]

    matches = search_datasets(enzyme=["trypsin", "unspecific"], **kwargs)
    assert [m["accession"] for m in matches] == ["PXD000001", "PXD000003"]

    # substring matching on instrument
    matches = search_datasets(instrument="Orbitrap", **kwargs)
    assert [m["accession"] for m in matches] == ["PXD000001"]

    # accession substring matches variant splits
    matches = search_datasets(accessions="PXD000002", **kwargs)
    assert [m["accession"] for m in matches] == ["PXD000002-Trypsin"]

    # combination of filters
    matches = search_datasets(acquisition_method="DDA", instrument="Q Exactive", **kwargs)
    assert [m["accession"] for m in matches] == ["PXD000003"]

    # no match
    assert search_datasets(enzyme="LysC", **kwargs) == []


def test_search_datasets_numeric_filters(metadata_server, tmp_path):
    kwargs = dict(cache_dir=tmp_path / "cache", max_age=0)

    assert [m["accession"] for m in search_datasets(min_psms=800, **kwargs)] == ["PXD000001"]
    assert [m["accession"] for m in search_datasets(max_psms=600, **kwargs)] == ["PXD000003"]
    assert [m["accession"] for m in search_datasets(max_runs=6, **kwargs)] == ["PXD000003"]
    assert [m["accession"] for m in search_datasets(min_runs=8, enzyme="Trypsin", **kwargs)] == ["PXD000001"]


def test_search_datasets_accepts_prefetched_metadata(metadata_server, tmp_path):
    datasets = fetch_collection_metadata(cache_dir=tmp_path / "cache", max_age=0)
    before = _MetadataHandler.request_count

    matches = search_datasets(datasets=datasets, enzyme="Trypsin")
    assert [m["accession"] for m in matches] == ["PXD000001"]
    assert _MetadataHandler.request_count == before


def test_download_datasets(metadata_server, tmp_path):
    paths = download_datasets(
        ["PXD000001", "PXD000003"],
        data_dir=tmp_path / "dl",
        files=["msnet"],
        progress=False,
    )

    assert len(paths) == 2
    assert (tmp_path / "dl" / "PXD000001" / "PXD000001-MSNet.parquet").read_bytes() == b"msnet-PXD000001"
    assert (tmp_path / "dl" / "PXD000003" / "PXD000003-MSNet.parquet").read_bytes() == b"msnet-PXD000003"


def test_download_datasets_empty_raises():
    with pytest.raises(ValueError):
        download_datasets([])


def test_download_by_metadata(metadata_server, tmp_path):
    paths = download_by_metadata(
        data_dir=tmp_path / "dl",
        files=["dataset"],
        enzyme="Trypsin",
        max_age=0,
        cache_dir=tmp_path / "cache",
        progress=False,
    )

    assert len(paths) == 1
    assert paths[0].name == "PXD000001.dataset.parquet"
    assert paths[0].read_bytes() == b"dataset-PXD000001"
    assert not (tmp_path / "dl" / "PXD000003").exists()


def test_download_by_metadata_requires_filter(metadata_server, tmp_path):
    with pytest.raises(ValueError, match="At least one filter"):
        download_by_metadata(data_dir=tmp_path, files=["dataset"])


def test_download_by_metadata_unknown_filter_raises(metadata_server, tmp_path):
    with pytest.raises(ValueError, match="Unknown filter"):
        download_by_metadata(data_dir=tmp_path, files=["dataset"], enzime="Trypsin")


def test_download_by_metadata_no_match_raises(metadata_server, tmp_path):
    with pytest.raises(ValueError, match="No dataset matches"):
        download_by_metadata(data_dir=tmp_path, files=["dataset"], enzyme="LysC", max_age=0)


@pytest.mark.skipif(
    not os.environ.get("MSNETLOADER_LIVE_TESTS"),
    reason="live network test; set MSNETLOADER_LIVE_TESTS=1 to run",
)
def test_metadata_filters_live(tmp_path):
    """Smoke test against the real quantms portal + FTP mirror."""
    datasets = fetch_collection_metadata(cache_dir=tmp_path, max_age=0)

    accessions = {item["accession"] for item in datasets}
    assert len(accessions) >= 100
    assert "PXD021013" in accessions

    # PXD021013: unspecific cleavage, Orbitrap Fusion Lumos, ~16.3M PSMs
    matches = search_datasets(datasets=datasets, enzyme="unspecific", instrument="Lumos", min_psms=10_000_000)
    assert any(m["accession"] == "PXD021013" for m in matches)
