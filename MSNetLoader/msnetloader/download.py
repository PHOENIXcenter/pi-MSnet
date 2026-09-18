"""Download π-MSNet dataset files for a given project accession.

Datasets are hosted by the quantms project. Two sources are supported:

* ``ftp`` (default): the official EBI FTP mirror, e.g.
  ``https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/quantms-collections/msnet/PXD021013/``
* ``browse``: the browse.quantms.org listing, e.g.
  ``https://browse.quantms.org/quantms/datasets/PXD021013/1ed4852abc60/``

Example
-------
>>> from msnetloader import download_dataset
>>> downloaded = download_dataset("PXD021013", files=["dataset", "run"])
>>> downloaded  # doctest: +SKIP
[WindowsPath('data/PXD021013/PXD021013.dataset.parquet'), ...]
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import unquote

__all__ = [
    "BROWSE_BASE",
    "DEFAULT_FTP_BASE",
    "FILE_SPECS",
    "PORTAL_BASE",
    "download_by_metadata",
    "download_dataset",
    "download_datasets",
    "fetch_collection_metadata",
    "list_remote_files",
    "search_datasets",
]

DEFAULT_FTP_BASE = (
    "https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/"
    "quantms-collections/msnet"
)
BROWSE_BASE = "https://browse.quantms.org/quantms/datasets"
PORTAL_BASE = "https://portal.quantms.org"

#: How long a cached copy of the collection metadata stays valid (seconds).
DEFAULT_METADATA_MAX_AGE = 24 * 3600

#: Files shipped for every project in the quantms/msnet collection.
#: ``filename`` may contain the ``{accession}`` placeholder.
FILE_SPECS = {
    "msnet": {
        "filename": "{accession}-MSNet.parquet",
        "description": "main PSM data used by msnetloader datasets (can be tens of GB)",
    },
    "dataset": {
        "filename": "{accession}.dataset.parquet",
        "description": "dataset metadata",
    },
    "ontology": {
        "filename": "{accession}.ontology.parquet",
        "description": "ontology annotations",
    },
    "provenance": {
        "filename": "{accession}.provenance.parquet",
        "description": "provenance metadata",
    },
    "run": {
        "filename": "{accession}.run.parquet",
        "description": "MS run metadata",
    },
    "sample": {
        "filename": "{accession}.sample.parquet",
        "description": "sample metadata",
    },
    "provenance_json": {
        "filename": "provenance.json",
        "description": "provenance information as JSON",
    },
}

_ACCESSION_RE = re.compile(r"(?:^|/)(PXD\d+|PDC\d+|IPX\d+|RPXD\d+)(?:/|$)")
_HASH_DIR_RE = re.compile(r'href="(?:[^"]*?/)?([0-9a-f]{12})/"')
_HREF_RE = re.compile(r'href="([^"?]+)"')

_USER_AGENT = "msnetloader/0.1"


def _urlopen(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    method: Optional[str] = None,
):
    """Open *url* with retries and exponential backoff."""
    request_headers = {"User-Agent": _USER_AGENT}
    if headers:
        request_headers.update(headers)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, headers=request_headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError:
            # HTTP error codes are not transient; retrying would just waste time
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt == max_retries:
                break
            time.sleep(2 ** attempt)
    raise ConnectionError(f"Failed to reach {url} after {max_retries + 1} attempts") from last_error


def _extract_accession(url: str) -> Optional[str]:
    """Try to recover the project accession (e.g. ``PXD021013``) from a URL."""
    match = _ACCESSION_RE.search(url.rstrip("/"))
    if match:
        return match.group(1)
    return None


def _resolve_input(accession_or_url: str) -> tuple[str, Optional[str]]:
    """Split user input into ``(accession, base_url)``.

    Plain accessions (``PXD021013``) yield ``base_url=None``; a full URL is kept
    as-is and its accession is extracted for naming the local data directory.
    """
    if accession_or_url.startswith(("http://", "https://")):
        base_url = accession_or_url.rstrip("/")
        accession = _extract_accession(base_url)
        if accession is None:
            segments = [s for s in base_url.split("/") if s]
            if segments and re.fullmatch(r"[0-9a-f]{12}", segments[-1]):
                segments.pop()
            accession = segments[-1] if segments else accession_or_url
        return accession, base_url
    return accession_or_url, None


def _list_directory(url: str, timeout: float, max_retries: int) -> list[str]:
    """Return the dataset file names of an HTML directory listing at *url*."""
    response = _urlopen(url, timeout=timeout, max_retries=max_retries)
    html = response.read().decode("utf-8", errors="replace")
    names = []
    for href in _HREF_RE.findall(html):
        name = unquote(href.rstrip("/").split("/")[-1])
        if name.endswith((".parquet", ".json")):
            names.append(name)
    return sorted(set(names))


def _resolve_base_url(
    accession: str,
    source: str,
    base_url: Optional[str],
    timeout: float,
    max_retries: int,
) -> str:
    """Compute the base URL hosting the files of *accession*."""
    if base_url:
        return base_url

    if source == "ftp":
        return f"{DEFAULT_FTP_BASE}/{accession}"

    if source == "browse":
        listing_url = f"{BROWSE_BASE}/{accession}/"
        html = _urlopen(listing_url, timeout=timeout, max_retries=max_retries).read().decode(
            "utf-8", errors="replace"
        )
        subdirs = sorted(set(_HASH_DIR_RE.findall(html)), reverse=True)
        if not subdirs:
            raise FileNotFoundError(
                f"No dataset version found for {accession!r} at {listing_url}. "
                f"Pass a full dataset URL as `base_url` instead."
            )
        return f"{BROWSE_BASE}/{accession}/{subdirs[0]}"

    raise ValueError(f"Unknown source {source!r}; expected 'ftp' or 'browse'")


def _remote_size(url: str, timeout: float, max_retries: int) -> Optional[int]:
    """Return the remote file size in bytes, or ``None`` if unknown."""
    try:
        response = _urlopen(url, timeout=timeout, max_retries=max_retries, method="HEAD")
        length = response.headers.get("Content-Length")
        if length is not None:
            return int(length)
    except Exception:
        pass

    # Some servers reject HEAD; fall back to a 1-byte ranged GET.
    try:
        response = _urlopen(url, headers={"Range": "bytes=0-0"}, timeout=timeout, max_retries=max_retries)
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)\s*$", content_range)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def _format_bytes(num_bytes: Union[int, float]) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _progress_line(downloaded: int, total: Optional[int]) -> str:
    if total:
        percent = 100.0 * downloaded / total
        return f"\r    {_format_bytes(downloaded)} / {_format_bytes(total)} ({percent:5.1f}%)"
    return f"\r    {_format_bytes(downloaded)} downloaded"


def _download_one(
    url: str,
    dest: Path,
    force: bool,
    resume: bool,
    progress: bool,
    chunk_size: int,
    timeout: float,
    max_retries: int,
) -> Path:
    """Download a single file, resuming from ``dest.part`` when possible."""
    part_path = dest.with_suffix(dest.suffix + ".part")
    remote_size = _remote_size(url, timeout, max_retries)

    if dest.exists() and not force:
        local_size = dest.stat().st_size
        if remote_size is None or local_size == remote_size:
            print(f"  {dest.name} already present, skipping")
            return dest
        print(f"  {dest.name} present but incomplete ({local_size} != {remote_size} bytes), re-downloading")

    if dest.exists() and force:
        dest.unlink()
        part_path.unlink(missing_ok=True)

    downloaded = 0
    headers = {}
    if resume and part_path.exists():
        downloaded = part_path.stat().st_size
        if remote_size is not None and downloaded >= remote_size:
            part_path.unlink()
            downloaded = 0
        else:
            headers["Range"] = f"bytes={downloaded}-"

    response = _urlopen(url, headers=headers, timeout=timeout, max_retries=max_retries)
    append_mode = response.status == 206 and downloaded > 0

    if not append_mode:
        downloaded = 0
        mode = "wb"
    else:
        mode = "ab"

    print(f"  downloading {dest.name}" + (f" ({_format_bytes(remote_size)})" if remote_size else ""))
    part_path.parent.mkdir(parents=True, exist_ok=True)
    with part_path.open(mode) as handle, response:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if progress:
                print(_progress_line(downloaded, remote_size), end="", flush=True)
    if progress:
        print()

    if remote_size is not None and downloaded != remote_size:
        raise OSError(
            f"Incomplete download for {dest.name}: got {downloaded} bytes, expected {remote_size}. "
            f"Re-run the same command to resume."
        )

    part_path.replace(dest)
    return dest


def download_dataset(
    accession: str,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    files: Union[str, Sequence[str]] = "all",
    source: str = "ftp",
    base_url: Optional[str] = None,
    force: bool = False,
    resume: bool = True,
    progress: bool = True,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[Path]:
    """Download the π-MSNet dataset files of a project accession.

    Parameters
    ----------
    accession:
        Project accession such as ``"PXD021013"``, or a full dataset URL such as
        ``"https://browse.quantms.org/quantms/datasets/PXD021013/1ed4852abc60/"``.
    data_dir:
        Local destination directory. Defaults to ``<cwd>/data/<accession>``.
    files:
        Which files to download: ``"all"`` or a subset of
        ``["msnet", "dataset", "ontology", "provenance", "run", "sample", "provenance_json"]``.
    source:
        ``"ftp"`` (default, official EBI mirror) or ``"browse"`` (browse.quantms.org).
        Ignored when *accession* is a full URL.
    base_url:
        Explicit base URL hosting the files. Overrides *source*.
    force:
        Re-download files that already exist locally.
    resume:
        Resume interrupted downloads from the ``*.part`` file.
    progress:
        Print a textual progress indicator while downloading.
    chunk_size, timeout, max_retries:
        Tuning knobs for the HTTP transfer.

    Returns
    -------
    list[Path]
        Paths of the files downloaded (or already present locally).
    """
    if isinstance(files, str):
        files = [files]
    selected = set(files)
    if "all" in selected:
        selected = set(FILE_SPECS)
    unknown = selected - set(FILE_SPECS)
    if unknown:
        raise ValueError(f"Unknown file type(s) {sorted(unknown)}; choose from {sorted(FILE_SPECS)}")

    accession, parsed_base_url = _resolve_input(accession)
    if base_url is None:
        base_url = parsed_base_url
    resolved_base = _resolve_base_url(accession, source, base_url, timeout, max_retries)

    if data_dir is None:
        data_dir = Path.cwd() / "data" / accession
    data_dir = Path(data_dir)

    results: list[Path] = []
    for key in sorted(selected):
        filename = FILE_SPECS[key]["filename"].format(accession=accession)
        url = f"{resolved_base}/{filename}"
        dest = data_dir / filename
        results.append(
            _download_one(
                url, dest, force=force, resume=resume, progress=progress,
                chunk_size=chunk_size, timeout=timeout, max_retries=max_retries,
            )
        )
    return results


def list_remote_files(
    accession: str,
    source: str = "ftp",
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[str]:
    """List the files available for a project accession on the remote host.

    Returns the base file names, e.g.
    ``["PXD021013-MSNet.parquet", "PXD021013.dataset.parquet", ...]``.
    """
    accession, parsed_base_url = _resolve_input(accession)
    if base_url is None:
        base_url = parsed_base_url
    resolved_base = _resolve_base_url(accession, source, base_url, timeout, max_retries)
    return _list_directory(resolved_base, timeout, max_retries)


def _fetch_json(url: str, timeout: float, max_retries: int) -> dict:
    response = _urlopen(url, timeout=timeout, max_retries=max_retries)
    return json.loads(response.read().decode("utf-8"))


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "msnetloader"


def _list_collection_folders(timeout: float, max_retries: int) -> list[str]:
    """Enumerate the dataset folders of the quantms/msnet FTP collection."""
    url = f"{DEFAULT_FTP_BASE}/"
    response = _urlopen(url, timeout=timeout, max_retries=max_retries)
    html = response.read().decode("utf-8", errors="replace")
    folders = []
    for href in _HREF_RE.findall(html):
        name = href.rstrip("/")
        if not name or name.startswith(("/", "?")) or "/" in name or name in ("..", "."):
            continue
        folders.append(unquote(name))
    return sorted(set(folders))


def fetch_collection_metadata(
    collection: str = "msnet",
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    max_age: Optional[float] = DEFAULT_METADATA_MAX_AGE,
    timeout: float = 60.0,
    max_retries: int = 3,
    workers: int = 8,
) -> list[dict]:
    """Return the metadata of every dataset in a quantms collection.

    The dataset folders are enumerated from the quantms FTP mirror and each
    dataset's metadata JSON (``enzyme``, ``instrument``, ``species``,
    ``superkingdom``, ``fragment_method``, ``label``, ``acquisition_method``,
    ``psm_count``, ``runs_total``, ...) is fetched from the quantms portal.

    Parameters
    ----------
    collection:
        Collection name; per-dataset metadata is read from
        ``<portal>/data/collections/<collection>/datasets/<accession>.json``.
    cache_dir:
        Directory holding the metadata cache file. Defaults to
        ``~/.cache/msnetloader``.
    max_age:
        Maximum age of the cache in seconds before refetching
        (default: 24 h). ``None`` or ``0`` always refetches.
    timeout, max_retries:
        Tuning knobs for the HTTP requests.
    workers:
        Number of parallel metadata fetches.

    Returns
    -------
    list[dict]
        One metadata dict per dataset, sorted by accession. Datasets without
        a portal metadata entry (e.g. ``PXD000865-Chymotrypsin`` splits) are
        returned as ``{"accession": <name>}``.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_file = cache_dir / f"{collection}_metadata.json"

    if max_age is not None and max_age > 0 and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - float(cache["generated_at"]) < max_age:
                return cache["datasets"]
        except (ValueError, KeyError, OSError):
            pass

    folders = _list_collection_folders(timeout, max_retries)
    if not folders:
        raise ConnectionError(f"No dataset folders found at {DEFAULT_FTP_BASE}/")

    def fetch_one(folder: str) -> dict:
        url = f"{PORTAL_BASE}/data/collections/{collection}/datasets/{folder}.json"
        try:
            metadata = _fetch_json(url, timeout, max_retries)
            metadata.setdefault("accession", folder)
            return metadata
        except Exception:
            # variant folders (e.g. PXD009449-CID) have no portal entry
            return {"accession": folder}

    print(f"Fetching metadata for {len(folders)} dataset(s) from {PORTAL_BASE} ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        datasets = list(pool.map(fetch_one, folders))

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"generated_at": time.time(), "datasets": datasets}, indent=2),
        encoding="utf-8",
    )
    return datasets


def _matches(value, pattern: Union[str, Sequence[str]]) -> bool:
    """Case-insensitive substring match; a list of patterns matches if any does."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    text = str(value).lower()
    return any(str(p).lower() in text for p in patterns)


#: Filters understood by :func:`search_datasets` and :func:`download_by_metadata`.
SEARCHABLE_FILTERS = frozenset(
    {
        "accessions",
        "acquisition_method",
        "enzyme",
        "fragment_method",
        "instrument",
        "label",
        "max_psms",
        "max_runs",
        "min_psms",
        "min_runs",
        "species",
        "superkingdom",
    }
)


def search_datasets(
    datasets: Optional[Sequence[dict]] = None,
    collection: str = "msnet",
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    max_age: Optional[float] = DEFAULT_METADATA_MAX_AGE,
    enzyme: Optional[Union[str, Sequence[str]]] = None,
    instrument: Optional[Union[str, Sequence[str]]] = None,
    species: Optional[Union[str, Sequence[str]]] = None,
    superkingdom: Optional[Union[str, Sequence[str]]] = None,
    fragment_method: Optional[Union[str, Sequence[str]]] = None,
    label: Optional[Union[str, Sequence[str]]] = None,
    acquisition_method: Optional[Union[str, Sequence[str]]] = None,
    accessions: Optional[Union[str, Sequence[str]]] = None,
    min_psms: Optional[int] = None,
    max_psms: Optional[int] = None,
    min_runs: Optional[int] = None,
    max_runs: Optional[int] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[dict]:
    """Filter the collection metadata by experimental properties.

    Text filters (``enzyme``, ``instrument``, ``species``, ``superkingdom``,
    ``fragment_method``, ``label``, ``acquisition_method``, ``accessions``)
    match case-insensitively on substrings, so ``instrument="Orbitrap"``
    matches every Orbitrap model and ``accessions="PXD014877"`` matches all
    species splits of that project. Numeric filters bound ``psm_count`` and
    ``runs_total``.

    Parameters
    ----------
    datasets:
        Pre-fetched metadata (see :func:`fetch_collection_metadata`). When
        ``None`` the metadata is fetched (and cached) automatically.
    collection, cache_dir, max_age, timeout, max_retries:
        Passed to :func:`fetch_collection_metadata` when *datasets* is None.

    Returns
    -------
    list[dict]
        The metadata dicts of the matching datasets, sorted by accession.
    """
    if datasets is None:
        datasets = fetch_collection_metadata(
            collection=collection, cache_dir=cache_dir, max_age=max_age,
            timeout=timeout, max_retries=max_retries,
        )

    text_filters = {
        "enzyme": enzyme,
        "instrument": instrument,
        "species": species,
        "superkingdom": superkingdom,
        "fragment_method": fragment_method,
        "label": label,
        "acquisition_method": acquisition_method,
        "accession": accessions,
    }

    def keep(item: dict) -> bool:
        for field, pattern in text_filters.items():
            if pattern is not None and not _matches(item.get(field), pattern):
                return False
        psm_count = item.get("psm_count")
        if min_psms is not None and not isinstance(psm_count, (int, float)):
            return False
        if max_psms is not None and not isinstance(psm_count, (int, float)):
            return False
        if isinstance(psm_count, (int, float)):
            if min_psms is not None and psm_count < min_psms:
                return False
            if max_psms is not None and psm_count > max_psms:
                return False
        runs_total = item.get("runs_total")
        if min_runs is not None and not isinstance(runs_total, (int, float)):
            return False
        if max_runs is not None and not isinstance(runs_total, (int, float)):
            return False
        if isinstance(runs_total, (int, float)):
            if min_runs is not None and runs_total < min_runs:
                return False
            if max_runs is not None and runs_total > max_runs:
                return False
        return True

    return [item for item in datasets if keep(item)]


def download_datasets(
    accessions: Sequence[str],
    data_dir: Optional[Union[str, os.PathLike]] = None,
    files: Union[str, Sequence[str]] = "all",
    source: str = "ftp",
    force: bool = False,
    resume: bool = True,
    progress: bool = True,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[Path]:
    """Download the dataset files of several project accessions.

    Thin wrapper around :func:`download_dataset`; each dataset lands in
    ``<data_dir>/<accession>/`` (``data_dir`` defaults to ``<cwd>/data``).
    """
    accessions = list(accessions)
    if not accessions:
        raise ValueError("accessions must not be empty")

    base_dir = Path.cwd() / "data" if data_dir is None else Path(data_dir)
    downloaded: list[Path] = []
    for accession in accessions:
        downloaded += download_dataset(
            accession,
            data_dir=base_dir / accession,
            files=files,
            source=source,
            force=force,
            resume=resume,
            progress=progress,
            chunk_size=chunk_size,
            timeout=timeout,
            max_retries=max_retries,
        )
    return downloaded


def download_by_metadata(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    files: Union[str, Sequence[str]] = "all",
    source: str = "ftp",
    collection: str = "msnet",
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    max_age: Optional[float] = DEFAULT_METADATA_MAX_AGE,
    force: bool = False,
    resume: bool = True,
    progress: bool = True,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
    **filters: Union[str, Sequence[str], int, None],
) -> list[Path]:
    """Search datasets by experimental metadata and download the matches.

    Filters are the same as :func:`search_datasets`, e.g.
    ``enzyme="Trypsin"``, ``instrument="Orbitrap Fusion Lumos"``,
    ``species="Homo sapiens"``, ``min_psms=1_000_000``. At least one filter is
    required, and the matched datasets are printed before downloading.

    >>> from msnetloader import download_by_metadata
    >>> download_by_metadata(  # doctest: +SKIP
    ...     files=["dataset", "run"],
    ...     instrument="Orbitrap",
    ...     enzyme="Trypsin",
    ... )
    """
    unknown = set(filters) - SEARCHABLE_FILTERS
    if unknown:
        raise ValueError(f"Unknown filter(s) {sorted(unknown)}; choose from {sorted(SEARCHABLE_FILTERS)}")
    if not filters:
        raise ValueError("At least one filter is required; pass none to download every dataset")

    datasets = search_datasets(
        collection=collection, cache_dir=cache_dir, max_age=max_age,
        timeout=timeout, max_retries=max_retries, **filters,
    )
    if not datasets:
        raise ValueError(f"No dataset matches the filters: {filters}")

    print(f"Matched {len(datasets)} dataset(s):")
    for item in datasets:
        print(
            f"  {item.get('accession')} | enzyme={item.get('enzyme')} | "
            f"instrument={item.get('instrument')} | psms={item.get('psm_count')}"
        )

    return download_datasets(
        [item["accession"] for item in datasets],
        data_dir=data_dir,
        files=files,
        source=source,
        force=force,
        resume=resume,
        progress=progress,
        chunk_size=chunk_size,
        timeout=timeout,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    print("Files available for PXD021013:")
    for name in list_remote_files("PXD021013"):
        print(" -", name)
