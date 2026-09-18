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

import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import unquote

__all__ = [
    "BROWSE_BASE",
    "DEFAULT_FTP_BASE",
    "FILE_SPECS",
    "download_dataset",
    "list_remote_files",
]

DEFAULT_FTP_BASE = (
    "https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/"
    "quantms-collections/msnet"
)
BROWSE_BASE = "https://browse.quantms.org/quantms/datasets"

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


if __name__ == "__main__":
    print("Files available for PXD021013:")
    for name in list_remote_files("PXD021013"):
        print(" -", name)
