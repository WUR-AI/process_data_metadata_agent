"""
pangaea_sampler.py
==================
Stratified random sampler for PANGAEA datasets across all 15 official topics.

Abstract
--------
Queries the PANGAEA search API using the ``topic:`` field prefix inside the
``q`` parameter (e.g. ``q='topic:Oceans'``) to sample datasets stratified by
domain. For each topic a random page offset is chosen so no single region of
the catalogue is over-represented.

Only datasets that pass ALL of the following are retained:

* Non-collection (individual tabular datasets)
* Open licence: CC-BY or CC0 (no NC / ND)
* Non-empty abstract (``ds.abstract`` is not None / blank)
* Non-empty keywords (``ds.keywords`` is a non-empty list)

Because those quality checks require loading each ``PanDataSet``, the sampler
over-draws from the search results and discards failures, continuing until the
per-topic quota is filled or the catalogue is exhausted.

Keywords
--------
PANGAEA, sampling, stratification, metadata, benchmark, earth science

License
-------
CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Usage
-----
    python pangaea_sampler.py                     # 100 DOIs, JSON to stdout
    python pangaea_sampler.py --n 50 --seed 7
    python pangaea_sampler.py --out sample.json
    python pangaea_sampler.py --no-load           # skip PanDataSet validation
                                                  # (faster, no quality filter)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import datetime
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PANGAEA's 15 canonical topic names. "Atomosphere" is their own typo.
TOPICS: list[str] = [
    "Agriculture",
    "Atmosphere",
    "Biological Classification",
    "Biosphere",
    "Chemistry",
    "Cryosphere",
    "Ecology",
    "Fisheries",
    "Geophysics",
    "Human Dimensions",
    "Lakes & Rivers",
    "Land Surface",
    "Lithosphere",
    "Oceans",
    "Paleontology",
]

# OPEN_LICENSES: tuple[str, ...] = ("CC-BY", "CC0", "CC BY")

SEARCH_URL = "https://www.pangaea.de/advanced/search.php"
PAGE_SIZE = 20    # results per API call; keep ≤ 50 to avoid timeouts
DELAY = 0.11       # seconds between requests (PANGAEA fair-use policy)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(params: dict) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=(5, 15))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"  [warn] attempt {attempt + 1} failed: {exc}")
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Topic querying
# ---------------------------------------------------------------------------

def topic_total(topic: str) -> int:
    """Return total search-result count for a topic."""
    data = _get({"q": f"topic:{topic}", "count": 1, "offset": 0})
    return data.get("totalCount", 0) if data else 0


def _iter_pages(topic: str, total: int) -> Iterator[list[dict]]:
    """
    Yield pages of non-collection search results in random offset order
    until all offsets have been visited.
    """
    if total == 0:
        return

    max_offset = max(0, total - PAGE_SIZE)
    # Generate all possible page-start positions and shuffle them
    step = PAGE_SIZE
    offsets = list(range(0, max_offset + 1, step))
    random.shuffle(offsets)

    for offset in offsets:
        data = _get({"q": f"topic:{topic}", "count": PAGE_SIZE, "offset": offset})
        time.sleep(DELAY)
        if data is None:
            continue
        results = [r for r in data.get("results", []) if r.get("type") != "collection"]
        if results:
            yield results


# ---------------------------------------------------------------------------
# Dataset quality check
# ---------------------------------------------------------------------------

def _is_open(licence: str | None) -> bool:
    if not licence:
        return False
    upper = licence.label.upper()
    # return any(tag in upper for tag in OPEN_LICENSES)
    return any(upper.startswith(tag) for tag in ['CC'])


def _load_and_validate(doi: str) -> bool:
    """
    Load a PanDataSet and return True only if it passes quality checks:
      - open licence (CC-BY or CC0)
      - non-empty abstract
      - non-empty keywords list
    """
    try:
        import pangaeapy.pandataset as pds
        ds = pds.PanDataSet(doi, include_data=False)  # skip data download for speed
    except Exception as exc:
        print(f"    [skip] {doi}: load error — {exc}")
        return False

    if not _is_open(ds.licence):
        if ds.licence:
            failure = f'license={ds.licence.label!r}'
        else:
            failure = 'license=None'
        return False, failure
    if not ds.abstract or not ds.abstract.strip():
        return False, 'no abstract'
    if not ds.keywords:
        return False, 'no keywords'

    return True, None


# ---------------------------------------------------------------------------
# Quota allocation
# ---------------------------------------------------------------------------

def allocate_quota(totals: dict[str, int], n: int, distribute_evenly=True) -> dict[str, int]:
    """
    Proportional allocation: each non-empty topic gets ≥ 1 slot;
    remainder distributed proportionally to catalogue size.
    """
    active = {t: c for t, c in totals.items() if c > 0}
    if not active:
        raise ValueError("No datasets found for any topic.")

    if n <= len(active):
        return {t: 1 for t in list(active)[:n]}

    if distribute_evenly:
        quota: dict[str, int] = {t: n // len(active) for t in active}
        remaining = n - sum(quota.values())
        for topic in sorted(active, key=active.__getitem__, reverse=True):
            if remaining <= 0:
                break
            quota[topic] += 1
            remaining -= 1
    else:
        quota: dict[str, int] = {t: 1 for t in active}
        remaining = n - len(active)
        total_ds = sum(active.values())

        for topic, count in active.items():
            quota[topic] += math.floor(remaining * count / total_ds)

        # Fix rounding shortfall
        deficit = n - sum(quota.values())
        for topic in sorted(active, key=active.__getitem__, reverse=True):
            if deficit <= 0:
                break
            quota[topic] += 1
            deficit -= 1

    return quota


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def sample_topic(
    topic: str,
    total: int,
    k: int,
    seen_dois: set[str],
    validate: bool = True,
) -> list[str]:
    """
    Collect k unique, quality-passing DOIs from a topic.

    Iterates pages in random order; within each page shuffles results.
    Continues until quota is met or the catalogue is exhausted.
    """
    collected: list[str] = []
    n_failed = 0

    for page in _iter_pages(topic, total):
        if len(collected) >= k:
            break

        random.shuffle(page)
        for result in page:
            if len(collected) >= k:
                break

            doi = result.get("URI", "")
            if not doi or doi in seen_dois:
                continue

            if validate:
                time.sleep(DELAY)  # extra delay for the PanDataSet call
                is_valid, failure_reason = _load_and_validate(doi)
                if not is_valid: 
                    print(f"    [skip ({len(collected)}/{k})] {doi}: {failure_reason}")
                    if failure_reason in ['abstract', 'keywords']:
                        n_failed += 1
                    continue

            seen_dois.add(doi)
            collected.append(doi)
            print(f"    [{len(collected)}/{k}] {doi}")

    if len(collected) < k:
        print(f"  [warn] {topic}: only collected {len(collected)}/{k}")

    return collected, n_failed


def sample(
    n: int = 100,
    seed: int | None = None,
    validate: bool = True,
) -> dict[str, list[str]]:
    """
    Return ``{topic: [doi, ...]}`` with ~n total DOIs stratified across topics.

    Parameters
    ----------
    n : int
        Target number of datasets.
    seed : int | None
        Random seed for reproducibility.
    validate : bool
        If True (default), load each PanDataSet and filter by licence,
        abstract, and keywords.  Set False for a fast dry-run.
    """
    if seed is not None:
        random.seed(seed)

    print("── Fetching topic totals ──────────────────────────────")
    totals: dict[str, int] = {}
    for topic in TOPICS:
        totals[topic] = topic_total(topic)
        print(f"  {topic}: {totals[topic]:,}")
        time.sleep(DELAY)

    quota = allocate_quota(totals, n)
    print(f"\n── Quota allocation ───────────────────────────────────")
    for t, q in quota.items():
        print(f"  {t}: {q}")
    print(f"  TOTAL TARGET: {sum(quota.values())}\n")

    seen_dois: set[str] = set()
    result: dict[str, list[str]] = {}

    dict_n_failed = {}
    for topic, k in quota.items():
        print(f"\n── Sampling '{topic}' (need {k}) ─────────────────────")
        dois, n_failed = sample_topic(topic, totals[topic], k, seen_dois, validate=validate)
        result[topic] = dois
        dict_n_failed[topic] = {topic: n_failed}

        write_json(result, dict_n_failed, out_path='tmp_save.json')
    return result, dict_n_failed

def write_json(result, dict_n_failed, out_path: str | None = None) -> None:
    flat = [doi for dois in result.values() for doi in dois]
    output = {"by_topic": result, "all_dois": flat, "total": len(flat), "n_failed": dict_n_failed}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Written to {out_path}")
    else:
        print(json.dumps(output, indent=2))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Target sample size (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--no-load", action="store_true",
                        help="Skip PanDataSet validation (faster, no quality filter)")
    args = parser.parse_args()

    result, dict_n_failed = sample(n=args.n, seed=args.seed, validate=not args.no_load)
    
    print(f"\n── Done: {sum(len(dois) for dois in result.values())} datasets collected ──────────────────")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.out:
        out_path = args.out.rstrip('.json') + f"_{timestamp}.json"
    else:
        out_path = f'samples_{timestamp}.json'
    write_json(result, dict_n_failed, out_path=out_path)


if __name__ == "__main__":
    main()