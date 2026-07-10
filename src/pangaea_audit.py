"""
pangaea_audit.py
================
Metadata coverage audit for PANGAEA datasets, stratified by topic.

Abstract
--------
Unlike ``pangaea_sampler.py``, which *filters* datasets down to those meeting
a quality bar, this module *measures* how often each metadata property is
present.  It draws a fixed number of datasets per topic uniformly at random
(no quality filtering, so the sample is representative of the catalogue) and
records, for every dataset, whether each Croissant-relevant property is
populated.  Results are aggregated into a ``pandas.DataFrame`` of coverage
fractions with topics as rows and properties as columns.

Properties audited
------------------
name              ds.title              non-empty string
description       ds.abstract           non-empty string (after strip)
keywords          ds.keywords           non-empty list
license           ds.licence.name       licence object present with a name
temporalCoverage  (mintime, maxtime)    at least one bound present
spatialCoverage   ds.geometryextent     non-empty dict
supplement_to     ds.supplement_to      non-empty dict

Keywords
--------
PANGAEA, metadata coverage, audit, stratified sampling, Croissant

License
-------
CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Usage
-----
    python pangaea_audit.py --per-topic 30 --seed 42 --out coverage.csv
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time

import pandas as pd
import pangaeapy.pandataset as pds

from pangaea_sampler_postcheck import (
    TOPICS,
    DELAY,
    FOLDER_SAVE,
    _iter_pages,
    topic_total,
)

# ---------------------------------------------------------------------------
# Property presence checks
# ---------------------------------------------------------------------------

# Order determines column order in the output DataFrame.
PROPERTIES: tuple[str, ...] = (
    "name",
    "description",
    "keywords",
    "license",
    "temporalCoverage",
    "spatialCoverage",
    "supplement_to",
)


def _has_name(ds) -> bool:
    return bool(ds.title and str(ds.title).strip())


def _has_description(ds) -> bool:
    return bool(ds.abstract and ds.abstract.strip())


def _has_keywords(ds) -> bool:
    return bool(ds.keywords)


def _has_license(ds) -> bool:
    # ds.licence is a PanLicence object (label / name / URI) or None.
    return ds.licence is not None and bool(getattr(ds.licence, "name", None))


def _has_temporal(ds) -> bool:
    # Present if EITHER bound is set.  Use `and` instead of `or` below if you
    # require a fully bounded interval.
    return ds.mintimeextent is not None or ds.maxtimeextent is not None


def _has_spatial(ds) -> bool:
    # geometryextent is {} when the dataset carries no georeference.
    return bool(ds.geometryextent)


def _has_supplement_to(ds) -> bool:
    # supplement_to is {} unless the dataset supplements a publication.
    return bool(ds.supplement_to)


CHECKS = {
    "name": _has_name,
    "description": _has_description,
    "keywords": _has_keywords,
    "license": _has_license,
    "temporalCoverage": _has_temporal,
    "spatialCoverage": _has_spatial,
    "supplement_to": _has_supplement_to,
}


def audit_dataset(doi: str) -> dict | None:
    """
    Load one dataset and return a flat record of property presence.

    Returns None if the dataset fails to load or is a collection, so that
    such records never enter the coverage denominator.
    """
    try:
        ds = pds.PanDataSet(doi, include_data=False)
    except Exception as exc:
        print(f"    [error] {doi}: {exc}")
        return None

    if getattr(ds, "isCollection", False):
        return None

    record = {"doi": doi}
    record.update({prop: check(ds) for prop, check in CHECKS.items()})

    # Keep the raw licence name so licence coverage can be broken down later.
    record["license_name"] = getattr(ds.licence, "name", None) if ds.licence else None
    return record


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def audit_topic(
    topic: str,
    total: int,
    n_per_topic: int,
    seen_dois: set[str],
) -> tuple[list[dict], int]:
    """
    Draw ``n_per_topic`` loadable, non-collection datasets from ``topic`` and
    audit each one.  No quality filtering is applied, so presence rates are
    unbiased estimates of catalogue coverage.

    Returns (records, n_skipped) where n_skipped counts load errors and
    collections encountered along the way.
    """
    records: list[dict] = []
    n_skipped = 0

    for page in _iter_pages(topic, total):
        if len(records) >= n_per_topic:
            break

        random.shuffle(page)
        for result in page:
            if len(records) >= n_per_topic:
                break

            doi = result.get("URI", "")
            if not doi or doi in seen_dois:
                continue
            seen_dois.add(doi)

            time.sleep(DELAY)
            record = audit_dataset(doi)
            if record is None:
                n_skipped += 1
                continue

            record["topic"] = topic
            records.append(record)
            print(f"   [audit] ({len(records)}/{n_per_topic}, {topic}) {doi}")

    if len(records) < n_per_topic:
        print(f"  [warn] {topic}: audited {len(records)}/{n_per_topic} "
              f"(catalogue exhausted)")

    return records, n_skipped


def audit(
    n_per_topic: int = 30,
    seed: int | None = 42,
    topics: list[str] | None = None,
    dedupe_across_topics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit ``n_per_topic`` randomly drawn datasets from each topic.

    Parameters
    ----------
    dedupe_across_topics : bool
        PANGAEA topics overlap: one dataset can be indexed under several.
        If True, a dataset drawn for an earlier topic is excluded from later
        ones, which makes the pooled ``ALL`` row a clean sample but distorts
        per-topic coverage (later topics are drawn from a depleted pool).
        Default False: each topic is sampled independently, so per-topic rows
        are unbiased and only ``ALL`` may double-count shared datasets.

    Returns
    -------
    per_dataset : pd.DataFrame
        One row per audited dataset; boolean columns for each property.
    coverage : pd.DataFrame
        Topics as rows, properties as columns; each cell is the fraction of
        audited datasets in that topic possessing the property.  Trailing
        columns ``n_checked`` (sample size / denominator) and ``n_skipped`` (load
        errors and collections encountered).  The ``ALL`` row pools datasets.
    """
    if seed is not None:
        random.seed(seed)

    topics = topics or TOPICS

    print("── Fetching topic totals ──────────────────────────────")
    totals: dict[str, int] = {}
    for topic in topics:
        totals[topic] = topic_total(topic)
        print(f"  {topic}: {totals[topic]:,}")
        time.sleep(DELAY)

    global_seen: set[str] = set()
    all_records: list[dict] = []
    skipped: dict[str, int] = {}

    for topic in topics:
        if totals[topic] == 0:
            print(f"\n[skip] {topic}: no datasets")
            skipped[topic] = 0
            continue

        print(f"\n── Auditing '{topic}' (n={n_per_topic}) ──────────────")
        seen = global_seen if dedupe_across_topics else set()
        records, n_skipped = audit_topic(
            topic, totals[topic], n_per_topic, seen
        )
        if not dedupe_across_topics:
            global_seen |= seen

        all_records.extend(records)
        skipped[topic] = n_skipped

    if not all_records:
        raise RuntimeError("No datasets audited.")

    per_dataset = pd.DataFrame(all_records)
    coverage = summarise(per_dataset, skipped)
    return per_dataset, coverage


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(
    per_dataset: pd.DataFrame,
    skipped: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Collapse the per-dataset audit into a topic x property coverage table.

    Cells are fractions in [0, 1].  The ``n_checked`` column is the per-topic sample
    size (the denominator).  ``n_skipped`` counts datasets that failed to load
    or were collections; these are excluded from the denominator.  The final
    ``ALL`` row pools every dataset, so it is weighted by realised per-topic
    sample sizes, not by catalogue size.
    """
    props = list(PROPERTIES)
    skipped = skipped or {}

    coverage = per_dataset.groupby("topic")[props].mean()
    coverage["n_checked"] = per_dataset.groupby("topic").size()
    coverage["n_skipped"] = pd.Series(skipped).reindex(coverage.index).fillna(0)

    pooled = per_dataset[props].mean()
    pooled["n_checked"] = len(per_dataset)
    pooled["n_skipped"] = sum(skipped.values())
    coverage.loc["ALL"] = pooled

    coverage = coverage[props + ["n_checked", "n_skipped"]]
    coverage[["n_checked", "n_skipped"]] = coverage[["n_checked", "n_skipped"]].astype(int)
    return coverage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--per-topic", type=int, default=30,
                        help="Datasets to audit per topic (default 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None,
                        help="Basename for output files (timestamped)")
    args = parser.parse_args()

    per_dataset, coverage = audit(n_per_topic=args.per_topic, seed=args.seed)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\n── Coverage (fraction of datasets with property) ──────")
    print(coverage)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (args.out or "audit").removesuffix(".csv")
    cov_path = os.path.join(FOLDER_SAVE, f"{base}_coverage_{timestamp}.csv")
    raw_path = os.path.join(FOLDER_SAVE, f"{base}_per_dataset_{timestamp}.csv")

    coverage.to_csv(cov_path)
    per_dataset.to_csv(raw_path, index=False)
    print(f"\nWritten to {cov_path}\n         and {raw_path}")


if __name__ == "__main__":
    main()