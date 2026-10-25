"""Verify committed image provenance records and write source audit summaries.

These summaries are derived from the repository's manifest and image files.
They do not substitute for the original provider exports or their README files.
"""

from collections import Counter
import re

from common import DATA, ROOT, read_csv, sha256_file, write_json


def main() -> None:
    sources = read_csv(DATA / "sources.csv", ["source_id", "source_name",
                                              "source_url", "version_url",
                                              "license", "number_selected",
                                              "archive_sha256"])
    images = read_csv(DATA / "metadata.csv", ["image_id", "source_id", "local_path",
                                               "dish_label", "sha256", "review_status"])
    if not sources or not images:
        raise RuntimeError("Source audit requires populated source and image manifests")

    by_source = {}
    for row in images:
        by_source.setdefault(row["source_id"], []).append(row)
    source_ids = {row["source_id"] for row in sources}
    if len(source_ids) != len(sources) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", sid)
                                              for sid in source_ids):
        raise RuntimeError("Source IDs must be unique and safe for evidence filenames")
    unknown = sorted(set(by_source) - source_ids)
    if unknown:
        raise RuntimeError(f"Image records have unknown source IDs: {', '.join(unknown)}")

    summaries = {}
    for source in sources:
        source_id = source["source_id"]
        rows = by_source.get(source_id, [])
        if len(rows) != int(source["number_selected"]):
            raise RuntimeError(f"{source_id}: manifest count does not match number_selected")
        for row in rows:
            path = (ROOT / row["local_path"]).resolve()
            if not path.is_relative_to((DATA / "images").resolve()) or not path.is_file():
                raise RuntimeError(f"{row['image_id']}: image missing or outside data/images")
            if sha256_file(path) != row["sha256"].strip().lower():
                raise RuntimeError(f"{row['image_id']}: image hash does not match metadata")
        summaries[source_id] = {
            "source_id": source_id,
            "source_name": source["source_name"],
            "source_url": source["source_url"],
            "version_url": source["version_url"],
            "uploader_stated_license": source["license"],
            "selected_image_count": len(rows),
            "selected_images_by_dish": dict(sorted(Counter(r["dish_label"] for r in rows).items())),
            "selected_image_hashes_verified_against_manifest": len(rows),
            "review_status_counts": dict(sorted(Counter(r["review_status"] for r in rows).items())),
            "archive_sha256_recorded_but_not_rechecked": source["archive_sha256"],
            "evidence_basis": "Derived from repository sources.csv, metadata.csv and committed image files",
            "limitations": [
                "Original provider export archive is not preserved in this repository.",
                "Original provider README is not preserved in this repository.",
                "The recorded archive SHA-256 cannot be rechecked without that archive.",
                "This summary is not evidence of a completed second human label review.",
            ],
        }

    evidence = ROOT / "evidence"
    wanted = {f"{source_id}_summary.json" for source_id in summaries}
    if evidence.exists():
        for old in evidence.glob("*_summary.json"):
            if old.name not in wanted:
                old.unlink()
    for source_id, summary in summaries.items():
        write_json(evidence / f"{source_id}_summary.json", summary)
    print(f"Verified {len(images)} selected image hashes across {len(sources)} sources.")
    print("Wrote repository-derived summaries to evidence/.")


if __name__ == "__main__":
    main()
