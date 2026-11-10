"""Capture real Git author/file evidence for the Deliverable 2 contribution table."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = [
    ("Hassan Mehmood (24L-2559)", "hassaanmehmood68@gmail.com",
     "models/clip_classifier.py, train.py, evaluate.py, metrics.py, infer.py"),
    ("Adnan Zaka (24i-2537)", "zakaadnan076@gmail.com",
     "data scripts/clean_data.py, preprocess.py, split_data.py, eda.py"),
    ("Mohid Israr (24L-2510)", "l242510@lhr.nu.edu.pk",
     "data/images/, data/metadata.csv, data/sources.csv"),
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def main():
    history = git("log", "--format=%x1e%h%x1f%an%x1f%ae%x1f%s", "--name-only", "HEAD")
    commits = []
    for record in history.split("\x1e"):
        if not record.strip():
            continue
        lines = record.strip().splitlines()
        commit, author, email, subject = lines[0].split("\x1f", 3)
        files = [line for line in lines[1:] if line]
        commits.append({"commit": commit, "author": author, "email": email, "subject": subject,
                        "code_files": [f for f in files if f.endswith(".py")],
                        "manifest_files": [f for f in files if f.endswith(".csv")],
                        "image_count": sum(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)})
    members = []
    for student, email, modules in TEAM:
        authored = [c for c in commits if c["email"].casefold() == email]
        coding = [c for c in authored if c["code_files"]]
        evidence = coding[:2] or authored[:1]
        members.append({
            "student": student, "git_author": authored[0]["author"] if authored else "No author found",
            "commit_evidence": ", ".join(c["commit"] for c in evidence) or "None",
            "modules": modules,
            "coding_status": "Code commits verified" if coding else "No coding commit found",
            "coding_commits": coding, "authored_commit_count": len(authored),
        })
    snapshot = {"head": git("rev-parse", "HEAD"),
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "members": members,
                "manual_upload_indicators": [c["commit"] for c in commits
                                             if "files via upload" in c["subject"].casefold()]}
    destination = ROOT / "docs" / "contributions.json"
    destination.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {destination.relative_to(ROOT)} from HEAD {snapshot['head'][:12]}.")
    for member in members:
        print(f"{member['student']}: {member['coding_status']} ({member['commit_evidence']})")


if __name__ == "__main__":
    main()
