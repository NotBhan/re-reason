"""Documentation-governance checks for RE:Reason (see docs/DOCUMENTATION_POLICY.md §14).

These tests enforce the append-only research-documentation policy:

* required governance documents exist and keep their mandated sections/fields;
* the research ledger is complete (every completed report is referenced) and
  every entry carries the required fields with non-decreasing dates;
* the current-status synthesis keeps its required structure;
* repository-relative paths cited by governance documents resolve;
* governance documents contain no machine-specific absolute paths;
* every completed report matches the SHA-256 baseline — a silent rewrite fails
  the suite until it is either reverted or registered as an appended amendment.
"""

import hashlib
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = os.path.join(ROOT, "docs", "DOCUMENTATION_POLICY.md")
LEDGER = os.path.join(ROOT, "docs", "RESEARCH_LOG.md")
STATUS = os.path.join(ROOT, "docs", "RESEARCH_STATUS.md")
BASELINE = os.path.join(ROOT, "docs", "reports", "HISTORICAL_BASELINE.json")
AGENTS = os.path.join(ROOT, "AGENTS.md")
REPORTS_DIR = os.path.join(ROOT, "docs", "reports")

GOVERNANCE_DOCS = [POLICY, LEDGER, STATUS, AGENTS]

POLICY_CLAUSES = [
    "append-only",
    "historical immutability",
    "corrections are append-only amendments",
    "failed hypotheses",
    "numerical immutability",
    "experiment numbering",
    "traceability",
    "guardrails",
    "research_log",
    "research_status",
    "historical_baseline",
]

LEDGER_ENTRY_FIELDS = [
    "Status:",
    "Research question:",
    "Configuration:",
    "Key result:",
    "Decision:",
    "Evidence:",
    "Changes to project state:",
    "Open questions:",
    "Next experiment:",
]

STATUS_SECTIONS = [
    "## 1. Current validated state",
    "## 2. Historical progression",
    "## 3. Rejected",
    "## 4. Open questions",
    "## 5. Active / next experiment",
    "## 6. Guardrails",
]


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_files():
    return sorted(f for f in os.listdir(REPORTS_DIR) if f.endswith(".md"))


def test_governance_files_exist():
    for path in GOVERNANCE_DOCS + [BASELINE]:
        assert os.path.exists(path), f"missing governance artifact: {os.path.relpath(path, ROOT)}"


def test_policy_contains_required_clauses():
    text = _read(POLICY).lower()
    for clause in POLICY_CLAUSES:
        assert clause in text, f"policy is missing required clause: {clause!r}"


@pytest.mark.parametrize("label,path", [("policy", POLICY), ("ledger", LEDGER),
                                        ("status", STATUS), ("agents", AGENTS)])
def test_governance_docs_have_no_absolute_machine_paths(label, path):
    text = _read(path)
    assert "/home/" not in text, f"{label} contains a machine-specific /home/ path"
    assert "file:///" not in text, f"{label} contains an absolute file:/// link"


def test_ledger_entries_complete_and_chronological():
    text = _read(LEDGER)
    entries = re.split(r"(?m)^## ", text)[1:]
    entries = [e for e in entries if re.match(r"^\d{4}-\d{2}-\d{2}\s*[—-]\s*", e)]
    assert len(entries) >= 13, f"expected at least 13 dated ledger entries, found {len(entries)}"
    dates = []
    for entry in entries:
        header = entry.splitlines()[0]
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})\s*[—-]\s*(.+)$", header)
        assert m, f"ledger entry header malformed: {header!r}"
        dates.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        for field in LEDGER_ENTRY_FIELDS:
            assert field in entry, f"ledger entry {header!r} is missing field {field!r}"
    assert dates == sorted(dates), "ledger entries are not in non-decreasing date order"


def test_every_completed_report_is_referenced_in_ledger():
    text = _read(LEDGER)
    for name in _report_files():
        assert name in text, f"report {name} is not referenced in docs/RESEARCH_LOG.md"


def test_status_document_structure_and_references():
    text = _read(STATUS)
    for section in STATUS_SECTIONS:
        assert section in text, f"status document is missing section: {section!r}"
    referenced = {name for name in _report_files() if name in text}
    assert len(referenced) >= 8, (
        f"status document should cite at least 8 reports, cites {len(referenced)}")


def test_cited_repo_relative_paths_exist():
    pattern = re.compile(r"`((?:docs|experiments|src|tests)/[A-Za-z0-9_./-]+\.(?:md|py|json|npz|png|pt))`")
    missing = []
    for doc in GOVERNANCE_DOCS:
        for rel in sorted(set(pattern.findall(_read(doc)))):
            if not os.path.exists(os.path.join(ROOT, rel)):
                missing.append((os.path.relpath(doc, ROOT), rel))
    assert not missing, f"governance docs cite paths that do not exist: {missing}"


def test_agents_document_contains_permanent_rule():
    text = _read(AGENTS)
    assert "RESEARCH DOCUMENTATION POLICY" in text
    assert "append-only" in text
    assert "Never silently rewrite" in text


def test_global_plan_file_present_and_well_formed():
    plan = os.path.join(ROOT, "PLAN.md")
    assert os.path.exists(plan), "missing repository-global plan file: PLAN.md (see AGENTS.md)"
    text = _read(plan)
    assert "Last updated:" in text, "PLAN.md must carry a 'Last updated:' line"
    assert "/home/" not in text, "PLAN.md contains a machine-specific /home/ path"
    assert "file:///" not in text, "PLAN.md contains an absolute file:/// link"


def test_historical_reports_match_sha256_baseline():
    with open(BASELINE, encoding="utf-8") as f:
        baseline = json.load(f)
    assert isinstance(baseline.get("reports"), dict) and baseline["reports"]
    assert isinstance(baseline.get("history"), list) and baseline["history"]

    current = {os.path.join("docs", "reports", name) for name in _report_files()}
    baselined = set(baseline["reports"])
    new_files = current - baselined
    assert not new_files, (
        f"new report(s) not registered in HISTORICAL_BASELINE.json: {sorted(new_files)}. "
        "At completion, add their SHA-256 to the baseline and note it in the baseline history."
    )

    mismatches = []
    for rel in sorted(current):
        path = os.path.join(ROOT, rel)
        actual = _sha256(path)
        expected = baseline["reports"].get(rel)
        if expected is None:
            mismatches.append((rel, "not baselined", "-"))
        elif actual != expected:
            mismatches.append((rel, expected[:12], actual[:12]))
    assert not mismatches, (
        "Completed report(s) differ from the SHA-256 baseline: "
        f"{mismatches}. If this was NOT an intentional appended amendment, revert the file. "
        "If it WAS an intentional append-only amendment: (1) record it in docs/RESEARCH_LOG.md, "
        "(2) update the hash in docs/reports/HISTORICAL_BASELINE.json, and (3) add an entry to "
        "the baseline 'history' array."
    )
