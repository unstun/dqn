from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    level: str  # FAIL/WARN
    message: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_one(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _looks_like_placeholder(token: str) -> bool:
    t = str(token)
    return any(ch in t for ch in ("<", ">", "{", "}", "$"))


def _extract_profiles_from_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"--profile\s+([^\s\\]+)", text):
        tok = m.group(1).strip()
        if not tok or _looks_like_placeholder(tok):
            continue
        out.add(tok)
    return out


def _extract_rand_pairs_paths(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"--rand-pairs-json\s+([^\s\\]+)", text):
        tok = m.group(1).strip()
        if not tok or _looks_like_placeholder(tok):
            continue
        out.add(tok)
    return out


def _check_required_paths(root: Path) -> list[Finding]:
    findings: list[Finding] = []

    required_files = [
        Path("AGENTS.md"),
        Path("README.md"),
        Path("README.zh-CN.md"),
        Path("train.py"),
        Path("infer.py"),
        Path("game.py"),
        Path("configs/README.md"),
        Path("forest_vehicle_dqn/cli/train.py"),
        Path("forest_vehicle_dqn/cli/infer.py"),
        Path("forest_vehicle_dqn/cli/game.py"),
        Path("docs/versions/README.md"),
    ]
    required_dirs = [
        Path("configs"),
        Path("paper"),
        Path("docs/versions"),
    ]

    for rel in required_files:
        p = root / rel
        if not p.is_file():
            findings.append(Finding("FAIL", f"Missing required file: {rel}"))

    for rel in required_dirs:
        p = root / rel
        if not p.is_dir():
            findings.append(Finding("FAIL", f"Missing required dir: {rel}"))

    # runs/ is an output directory (gitignored) and may not exist in a fresh clone.
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        findings.append(Finding("WARN", "runs/ directory is missing (will be created on first run)"))

    return findings


def _check_version_archives(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    versions_dir = root / "docs" / "versions"
    if not versions_dir.is_dir():
        return [Finding("FAIL", "Missing docs/versions/ directory")]

    version_re = re.compile(r"v\d+(?:p\d+)*\Z")
    required = [Path("README.md"), Path("CHANGES.md"), Path("RESULTS.md"), Path("runs/README.md")]
    for child in sorted(versions_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not version_re.match(child.name):
            continue
        for req in required:
            p = child / req
            if not p.is_file():
                findings.append(Finding("FAIL", f"Version archive missing: {p.relative_to(root)}"))
    return findings


def _check_readme_alignment(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    en = _read_text(root / "README.md")
    zh = _read_text(root / "README.zh-CN.md")

    en_updated = _find_one(r"^Last updated:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$", en)
    zh_updated = _find_one(r"^最后更新：\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$", zh)
    if en_updated is None:
        findings.append(Finding("FAIL", "README.md missing 'Last updated: YYYY-MM-DD' line"))
    if zh_updated is None:
        findings.append(Finding("FAIL", "README.zh-CN.md missing '最后更新：YYYY-MM-DD' line"))
    if en_updated is not None and zh_updated is not None and en_updated != zh_updated:
        findings.append(Finding("FAIL", f"README last-updated mismatch: en={en_updated} zh={zh_updated}"))

    en_prof = _find_one(r"^Current recommended train profile:\s*`([^`]+)`\s*$", en)
    zh_prof = _find_one(r"^当前推荐训练 profile：\s*`([^`]+)`\s*$", zh)
    if en_prof is None:
        findings.append(Finding("FAIL", "README.md missing 'Current recommended train profile: `...`' line"))
    if zh_prof is None:
        findings.append(Finding("FAIL", "README.zh-CN.md missing '当前推荐训练 profile：`...`' line"))
    if en_prof is not None and zh_prof is not None and en_prof != zh_prof:
        findings.append(Finding("FAIL", f"README recommended profile mismatch: en={en_prof} zh={zh_prof}"))

    if en_prof is not None:
        prof_path = root / "configs" / f"{en_prof}.json"
        if not prof_path.is_file():
            findings.append(Finding("FAIL", f"Recommended profile not found: {prof_path.relative_to(root)}"))

    # Check that documented profiles/pairs exist (warn-only unless strict mode is used).
    profiles = _extract_profiles_from_text(en) | _extract_profiles_from_text(zh)
    for prof in sorted(profiles):
        prof_path = root / "configs" / f"{prof}.json"
        if not prof_path.is_file():
            findings.append(Finding("WARN", f"Doc references missing profile: {prof} -> {prof_path.relative_to(root)}"))

    pairs_paths = _extract_rand_pairs_paths(en) | _extract_rand_pairs_paths(zh)
    for p in sorted(pairs_paths):
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            findings.append(Finding("WARN", f"Doc references missing rand-pairs json: {p}"))

    return findings


def _check_configs_doc(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = _read_text(root / "configs" / "README.md")
    if "game.py" not in text:
        findings.append(Finding("WARN", "configs/README.md does not mention game.py profiles (game section)"))
    if "game" not in text:
        findings.append(Finding("WARN", "configs/README.md does not mention 'game' section"))
    return findings


def _check_git_status(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(root), text=True)
    except Exception:
        return findings
    if out.strip():
        findings.append(Finding("WARN", "git working tree is not clean"))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repo doctor: static checks for docs/config alignment.")
    ap.add_argument("--strict", action="store_true", help="Treat WARN as FAIL.")
    args = ap.parse_args(argv)

    root = _repo_root()
    findings: list[Finding] = []
    findings += _check_required_paths(root)
    findings += _check_version_archives(root)
    findings += _check_readme_alignment(root)
    findings += _check_configs_doc(root)
    findings += _check_git_status(root)

    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    for f in fails + warns:
        print(f"[{f.level}] {f.message}", file=sys.stderr)

    if fails:
        print(f"[doctor] FAIL ({len(fails)} fails, {len(warns)} warns)", file=sys.stderr)
        return 1
    if warns and bool(args.strict):
        print(f"[doctor] FAIL(strict) ({len(warns)} warns)", file=sys.stderr)
        return 2

    print(f"[doctor] ok ({len(warns)} warns)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
