"""Create a code-only ZIP without data, results, credentials or Git history."""
import argparse
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md","train.py","test.py","requirements.txt","requirements-dev.txt","requirements-profile.txt",".gitignore",".gitattributes"}
PRIVATE_PATTERNS = [r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)", r"/"+r"root/(?:autodl|data|workspace)",
                    r"gh[pousr]_[A-Za-z0-9]{25,}", r"github_pat_[A-Za-z0-9_]{25,}",
                    r"sk-[A-Za-z0-9]{25,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]


def release_files(root=ROOT):
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file(): continue
        relative = path.relative_to(root)
        parts = relative.parts
        if any(p in {".git","__pycache__",".pytest_cache","local",".venv"} for p in parts): continue
        allowed = (relative.as_posix() in ROOT_FILES or
                   (parts[0] in {"models","utils","tests"} and path.suffix==".py") or
                   (parts[0]=="scripts" and path.suffix==".py") or
                   (parts[0]=="configs" and path.suffix==".yaml") or
                   (relative.as_posix() in {"checkpoints/README.md","splits/README.md"}) or
                   (parts[:2]==(".github","workflows") and path.suffix in {".yaml",".yml"}))
        if not allowed:
            if parts[0] in {"runs","results","outputs","data","datasets","checkpoints"}: continue
            raise ValueError(f"Unexpected file in the code release: {relative.as_posix()}")
        text = path.read_text(encoding="utf-8")
        if any(re.search(pattern,text) for pattern in PRIVATE_PATTERNS):
            raise ValueError(f"Private path or credential pattern in {relative.as_posix()}")
        result.append(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    files = release_files()
    if args.check:
        print(f"Code-only checks passed: {len(files)} files")
    if args.output:
        output = args.output.resolve()
        if output==ROOT or ROOT in output.parents:
            raise ValueError("Write the archive outside the source directory")
        output.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(output,"x",compression=ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file,arcname="MORPH-Polyp-Segmentation/"+file.relative_to(ROOT).as_posix())
        with ZipFile(output) as archive:
            if archive.testzip() is not None: raise RuntimeError("ZIP CRC check failed")
        print(f"Created code-only archive with {len(files)} files")
    if not args.check and not args.output:
        parser.error("Supply --check or --output")


if __name__ == "__main__":
    main()
