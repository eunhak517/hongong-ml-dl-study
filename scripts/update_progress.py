from __future__ import annotations

import re
from pathlib import Path

README_PATH = Path("README.md")
NOTEBOOK_ROOT = Path("notebooks")
TOTAL_SECTIONS = 28

PROGRESS_PATTERN = re.compile(
    r"<!-- PROGRESS_START -->.*?<!-- PROGRESS_END -->",
    flags=re.DOTALL,
)

SECTION_PATTERN = re.compile(r"^(?P<chapter>\d{2})_(?P<section>\d{2})_.*\.ipynb$")


def find_completed_sections() -> set[tuple[int, int]]:
    completed: set[tuple[int, int]] = set()

    if not NOTEBOOK_ROOT.exists():
        return completed

    for notebook in NOTEBOOK_ROOT.rglob("*.ipynb"):
        match = SECTION_PATTERN.match(notebook.name)
        if not match:
            continue

        chapter = int(match.group("chapter"))
        section = int(match.group("section"))
        completed.add((chapter, section))

    return completed


def make_progress_block(completed_count: int) -> str:
    percentage = round(completed_count / TOTAL_SECTIONS * 100)
    filled = round(percentage / 5)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty

    return f"""<!-- PROGRESS_START -->
![Progress](https://img.shields.io/badge/Progress-{completed_count}%20%2F%20{TOTAL_SECTIONS}-2EA44F?style=for-the-badge)

```text
{bar}  {percentage}%
```

**완료 {completed_count}개 · 전체 {TOTAL_SECTIONS}개 절**
<!-- PROGRESS_END -->"""


def main() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError("README.md 파일을 찾을 수 없습니다.")

    readme = README_PATH.read_text(encoding="utf-8")
    completed = find_completed_sections()
    progress_block = make_progress_block(len(completed))

    updated, count = PROGRESS_PATTERN.subn(progress_block, readme)

    if count != 1:
        raise RuntimeError(
            "README.md에서 진행률 마커를 정확히 한 쌍 찾지 못했습니다."
        )

    README_PATH.write_text(updated, encoding="utf-8")

    formatted = ", ".join(
        f"{chapter:02d}-{section:02d}"
        for chapter, section in sorted(completed)
    )
    print(f"Completed sections: {len(completed)}/{TOTAL_SECTIONS}")
    print(f"Detected: {formatted or 'none'}")


if __name__ == "__main__":
    main()
