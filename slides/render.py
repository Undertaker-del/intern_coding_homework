"""slides.html を Edge headless で PoC_report.pdf に変換する。

先に build_slides.py を実行して HTML を生成してから呼ぶ。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "slides.html"
PDF = HERE / "PoC_report.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge() -> str:
    for c in EDGE_CANDIDATES:
        if Path(c).exists():
            return c
    raise FileNotFoundError("msedge.exe が見つかりません")


def render() -> Path:
    if not HTML.exists():
        raise FileNotFoundError(f"{HTML} がありません。先に build_slides.py を実行してください")
    edge = find_edge()
    cmd = [
        edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}", HTML.as_uri(),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    print(f"[saved] {PDF}  ({PDF.stat().st_size // 1024} KB)")
    return PDF


if __name__ == "__main__":
    render()
