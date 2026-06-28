"""matplotlib の日本語フォント設定（Windows 環境の和文グリフ対応）。"""
from __future__ import annotations


def use_japanese_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if cand in available:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
