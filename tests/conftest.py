import sys
from pathlib import Path

# src/ をインポートパスへ追加
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
