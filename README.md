# 変圧器オイル温度予測 PoC（ETT データセット）

ATHENA TECHNOLOGIES INC. ML エンジニア選考課題。電力用変圧器のオイル温度 (OT) を
将来予測し、予防保全への移行価値を検証する PoC の実装・報告一式。

- **報告スライド（提出物②）**: [`slides/PoC_report.pdf`](slides/PoC_report.pdf)
- **実装（提出物①）**: `src/`（パイプライン）+ `tests/`（リーク検証）

## 結論（要約）

| 観点 | 結果 |
|---|---|
| 短期(1h先) | 油温の自己相関が極めて高く（lag1≈0.99）、**持続予測が既に高精度**。ML の上乗せは小。 |
| 中期(6〜12h先) | 持続予測が崩れる領域で **LightGBM が MAE を ETTh1 約+11% / ETTh2 約+58% 改善**。価値の本体。 |
| 長期(24h先) | 日次周期が支配し naive≒季節持続。**予測可能性が頭打ち**（外部データが必要）。 |
| 最大リスク | 季節による**分布シフト**。固定閾値は不適、**定期再学習**が前提。 |

数値はすべて `reports/metrics_*.json` から自動生成（コードと報告の乖離なし）。

## セットアップ

```bash
python -m pip install -r requirements.txt   # Python 3.12 推奨
```

## 再現（ワンコマンド）

```bash
py -3.12 run_all.py
```

個別実行:

```bash
py -3.12 -m pytest tests -q                 # リーク検証含む全テスト（VerifyCommand）
py -3.12 src/eda.py --dataset ETTh1         # EDA 図表・サマリ
py -3.12 src/pipeline.py --dataset ETTh1    # 学習・評価 → reports/metrics_ETTh1.json
py -3.12 src/report_figures.py              # ホライズン依存の結果図
py -3.12 slides/build_slides.py             # 報告スライド HTML
py -3.12 slides/render.py                   # Edge headless で PDF 化
```

## 設計の要点

- **時系列リーケージの体系的遮断**（本 PoC の中核）。時系列順分割・シャッフル禁止・
  train 限定スケーリング・因果的特徴量。`tests/test_leakage.py` で恒常的に検証。
- **差分予測（persistence-anchored）**: 目的を OT(t+h) ではなく変化量 Δ=OT(t+h)−OT(t) に
  置き、持続予測を錨にして系統的変化のみを学習。直接予測ではベースライン未達だったが、
  差分予測で全ホライズンで上回った。
- **未来負荷を既知とした上限実験**で「どの情報が価値を生むか」を切り分け。

詳細は [`docs/spec.md`](docs/spec.md)（仕様契約）と [`docs/design.md`](docs/design.md)（設計判断）。

## ディレクトリ

```
data/                ETT CSV（ETTh1/h2/m1/m2）
src/                 config / data / features / models / evaluate / pipeline / eda / ...
tests/               リーク検証・統合テスト（pytest）
reports/             metrics_*.json, eda_*.json, figures/
slides/              build_slides.py, render.py, slides.html, PoC_report.pdf
docs/                spec.md, design.md
```

## データ出典

- ETT (Electricity Transformer Temperature) dataset — zhouhaoyi/ETDataset (GitHub, MIT License)。
  Informer (Zhou et al., AAAI 2021) で公開されたベンチマーク。
- 追加データは未使用。利用する場合は本 README に出典を明記する。

## セキュリティ / データ保護

- 本リポジトリで扱うのは**公開ベンチマークデータのみ**（顧客実データではない）。
- 秘密情報・認証情報・PII は一切含まない。`.gitignore` で `.env` 等を除外。
- 顧客実データへ適用する際は、データを repo に含めず（パス参照のみ）、
  アクセス制御・匿名化・監査ログを別途設計する想定。
