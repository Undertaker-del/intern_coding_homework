# 変圧器オイル温度予測 PoC（ETT データセット）

ATHENA TECHNOLOGIES INC. ML エンジニア選考課題。電力用変圧器のオイル温度 (OT) を
将来予測し、予防保全への移行価値を検証する PoC の実装・報告一式。

- **報告スライド（提出物②）**: [`slides/PoC_report.pdf`](slides/PoC_report.pdf)
- **実装（提出物①）**: `src/`（パイプライン）+ `tests/`（リーク検証）

## 結論（要約）

評価は **5フォールドのローリング起点バックテスト**（naive 比 MAE 改善率 mean±std、正のフォールド数）。

| ホライズン | ETTh1 | ETTh2 | 解釈 |
|---|---|---|---|
| 1h先 | +3.8%±1.7% (5/5) | +57.0%±3.2% (5/5) | 持続予測が既に強く改善は堅実だが小（自己相関 lag1≈0.99） |
| 6h先 | +9.3%±11.7% (4/5) | +57.2%±2.9% (5/5) | **価値の本体**：持続予測が崩れる領域 |
| 12h先 | +9.2%±7.7% (4/5) | +54.2%±3.1% (5/5) | **価値の本体**：多/全フォールドで正 |
| 24h先 | −2.1%±5.1% (3/5) | +0.3%±5.5% (3/5) | **頭打ち**：持続予測を安定的に超えない（外部データが必要） |

**分割方法への感度（重要）**：時系列60/20/20 と文献標準 Informer 12/4/4 を比較すると、
**ETTh2 はどちらでも +54〜60%（頑健）**だが、**ETTh1 は時系列分割で +9%・文献分割で −6〜−9% と符号反転**＝
中期改善は**評価期間に依存して頑健でない**。よって結論は「**ML の価値は変圧器の変動性レジームに依存**：
変動大なら大きく頑健、穏やかなら不確実」。導入判断は資産別・期間横断の検証が前提。

- 24h 先はどの分割でも naive 比ほぼ 0〜負＝**予測可能性の天井**（外部データが必要）。
- 結論はバックテスト・**シード(1/42/123)**・**分割方法**の感度分析で裏取り済み（過大評価を回避）。
- 最大リスクは季節による**分布シフト**（ETTh1: train 平均 17.3°C → test 7.7°C）。固定閾値は不適、**定期再学習**が前提。
- 点予測に加え**予測区間(P10/P90)**と**MASE**も算出。数値は `reports/*.json` から自動生成。

### 機構（なぜ ETTh2 だけ効くか・実務示唆）
ML の価値は**日内温度サイクルの振幅**で決まる：ETTh2=9.4°C（大）→ +54%、ETTh1=2.4°C（小）→ +9%。
特徴量アブレーションでもカレンダー（日内/季節）特徴を外すと skill が 57%→35% に半減＝**日内サイクルが主シグナル**
（負荷特徴は無用）。**実務示唆**：導入前に各変圧器の日内振幅を測るだけで ML 価値を事前スクリーニングできる。

### 検証の網羅性（`src/verify_all.py`・`reports/verification_report.json`）
| 観点 | 方法 | 結果 |
|---|---|---|
| 再現性 | 5フォールド・バックテスト、fold数(3/5/8)感度 | 結論不変 |
| 分割依存 | 時系列 vs 文献Informer12/4/4（×シード） | ETTh2頑健・ETTh1非頑健（seed不変） |
| シード | LightGBM seed 1/42/123 | 結論不変 |
| 機構 | 日内振幅・特徴量アブレーション | 日内サイクルが主シグナル |
| モデル | Ridge vs LGBM | LGBM が頑健に優位（ETTh1でRidgeは負） |
| 目的 | 差分 vs 直接予測 | 差分が必須（直接は naive に大敗 −32〜−43%） |
| 粒度 | ETTm1 バックテスト | ETTh1 と同型（中期+9%・24h天井） |
| 正しさ | 指標を生lightgbm/手計算で独立再計算 | 完全一致（実装バグなし） |
| リーク | ノイズ下限・陽性対照・決定性 | 漏洩なしを裏取り |
| データ品質 | 張り付き・外れ値 | ETTm2 は OT 張り付き21.7%（要注意）、他は良好 |

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

## 設計の要点（検証ループの実践）

- **時系列リーケージの体系的遮断**（本 PoC の中核）。時系列順分割・シャッフル禁止・
  train 限定スケーリング・因果的特徴量。`tests/test_leakage.py` で恒常的に検証（pytest 17件）。
- **差分予測（persistence-anchored）**: 目的を OT(t+h) ではなく変化量 Δ=OT(t+h)−OT(t) に
  置き、持続予測を錨にして系統的変化のみを学習。直接予測ではベースライン未達だったが、
  差分予測で中期まで安定して上回った。
- **ローリング起点バックテスト**（5フォールド）で単一分割の楽観を排し再現性を担保。
- **仮説の棄却**: 負荷の移動平均（熱蓄積）特徴を検証 → ETTh1 は改善するが ETTh2 は悪化し、
  交差検証上の頑健な改善が無いため**不採用**（`config.LOAD_ROLL_WINDOWS=[]`、再現は config 変更のみ）。
- **予測区間(分位点回帰 P10/P90)・MASE・未来負荷の上限実験**で価値と限界を多面的に定量化。

詳細は [`docs/spec.md`](docs/spec.md)（仕様契約）と [`docs/design.md`](docs/design.md)（設計判断）。

## ディレクトリ

```
data/                ETT CSV（ETTh1/h2/m1/m2）
src/                 config / data / features / models / evaluate / pipeline /
                     backtest / eda / report_figures / plotstyle
tests/               リーク検証・統合・指標テスト（pytest 17件）
reports/             metrics_*.json, backtest_*.json, eda_*.json, figures/
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
