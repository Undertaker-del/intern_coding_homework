# design.md — 設計と意思決定（HOW）

## Scope
ETT の油温 (OT) 予測 PoC。リーケージ防止を最優先に、ベースライン→勾配ブースティングの
段階的検証で「予測がどの程度の業務価値を出せるか」を定量化する。

## Background（EDA からの初期示唆）
- データはクリーン（欠損・重複なし、時刻単調）。ETTh=17,420時間（約2年）、ETTm=15分粒度69,680点。
- **OT と負荷6特徴量の同時刻相関は弱い（|r|≤0.22）**。→ 瞬時負荷だけでは OT を説明できない。
- 油温は**熱慣性**を持ち、過去の負荷の蓄積・自己の過去値・日次/年次（外気温）周期で決まると仮説。
  → 自己回帰ラグ・移動統計・周期特徴量が支配的シグナルになると予測（EDA で検証）。

## Problem Statement
将来の OT を、未来情報を一切漏らさずに予測し、ベースライン（素朴・季節持続）に対する
改善幅（skill）を業務価値（早期警報リードタイム）に翻訳する。

## Proposal（採用案）
**段階的モデリング + 厳格な時系列検証**
1. 時系列分割: 全系列を時刻順に train 60% / val 20% / test 20%。シャッフル禁止。
   （文献の 12/4/4ヶ月分割も `--split informer` で選択可能にし、結果の頑健性を確認。）
2. 特徴量（すべて因果的 = 時刻 t 以前のみ。`shift(≥1)`・`rolling` は `center=False`）:
   - OT 自己ラグ {1,2,3,6,12,24,48,168}、負荷6列のラグ {1,24}。
   - OT 移動平均/標準偏差（窓 {3,6,24}）、OT 差分。
   - カレンダー（hour, dayofweek, month）と Fourier 項（日次・年次）。
3. 目的変数: `OT(t+h)`。学習行は h だけ将来にずらし、系列端の NaN 行を除去。
4. モデル:
   - **Naive persistence**: ŷ(t+h)=OT(t)（ベースライン下限）。
   - **Seasonal naive**: ŷ(t+h)=OT(t+h-24)（日次周期ベースライン）。
   - **Ridge**: 標準化済みラグ窓への線形回帰（線形上限の目安・DLinear 的）。
   - **LightGBM**: 勾配ブースティング本命。GPU 不要・高速・特徴量重要度で説明可能。
5. 評価: MAE / RMSE を**元の°C 単位**（逆標準化後）で算出。OT は 0/負を取るため MAPE は不採用。
   skill = 1 − MAE_model / MAE_naive。早期警報の業務価値に翻訳。

### 代替案と不採用理由
- **代替1: Informer/Autoformer 等の Transformer 系**。長期予測 SOTA だが GPU 前提・実装/学習コスト大、
  PoC の意思決定（価値があるか）には過剰。線形/DLinear が ETT で強いという既知結果もあり、
  まず軽量モデルで価値を示し、改善方針として将来拡張に回す（YAGNI）。
- **代替2: 同時刻回帰のみ（ソフトセンサ）を主軸**。EDA で同時刻相関が弱く、予防保全の
  「先読み」価値を示せない。→ 補助タスク(T-NOWCAST)に格下げし、主軸は将来予測に。
- **代替3: 全データシャッフル CV**。時系列リークを生むため**禁止**。`TimeSeriesSplit` のみ使用。

## リーケージ防止設計（本 PoC の中核）
| リスク | 対策 |
|---|---|
| 未来→過去の情報漏洩 | 目的を `shift(-h)`、特徴量は `shift(≥1)`/因果 rolling のみ |
| train/test 汚染 | 時刻順分割・シャッフル禁止・境界で重複なし |
| スケーラ/統計の漏洩 | StandardScaler は train のみ fit、val/test は transform のみ |
| CV の漏洩 | `TimeSeriesSplit`（expanding window）でハイパラ選択 |
| セキュリティ(Public repo) | `.gitignore` で秘密/生成物除外、コード内に認証情報なし、公開データのみ |

## アーキテクチャ（多数の小ファイル）
```
src/
  config.py     # 定数・特徴量定義・分割比・シード
  data.py       # CSV読込・時系列分割・標準化(train fit)
  features.py   # 因果的特徴量生成（リーク安全の単一責務）
  models.py     # baseline群 + Ridge + LightGBM ラッパ
  evaluate.py   # MAE/RMSE/skill・業務価値翻訳
  pipeline.py   # end-to-end 実行（CLI, metrics.json/figures 出力）
  eda.py        # EDA 図表生成
tests/          # リーク検証中心の pytest
```

## Verify（spec の VerifyCommand に対応）
`pytest tests -q` green → `pipeline.py` で AC-2/4 → `render.py` で PDF。
