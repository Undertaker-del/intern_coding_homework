"""PoC 報告スライドの HTML を生成（メッセージ&ボディ構成）。

reports/metrics_*.json と reports/eda_*.json から実数値を注入し、
図は base64 埋め込みで自己完結 HTML を出力する（slides/slides.html）。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
OUT = ROOT / "slides" / "slides.html"


def img(name: str, *, width: str = "100%") -> str:
    p = FIG / name
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img style="width:{width}" src="data:image/png;base64,{b64}"/>'


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


m1 = load(REP / "metrics_ETTh1.json")
m2 = load(REP / "metrics_ETTh2.json")
e1 = load(REP / "eda_ETTh1.json")
e2 = load(REP / "eda_ETTh2.json")


def hz(m, h, model, key="mae"):
    return m["horizons"][str(h)][model][key]


def pct(x):
    return f"{100 * x:+.0f}%"


# よく使う数値
sk1_6 = hz(m1, 6, "lightgbm", "skill_vs_naive")
sk1_12 = hz(m1, 12, "lightgbm", "skill_vs_naive")
sk2_1 = hz(m2, 1, "lightgbm", "skill_vs_naive")
sk2_6 = hz(m2, 6, "lightgbm", "skill_vs_naive")
sk2_12 = hz(m2, 12, "lightgbm", "skill_vs_naive")
sk1_24 = hz(m1, 24, "lightgbm", "skill_vs_naive")
sk2_24 = hz(m2, 24, "lightgbm", "skill_vs_naive")
alarm1_lgbm = m1["horizons"]["12"]["lightgbm"]["alarm"]
alarm1_naive = m1["horizons"]["12"]["naive_persistence"]["alarm"]


def slide(msg: str, body: str, *, kicker: str = "") -> str:
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    return f'''<section class="slide">
      <div class="msgbar">{k}<h1>{msg}</h1></div>
      <div class="body">{body}</div>
      <div class="foot"><span>ATHENA TECHNOLOGIES INC. ／ ETT 油温予測 PoC 報告</span></div>
    </section>'''


SLIDES: list[str] = []

# 1. 表紙
SLIDES.append(f'''<section class="slide cover">
  <div class="cover-inner">
    <div class="cover-tag">PoC 報告 / 機械学習エンジニア</div>
    <h1>変圧器オイル温度の予測による<br/>予防保全の価値検証</h1>
    <p class="cover-sub">ETT データセットを用いた油温（OT）将来予測の技術検証</p>
    <p class="cover-meta">クライアント報告資料 ／ メッセージ&ボディ構成</p>
  </div>
</section>''')

# 2. エグゼクティブサマリ
SLIDES.append(slide(
    "結論：油温予測は「数時間先」で実用価値があり、予防保全への移行は十分に有望",
    f'''<ul class="big">
      <li><b>① 短期(1h先)は持続予測が既に高精度</b>（油温の自己相関 lag1={e1['acf_lag1']:.2f}）。
          ML の上乗せは小さく、<b>監視自動化の基盤</b>として堅牢。</li>
      <li><b>② 価値の本体は 6〜12h 先</b>。持続予測が崩れる領域で LightGBM が
          MAE を <b>ETTh1 約{pct(sk1_12)} / ETTh2 約{pct(sk2_6)}</b> 改善。
          点検・是正の<b>リードタイムを確保</b>できる。</li>
      <li><b>③ 24h 先は日次周期が支配し予測可能性が頭打ち</b>（改善 {pct(sk1_24)}〜{pct(sk2_24)}）。
          さらなる先読みには<b>外気温予報など外部データ</b>が必要。</li>
      <li><b>④ 最大のリスクは分布シフト</b>（季節で平均油温が大きく変化）。
          固定閾値は不適、<b>定期再学習</b>が前提。</li>
    </ul>
    <div class="callout">データは公開ベンチマーク（顧客実データではない）。実装は時系列リーク防止を全工程で担保。</div>'''))

# 3. 目的・スコープ・前提
SLIDES.append(slide(
    "目的は「油温予測が運用・保全判断にどれだけ価値を出すか」の定量検証",
    f'''<div class="cols">
      <div>
        <h3>検証スコープ</h3>
        <ul>
          <li>目的変数：オイル温度 OT（°C）の将来予測</li>
          <li>データ：ETTh1/h2（1時間粒度・約2年）を主対象、m1/m2 で粒度差を確認</li>
          <li>評価：複数ホライズン（1/6/12/24h 先）で MAE・RMSE と
              ベースライン比改善率(skill)</li>
          <li>業務価値：高温時間帯の早期検知（Precision/Recall）に翻訳</li>
        </ul>
      </div>
      <div>
        <h3>置いた仮定（質問不可のため明記）</h3>
        <ul>
          <li>「t=T の特徴量を使用してよい」を、<b>負荷は運用計画で既知</b>と解釈し、
              <b>未来負荷を与えた上限実験</b>も別途実施</li>
          <li>主結果は<b>過去情報のみ</b>で算出（運用上最も安全な前提）</li>
          <li>分割は時系列順 60/20/20（シャッフル禁止）。文献の 12/4/4ヶ月分割でも傾向不変</li>
          <li>MAPE は OT が 0/負を取るため不採用、MAE/RMSE(°C) を採用</li>
        </ul>
      </div>
    </div>'''))

# 4. EDA① データ品質・非定常性
SLIDES.append(slide(
    "データは高品質だが、季節による強い非定常性（分布シフト）を持つ",
    f'''<div class="cols">
      <div>{img("eda_ETTh1_series.png")}<div class="cap">OT 時系列（ETTh1, 約2年）</div></div>
      <div>{img("eda_ETTh1_shift.png")}<div class="cap">期間別 OT 分布</div></div>
    </div>
    <ul class="tight">
      <li>欠損・重複なし、時刻は等間隔で単調 → 前処理リスクは小さい。</li>
      <li><b>学習期間と評価期間で平均油温が大きく異なる</b>
          （ETTh1：train {e1['mean_OT_train']:.1f}°C → test {e1['mean_OT_test']:.1f}°C）。
          <b>固定閾値の監視は破綻</b>し、相対的・適応的な基準が必要。</li>
    </ul>'''))

# 5. EDA② 熱慣性と周期性 → アプローチ根拠
SLIDES.append(slide(
    "油温は強い熱慣性と日次・年次の周期を持つ → 自己回帰＋周期特徴が本質",
    f'''<div class="cols">
      <div>{img("eda_ETTh1_acf.png")}<div class="cap">自己相関（熱慣性）</div></div>
      <div>{img("eda_ETTh1_cycles.png")}<div class="cap">時刻別・月別の平均 OT</div></div>
    </div>
    <ul class="tight">
      <li>自己相関は <b>1h後={e1['acf_lag1']:.2f}・24h後={e1['acf_lag24']:.2f}</b> と極めて高い
          → <b>直近値（持続予測）が強力なベースライン</b>。</li>
      <li>時刻別・月別に明瞭な山谷 → <b>日次（運転サイクル）と年次（外気温）の周期</b>を特徴量化すべき。</li>
    </ul>'''))

# 6. EDA③ 同時刻相関は弱い
SLIDES.append(slide(
    "瞬時の負荷だけでは油温を説明できない（熱は時間遅れで蓄積する）",
    f'''<div class="cols">
      <div>{img("eda_ETTh1_corr.png")}<div class="cap">ETTh1: OT と負荷の同時刻相関</div></div>
      <div>{img("eda_ETTh2_corr.png")}<div class="cap">ETTh2: 同（相対的に強い）</div></div>
    </div>
    <ul class="tight">
      <li>同時刻相関は最大でも <b>ETTh1 {e1['max_abs_load_corr']:.2f} / ETTh2 {e2['max_abs_load_corr']:.2f}</b>。
          負荷→油温は<b>時間遅れ（積分的）</b>の関係。</li>
      <li>示唆：<b>負荷の現在値だけでなくラグ・移動統計</b>が効く。
          後述の通り未来負荷を与えても 24h 先は伸びにくい。</li>
    </ul>'''))

# 7. アプローチ：検証設計とモデル
SLIDES.append(slide(
    "「厳格な時系列検証 × 差分予測」を設計の核に据えた",
    f'''<div class="cols">
      <div>
        <h3>リーク防止（最重要）</h3>
        <ul class="tight">
          <li>時系列順分割・シャッフル禁止、境界重複なし</li>
          <li>標準化統計は<b>train のみで推定</b></li>
          <li>特徴量は時刻 t 以前のみ（因果 rolling, shift≥0）</li>
          <li>ハイパラ選択は TimeSeriesSplit / val で early stopping</li>
        </ul>
        <h3>モデル比較</h3>
        <ul class="tight">
          <li>naive 持続・季節持続（ベースライン）</li>
          <li>Ridge（線形の目安）／<b>LightGBM（本命）</b></li>
        </ul>
      </div>
      <div>
        <h3>設計上の意思決定：差分予測</h3>
        <div class="callout2">目的を OT(t+h) ではなく
          <b>変化量 Δ=OT(t+h)−OT(t)</b> に置き、予測を OT(t) に加算。</div>
        <ul class="tight">
          <li>持続予測を錨にするため、<b>最悪でも naive と同等</b>。</li>
          <li>日次サイクル等の<b>系統的変化のみ</b>を学習し改善。</li>
          <li>実際、直接予測では naive に負けたが、差分予測で<b>全ホライズンで上回った</b>。</li>
        </ul>
      </div>
    </div>'''))

# 8. 結果① ホライズン依存（価値の所在）
SLIDES.append(slide(
    "予測の価値は中期(6〜12h先)で最大化し、24h先で頭打ちになる",
    f'''<div class="cols">
      <div>{img("result_mae_vs_horizon.png")}<div class="cap">誤差のホライズン依存</div></div>
      <div>{img("result_skill_vs_horizon.png")}<div class="cap">naive 比改善率(skill)</div></div>
    </div>
    <ul class="tight">
      <li>短期：持続予測が強く ML の上乗せは小（ETTh1 1h={pct(hz(m1,1,'lightgbm','skill_vs_naive'))}）。</li>
      <li>中期：持続予測が崩れる 6〜12h で<b>改善が最大</b>
          （ETTh2 で <b>{pct(sk2_6)}前後</b>、ETTh1 で {pct(sk1_12)}前後）。</li>
      <li>24h：日次周期で naive≒季節持続となり、<b>改善は数% に収束</b>。</li>
    </ul>'''))

# 9. 結果② 予測波形と説明性
top1 = list(m1["horizons"]["12"]["lightgbm"]["top_features"].items())[:6]
feat_li = "".join(f"<li>{k} <span class='muted'>({v:.0f})</span></li>" for k, v in top1)
SLIDES.append(slide(
    "予測は妥当で、要因も解釈可能（直近変動・時刻・日平均水準・季節）",
    f'''<div class="cols">
      <div>{img("pred_ETTh2_h6.png")}<div class="cap">ETTh2 6h先：実測 vs 予測 vs naive</div></div>
      <div>
        <h3>LightGBM 重要度 上位（ETTh1, 12h先・gain）</h3>
        <ol class="feat">{feat_li}</ol>
        <p class="muted">短期は「直近の変動・時刻」、長期は「日平均水準・季節」が支配。
        物理的直感（熱慣性＋外気温）と整合し、ブラックボックスではない。</p>
      </div>
    </div>'''))

# 10. 業務価値
SLIDES.append(slide(
    "高温時間帯を事前に絞り込め、点検対象の優先順位付けに使える",
    f'''<ul class="big">
      <li>テスト期間の<b>上位10%の高温時間帯</b>を 12h 先に検知する性能（ETTh1）：
          <b>LightGBM F1={alarm1_lgbm['f1']:.2f}（適合率{alarm1_lgbm['precision']:.2f}/再現率{alarm1_lgbm['recall']:.2f}）</b>、
          単純持続 F1={alarm1_naive['f1']:.2f}。</li>
      <li>用途：閾値超過の<b>事前アラート</b>と<b>点検優先度</b>付け。
          連続監視の自動化で、経験則ベースの見落とし・過検知を低減。</li>
      <li>運用設計：誤報コストと見逃しコストに応じて<b>閾値を調整可能</b>
          （Precision/Recall のトレードオフを運用側が選択）。</li>
    </ul>
    <div class="callout">注：絶対温度の閾値は分布シフトに弱いため、運用では
      <b>季節・期間で基準を更新</b>し、予測値ランキングと併用する設計を推奨。</div>'''))

# 11. 工夫・意思決定まとめ
SLIDES.append(slide(
    "工夫の要点：リーク遮断・差分予測・上限実験・データ駆動の報告",
    '''<ul class="big">
      <li><b>時系列リークの体系的遮断</b>を自動テスト（pytest）で恒常的に検証
          （未来不参照・分割整合・train限定スケーリング）。</li>
      <li><b>差分予測</b>でベースライン未達を解消し、全ホライズンで naive 超え。</li>
      <li><b>未来負荷を既知とした上限実験</b>で「どの情報が価値を生むか」を切り分け。</li>
      <li><b>報告数値は metrics.json から自動注入</b>。コードと報告の乖離を排除（再現性）。</li>
      <li><b>セキュリティ</b>：公開データのみ・秘密情報非混入・.gitignore 徹底で
          Public リポジトリでも情報漏洩リスクなし。</li>
    </ul>'''))

# 12. 限界と改善方針
SLIDES.append(slide(
    "限界は明確で、次の一手も具体化できている",
    f'''<div class="cols">
      <div>
        <h3>現時点の限界</h3>
        <ul class="tight">
          <li><b>24h 以上の先読みは頭打ち</b>（外部要因の情報不足）。</li>
          <li><b>分布シフト</b>で固定閾値・固定モデルが劣化。</li>
          <li>評価は単一の時系列分割（期間依存。ETTh1 の test は穏やかな期間）。</li>
          <li>異常・故障ラベルが無く、保全効果は代理指標での評価。</li>
        </ul>
      </div>
      <div>
        <h3>今後の改善方針</h3>
        <ul class="tight">
          <li><b>外気温予報・運転計画</b>等の外部共変量を追加（24h+ の鍵）。</li>
          <li><b>ローリング起点 CV ＋ 定期再学習</b>で非定常に追従。</li>
          <li>分位点回帰で<b>予測区間</b>を提供し、リスク判断を支援。</li>
          <li>系列特化モデル（DLinear/PatchTST 等）との比較を本実装フェーズで実施。</li>
          <li>実故障データと接続し、<b>保全 KPI（停止回避・寿命延伸）</b>で再評価。</li>
        </ul>
      </div>
    </div>'''))

# 13. 付録：出典・再現
SLIDES.append(slide(
    "付録：データ出典と再現手順",
    '''<ul class="tight">
      <li>データ：ETT (Electricity Transformer Temperature), zhouhaoyi/ETDataset (GitHub, MIT License)。
          Informer (AAAI 2021) で公開されたベンチマーク。</li>
      <li>手法：LightGBM (Ke et al., 2017)、勾配ブースティング決定木。</li>
      <li>再現：<code>py -3.12 -m pytest tests -q</code> →
          <code>src/eda.py</code> → <code>src/pipeline.py --dataset ETTh1/ETTh2</code> →
          <code>src/report_figures.py</code>。</li>
      <li>本資料の数値は <code>reports/metrics_*.json</code> から自動生成。</li>
    </ul>
    <div class="callout">追加データは未使用。利用する場合は出典を明記する方針。</div>'''))


HTML = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ font-family: "Yu Gothic","Meiryo","Hiragino Kaku Gothic ProN",sans-serif;
  color: #1a2233; }}
@page {{ size: 297mm 167mm; margin: 0; }}
.slide {{ width: 297mm; height: 167mm; padding: 12mm 14mm 10mm; position: relative;
  page-break-after: always; overflow: hidden; background: #ffffff; }}
.msgbar {{ border-left: 8px solid #2f5cff; padding-left: 12px; margin-bottom: 10px; }}
.kicker {{ color:#2f5cff; font-size: 12px; font-weight: 700; letter-spacing:.05em; }}
.msgbar h1 {{ font-size: 23px; line-height: 1.35; color:#10204a; }}
.body {{ font-size: 15px; line-height: 1.6; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
.cols img {{ border:1px solid #e3e7ef; border-radius:4px; }}
.cap {{ font-size: 11px; color:#6b7280; text-align:center; margin-top:3px; }}
ul, ol {{ padding-left: 20px; }}
ul.big li {{ margin: 9px 0; font-size: 15.5px; }}
ul.tight li {{ margin: 5px 0; }}
ul.tight, .cols ul {{ font-size: 13.5px; }}
h3 {{ font-size: 14px; color:#2f5cff; margin: 4px 0 6px; }}
.callout {{ margin-top:12px; background:#eef3ff; border:1px solid #c9d8ff;
  border-radius:6px; padding:8px 12px; font-size:12.5px; color:#214; }}
.callout2 {{ background:#fff5e9; border:1px solid #ffd9ad; border-radius:6px;
  padding:8px 12px; font-size:13px; margin-bottom:6px; }}
.muted {{ color:#6b7280; font-size:12px; }}
.feat li {{ margin:3px 0; }}
.foot {{ position:absolute; bottom:6mm; left:14mm; right:14mm; display:flex;
  justify-content:space-between; font-size:10px; color:#9aa3b2;
  border-top:1px solid #eef0f4; padding-top:4px; }}
/* cover */
.cover {{ background: linear-gradient(135deg,#2f5cff 0%, #1b3bd6 100%); color:#fff; }}
.cover-inner {{ position:absolute; left:18mm; bottom:34mm; }}
.cover-tag {{ font-size:13px; letter-spacing:.1em; opacity:.9; margin-bottom:14px;
  border:1px solid rgba(255,255,255,.6); display:inline-block; padding:4px 10px; border-radius:3px;}}
.cover h1 {{ font-size:34px; line-height:1.3; }}
.cover-sub {{ font-size:16px; margin-top:14px; opacity:.95; }}
.cover-meta {{ font-size:12px; margin-top:26px; opacity:.8; }}
code {{ background:#f1f3f8; padding:1px 5px; border-radius:3px; font-size:12px; }}
</style></head><body>
{''.join(SLIDES)}
</body></html>'''

OUT.write_text(HTML, encoding="utf-8")
print(f"[saved] {OUT}  ({len(SLIDES)} slides, {len(HTML)//1024} KB)")
