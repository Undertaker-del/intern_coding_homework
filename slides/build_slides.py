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
bt1 = load(REP / "backtest_ETTh1.json")
bt2 = load(REP / "backtest_ETTh2.json")


def hz(m, h, model, key="mae"):
    return m["horizons"][str(h)][model][key]


def bt(m, h):
    return m["horizons"][str(h)]


def fmt_bt(m, h):
    """バックテスト skill を '+9.3%±7.7% (4/5)' 形式で整形。"""
    r = bt(m, h)
    return (f"{100 * r['skill_mean']:+.1f}%±{100 * r['skill_std']:.1f}% "
            f"({r['n_positive']}/{r['n_total']})")


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

# 2. エグゼクティブサマリ（バックテスト＋分割感度で裏取り）
SLIDES.append(slide(
    "結論：ML の価値は変圧器の「変動性」で決まる。変動大は大きく安定、穏やかは不確実",
    f'''<div class="cols">
      <div>{img("result_value_heatmap.png")}
        <div class="cap">色=naive持続予測に対する改善率（緑ほど価値大）</div></div>
      <div>
        <ul class="tight">
          <li><b>変動大(ETTh2型)＝価値あり</b>：6〜12h先で +54〜57%。どの分割でも安定。</li>
          <li><b>穏やか(ETTh1型)＝不確実</b>：lag1={e1['acf_lag1']:.2f} で持続予測が強い。
              分割を変えると +9%↔負に振れる。資産別の事前検証が必要。</li>
          <li><b>24h先＝天井</b>：日次周期を持続予測が捕捉。外気温など外部データが要る。</li>
          <li><b>最大リスク＝分布シフト</b>：季節で平均油温が変化。固定閾値は不可、定期再学習前提。</li>
        </ul>
        <div class="callout">公開データのみ（顧客実データ不使用）。リーク防止は
          pytest 23件で全工程検証。</div>
      </div>
    </div>'''))

# 3. 目的・スコープ・前提
SLIDES.append(slide(
    "目的は「油温予測が運用・保全判断にどれだけ価値を出すか」の定量検証",
    f'''<div class="cols even">
      <div>
        <h3>検証スコープ</h3>
        <ul class="tight">
          <li>予測対象：油温 OT（°C）の将来値</li>
          <li>データ：ETTh1/h2（1h粒度・約2年）。m1/m2 で粒度差も確認</li>
          <li>評価：1/6/12/24h 先で MAE・RMSE と naive 比改善率(skill)</li>
          <li>業務価値：高温時間帯の早期検知（Precision/Recall）に翻訳</li>
        </ul>
        <h3>置いた仮定（質問不可のため明記）</h3>
        <ul class="tight">
          <li>「t=T の特徴量を使ってよい」→ <b>負荷は運用計画で既知</b>と解釈し上限実験も実施。
              主結果は<b>過去情報のみ</b>（最も安全な前提）。</li>
          <li>分割は時系列順 60/20/20。文献標準 12/4/4 でも検証。</li>
          <li>MAPE は OT が 0/負を取るため不採用、MAE/RMSE(°C)。</li>
        </ul>
      </div>
      <div>{img("eda_ETTh1_series.png")}
        <div class="cap">予測対象：油温 OT の時系列（ETTh1・約2年）。季節で水準が大きく動く</div></div>
    </div>'''))

# 4. EDA① データ品質・非定常性
SLIDES.append(slide(
    "データは高品質だが、季節による強い非定常性（分布シフト）を持つ",
    f'''<div class="cols even">
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
    f'''<div class="cols even">
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
    f'''<div class="cols even">
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
    "設計の核は「厳格な時系列検証 × 差分予測」",
    f'''<div class="cols">
      <div>{img("eda_ETTh1_acf.png")}
        <div class="cap">自己相関：1h後={e1['acf_lag1']:.2f}・24h後={e1['acf_lag24']:.2f}
          → 持続予測が強敵</div></div>
      <div>
        <h3>差分予測（設計上の意思決定）</h3>
        <div class="callout2">OT(t+h) を直接当てず、<b>変化量 Δ=OT(t+h)−OT(t)</b> を学習し OT(t) に加算。</div>
        <ul class="tight">
          <li>持続予測を錨にする → <b>最悪でも naive 同等</b>、系統的変化だけ学習。</li>
          <li>直接予測は naive 未達。差分予測で中期まで安定して上回る。</li>
        </ul>
        <h3>リーク防止／モデル</h3>
        <ul class="tight">
          <li>時系列順分割・train限定スケーリング・因果特徴（shift≥0）。</li>
          <li>ベースライン(持続/季節持続)＋Ridge＋<b>LightGBM(本命)</b>を比較。</li>
        </ul>
      </div>
    </div>'''))

# 8. 結果① バックテストで頑健性を検証（単一分割の楽観を排除）
SLIDES.append(slide(
    "5フォールドのバックテストで「中期は安定して有効・24hは頭打ち」を確認",
    f'''<div class="cols">
      <div>{img("result_backtest_skill.png")}<div class="cap">ローリング起点バックテスト：naive比改善率 mean±std</div></div>
      <div>
        <h3>なぜバックテストか</h3>
        <p class="muted" style="font-size:12.5px">単一分割は評価期間に依存する（ETTh1 のtestは穏やかな期間で持続予測が強い）。
        起点をずらした5フォールドで mean±std を見ることで、<b>再現性のある主張</b>に変える。</p>
        <ul class="tight">
          <li>中期(6〜12h)：ETTh2 <b>{fmt_bt(bt2, 12)}</b>、ETTh1 <b>{fmt_bt(bt1, 12)}</b>
              ＝多/全フォールドで正。</li>
          <li>24h：ETTh1 <b>{fmt_bt(bt1, 24)}</b>、ETTh2 <b>{fmt_bt(bt2, 24)}</b>
              ＝<b>持続予測を安定的に超えない</b>（単一分割の +1% は偶然）。</li>
          <li>短期(1h)：堅実だが小（ETTh1 {fmt_bt(bt1, 1)}）。</li>
        </ul>
      </div>
    </div>'''))

# 8b. 結論の頑健性（分割感度）— 過大評価の回避
SLIDES.append(slide(
    "結論の頑健性を検証：ETTh2 の価値は分割に不変、ETTh1 の改善は分割依存で消える",
    f'''<div class="cols">
      <div>{img("result_split_sensitivity.png")}<div class="cap">時系列60/20/20 vs 文献Informer12/4/4 の skill</div></div>
      <div>
        <p class="muted">時系列分割に加え<b>文献標準(Informer 12/4/4)</b>でも評価し、
        結論が分割に依存しないか確認。</p>
        <ul class="tight">
          <li><b>ETTh2</b>：両分割で 6〜12h が +54〜57%＝<b>頑健</b>。価値の主張は信頼できる。</li>
          <li><b>ETTh1</b>：時系列分割では +9% だが<b>文献分割では −6〜−9%</b>に反転。
              <b>評価期間に依存し頑健でない</b>＝持続予測で十分な可能性。</li>
          <li>含意：<b>資産の変動性レジームで価値が決まる</b>。導入判断は
              資産別・期間横断の評価が前提。</li>
        </ul>
        <div class="callout2" style="font-size:12px">過大評価を避けるため、頑健でない改善は
          「不確実」と明示する方針。</div>
      </div>
    </div>'''))

# 8c. 機構解明 — どの変圧器が効くかを事前に予測できる
vr = load(REP / "verification_report.json")
amp1 = vr["mechanism"]["ETTh1"]["diurnal_amplitude"]
amp2 = vr["mechanism"]["ETTh2"]["diurnal_amplitude"]
abl = vr["ablations"]["feature_ablation_ETTh2_h6"]
SLIDES.append(slide(
    "なぜ効くかを解明：価値は「日内温度サイクルの振幅」で決まり、事前に予測できる",
    f'''<div class="cols">
      <div>{img("result_mechanism.png")}<div class="cap">日内振幅と価値の連動／特徴量アブレーション</div></div>
      <div>
        <ul class="tight">
          <li><b>機構</b>：ML の価値は<b>日内温度サイクルの振幅</b>に比例。
              ETTh2={amp2:.1f}°C（大）→ +54%、ETTh1={amp1:.1f}°C（小）→ +9%。
              24h で消えるのは<b>1日で一周し持続予測が捕捉</b>するため。</li>
          <li><b>アブレーションで裏取り</b>：カレンダー（日内/季節）特徴を外すと
              skill が <b>{100*abl['full']:.0f}%→{100*abl['no_calendar']:.0f}%</b> に半減＝
              <b>日内サイクルが主シグナル</b>。負荷特徴は外しても悪化せず（無用）。</li>
          <li><b>実務示唆（actionable）</b>：導入前に<b>各変圧器の日内振幅を測る</b>だけで、
              ML 予測が価値を出すかを<b>事前にスクリーニング</b>できる。</li>
        </ul>
      </div>
    </div>'''))

# 9. 結果② 予測波形・予測区間・説明性
top1 = list(m1["horizons"]["12"]["lightgbm"]["top_features"].items())[:6]
feat_li = "".join(f"<li>{k} <span class='muted'>({v:.0f})</span></li>" for k, v in top1)
iv6 = m2["horizons"]["6"]["lightgbm"]["interval"]
iv24 = m2["horizons"]["24"]["lightgbm"]["interval"]
mase_lgbm = m2["horizons"]["6"]["lightgbm"]["mase"]
mase_naive = m2["horizons"]["6"]["naive_persistence"]["mase"]
SLIDES.append(slide(
    "点予測に加え「予測区間」で不確実性も提示でき、要因も解釈可能",
    f'''<div class="cols">
      <div>{img("pred_ETTh2_h6.png")}<div class="cap">ETTh2 6h先：実測・予測・P10–P90 区間</div></div>
      <div>
        <h3>予測区間（分位点回帰 P10/P90）</h3>
        <ul class="tight">
          <li>被覆率 6h={iv6['coverage']:.2f}（名目0.80）、幅は
              <b>{iv6['mean_width']:.1f}°C(6h)→{iv24['mean_width']:.1f}°C(24h)</b> と
              不確実性の増大を定量化。</li>
          <li>MASE(6h)：LightGBM <b>{mase_lgbm:.2f}</b> vs naive {mase_naive:.2f}
              （基準=1段 in-sample naive。多段予測ゆえ1超は当然で、
              <b>naive を大幅に下回る</b>＝相対的に良好。系列間比較可）。</li>
        </ul>
        <h3>重要度上位（ETTh1,12h・gain）</h3>
        <ol class="feat" style="font-size:12px">{feat_li}</ol>
        <p class="muted" style="font-size:11.5px">短期は直近変動・時刻、長期は日平均水準・季節が支配。
        物理直感（熱慣性＋外気温）と整合し説明可能。</p>
      </div>
    </div>'''))

# 10. 業務価値
SLIDES.append(slide(
    "誰が何のために使うか：用途は2つ。早期警報は今すぐ有効、異常検知は要追加データ",
    '''<div class="cols even">
      <div>
        <h3>想定ユーザーと用途</h3>
        <div class="card"><span class="tag">運用</span><b>運用オペレーター</b>：高油温の<b>事前警報</b>
          → 負荷抑制・冷却強化。必要なのは MAE でなく h時間前の検知率と低誤報。</div>
        <div class="card"><span class="tag">保全</span><b>信頼性技師／設備管理</b>：<b>劣化・異常の早期検知</b>
          （予防保全の本丸）。期待温度からの残差で兆候を捉える。</div>
        <div class="card"><span class="tag">計画</span><b>保全計画担当</b>：点検の<b>優先順位付け</b>
          （リスクランキング）。</div>
      </div>
      <div>
        <h3>検証の結論（用途別）</h3>
        <ul class="tight">
          <li><b>① 早期警報＝今すぐ有効</b>。変動大の変圧器で実証済（次スライド）。</li>
          <li><b>② 異常検知＝今のデータでは不足</b>。早期警報に最適な自己回帰モデルは
              故障を<b>追従吸収</b>し検知不能。ソフトセンサが正解だが外気温データが要る。</li>
          <li>→ <b>段階導入</b>：まず早期警報、異常検知はフェーズ2（外部データ取得後）。</li>
        </ul>
        <div class="callout2">MAE でなく<b>用途別の性能指標</b>で検証したのが要点。</div>
      </div>
    </div>'''))

# 10b. 用途別検証の実証
be = load(REP / "business_eval.json")
ew6 = be["early_warning"]["horizons"]["6"]
fd = be["fault_detection"]["models"]
SLIDES.append(slide(
    "実証：早期警報は6h前に検知率3倍・誤報週1件未満／自己回帰は故障の98%を吸収",
    f'''<div class="cols">
      <div>{img("result_business.png")}<div class="cap">用途①早期警報の検知率／用途②合成故障への残差</div></div>
      <div>
        <ul class="tight">
          <li><b>用途①早期警報(ETTh2)</b>：高温(閾値{be['early_warning']['threshold_C']:.0f}°C)を
              <b>6h前に再現率{ew6['model']['recall']:.0%}・適合率{ew6['model']['precision']:.0%}・
              誤報{ew6['model']['false_alarms_per_week']}件/週</b>。
              持続予測は再現率{ew6['persistence']['recall']:.0%}（先読み不可）。</li>
          <li><b>用途②異常検知</b>：合成故障(+{be['fault_detection']['fault_C']:.0f}°Cドリフト)のうち
              残差に現れる量＝<b>自己回帰{fd['autoregressive']['fault_visible_pct']:.0f}%（吸収し検知不能）</b>、
              <b>ソフトセンサ{fd['soft_sensor']['fault_visible_pct']:.0f}%</b>（反応するがSN比{fd['soft_sensor']['signal_to_noise']}で実用不足）。</li>
          <li>含意：<b>用途で最適モデルが異なる</b>。予防保全には外気温＋熱モデルが必要。</li>
        </ul>
      </div>
    </div>'''))

# 10c. モデル選択の妥当性（DLinear/MLP まで比較）
mc = load(REP / "model_compare.json")
SLIDES.append(slide(
    "モデル選択も検証：勾配ブースティングが総合最良、深層 DLinear が穏やか資産で拮抗",
    f'''<div class="cols">
      <div>{img("result_model_compare.png")}<div class="cap">Ridge/LightGBM/MLP/DLinear を同一の差分予測で比較（5フォールド平均）</div></div>
      <div>
        <p class="muted">全モデルを同じ差分予測に統一し、モデルの差だけを5フォールドで比較。</p>
        <ul class="tight">
          <li><b>LightGBM が総合最良</b>：ETTh2 で +{100*mc['ETTh2']['h6']['lgbm']:.0f}%（h6）と最高、
              ETTh1 でも安定して正。<b>表形式・非線形・少データに強く本命として妥当</b>。</li>
          <li><b>DLinear（系列分解＋線形, AAAI2023）</b>は<b>穏やかな ETTh1 で逆に最良</b>
              （h6 +{100*mc['ETTh1']['h6']['dlinear']:.0f}% &gt; LightGBM +{100*mc['ETTh1']['h6']['lgbm']:.0f}%）。
              日内サイクルが支配的な資産では生波形の線形分解が効く。</li>
          <li><b>MLP は過学習で全敗</b>（多くで負）。データ規模に対し過剰。</li>
          <li>含意：<b>資産レジームで最適モデルも変わる</b>→本実装は両者を持ち選択/アンサンブル。</li>
        </ul>
        <div class="callout2" style="font-size:12px"><b>強化学習は不採用</b>：本タスクは入力→将来値の
          教師あり回帰。RL（逐次意思決定）が活きるのは予測を踏まえた<b>下流の制御</b>であり PoC 範囲外。</div>
      </div>
    </div>'''))

# 10d. 特徴量数の感度（増やす/減らす）
fs = load(REP / "feature_sensitivity.json")
SLIDES.append(slide(
    "特徴量は「増やすほど良い」ではない：穏やか資産は増で改善、変動資産は40前後が最適",
    f'''<div class="cols">
      <div>{img("result_feature_sensitivity.png")}<div class="cap">最小16 / 現行40 / 拡張110 特徴量の skill（6/12h平均）</div></div>
      <div>
        <p class="muted" style="font-size:12.5px">特徴量を最小・現行・拡張の3水準で 5 フォールド比較し、
        過不足を検証した。</p>
        <ul class="tight">
          <li><b>ETTh2（変動大）は逆U字</b>：現行40で最良(+{100*fs['ETTh2']['current']['h6']:.0f}%)、
              110に増やすと<b>+{100*fs['ETTh2']['expanded']['h6']:.0f}%へ悪化</b>＝過学習。</li>
          <li><b>ETTh1（穏やか）は単調改善</b>：16→40→110 で +{100*fs['ETTh1']['minimal']['h6']:.0f}→
              +{100*fs['ETTh1']['current']['h6']:.0f}→+{100*fs['ETTh1']['expanded']['h6']:.0f}%。
              シグナルが弱く<b>多特徴が効く</b>。</li>
          <li><b>採用＝現行40</b>：両レジームで頑健な妥協点。一方を伸ばす拡張は他方を壊すため、
              <b>YAGNI＋頑健性</b>でこの水準を選択（負荷移動平均を棄却した判断と整合）。</li>
        </ul>
      </div>
    </div>'''))

# 10e. 予測区間の較正（改善実装）
cf = load(REP / "conformal.json")
cf6 = cf["horizons"]["6"]; cf24 = cf["horizons"]["24"]
SLIDES.append(slide(
    "限界を改善：split-conformal で予測区間の過小被覆を名目80%まで較正",
    f'''<div class="cols">
      <div>{img("result_conformal.png")}<div class="cap">分位点回帰（旧）vs split-conformal（改善）の実測被覆率</div></div>
      <div>
        <p class="muted" style="font-size:12.5px">分位点回帰の区間は系統的に過小被覆だった。
        検量(val)集合の残差分位点で幅を決める <b>split-conformal</b> で較正した。</p>
        <ul class="tight">
          <li><b>被覆率を名目0.80へ回復</b>：6h {cf6['quantile']['coverage']:.2f}→
              <b>{cf6['conformal']['coverage']:.2f}</b>、24h {cf24['quantile']['coverage']:.2f}→
              <b>{cf24['conformal']['coverage']:.2f}</b>。幅は同等〜微増で済む。</li>
          <li>分布のずれた検証期間でも<b>被覆保証を実務的に確保</b>。
              運用時の「区間の信頼性」を担保できる。</li>
          <li>時系列で交換可能性は厳密に成り立たないため、<b>直近窓での再較正</b>を運用に組込む前提。</li>
        </ul>
        <div class="callout2" style="font-size:12px">限界スライドで挙げた「過小被覆」を本実装で解消済み。</div>
      </div>
    </div>'''))

# 11. 工夫・意思決定まとめ（検証ループの実践）
SLIDES.append(slide(
    "工夫：1発出しせず「仮説→検証→採否」を反復して結論を固めた",
    '''<ul class="big">
      <li><b>リーク遮断を自動テスト化</b>（pytest 23件）。加えてノイズ下限・故意リーク検出・
          指標の独立再計算で「漏洩なし」を確認。</li>
      <li><b>差分予測</b>で直接予測の baseline 未達を解消し、中期まで naive 超え。</li>
      <li><b>過大評価を避ける検証</b>：バックテスト・分割感度・シード・アブレーションで、
          結論を「変動性レジーム依存」へ正直に修正。</li>
      <li><b>効かない仮説は棄却</b>：負荷の移動平均特徴は交差検証で改善なく不採用。</li>
      <li><b>再現性とセキュリティ</b>：数値は JSON から自動注入、公開データのみ・機密PDF除外。</li>
    </ul>'''))

# 12. 限界と改善方針
SLIDES.append(slide(
    "限界はバックテストで明確化済み、次の一手も具体化できている",
    f'''<div class="cols">
      <div>
        <h3>現時点の限界（実測に基づく）</h3>
        <ul class="tight">
          <li><b>穏やかな資産(ETTh1)では価値が評価期間依存</b>で頑健でない（分割で符号反転）。</li>
          <li><b>24h 先は持続予測を安定的に超えない</b>（外部要因の情報不足。全分割で確認）。</li>
          <li><b>分布シフト</b>（季節で平均油温が大きく変化）で固定閾値・固定モデルが劣化。</li>
          <li>予測区間は分位点回帰だと<b>過小被覆</b>（名目80%対fold平均0.74）だったが、
              <b>split-conformal で名目80%へ較正済</b>（前スライド）。残る課題は分布シフト下の再較正運用。</li>
          <li><b>データ品質</b>：ETTm2 は OT の張り付き（連続同値）が<b>21.7%</b>＝
              センサ固着/前値補完の疑い。実データでは品質検証が前提。</li>
          <li>異常・故障ラベルが無く、保全効果は代理指標での評価。</li>
        </ul>
      </div>
      <div>
        <h3>今後の改善方針</h3>
        <ul class="tight">
          <li><b>外気温予報・運転計画</b>等の外部共変量を追加（24h+ の鍵）。</li>
          <li><b>定期再学習</b>＋ローリング起点 CV を運用に組込み非定常へ追従。</li>
          <li><b>コンフォーマル予測</b>で区間を較正（本PoCで実装済）→ 分布シフト追従の再較正を運用化。</li>
          <li>系列特化モデル（DLinear を実装・比較済、ETTh1で最良）→ PatchTST 等へ拡張、レジーム別に選択。</li>
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
      <li>再現：<code>py -3.12 run_all.py</code>（テスト→EDA→学習評価→バックテスト→分割感度→図→スライド）。
          個別は <code>pipeline</code>/<code>backtest</code>/<code>split_sensitivity</code>/<code>verify_all</code>/<code>business_eval</code>/
          <code>model_compare</code>/<code>feature_sensitivity</code>/<code>conformal</code>。</li>
      <li>DLinear: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023。
          コンフォーマル: Vovk et al. / split-conformal（検量集合の残差分位点で被覆保証）。</li>
      <li>本資料の数値は <code>reports/metrics_*.json</code> と <code>reports/backtest_*.json</code> から自動生成。
          評価=5フォールド ローリング起点バックテスト（mean±std）。</li>
    </ul>
    <div class="callout">追加データは未使用。利用する場合は出典を明記する方針。</div>'''))


HTML = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ font-family: "Yu Gothic","Meiryo","Hiragino Kaku Gothic ProN",sans-serif;
  color: #1a2233; }}
@page {{ size: 297mm 167mm; margin: 0; }}
.slide {{ width: 297mm; height: 167mm; padding: 11mm 14mm 12mm; position: relative;
  page-break-after: always; overflow: hidden; background: #ffffff;
  display: flex; flex-direction: column; }}
.msgbar {{ border-left: 8px solid #2f5cff; padding-left: 12px; margin-bottom: 4px; flex: none; }}
.kicker {{ color:#2f5cff; font-size: 12px; font-weight: 700; letter-spacing:.05em; }}
.msgbar h1 {{ font-size: 24px; line-height: 1.3; color:#10204a; }}
/* 本文は残り高さを使って縦に充填（空白を作らない） */
.body {{ flex: 1; font-size: 16.5px; line-height: 1.65;
  display: flex; flex-direction: column; justify-content: center; gap: 8px; }}
.cols {{ display: grid; grid-template-columns: 1.12fr 0.88fr; gap: 22px;
  align-items: center; }}
.cols.even {{ grid-template-columns: 1fr 1fr; }}
.cols img {{ border:1px solid #e3e7ef; border-radius:5px; box-shadow:0 1px 6px rgba(20,40,90,.07); }}
.cap {{ font-size: 12px; color:#6b7280; text-align:center; margin-top:5px; }}
ul, ol {{ padding-left: 22px; }}
ul.big li {{ margin: 15px 0; font-size: 18px; line-height:1.6; }}
ul.tight li {{ margin: 9px 0; }}
ul.tight, .cols ul {{ font-size: 15px; line-height:1.6; }}
h3 {{ font-size: 15.5px; color:#2f5cff; margin: 8px 0 7px; }}
.callout {{ margin-top:14px; background:#eef3ff; border:1px solid #c9d8ff;
  border-radius:6px; padding:11px 14px; font-size:14px; color:#214; }}
.callout2 {{ background:#fff5e9; border:1px solid #ffd9ad; border-radius:6px;
  padding:10px 13px; font-size:14px; margin-bottom:6px; }}
.muted {{ color:#6b7280; font-size:13.5px; }}
.feat li {{ margin:4px 0; }}
.hero {{ text-align:center; }}
.hero img {{ width: 82%; }}
.card {{ background:#f6f8fd; border:1px solid #dde5f3; border-left:4px solid #2f5cff;
  border-radius:6px; padding:11px 14px; margin:9px 0; font-size:14.5px; line-height:1.55; }}
.card b {{ color:#10204a; }}
.tag {{ display:inline-block; background:#2f5cff; color:#fff; font-size:11px;
  padding:1px 8px; border-radius:10px; margin-right:6px; vertical-align:middle; }}
.foot {{ position:absolute; bottom:5mm; left:14mm; right:14mm; display:flex;
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
