"""PhoneMatch: an explainable smartphone recommender.   streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.config import DIM_LABELS, DIMS, FEATURE_LABELS, PERSONAS, RESULTS_DIR
from src.data import load_dataset_info
from src.pipeline import build
from src.sentiment import aspects_in, split_sentences
from src.value_model import explain_price

st.set_page_config(page_title="PhoneMatch", page_icon="📱", layout="wide")

INK, COBALT, MINT, AMBER, CORAL, VIOLET, MUTED = "#131A2A", "#2D55FF", "#0FA68E", "#E8912D", "#E5534B", "#7A5CFA", "#6B7385"
FONT_STACK = "DM Sans, Source Sans Pro, Helvetica, Arial, sans-serif"
pio.templates["phonematch"] = go.layout.Template(layout=dict(
    colorway=[COBALT, MINT, AMBER, CORAL, VIOLET, INK], font=dict(family=FONT_STACK, color=INK)))
pio.templates.default = "plotly+phonematch"
DIM_COLORS = {"performance": COBALT, "camera": VIOLET, "battery": MINT, "display": AMBER, "value": CORAL}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--ink:#131A2A;--line:#E3E6EE;--cobalt:#2D55FF;--mint:#0FA68E;--amber:#E8912D;--coral:#E5534B;--muted:#6B7385}
.stApp{font-family:'DM Sans',system-ui,sans-serif;color:var(--ink)}
h1,h2,h3,.pm-title,.pm-rank,.pm-ring span,.pm-brand{font-family:'Bricolage Grotesque','DM Sans',sans-serif;letter-spacing:-0.02em}
footer{visibility:hidden}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.pm-hero{padding:4px 0 10px}
.stMarkdown p.pm-brand{font-size:46px;font-weight:800;line-height:1.05;margin:0}
.stMarkdown p.pm-tag{color:var(--muted);font-size:17px;max-width:640px;margin:6px 0 0}
.pm-note{background:#FFF6E6;border:1px solid #F5DDB0;color:#7A4A06;border-radius:12px;padding:10px 14px;font-size:13.5px;margin:12px 0 4px}
.pm-card{display:grid;grid-template-columns:46px 1.25fr 1.15fr 92px;gap:22px;align-items:center;background:#fff;
  border:1px solid var(--line);border-radius:18px;padding:18px 22px;margin:0 0 12px}
.pm-top{border-color:var(--cobalt)!important;box-shadow:0 0 0 3px rgba(45,85,255,.08)}
.pm-rank{font-size:30px;font-weight:800;color:var(--cobalt)}
.pm-title{font-size:20px;font-weight:700;line-height:1.2}
.pm-price{font-size:17px;font-weight:700;margin-top:2px}
.pm-price small{color:var(--muted);font-weight:400;font-size:12.5px;margin-left:6px}
.pm-chip{display:inline-block;background:#EEF1F8;border-radius:999px;padding:3px 10px;font-size:12.5px;margin:7px 6px 0 0;color:#2b3550}
.pm-good{background:#FFF1DC;color:#8A4F08}.pm-bad{background:#FDE7E5;color:#9B2C26}.pm-seg{background:#E8EDFF;color:#1F3BC4}
.pm-bar{display:grid;grid-template-columns:104px 1fr 30px;gap:8px;align-items:center;font-size:12.5px;margin:4px 0}
.pm-track{height:7px;background:#EDF0F6;border-radius:99px;overflow:hidden}
.pm-fill{height:100%;border-radius:99px}
.pm-off{opacity:.4}
.pm-ring{width:84px;height:84px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--cobalt) calc(var(--p)*1%),#E6E9F2 0)}
.pm-ring span{width:64px;height:64px;border-radius:50%;background:#fff;display:grid;place-items:center;font-size:20px;font-weight:800}
@media(max-width:900px){.pm-card{grid-template-columns:34px 1fr;gap:12px}.pm-bars,.pm-ring{grid-column:1/-1}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def inr(x):
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return "₹" + s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts + [tail])


@st.cache_resource(show_spinner="Training the models (first launch only)...")
def get_bundle():
    return build()


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(family=FONT_STACK, color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#E9ECF3", zerolinecolor="#E9ECF3")
    fig.update_yaxes(gridcolor="#E9ECF3", zerolinecolor="#E9ECF3")
    return fig


def card_html(rank, r, weights):
    gap = r.value_gap * 100
    if gap >= 10:
        badge = f'<span class="pm-chip pm-good">Underpriced by {gap:.0f}%</span>'
    elif gap <= -10:
        badge = f'<span class="pm-chip pm-bad">Priced {abs(gap):.0f}% above its specs</span>'
    else:
        badge = '<span class="pm-chip">Fairly priced</span>'
    chips = (f'<span class="pm-chip">{r.ram_gb} GB + {r.storage_gb} GB</span>'
             f'<span class="pm-chip">{r.battery_mah} mAh, {r.charging_w} W</span>'
             f'<span class="pm-chip">{r.rear_camera_mp} MP camera</span>'
             f'<span class="pm-chip">{r.refresh_hz} Hz</span>' + ('<span class="pm-chip">5G</span>' if r.is_5g else ""))
    bars = ""
    for d in DIMS:
        off = " pm-off" if weights.get(d, 0) == 0 else ""
        v = getattr(r, f"b_{d}")
        bars += (f'<div class="pm-bar{off}"><span>{DIM_LABELS[d]}</span><div class="pm-track"><div class="pm-fill" '
                 f'style="width:{v:.0f}%;background:{DIM_COLORS[d]}"></div></div><b>{v:.0f}</b></div>')
    top = " pm-top" if rank == 1 else ""
    return (f'<div class="pm-card{top}"><div class="pm-rank">{rank}</div>'
            f'<div><div class="pm-title">{r.model}</div>'
            f'<div class="pm-price">{inr(r.price_inr)}<small>fair price about {inr(r.pred_price)}</small></div>'
            f'{chips}<br>{badge}<span class="pm-chip pm-seg">{r.cluster_name}</span></div>'
            f'<div class="pm-bars">{bars}</div>'
            f'<div class="pm-ring" style="--p:{r.match:.0f}" aria-label="Match score {r.match:.0f} out of 100"><span>{r.match:.0f}</span></div></div>')


def load_result(name):
    f = RESULTS_DIR / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


B = get_bundle()
ph, rec = B.phones, B.recommender
info = load_dataset_info()

# ----------------------------------------------------------------------------- sidebar
def apply_preset():
    p = PERSONAS.get(st.session_state.preset)
    if p:
        for d in DIMS:
            st.session_state[f"w_{d}"] = p["weights"][d]
        st.session_state["budget"] = p["budget"]


with st.sidebar:
    st.markdown("### Your priorities")
    st.selectbox("Start from a buyer profile", ["Custom"] + list(PERSONAS), key="preset", on_change=apply_preset)
    for d in DIMS:
        st.slider(DIM_LABELS[d], 0, 10, 5, key=f"w_{d}")
    st.markdown("### Budget and filters")
    st.slider("Budget (₹)", 5000, 150000, (15000, 30000), step=1000, key="budget")
    brands = st.multiselect("Brands (all if empty)", sorted(ph.brand.unique()))
    min_ram = st.select_slider("Minimum RAM (GB)", options=[0, 4, 6, 8, 12, 16], value=0)
    only_5g = st.toggle("5G only", value=False)
    st.markdown("### Method")
    trust = st.slider("Trust in owner reviews", 0, 100, 30, format="%d%%",
                      help="0% ranks by specs alone. 100% ranks by what owners say about each aspect.")
    k = st.slider("How many results", 3, 10, 5)

weights = {d: st.session_state.get(f"w_{d}", 5) for d in DIMS}
lo, hi = st.session_state.get("budget", (15000, 30000))
res = rec.recommend(weights, (lo, hi), min_ram, brands or None, only_5g, trust / 100, k)

# ----------------------------------------------------------------------------- header
st.markdown('<div class="pm-hero"><p class="pm-brand">PhoneMatch</p>'
            '<p class="pm-tag">Rank phones by what you care about, using specs, what owners say in reviews, '
            'and whether the price is fair.</p></div>', unsafe_allow_html=True)
if info.get("type") == "synthetic":
    st.markdown('<div class="pm-note"><b>Demo data.</b> The bundled catalogue is synthetic: fictional brands with '
                'simulated specs, prices and reviews. Swap in real data to use it for real decisions (see README).</div>',
                unsafe_allow_html=True)

tabs = st.tabs(["Find a phone", "Compare", "Market map", "Value check", "Review insights", "Model lab", "About the data"])

# ----------------------------------------------------------------------------- 1. find
with tabs[0]:
    if res.empty:
        st.warning("No phones match these filters. Widen the budget, lower the minimum RAM or clear the brand filter.")
    else:
        st.subheader(f"Top {len(res)} between {inr(lo)} and {inr(hi)}")
        for i, (_, r) in enumerate(res.iterrows(), 1):
            st.markdown(card_html(i, r, weights), unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("##### Why these phones ranked here")
            long = res.melt(id_vars="model", value_vars=[f"c_{d}" for d in DIMS], var_name="dim", value_name="pts")
            long["dim"] = long.dim.str.replace("c_", "").map(DIM_LABELS)
            fig = px.bar(long, y="model", x="pts", color="dim", orientation="h",
                         color_discrete_map={DIM_LABELS[d]: DIM_COLORS[d] for d in DIMS},
                         labels={"pts": "Points toward match score", "model": "", "dim": ""})
            fig.update_yaxes(categoryorder="array", categoryarray=res.model.tolist()[::-1])
            st.plotly_chart(style(fig, 330), width="stretch")
        with right:
            st.markdown("##### Inspect a phone")
            pick = st.selectbox("Choose a result", res.model.tolist(), label_visibility="collapsed")
            pid = int(res[res.model == pick].phone_id.iloc[0])
            eff = explain_price(B.value_model, ph, pid)
            eff.index = [FEATURE_LABELS[i] for i in eff.index]
            fig = go.Figure(go.Bar(x=eff.values, y=eff.index, orientation="h",
                                   marker_color=[MINT if v >= 0 else CORAL for v in eff.values]))
            fig.update_layout(title="How each spec moves the fair price vs a typical phone (%)")
            st.plotly_chart(style(fig, 330), width="stretch")
        st.markdown(f"##### Phones similar to {pick}")
        sim = rec.similar(pid, 5)
        st.dataframe(sim.assign(Price=sim.price_inr.map(inr), Similarity=(sim.similarity * 100).round(0).astype(int).astype(str) + "%")
                     [["model", "Price", "ram_gb", "battery_mah", "camera_score", "refresh_hz", "Similarity"]]
                     .rename(columns={"model": "Phone", "ram_gb": "RAM (GB)", "battery_mah": "Battery (mAh)",
                                      "camera_score": "Camera score", "refresh_hz": "Refresh (Hz)"}),
                     hide_index=True, width="stretch")
        st.download_button("Download these results (CSV)",
                           res[["model", "brand", "price_inr", "pred_price", "match"]].round(1).to_csv(index=False),
                           "phonematch_results.csv", "text/csv")

# ----------------------------------------------------------------------------- 2. compare
with tabs[1]:
    st.subheader("Compare phones side by side")
    names = ph.sort_values("model").model.tolist()
    default = res.model.tolist()[:3] if not res.empty else names[:3]
    chosen = st.multiselect("Pick 2 to 4 phones", names, default=default, max_selections=4)
    if len(chosen) < 2:
        st.info("Pick at least two phones to compare.")
    else:
        sel_ids = ph[ph.model.isin(chosen)].phone_id.tolist()
        rows, prof = rec.profile(sel_ids, trust / 100)
        fig = go.Figure()
        for (_, r), vals in zip(rows.iterrows(), prof):
            fig.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]], theta=[DIM_LABELS[d] for d in DIMS] + [DIM_LABELS[DIMS[0]]],
                                          name=r.model, fill="toself", opacity=0.55))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#E3E6EE")), title="Blended score per dimension (0 to 100)")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(style(fig, 430), width="stretch")
        with c2:
            t = rows.set_index("model")[["brand", "price_inr", "pred_price", "ram_gb", "storage_gb", "battery_mah", "charging_w",
                                         "rear_camera_mp", "front_camera_mp", "camera_score", "display_inch", "refresh_hz", "is_5g", "rating"]].T
            t.loc["price_inr"] = t.loc["price_inr"].map(inr)
            t.loc["pred_price"] = t.loc["pred_price"].map(inr)
            t.index = ["Brand", "Price", "Fair price (model)", "RAM (GB)", "Storage (GB)", "Battery (mAh)", "Charging (W)",
                       "Rear camera (MP)", "Selfie camera (MP)", "Camera score", "Screen (inch)", "Refresh (Hz)", "5G", "Store rating"]
            st.dataframe(t.astype(str), width="stretch", height=430)

# ----------------------------------------------------------------------------- 3. market map
with tabs[2]:
    cl = B.clusters
    st.subheader(f"The market falls into {cl['k']} segments")
    st.caption(f"K-Means on specs and price, projected to 2D with PCA ({cl['explained'].sum() * 100:.0f}% of variance shown). "
               "Stars mark your current top results.")
    fig = px.scatter(ph, x="pc1", y="pc2", color="cluster_name", hover_name="model",
                     hover_data={"price_inr": True, "pc1": False, "pc2": False, "cluster_name": False},
                     labels={"pc1": "Spec and price axis 1", "pc2": "Spec and price axis 2", "cluster_name": "Segment"},
                     opacity=0.75)
    if not res.empty:
        fig.add_trace(go.Scatter(x=res.pc1, y=res.pc2, mode="markers+text", text=list(range(1, len(res) + 1)), textposition="top center",
                                 marker=dict(symbol="star", size=17, color=INK), name="Your top results", hoverinfo="skip"))
    st.plotly_chart(style(fig, 470), width="stretch")
    summary = ph.groupby("cluster_name").agg(Phones=("phone_id", "count"), AvgPrice=("price_inr", "mean"),
                                             Performance=("spec_performance", "mean"), Camera=("spec_camera", "mean"),
                                             Battery=("spec_battery", "mean"), Display=("spec_display", "mean"),
                                             Value=("spec_value", "mean")).sort_values("AvgPrice")
    summary["AvgPrice"] = summary.AvgPrice.map(inr)
    for c in ["Performance", "Camera", "Battery", "Display", "Value"]:
        summary[c] = (summary[c] * 100).round(0).astype(int)
    st.markdown("##### Segment profiles (scores out of 100)")
    st.dataframe(summary, width="stretch")

# ----------------------------------------------------------------------------- 4. value check
with tabs[3]:
    st.subheader("Is the price fair for the specs?")
    st.caption("A gradient-boosting model predicts a fair price from specs alone. Each prediction comes from a model "
               "that never saw that phone (5-fold out-of-fold).")
    fig = px.scatter(ph, x="price_inr", y="pred_price", color="value_gap", hover_name="model", log_x=True, log_y=True,
                     color_continuous_scale=["#E5534B", "#F4F1E8", "#0FA68E"], range_color=(-0.4, 0.4),
                     labels={"price_inr": "Actual price (₹)", "pred_price": "Predicted fair price (₹)", "value_gap": "Value gap"})
    lim = [ph.price_inr.min(), ph.price_inr.max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=INK, dash="dash", width=1), name="Fair price line"))
    st.plotly_chart(style(fig, 430), width="stretch")
    inb = ph[ph.price_inr.between(lo, hi)]
    c1, c2 = st.columns(2)
    cols = {"model": "Phone", "price_inr": "Price", "pred_price": "Fair price", "value_gap": "Gap"}
    for col, title, d in [(c1, "Most underpriced in your budget", inb.nlargest(8, "value_gap")),
                          (c2, "Most overpriced in your budget", inb.nsmallest(8, "value_gap"))]:
        with col:
            st.markdown(f"##### {title}")
            t = d[list(cols)].rename(columns=cols).copy()
            t["Price"], t["Fair price"] = t.Price.map(inr), t["Fair price"].map(inr)
            t["Gap"] = (d.value_gap * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(t, hide_index=True, width="stretch")
    imp = load_result("permutation_importance")
    if imp is not None:
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     title="What drives price (permutation importance on held-out phones)")
        fig.update_traces(marker_color=COBALT)
        st.plotly_chart(style(fig, 360), width="stretch")

# ----------------------------------------------------------------------------- 5. reviews
with tabs[4]:
    st.subheader("What owners say")
    st.caption("A TF-IDF + logistic regression classifier scores every review sentence, and sentences are grouped by the "
               "aspect they mention. Scores are shrunk toward the catalogue average when a phone has few mentions.")
    who = st.selectbox("Phone", ph.sort_values("model").model.tolist(),
                       index=0 if res.empty else ph.sort_values("model").model.tolist().index(res.model.iloc[0]))
    row = ph[ph.model == who].iloc[0]
    c1, c2 = st.columns([2, 3])
    with c1:
        vals = [row[f"rev_raw_{d}"] * 100 for d in DIMS]
        fig = go.Figure(go.Bar(x=[DIM_LABELS[d] for d in DIMS], y=vals, marker_color=[DIM_COLORS[d] for d in DIMS],
                               text=[f"{v:.0f}% ({row[f'rev_n_{d}']} mentions)" for v, d in zip(vals, DIMS)], textposition="outside"))
        fig.update_layout(title="Share of positive sentiment by aspect", yaxis=dict(range=[0, 110]))
        st.plotly_chart(style(fig, 360), width="stretch")
    with c2:
        s = B.sentences[B.sentences.phone_id == row.phone_id].drop_duplicates("sentence")
        st.markdown("**Most positive sentences**")
        for _, x in s.nlargest(3, "p_pos").iterrows():
            st.markdown(f"> {x.sentence}  \n> <small>{DIM_LABELS[x.dim]}, {x.p_pos * 100:.0f}% positive</small>", unsafe_allow_html=True)
        st.markdown("**Most negative sentences**")
        for _, x in s.nsmallest(3, "p_pos").iterrows():
            st.markdown(f"> {x.sentence}  \n> <small>{DIM_LABELS[x.dim]}, {x.p_pos * 100:.0f}% positive</small>", unsafe_allow_html=True)
    st.divider()
    st.markdown("##### Try the classifier")
    text = st.text_area("Write a review", "The camera is stunning but the battery drains too fast. Great value for the price.", height=90)
    if text.strip():
        overall = B.sentiment.predict_proba([text])[0, 1]
        st.progress(float(overall), text=f"Whole review: {overall * 100:.0f}% positive")
        parts = []
        for sent in split_sentences(text):
            for d in aspects_in(sent) or ["(no aspect found)"]:
                parts.append({"Sentence": sent, "Aspect": DIM_LABELS.get(d, d),
                              "Positive": f"{B.sentiment.predict_proba([sent])[0, 1] * 100:.0f}%"})
        st.dataframe(pd.DataFrame(parts), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 6. model lab
with tabs[5]:
    st.subheader("Model lab: experiments and evaluation")
    st.caption("These tables are produced by `python run_experiments.py` and saved in the results folder.")
    if load_result("value_model_comparison") is None:
        st.info("Run `python run_experiments.py` to generate the evaluation tables.")
    else:
        st.markdown("#### 1. Fair-price model (5-fold cross-validation)")
        st.dataframe(load_result("value_model_comparison"), hide_index=True, width="stretch")
        tune = load_result("value_model_tuning")
        st.caption("Best Gradient Boosting settings from grid search: " + ", ".join(f"{c} = {tune[c].iloc[0]}" for c in tune.columns))
        ab = load_result("ablation")
        fig = px.bar(ab, x="Removed group", y="R2 drop", title="Ablation: R² lost when a feature group is removed")
        fig.update_traces(marker_color=COBALT)
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption("Near-zero drops mean the other specs carry overlapping information, which is common because specs move together with the price tier.")

        st.markdown("#### 2. Review sentiment classifiers (5-fold, stratified)")
        st.dataframe(load_result("sentiment_comparison"), hide_index=True, width="stretch")
        st.dataframe(load_result("sentiment_confusion"), hide_index=True, width="stretch")

        st.markdown("#### 3. Clustering validation")
        val = load_result("clustering_validation")
        metric = st.radio("Metric", ["silhouette", "davies_bouldin", "calinski_harabasz"], horizontal=True,
                          format_func=lambda m: {"silhouette": "Silhouette (higher is better)",
                                                 "davies_bouldin": "Davies-Bouldin (lower is better)",
                                                 "calinski_harabasz": "Calinski-Harabasz (higher is better)"}[m])
        fig = px.line(val, x="k", y=metric, color="algorithm", markers=True)
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Two clusters score best on silhouette but give only a coarse split, so the app picks the best K-Means k "
                   "between 3 and 6 to get segments a buyer can act on.")

        st.markdown("#### 4. Recommender comparison on simulated buyer personas")
        st.dataframe(load_result("recommender_comparison"), hide_index=True, width="stretch")
        st.caption("There are no real user ratings, so these are proxy metrics averaged over six personas. The top-rated baseline "
                   "wins on store rating by construction but ignores the buyer's priorities; the hybrid raises spec fit and review fit.")
        ts = load_result("trust_sweep")
        fig = px.line(ts, x="Trust in reviews", y=["Spec fit", "Review fit"], markers=True, title="Trading specs against reviews")
        st.plotly_chart(style(fig, 320), width="stretch")

# ----------------------------------------------------------------------------- 7. about data
with tabs[6]:
    st.subheader("About the data")
    ds = Path(__file__).parent / "DATASET.md"
    st.markdown(ds.read_text() if ds.exists() else "See DATASET.md")
    st.markdown("##### Phones (first 50 rows)")
    st.dataframe(ph[["phone_id", "brand", "model", "price_inr", "ram_gb", "storage_gb", "battery_mah", "camera_score",
                     "refresh_hz", "processor_score", "is_5g", "rating"]].head(50), hide_index=True, width="stretch")
    st.markdown("##### Reviews (first 30 rows)")
    st.dataframe(B.reviews.head(30), hide_index=True, width="stretch")
