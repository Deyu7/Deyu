# app.py — GSHS 行为问卷 · 抑郁风险预测 · 可视化
# Usage:
#   pip install streamlit plotly scikit-learn pandas numpy joblib
#   streamlit run app.py
#
# 左侧侧边栏可配置：
# - 行为模型 pkl：svm_pca_behavior_pipeline.pkl
# - 模型元数据 json：svm_pca_behavior_meta.json
# - 原始数据 CSV（用于计算人群均值/标准差/百分位）
# - （可选）文本模型 pkl：pipe_text_final.pkl / pipe_text_final2.pkl

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="青少年心理健康风险预测", page_icon="🧠", layout="wide")

# ========= Sidebar: 资源加载 =========
st.sidebar.header("⚙️ 配置 / 资源")
model_path = st.sidebar.text_input("行为模型文件（.pkl）", "svm_pca_behavior_pipeline.pkl")
meta_path  = st.sidebar.text_input("行为模型元数据（.json）", "svm_pca_behavior_meta.json")
data_path  = st.sidebar.text_input("数据集 CSV（用于计算人群基线）", "AAAcss project/CNAH2003_public_use.csv")
text_model_path = st.sidebar.text_input("（可选）文本模型（.pkl）", "")

st.sidebar.markdown("---")
st.sidebar.caption("提示：先在 Notebook 导出 pkl/json；数据集仅用于计算平均/标准差/百分位。")

# ========= 工具函数 =========
@st.cache_data(show_spinner=False)
def load_meta(meta_path: str):
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feats = meta.get("features", [])
        return meta, feats
    except Exception as e:
        st.sidebar.error(f"读取 meta 失败：{e}")
        return {}, []

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        pipe = joblib.load(model_path)
        return pipe
    except Exception as e:
        st.sidebar.error(f"加载模型失败：{e}")
        return None

@st.cache_data(show_spinner=False)
def load_dataset(data_path: str):
    try:
        df = pd.read_csv(data_path, low_memory=False)
        return df
    except Exception as e:
        st.sidebar.error(f"读取数据集失败：{e}")
        return None

def numeric_bounds(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return 0.0, 1.0, 0.5
    lo, hi = np.nanpercentile(x, [1, 99])
    mu, sd = np.nanmean(x), np.nanstd(x)
    default = float(mu) if np.isfinite(mu) else float(np.nanmedian(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(default):
        default = (lo + hi) / 2.0
    return float(lo), float(hi), float(default)

def compute_baseline(df: pd.DataFrame, features: list):
    feats_present = [c for c in features if c in df.columns]
    base = {}
    for c in feats_present:
        s = df[c]
        if s.dtype.kind in "biufc":  # numeric-like
            x = pd.to_numeric(s, errors="coerce")
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x))
            qs = {int(q): float(np.nanpercentile(x, q)) for q in range(1,100)}
            base[c] = {"kind": "num", "mean": mu, "std": sd, "quantiles": qs}
        else:
            vals = s.dropna().astype(str).value_counts().index.tolist()
            vals = vals[:50] if len(vals) > 50 else vals
            base[c] = {"kind": "cat", "choices": vals}
    return base

def zscore(x, mu, sd):
    if sd is None or sd == 0 or not np.isfinite(sd):
        return np.nan
    return (x - mu) / sd

def empirical_percentile(x, qs: dict):
    if not qs:
        return np.nan
    items = sorted(qs.items(), key=lambda kv: kv[0])
    loq, hiq = 1, 99
    for q, v in items:
        if x >= v:
            loq = q
        else:
            hiq = q
            break
    v_lo = qs.get(loq, None)
    v_hi = qs.get(hiq, None)
    if v_lo is None or v_hi is None or v_hi == v_lo:
        return float(loq)
    frac = (x - v_lo) / (v_hi - v_lo)
    return float(np.clip(loq + frac * (hiq - loq), 1, 99))

# ========= 加载资源 =========
meta, model_features = load_meta(meta_path)
pipe = load_model(model_path)
if pipe is None:
    st.stop()

df_all = load_dataset(data_path)
if df_all is None:
    st.stop()

# 仅用模型特征参与统计
def compute_baseline(df, features):
    feats_present = [c for c in features if c in df.columns]
    base = {}
    for c in feats_present:
        s = df[c]
        if s.dtype.kind in "biufc":
            x = pd.to_numeric(s, errors="coerce")
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
            qs = {int(q): float(np.nanpercentile(x, q)) for q in range(1,100)}
            base[c] = {"kind": "num", "mean": mu, "std": sd, "quantiles": qs}
        else:
            vals = s.dropna().astype(str).value_counts().index.tolist()
            base[c] = {"kind": "cat", "choices": vals[:50]}
    return base

baseline = compute_baseline(df_all, model_features)
feats_present = [c for c in model_features if c in baseline]
missing = [c for c in model_features if c not in feats_present]
if missing:
    st.warning(f"以下特征在数据集中未找到（控件将缺失，预测以缺失处理）：{missing[:12]}{' ...' if len(missing)>12 else ''}")

# ========= 页面结构 =========
st.title("🧠 青少年心理健康风险预测 · 问卷与可视化")
st.markdown("本应用基于你训练好的 **行为模型 Pipeline**。左侧配置 pkl/json/CSV 路径；填写问卷后生成 **概率、异常指标、z 分数、百分位、分布定位** 等可视化。")

tab_form, tab_result = st.tabs(["📝 填写问卷", "📊 结果与可视化"])

# ========= 问卷输入 =========
with tab_form:
    st.subheader("1) 行为问卷输入（仅显示模型使用的列）")
    cols = st.columns(2)
    user_input = {}
    for i, c in enumerate(feats_present):
        info = baseline[c]
        with cols[i % 2]:
            if info["kind"] == "num":
                lo, hi, default = numeric_bounds(df_all[c])
                step = 1.0 if (hi - lo) > 5 else 0.1
                val = st.slider(f"{c}", min_value=float(np.floor(lo)), max_value=float(np.ceil(hi)),
                                value=float(np.clip(default, lo, hi)), step=step,
                                help=f"均值≈{default:.2f}（数据集估计）")
                user_input[c] = val
            else:
                choices = ["<缺失>"] + info["choices"]
                val = st.selectbox(f"{c}", choices, index=0)
                user_input[c] = (None if val == "<缺失>" else val)

    st.subheader("2) （可选）行为日志 / 心情描述（自由文本）")
    st.caption("如提供文本模型 pkl，此处文本将得到一个“文本风险概率”，再与行为概率加权融合。")
    text_input = st.text_area("输入最近一周的情况（可留空）", height=140, placeholder="例：最近作息不规律，晚上刷手机到很晚；上课容易分心……")
    w_behavior = st.slider("融合加权（行为概率权重）", 0.0, 1.0, 0.7, 0.05)
    submitted = st.button("🚀 生成预测与图表", type="primary")

# ========= 预测与可视化 =========
if submitted:
    with tab_result:
        st.subheader("预测结果")
        X_one = pd.DataFrame([{c: user_input.get(c, np.nan) for c in model_features}])
        try:
            proba_behavior = float(pipe.predict_proba(X_one)[0, 1])
        except Exception as e:
            st.error(f"行为概率计算失败：{e}")
            st.stop()

        proba_text = None
        if text_model_path and Path(text_model_path).exists() and text_input.strip():
            try:
                pipe_text = joblib.load(text_model_path)
                proba_text = float(pipe_text.predict_proba(pd.Series([text_input]))[0, 1])
            except Exception as e:
                st.warning(f"文本模型不可用：{e}")

        if proba_text is not None:
            w_text = 1.0 - w_behavior
            proba_final = w_behavior * proba_behavior + w_text * proba_text
            st.caption(f"融合概率 = 行为 {w_behavior:.2f} × {proba_behavior:.3f} + 文本 {w_text:.2f} × {proba_text:.3f}")
        else:
            proba_final = proba_behavior
            st.caption("未提供文本模型或文本为空：仅使用行为概率。")

        if proba_final >= 0.6:
            verdict = "🔴 高风险（建议尽快联系专业人士/学校心理中心）"
        elif proba_final >= 0.3:
            verdict = "🟡 预警（建议进行线上随访或干预）"
        else:
            verdict = "🟢 安全（建议维持良好作息与支持）"

        g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_final*100,
            number={'suffix': '%', 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#636EFA'},
                'steps': [
                    {'range': [0, 30], 'color': '#E8F5E9'},
                    {'range': [30, 60], 'color': '#FFF9C4'},
                    {'range': [60, 100], 'color': '#FFEBEE'},
                ],
                'threshold': {'line': {'color': 'red', 'width': 3}, 'thickness': 0.75, 'value': 60}
            },
            title={'text': "总体风险概率"}
        ))
        st.plotly_chart(g, use_container_width=True)
        st.success(f"判定：{verdict}")

        st.markdown("---")
        st.subheader("个体位置 · 偏差与百分位")

        rows = []
        for c in feats_present:
            info = baseline[c]
            val = user_input.get(c, None)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            if info["kind"] == "num":
                mu, sd = info.get("mean"), info.get("std")
                z = zscore(float(val), mu, sd)
                pct = empirical_percentile(float(val), info.get("quantiles", {}))
                rows.append({"feature": c, "value": val, "mean": mu, "std": sd, "z": z, "percentile": pct})
        rep = pd.DataFrame(rows).sort_values("z", key=lambda s: np.abs(s), ascending=False)

        if rep.empty:
            st.info("暂无可计算 z 分数的数值型特征（可能全部为分类题）。")
        else:
            topK = min(15, len(rep))
            fig = px.bar(rep.head(topK), x="z", y="feature", orientation="h",
                         color=rep.head(topK)["z"].apply(lambda v: "高于均值" if v>=0 else "低于均值"),
                         color_discrete_map={"高于均值":"#EF553B", "低于均值":"#00CC96"},
                         title=f"行为指标 z 分数（Top {topK} |z|）")
            fig.update_layout(yaxis=dict(title="特征"), xaxis=dict(title="z 分数"))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(rep.head(topK), x="percentile", y="feature", orientation="h",
                          range_x=[0,100], title="行为指标百分位（Top 影响项）")
            st.plotly_chart(fig2, use_container_width=True)

            rep["abs_z"] = rep["z"].abs()
            flagged = rep[rep["abs_z"] >= 2].copy()
            if flagged.empty:
                st.info("未发现 |z| ≥ 2 的异常指标。")
            else:
                st.warning("⚠️ 异常指标（|z| ≥ 2）")
                st.dataframe(flagged[["feature","value","mean","std","z","percentile"]], use_container_width=True)

        st.markdown("—")
        with st.expander("📎 查看人群分布定位（最多 6 个数值题）", expanded=False):
            numeric_feats = [c for c in feats_present if baseline[c]["kind"]=="num"]
            pick = st.multiselect("选择要展示的特征（最多 6 个）", numeric_feats, default=numeric_feats[:6], max_selections=6)
            cols3 = st.columns(3)
            for i, c in enumerate(pick):
                with cols3[i%3]:
                    x = pd.to_numeric(df_all[c], errors="coerce")
                    figd = px.histogram(x.dropna(), nbins=30, title=c)
                    v = user_input.get(c, None)
                    if v is not None and np.isfinite(float(v)):
                        figd.add_vline(x=float(v), line_color="red")
                    st.plotly_chart(figd, use_container_width=True)

        if proba_text is not None:
            st.markdown("---")
            st.subheader("文本通道（可选）")
            st.write(f"文本风险概率：**{proba_text:.3f}**（融合按侧边栏权重）")
            st.caption("说明：文本模型为 TF-IDF + LogisticRegression；需在侧边栏填入 pkl 路径。")

        st.markdown("---")
        st.subheader("导出")
        if rep is not None and not rep.empty:
            csv = rep.to_csv(index=False).encode("utf-8")
            st.download_button("下载个人偏差报告（CSV）", data=csv, file_name="individual_deviation_report.csv", mime="text/csv")
