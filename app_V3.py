# app.py — GSHS 行为问卷 · 抑郁风险预测 · 可视化
# Usage:
#  
#   streamlit run app.py
#
# 左侧侧边栏可配置：
# - 行为模型 pkl：svm_pca_behavior_pipeline.pkl
# - 模型元数据 json：svm_pca_behavior_meta.json
# - 原始数据 CSV（用于计算人群均值/标准差/百分位）
# - （可选）文本模型 pkl：pipe_text_final.pkl / pipe_text_final2.pkl
# ---------- START: paste this BEFORE any call to load_dataset ----------
import os
import traceback
import pandas as pd
import streamlit as st

def _detect_file_format(path):
    """Return 'csv', 'xls', 'xlsx', or None."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"PK\x03\x04"):
        return "xlsx"
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return "xls"
    try:
        s = head.decode("utf-8", errors="ignore")
        first = s.splitlines()[0] if s.splitlines() else ""
        if "," in first or "\t" in first:
            return "csv"
    except Exception:
        pass
    return None

def load_dataset(path):
    """
    Robust dataset loader.
    - path: 文件路径，相对于 repo 根（或绝对路径）。
    返回 pandas.DataFrame 或在失败时抛异常（会被 Streamlit 捕获并记录到日志）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在：{path}")

    fmt = _detect_file_format(path)
# 不在页面显示检测信息，改为写入后端日志（便于调试但不打扰 UI）
print(f"load_dataset: detected format => {fmt}")
    try:
        # 优先按扩展名判断（更明确）
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv" or fmt == "csv":
            df = pd.read_csv(path, low_memory=False)
        elif ext == ".xls" or fmt == "xls":
            # 需要 xlrd>=2.0.1 在 requirements.txt 中
            df = pd.read_excel(path, engine="xlrd", low_memory=False)
        elif ext == ".xlsx" or fmt == "xlsx":
            # 需要 openpyxl 在 requirements.txt 中
            df = pd.read_excel(path, engine="openpyxl", low_memory=False)
        else:
            # 兜底：先试 csv，再试 excel（xlrd/openpyxl）
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception:
                df = pd.read_excel(path, engine="xlrd", low_memory=False)
        return df
    except Exception as e:
        tb = traceback.format_exc()
        # 输出到部署日志（Streamlit 的后端日志），并在 UI 显示简短信息
        print("load_dataset error traceback:\n", tb)
        st.error("读取数据集失败（详细信息见部署日志）。")
        raise

# ---------- END: paste this BEFORE any call to load_dataset ----------

import platform, streamlit as st
st.sidebar.caption(f"Python: {platform.python_version()}")

import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


import json, os
from pathlib import Path
import numpy as np
import pandas as pd

st.sidebar.caption(
    "环境版本："
    f" numpy {np.__version__} | pandas {pd.__version__}"
)
try:
    import sklearn, plotly  # noqa
    import joblib as _joblib
    st.sidebar.caption(
        f" scikit-learn {sklearn.__version__} | joblib {jb.__version__}"
    )
except Exception:
    pass


# —— 安全导入：joblib 不在环境时自动回退到 pickle（仅此为新增，其他逻辑不变）
try:
    import joblib as _joblib
except Exception:
    _joblib = None
import pickle as _pickle

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="青少年心理健康风险预测", page_icon="🧠", layout="wide")

# ========= 仅新增：把 Qxx 显示为中文题干 =========
QUESTION_LABELS = {
    # 基本信息
    "Q1": "你的年龄？",
    "Q2": "你的性别？",
    "Q3": "你现在的年级？",
    "Q4": "不穿鞋时的身高（米）",
    "Q5": "不穿鞋时的体重（公斤）",

    # 饮食/卫生
    "Q7":  "过去30天，因为家里没有足够的食物而挨饿的频率",
    "Q9":  "过去30天，你通常每天吃水果的次数",
    "Q10": "过去30天，你通常每天吃蔬菜的次数",
    "Q14": "过去30天，你通常每天刷牙的次数",
    "Q15": "过去30天，你饭前洗手的频率",
    "Q17": "过去30天，你如厕后洗手的频率",
    "Q19": "过去30天，你洗手时使用肥皂/香皂的频率",

    # 暴力/伤害与欺凌
    "Q23": "过去12个月，你被人身体攻击的次数",
    "Q24": "过去12个月，你参与肢体冲突的次数",
    "Q25": "过去12个月，你严重受伤的次数",
    "Q26": "过去12个月，你最严重一次受伤时正在做什么",
    "Q27": "过去12个月，你最严重一次受伤的主要原因",
    "Q28": "过去12个月，你最严重一次受伤是如何发生的",
    "Q29": "过去12个月，你最严重一次受伤的伤情类型",
    "Q30": "过去30天，你被欺凌的天数",
    "Q31": "过去30天，你最常遭遇的欺凌方式",

    # 心理健康（多用于标签，不进特征以防泄漏）
    "Q36": "过去12个月，你感到孤独的频率",
    "Q37": "过去12个月，你因担忧而晚上无法入睡的频率",
    "Q38": "过去12个月，你是否持续两周以上几乎每天感到悲伤/绝望并停止日常活动",
    "Q39": "过去12个月，你是否认真考虑过自杀",
    "Q40": "过去12个月，你是否制定过自杀计划",
    "Q41": "你有几个亲密朋友",

    # 烟草
    "Q44": "你第一次尝试吸烟时的年龄",
    "Q45": "过去30天，你吸烟的天数",
    "Q46": "过去30天，你使用其他形式烟草的天数",
    "Q47": "过去12个月，你是否尝试戒烟",
    "Q48": "过去7天，你身边有人吸烟的天数",
    "Q49": "你的父母/监护人是否使用任何形式的烟草",

    # 酒精/药物
    "Q53": "过去30天，你饮酒（至少一杯）的天数",
    "Q54": "过去30天，在饮酒的那些天里，你平均每天喝几杯",
    "Q55": "过去30天，你通常从哪里获得酒精饮料",
    "Q56": "你一生中，喝酒喝到“真的醉了”的次数",
    "Q58": "你一生中，因为喝酒而宿醉/不适/惹麻烦等的次数",
    "Q61": "你一生中使用毒品（摇头丸/MDMA/冰毒/甲基苯丙胺/大麻/海洛因）的次数",

    # 体力活动/久坐/通学
    "Q67": "过去7天，你有多少天每天进行≥60分钟的体力活动",
    "Q68": "平常一周，你有多少天每天进行≥60分钟的体力活动",
    "Q72": "典型的一天，你坐着（看电视/电脑/聊天/其他）的时间",
    "Q73": "过去7天，你有多少天步行或骑自行车上下学",
    "Q74": "过去7天，你每天往返上学通常花多长时间",
    "Q75": "过去30天，你无故缺课的天数",

    # 学校/家庭支持
    "Q76": "过去30天，你学校里的大多数同学是否友善并乐于助人（频率）",
    "Q77": "过去30天，你的父母/监护人检查你是否完成作业的频率",
    "Q78": "过去30天，你的父母/监护人理解你问题和担忧的频率",
    "Q79": "过去30天，你的父母/监护人是否真的了解你空闲时间在做什么（频率）",
}
def label_for(col_name: str) -> str:
    text = QUESTION_LABELS.get(str(col_name))
    return f"{text}（{col_name}）" if text else str(col_name)
# =========（新增部分结束，其它一律不动）=========

# ========= Sidebar: 资源加载 =========
st.sidebar.header("⚙️ 配置 / 资源")
model_path = st.sidebar.text_input("行为模型文件（.pkl）", "svm_pca_behavior_pipeline.pkl")
meta_path  = st.sidebar.text_input("行为模型元数据（.json）", "svm_pca_behavior_meta.json")
data_path  = st.sidebar.text_input("数据集 CSV（用于计算人群基线）", "CNAH2003_public_use.xls")
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
        if _joblib is not None:
            return _joblib.load(model_path)
        # 回退到 pickle（若 joblib 缺失）
        with open(model_path, "rb") as f:
            return _pickle.load(f)
    except Exception as e:
        msg = f"加载模型失败：{type(e).__name__}: {e}"
        if _joblib is None:
            msg += "（当前环境未安装 joblib，已尝试使用 pickle 但仍失败。请在 requirements.txt 中添加 joblib）"
        st.sidebar.error(msg)
        return None

#数据
# 替换你原来的读取代码为下面这段
import os
import traceback
import pandas as pd
import streamlit as st

data_path = "CNAH2003_public_use.xls"  # 确认路径（相对于 repo 根）

def detect_file_format(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    # XLSX (zip archive)
    if head.startswith(b"PK\x03\x04"):
        return "xlsx"
    # Old BIFF Compound File (xls)
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return "xls"
    # Heuristic for CSV / text: first line contains commas or tabs and ASCII letters
    try:
        s = head.decode("utf-8", errors="ignore")
        first = s.splitlines()[0] if s.splitlines() else ""
        if "," in first or "\t" in first:
            return "csv"
    except Exception:
        pass
    return None

if not os.path.exists(data_path):
    st.error(f"数据文件不存在：{data_path}。请确认已 push 到 repo 根目录，或使用正确的相对路径。")
    raise FileNotFoundError(data_path)

fmt = detect_file_format(data_path)
st.write(f"Detected format: {fmt}")

try:
    if fmt == "csv" or os.path.splitext(data_path)[1].lower() == ".csv":
        df = pd.read_csv(data_path, low_memory=False)
    elif fmt == "xls":
        # .xls 需要 xlrd>=2.0.1 已在 requirements.txt
        df = pd.read_excel(data_path, engine="xlrd", low_memory=False)
    elif fmt == "xlsx":
        # .xlsx 需要 openpyxl
        df = pd.read_excel(data_path, engine="openpyxl", low_memory=False)
    else:
        # 兜底尝试：先用 csv，再用 excel
        try:
            df = pd.read_csv(data_path, low_memory=False)
        except Exception:
            df = pd.read_excel(data_path, engine="xlrd", low_memory=False)
    st.success(f"读取成功，形状：{df.shape}")
    st.dataframe(df.head())
except Exception as e:
    # 在 app 页面显示友好信息，并将完整 traceback 输出到日志（streamlit 日志 / 部署日志可见）
    st.error("读取数据集失败：详细错误已记录到日志（查看 Manage App -> Logs）。")
    tb = traceback.format_exc()
    print(tb)          # 会进入部署日志
    raise



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

# 仅用模型特征参与统计（保持你原有结构，函数名相同无副作用）
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
st.title("🧠 青少年心理健康风险预测")
st.markdown("本应用基于预训练的机器学习模型。左侧配置 pkl/json/CSV 路径；填写问卷后生成 **概率、异常指标、z 分数、百分位、分布定位** 等可视化。")

tab_form, tab_result = st.tabs(["📝 填写问卷", "📊 结果与可视化"])

# ========= 问卷输入 =========
with tab_form:
    st.subheader("1) 请对行为问卷进行输入")
    cols = st.columns(2)
    user_input = {}
    for i, c in enumerate(feats_present):
        info = baseline[c]
        with cols[i % 2]:
            if info["kind"] == "num":
                lo, hi, default = numeric_bounds(df_all[c])
                step = 1.0 if (hi - lo) > 5 else 0.1
                # 使用中文题干
                val = st.slider(label_for(c), min_value=float(np.floor(lo)), max_value=float(np.ceil(hi)),
                                value=float(np.clip(default, lo, hi)), step=step,
                                help=f"均值≈{default:.2f}（数据集估计）")
                user_input[c] = val
            else:
                choices = ["<缺失>"] + info["choices"]
                # 使用中文题干
                val = st.selectbox(label_for(c), choices, index=0)
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
                if _joblib is not None:
                    pipe_text = _joblib.load(text_model_path)
                else:
                    with open(text_model_path, "rb") as f:
                        pipe_text = _pickle.load(f)
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
