import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =========================
# 页面设置
# =========================
st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 妊娠糖尿病风险预测工具")
st.markdown("本工具基于 XGBoost 模型预测个体是否患有 GDM。")


# =========================
# 当前脚本所在目录
# 你的 7.py、模型文件、metrics 文件在同一文件夹
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model_xgboost_14vars.pkl"
METRICS_PATH = BASE_DIR / "best_model_xgboost_14vars_metrics.json"


# =========================
# 默认阈值
# 若 metrics 文件中有 threshold，则自动覆盖
# =========================
THRESHOLD = 0.30


# =========================
# 模型内部实际使用的14个变量
# 顺序必须与训练模型完全一致
# =========================
feature_names = [
    "HDL-C",
    "TC",
    "LDL-C",
    "TG",
    "Glucose",
    "lnCu+10",
    "lnAs+10",
    "lnPb+10",
    "GGT",
    "SF",
    "ChE",
    "PTA",
    "lnCd+10",
    "Tg"
]


# =========================
# 网页输入名称（输入原始值）
# =========================
input_labels = {
    "HDL-C": "高密度脂蛋白胆固醇（HDL-C）",
    "TC": "总胆固醇（TC）",
    "LDL-C": "低密度脂蛋白胆固醇（LDL-C）",
    "TG": "甘油三酯（TG）",
    "Glucose": "葡萄糖（Glucose）",
    "Cu": "铜原始值（Cu）",
    "As": "砷原始值（As）",
    "Pb": "铅原始值（Pb）",
    "GGT": "γ-谷氨酰转肽酶（GGT）",
    "SF": "铁蛋白（SF）",
    "ChE": "胆碱酯酶（ChE）",
    "PTA": "凝血酶原活动度（PTA）",
    "Cd": "镉原始值（Cd）",
    "Tg": "甲状腺球蛋白（Tg）"
}


# =========================
# 模型变量中文名称（SHAP/回显）
# =========================
feature_labels = {
    "HDL-C": "高密度脂蛋白胆固醇（HDL-C）",
    "TC": "总胆固醇（TC）",
    "LDL-C": "低密度脂蛋白胆固醇（LDL-C）",
    "TG": "甘油三酯（TG）",
    "Glucose": "葡萄糖（Glucose）",
    "lnCu+10": "ln(Cu)+10（铜）",
    "lnAs+10": "ln(As)+10（砷）",
    "lnPb+10": "ln(Pb)+10（铅）",
    "GGT": "γ-谷氨酰转肽酶（GGT）",
    "SF": "铁蛋白（SF）",
    "ChE": "胆碱酯酶（ChE）",
    "PTA": "凝血酶原活动度（PTA）",
    "lnCd+10": "ln(Cd)+10（镉）",
    "Tg": "甲状腺球蛋白（Tg）"
}


# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# =========================
# 加载指标文件
# =========================
@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 原始值转换为 ln(x)+10
# =========================
def transform_ln_plus_10(x, var_name):
    if x <= 0:
        raise ValueError(f"{var_name} 原始值必须大于 0，才能进行 ln(x)+10 转换。")
    return np.log(x) + 10


# =========================
# 获取输入
# =========================
def get_input():
    st.subheader("📋 请输入 14 项临床指标")

    raw_inputs_display = {}
    col1, col2 = st.columns(2)

    input_order = [
        "HDL-C",
        "TC",
        "LDL-C",
        "TG",
        "Glucose",
        "Cu",
        "As",
        "Pb",
        "GGT",
        "SF",
        "ChE",
        "PTA",
        "Cd",
        "Tg"
    ]

    for i, item in enumerate(input_order):
        label = input_labels.get(item, item)
        with col1 if i % 2 == 0 else col2:
            value = st.number_input(
                label,
                value=0.0,
                step=0.1,
                format="%.4f" if item in ["Cu", "As", "Pb", "Cd"] else "%.2f",
                key=item
            )
            raw_inputs_display[item] = value

    try:
        model_inputs = [
            raw_inputs_display["HDL-C"],
            raw_inputs_display["TC"],
            raw_inputs_display["LDL-C"],
            raw_inputs_display["TG"],
            raw_inputs_display["Glucose"],
            transform_ln_plus_10(raw_inputs_display["Cu"], "Cu"),
            transform_ln_plus_10(raw_inputs_display["As"], "As"),
            transform_ln_plus_10(raw_inputs_display["Pb"], "Pb"),
            raw_inputs_display["GGT"],
            raw_inputs_display["SF"],
            raw_inputs_display["ChE"],
            raw_inputs_display["PTA"],
            transform_ln_plus_10(raw_inputs_display["Cd"], "Cd"),
            raw_inputs_display["Tg"]
        ]
    except ValueError as e:
        st.error(f"❌ 输入值错误：{e}")
        return None, raw_inputs_display

    return np.array([model_inputs], dtype=float), raw_inputs_display


# =========================
# 对 pipeline 做前处理
# 用于 SHAP
# =========================
def transform_for_base_model(model, user_input):
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        x = user_input.copy()

        for step_name in step_names[:-1]:
            step_obj = model.named_steps[step_name]
            if hasattr(step_obj, "transform"):
                x = step_obj.transform(x)

        base_model = model.named_steps[step_names[-1]]
        return base_model, x

    return model, user_input


# =========================
# 加载模型
# =========================
try:
    model = load_model()
    st.success("✅ 模型加载成功")
except Exception as e:
    st.error(f"❌ 模型加载失败：{e}")
    st.write("当前工作目录：", os.getcwd())
    st.write("脚本所在目录：", str(BASE_DIR))
    st.write("模型路径：", str(MODEL_PATH))
    st.stop()


# =========================
# 加载指标文件
# =========================
metrics_data = load_metrics()
if metrics_data is not None:
    try:
        threshold_from_metrics = metrics_data.get("metrics", {}).get("threshold", None)
        if threshold_from_metrics is not None:
            THRESHOLD = float(threshold_from_metrics)
    except Exception:
        pass


# =========================
# 路径调试信息
# =========================
with st.expander("🔧 文件路径调试信息"):
    st.write("当前工作目录：", os.getcwd())
    st.write("脚本所在目录：", str(BASE_DIR))
    st.write("模型路径：", str(MODEL_PATH))
    st.write("指标路径：", str(METRICS_PATH))
    st.write("模型文件是否存在：", MODEL_PATH.exists())
    st.write("指标文件是否存在：", METRICS_PATH.exists())


user_input, raw_inputs_display = get_input()


# =========================
# 预测
# =========================
if st.button("🔍 预测 GDM 风险"):
    try:
        if user_input is None:
            st.stop()

        if not hasattr(model, "predict_proba"):
            st.error("❌ 当前模型不支持 predict_proba。")
            st.stop()

        proba = model.predict_proba(user_input)[0][1]
        pred_label = 1 if proba >= THRESHOLD else 0
        pred_text = "阳性（GDM）" if pred_label == 1 else "阴性（非GDM）"

        st.markdown("---")
        st.markdown(f"## 🧪 预测结果：**{proba * 100:.2f}%**")
        st.markdown(f"### 🩺 判定：**{pred_text}**（阈值：{THRESHOLD:.2f}）")
        st.progress(min(max(float(proba), 0.0), 1.0))

        # SHAP解释
        st.markdown("---")
        st.subheader("🎯 特征贡献解释（SHAP）")

        try:
            base_model, input_data = transform_for_base_model(model, user_input)

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):
                shap_values_single = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                shap_values_single = shap_values[0]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            force_plot = shap.force_plot(
                expected_value,
                shap_values_single,
                input_data[0],
                feature_names=[feature_labels.get(f, f) for f in feature_names],
                matplotlib=False
            )

            st.components.v1.html(
                shap.getjs() + force_plot.html(),
                height=320,
                width=900
            )

            st.markdown("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

        except Exception as e:
            st.warning(f"⚠️ SHAP 解释暂时无法显示：{e}")

        # 输入回显
        st.markdown("---")
        st.subheader("🧾 本次输入值回显")

        display_df = pd.DataFrame({
            "网页输入项": [
                "HDL-C", "TC", "LDL-C", "TG", "Glucose", "Cu", "As", "Pb",
                "GGT", "SF", "ChE", "PTA", "Cd", "Tg"
            ],
            "中文名": [
                input_labels["HDL-C"],
                input_labels["TC"],
                input_labels["LDL-C"],
                input_labels["TG"],
                input_labels["Glucose"],
                input_labels["Cu"],
                input_labels["As"],
                input_labels["Pb"],
                input_labels["GGT"],
                input_labels["SF"],
                input_labels["ChE"],
                input_labels["PTA"],
                input_labels["Cd"],
                input_labels["Tg"]
            ],
            "输入原始值": [
                raw_inputs_display["HDL-C"],
                raw_inputs_display["TC"],
                raw_inputs_display["LDL-C"],
                raw_inputs_display["TG"],
                raw_inputs_display["Glucose"],
                raw_inputs_display["Cu"],
                raw_inputs_display["As"],
                raw_inputs_display["Pb"],
                raw_inputs_display["GGT"],
                raw_inputs_display["SF"],
                raw_inputs_display["ChE"],
                raw_inputs_display["PTA"],
                raw_inputs_display["Cd"],
                raw_inputs_display["Tg"]
            ],
            "模型实际使用值": user_input[0]
        })

        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 预测过程中出错：{e}")


# =========================
# 页脚
# =========================
st.markdown("---")
st.markdown("🔬 本工具仅用于科研/辅助判断，不作为医学诊断依据。")