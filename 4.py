import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import os

# =========================
# 页面设置
# =========================
st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 GDM 妊娠糖尿病风险预测工具")
st.markdown("本工具基于 XGBoost 模型预测个体是否患有 GDM。")

# =========================
# 模型路径
# =========================
MODEL_PATH = r"E:\新建文件夹\GDM相关\GDM\分析用的数据及代码\best_model_xgboost.pkl"

# =========================
# 该 pkl 实际使用的 9 个变量
# =========================
feature_names = [
    "TG",
    "HDL-C",
    "TC",
    "Glucose",
    "lnCu+10",
    "LDL-C",
    "lnPb+10",
    "GGT",
    "lnAs+10"
]

# =========================
# 中文显示名称
# =========================
feature_labels = {
    "TG": "甘油三酯（TG）",
    "HDL-C": "高密度脂蛋白胆固醇（HDL-C）",
    "TC": "总胆固醇（TC）",
    "Glucose": "葡萄糖（Glucose）",
    "lnCu+10": "ln(Cu)+10（铜）",
    "LDL-C": "低密度脂蛋白胆固醇（LDL-C）",
    "lnPb+10": "ln(Pb)+10（铅）",
    "GGT": "γ-谷氨酰转肽酶（GGT）",
    "lnAs+10": "ln(As)+10（砷）"
}

# =========================
# 阈值
# =========================
THRESHOLD = 0.30

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    return joblib.load(MODEL_PATH)

# =========================
# 构造输入
# =========================
def get_input():
    st.subheader("📋 请输入 9 项临床指标")

    inputs = []
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        label = feature_labels.get(feature, feature)
        with col1 if i % 2 == 0 else col2:
            value = st.number_input(
                label,
                value=0.0,
                step=0.1,
                format="%.2f",
                key=feature
            )
            inputs.append(value)

    return np.array([inputs], dtype=float)

# =========================
# 主程序
# =========================
try:
    model = load_model()
    st.success("✅ 模型加载成功")
except Exception as e:
    st.error(f"❌ 模型加载失败：{e}")
    st.stop()

user_input = get_input()

# =========================
# 预测
# =========================
if st.button("🔍 预测 GDM 风险"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0][1]
        else:
            st.error("❌ 当前模型不支持 predict_proba。")
            st.stop()

        pred_label = 1 if proba >= THRESHOLD else 0
        pred_text = "阳性（GDM）" if pred_label == 1 else "阴性（非GDM）"

        st.markdown("---")
        st.markdown(f"## 🧪 预测结果：**{proba * 100:.2f}%**")
        st.markdown(f"### 🩺 判定：**{pred_text}**（阈值：{THRESHOLD:.2f}）")
        st.progress(min(max(float(proba), 0.0), 1.0))

        st.markdown("---")
        st.subheader("🎯 特征贡献解释（SHAP）")

        if hasattr(model, "named_steps"):
            step_names = list(model.named_steps.keys())
            scaler = model.named_steps.get("scaler", None)
            last_step_name = step_names[-1]
            base_model = model.named_steps[last_step_name]
            input_data = scaler.transform(user_input) if scaler is not None else user_input
        else:
            base_model = model
            input_data = user_input

        try:
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

        except Exception as e1:
            st.warning(f"⚠️ TreeExplainer 失败，尝试通用 Explainer：{e1}")

            explainer = shap.Explainer(base_model, input_data)
            shap_values = explainer(input_data)

            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values.values[0],
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

        st.markdown("---")
        st.subheader("🧾 本次输入值回显")

        display_df = pd.DataFrame({
            "变量英文名": feature_names,
            "变量中文名": [feature_labels.get(f, f) for f in feature_names],
            "输入值": user_input[0]
        })

        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 预测过程中出错：{e}")

# =========================
# 页脚
# =========================
st.markdown("---")
st.markdown("🔬 本工具仅用于科研/辅助判断，不作为医学诊断依据。")