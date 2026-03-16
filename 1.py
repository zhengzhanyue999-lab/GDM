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
# GitHub / Streamlit Cloud 部署时必须使用相对路径
# =========================
MODEL_PATH = "best_model_xgboost.pkl"

# =========================
# 最终入模的14个变量
# ⚠️ 必须与你训练模型时的特征顺序一致
# =========================
feature_names = [
    'HDL-C',
    'TC',
    'LDL-C',
    'TG',
    'Glucose',
    'lnCu+10',
    'lnAs+10',
    'lnPb+10',
    'GGT',
    'SF',
    'ChE',
    'PTA',
    'lnCd+10',
    'Tg'
]

# =========================
# 中文显示名称
# =========================
feature_labels = {
    'HDL-C': '高密度脂蛋白胆固醇（HDL-C）',
    'TC': '总胆固醇（TC）',
    'LDL-C': '低密度脂蛋白胆固醇（LDL-C）',
    'TG': '甘油三酯（TG）',
    'Glucose': '葡萄糖（Glucose）',
    'lnCu+10': 'ln(Cu)+10（铜）',
    'lnAs+10': 'ln(As)+10（砷）',
    'lnPb+10': 'ln(Pb)+10（铅）',
    'GGT': 'γ-谷氨酰转肽酶（GGT）',
    'SF': '铁蛋白（SF）',
    'ChE': '胆碱酯酶（ChE）',
    'PTA': '凝血酶原活动度（PTA）',
    'lnCd+10': 'ln(Cd)+10（镉）',
    'Tg': '甲状腺球蛋白（Tg）'
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
    st.subheader("📋 请输入 14 项临床指标")
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

st.info(
    "⚠️ 请输入与模型训练时一致的数值类型。\n\n"
    "如果模型训练时使用的是原始临床值，请输入原始值；\n"
    "如果训练时使用的是标准化/变换后的值，请输入相同形式的数据。"
)

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
            st.caption(f"检测到 Pipeline 步骤：{step_names}")
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

st.markdown("---")
st.markdown("🔬 本工具仅用于科研/辅助判断，不作为医学诊断依据。")