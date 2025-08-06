import streamlit as st
import numpy as np
import joblib
import shap

st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 GDM 妊娠糖尿病风险预测工具")
st.markdown("本工具基于机器学习模型（XGBoost）预测个体是否患有 GDM。")

@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")

model = load_model()

feature_names = ['BMI', 'As', 'Cd', '低密度脂蛋白', '前白蛋白', '淋巴细胞百分比', '胆碱酯酶', '葡萄糖', '年龄']

st.subheader("📋 输入9项临床特征（标准化值）")
def get_input():
    return np.array([[st.number_input(name, value=0.0, step=0.1, format="%.2f") for name in feature_names]])

user_input = get_input()

if st.button("🔍 预测 GDM 风险"):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(user_input)[0][1]
    else:
        st.error("❌ 模型不支持 predict_proba。")
        st.stop()

    threshold = 0.3
    pred = "阳性 (GDM)" if proba >= threshold else "阴性 (非GDM)"
    st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
    st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

    st.markdown("🎯 **特征贡献力图（SHAP 力图）**")
    shap.initjs()

    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        base_model = model.named_steps["clf"]
        input_data = model.named_steps["scaler"].transform(user_input)
    else:
        base_model = model
        input_data = user_input

    explainer = shap.Explainer(base_model)
    shap_values = explainer(input_data)

    # 生成力图
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        input_data[0],
        feature_names=feature_names,
        matplotlib=False
    )

    # 用 HTML 显示，并调整比例（宽度100%，高度300px）
    st.components.v1.html(shap.getjs() + force_plot.html(), height=300)

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
