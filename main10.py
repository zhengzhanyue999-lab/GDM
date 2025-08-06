import streamlit as st
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")

model = load_model()

# 特征名
feature_names = ['BMI', 'As', 'Cd', '低密度脂蛋白', '前白蛋白', '淋巴细胞百分比', '胆碱酯酶', '葡萄糖', '年龄']

# 输入
st.subheader("📋 输入9项临床特征（标准化值）")
inputs = []
for name in feature_names:
    val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
    inputs.append(val)
user_input = np.array([inputs])

# 预测
if st.button("🔍 预测 GDM 风险"):
    proba = model.predict_proba(user_input)[0][1]
    pred = "阳性 (GDM)" if proba >= 0.3 else "阴性 (非GDM)"
    st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
    st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

    # 生成 SHAP 力图并嵌入 HTML
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        base_model = model.named_steps["clf"]
        input_data = model.named_steps["scaler"].transform(user_input)
    else:
        base_model = model
        input_data = user_input

    explainer = shap.Explainer(base_model)
    shap_values = explainer(input_data)

    st.markdown("### 🎯 特征贡献力图（SHAP 力图）")
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values.values,
        feature_names=feature_names,
        matplotlib=False
    ).html()

    components.html(force_html, height=300)

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")

