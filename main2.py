import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 设置页面标题
st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 GDM 妊娠糖尿病风险预测工具")
st.markdown("本工具基于机器学习模型（XGBoost）预测个体是否患有 GDM。")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")  # 添加 return

model = load_model()

# 加载特征名（中文列名对应你Excel的内容）
feature_names = ['BMI', 'As', 'Cd', '低密度脂蛋白', '前白蛋白', '淋巴细胞百分比', '胆碱酯酶', '葡萄糖', '年龄']

# 构建输入界面
st.subheader("📋 输入9项临床特征（标准化值）")

def get_input():
    inputs = []
    for name in feature_names:
        val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
        inputs.append(val)
    return np.array([inputs])

user_input = get_input()

# 预测逻辑
if st.button("🔍 预测 GDM 风险"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0][1]
        else:
            st.error("❌ 模型不支持概率预测（predict_proba）。")
            st.stop()

        threshold = 0.3
        pred = "阳性 (GDM)" if proba >= threshold else "阴性 (非GDM)"

        st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
        st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

        # 生成 SHAP 图（建议 bar 或 waterfall）
        st.markdown("---")
        st.markdown("🎯 **特征贡献力图（SHAP）**")

        explainer = shap.Explainer(model)
        shap_values = explainer(user_input)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        st.markdown("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
