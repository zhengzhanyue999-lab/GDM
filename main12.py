import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 初始化 SHAP JS（力图需要）
shap.initjs()

# 设置页面标题
st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 GDM 妊娠糖尿病风险预测工具")
st.markdown("本工具基于机器学习模型（XGBoost）预测个体是否患有 GDM。")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")

model = load_model()

# 特征名
feature_names = ['BMI', 'As', 'Cd', '低密度脂蛋白', '前白蛋白', '淋巴细胞百分比', '胆碱酯酶', '葡萄糖', '年龄']

# 输入界面
st.subheader("📋 输入9项临床特征（标准化值）")
def get_input():
    inputs = []
    for name in feature_names:
        val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
        inputs.append(val)
    return np.array([inputs])

user_input = get_input()

# 预测
if st.button("🔍 预测 GDM 风险"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0][1]
        else:
            st.error("❌ 模型不支持 predict_proba。")
            st.stop()

        threshold = 0.3
        pred = "阳性 (GDM)" if proba >= threshold else "阴性 (非GDM)"

        st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
        st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

        # 取出模型中的分类器 & 处理输入
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
            input_data = model.named_steps["scaler"].transform(user_input)
        else:
            base_model = model
            input_data = user_input

        # 创建 SHAP 力图
        explainer = shap.Explainer(base_model)
        shap_values = explainer(input_data)

        st.markdown("🎯 **特征贡献力图（SHAP 力图）**")
        force_plot_html = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            feature_names=feature_names,
            matplotlib=False
        )

        # 嵌入 HTML 力图
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        st.components.v1.html(shap_html, height=300)

        st.markdown("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
