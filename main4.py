import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置页面标题
st.set_page_config(page_title="GDM 风险预测工具", layout="centered")
st.title("🤰 GDM 妊娠糖尿病风险预测工具")
st.markdown("本工具基于机器学习模型（XGBoost）预测个体是否患有 GDM。")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")

model = load_model()

# 特征名称（与模型训练数据保持一致）
feature_names = ['BMI', 'As', 'Cd', '低密度脂蛋白', '前白蛋白', '淋巴细胞百分比', '胆碱酯酶', '葡萄糖', '年龄']

# 输入表单
st.subheader("📋 输入 9 项临床特征（标准化值）")

def get_input():
    inputs = []
    for name in feature_names:
        val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
        inputs.append(val)
    return np.array([inputs])

user_input = get_input()

# 预测 + SHAP 力图
if st.button("🔍 预测 GDM 风险"):
    try:
        if not hasattr(model, "predict_proba"):
            st.error("❌ 当前模型不支持概率预测（predict_proba）")
            st.stop()

        # 预测概率
        proba = model.predict_proba(user_input)[0][1]
        pred = "阳性 (GDM)" if proba >= 0.3 else "阴性 (非GDM)"

        # 显示预测结果
        st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
        st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

        # SHAP 力图解释
        st.markdown("---")
        st.markdown("🎯 **特征贡献力图（SHAP）**")

        # 取出 xgboost 模型（如果是 pipeline）
        if hasattr(model, "named_steps"):
            xgb_model = model.named_steps["clf"]
        else:
            xgb_model = model  # 如果本身就是模型

        # 解释器（用当前样本做 background）
        explainer = shap.Explainer(xgb_model, user_input)
        shap_values = explainer(user_input)

        # 显示 force plot
        shap.initjs()
        html = shap.plots.force(
            explainer.expected_value,
            shap_values.values[0],
            matplotlib=False,
            feature_names=feature_names,
            show=False
        ).html()

        components.html(shap.getjs() + html, height=300)

        st.markdown("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
