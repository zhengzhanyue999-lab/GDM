import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ✅ 重要：强制使用 matplotlib 后端
shap.initjs()  # 可删除，因 matplotlib 不用 JS

# 页面设置
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

# 输入
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
        # 获取模型与输入
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
            input_data = model.named_steps["scaler"].transform(user_input)
        else:
            base_model = model
            input_data = user_input

        # 概率预测
        proba = model.predict_proba(user_input)[0][1]
        threshold = 0.3
        pred = "阳性 (GDM)" if proba >= threshold else "阴性 (非GDM)"

        st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
        st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

        # SHAP force_plot 可视化（静态图）
        st.markdown("---")
        st.markdown("🎯 **特征贡献力图（SHAP 力图）**")

        explainer = shap.Explainer(base_model)
        shap_values = explainer(input_data)

        # force_plot 生成 matplotlib 图像
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

        st.caption("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
