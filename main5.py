import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互后端，避免服务器绘图报错

# 设置页面
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

# 构建输入
st.subheader("📋 输入9项临床特征（标准化值）")

def get_input():
    values = []
    for feat in feature_names:
        val = st.number_input(feat, value=0.0, step=0.1, format="%.2f")
        values.append(val)
    return np.array([values])

user_input = get_input()

# 预测并显示 SHAP 力图
if st.button("🔍 预测 GDM 风险"):
    try:
        # 若模型是管道（如有 scaler）
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
            input_data = model.named_steps["scaler"].transform(user_input)
        else:
            base_model = model
            input_data = user_input

        # 预测概率
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0][1]
        else:
            st.error("❌ 模型不支持 predict_proba。")
            st.stop()

        threshold = 0.3
        pred = "阳性 (GDM)" if proba >= threshold else "阴性 (非GDM)"

        st.markdown(f"### 🧪 预测结果：**{proba*100:.2f}%** GDM 概率")
        st.markdown(f"### 🩺 判定：**{pred}** （阈值：0.3）")

        # SHAP 力图
        st.markdown("---")
        st.markdown("🎯 **特征贡献力图（SHAP Force Plot）**")

        explainer = shap.Explainer(base_model)
        shap_values = explainer(input_data)

        fig = plt.figure()
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(fig)

        st.caption("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")

