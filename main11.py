import streamlit as st
import numpy as np
import joblib
import shap
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

# 特征名（应与训练模型一致）
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

# 预测逻辑
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

        # -------- SHAP 力图部分 --------
        st.markdown("---")
        st.markdown("🎯 **特征贡献力图（SHAP 力图）**")

        # 如果是 Pipeline，取出分类器并转换输入
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
            input_data = model.named_steps["scaler"].transform(user_input)
        else:
            base_model = model
            input_data = user_input

        explainer = shap.Explainer(base_model)
        shap_values = explainer(input_data)

        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0, :],
            feature_names=feature_names
        )

        # 在 Streamlit 中显示 HTML
        components.html(shap.getjs() + force_plot.html(), height=300)

        st.markdown("🔴 红色特征推动预测为 GDM，🔵 蓝色特征推动预测为非 GDM。")

    except Exception as e:
        st.error(f"❌ 出错了：{e}")

st.markdown("---")
st.markdown("🔬 本工具用于科研/辅助判断，不作为医学诊断依据。")
