import streamlit as st
import pandas as pd
import joblib
import os
import mlflow
from datasets_config import DATASET_CONFIGS, detect_dataset_type

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://host.docker.internal:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction_app")
except Exception:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

st.set_page_config(page_title="Churn Predictor - Multi Dataset", page_icon="📡", layout="wide")
st.title("📡 Hệ Thống Dự Đoán Churn - Đa Dataset")
st.markdown("*Tự động nhận diện dataset và sử dụng model phù hợp*")

@st.cache_resource
def load_models():
    models = {}
    for dataset_key, config in DATASET_CONFIGS.items():
        model_path = config['model_path']
        if os.path.exists(model_path):
            try:
                models[dataset_key] = {
                    'model': joblib.load(model_path),
                    'config': config
                }
            except Exception as e:
                st.warning(f"⚠️ Không load được model {dataset_key}: {e}")
    return models

models_dict = load_models()

if not models_dict:
    st.warning("⚠️ Chưa có model mặc định, nhưng bạn vẫn có thể dùng tính năng Auto-Train bên dưới!")

with st.expander("📦 Trạng thái Models", expanded=False):
    for key, info in models_dict.items():
        st.success(f"✅ {DATASET_CONFIGS[key]['name']}")

with st.sidebar:
    st.header("📝 Dự Đoán Thủ Công")
    
    if not models_dict:
        st.warning("Chưa có model nào sẵn sàng")
    else:
        selected_type = st.selectbox(
            "Chọn loại dataset",
            options=list(models_dict.keys()),
            format_func=lambda x: DATASET_CONFIGS[x]['name']
        )
        
        config = models_dict[selected_type]['config']
        model = models_dict[selected_type]['model']
        
        st.info(f"📊 {config['name']}")
        
        inputs = {}
        st.subheader("Nhập thông tin")
        
        for feature in config['numeric_features']:
            if 'tenure' in feature.lower() or 'length' in feature.lower():
                default_val = 24.0
            elif 'charges' in feature.lower() or 'minutes' in feature.lower():
                default_val = 65.0
            else:
                default_val = 0.0
            
            step_val = 0.1 if 'charges' in feature.lower() or 'minutes' in feature.lower() else 1.0
            inputs[feature] = st.number_input(feature, value=default_val, step=step_val)
        
        for feature in config['categorical_features']:
            options = ['No', 'Yes', 'No internet service'] if 'internet' in feature.lower() else ['No', 'Yes']
            inputs[feature] = st.selectbox(feature, options)
        
        if st.button("🔍 Dự Đoán", type="primary", use_container_width=True):
            try:
                input_df = pd.DataFrame({k: [v] for k, v in inputs.items()})
                prob = model.predict_proba(input_df)[0][1]
                pred = model.predict(input_df)[0]
                
                st.divider()
                st.subheader("📊 Kết Quả")
                pred_val = 1 if (pred == 1 or str(pred).lower() == 'yes') else 0
                if pred_val == 1:
                    st.error(f"⚠️ Nguy cơ churn: **{prob*100:.1f}%**")
                    st.info("💡 Khuyến nghị: Liên hệ khách hàng để giữ chân")
                else:
                    st.success(f"✅ Khách hàng trung thành. Nguy cơ: **{prob*100:.1f}%**")
                    st.info("💡 Tiếp tục chăm sóc và upsell dịch vụ")
            except Exception as e:
                st.error(f"❌ Lỗi dự đoán: {str(e)}")

st.divider()
st.subheader("📤 Dự Đoán Hàng Loạt (Upload CSV)")

uploaded_file = st.file_uploader("Kéo thả file CSV vào đây hoặc click để chọn file", type=["csv"])

if uploaded_file is not None:
    try:
        df_batch = pd.read_csv(uploaded_file)
        
        with st.expander("📋 Preview dữ liệu (5 dòng đầu)", expanded=True):
            st.dataframe(df_batch.head(5), use_container_width=True)
            st.caption(f"Tổng số dòng: {len(df_batch)} | Cột: {list(df_batch.columns)}")
        
        dataset_type = detect_dataset_type(df_batch)
        
        if dataset_type is None:
            st.error("❌ Không nhận diện được định dạng dataset!")
            with st.expander("📖 Các định dạng được hỗ trợ"):
                st.markdown("""
                **1. Telco IBM** (cần có): `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`...
                
                **2. Call Details/BigML** (cần có): `Account length`, `Total day minutes`, `Total eve minutes`...
                """)
            st.stop()
        
        st.success(f"✅ Phát hiện dataset: **{DATASET_CONFIGS[dataset_type]['name']}**")
        
        if dataset_type not in models_dict:
            st.error(f"❌ Chưa có model cho dataset này!")
            st.info(f"💡 Vui lòng train model '{dataset_type}' bằng notebook train_model.py")
            st.stop()
        
        model = models_dict[dataset_type]['model']
        config = models_dict[dataset_type]['config']
        
        df_clean = df_batch.copy()
        
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            dropped = df_clean['TotalCharges'].isna().sum()
            df_clean = df_clean.dropna(subset=['TotalCharges'])
            if dropped > 0:
                st.warning(f"⚠️ Đã loại {dropped} dòng do TotalCharges không hợp lệ")
        
        missing_cols = [col for col in config['required_columns'] if col not in df_clean.columns]
        if missing_cols:
            st.error(f"❌ File thiếu các cột bắt buộc: {missing_cols}")
            st.info(f"✅ Cần có đủ: {config['required_columns']}")
            st.stop()
        
        if st.button("🔮 Bắt đầu dự đoán hàng loạt", type="secondary", use_container_width=True):
            with st.spinner('🔄 Đang xử lý...'):
                try:
                    feature_cols = config['numeric_features'] + config['categorical_features']
                    input_data = df_clean[feature_cols]
                    
                    preds_proba = model.predict_proba(input_data)
                    df_clean["Churn_Probability"] = preds_proba[:, 1]
                    
                    raw_preds = model.predict(input_data)
                    if raw_preds.dtype == 'object':
                        df_clean["Churn_Prediction"] = pd.Series(raw_preds).map({
                            'Yes': 1, 'No': 0, 'True': 1, 'False': 0
                        }).fillna(0).astype(int).values
                    else:
                        df_clean["Churn_Prediction"] = raw_preds
                    
                    churn_rate = float(df_clean["Churn_Prediction"].mean()) * 100
                    avg_prob = float(df_clean["Churn_Probability"].mean()) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tổng khách hàng", f"{len(df_clean):,}")
                    with col2:
                        st.metric("Dự đoán Churn", f"{int(df_clean['Churn_Prediction'].sum()):,} ({churn_rate:.1f}%)")
                    with col3:
                        st.metric("Xác suất trung bình", f"{avg_prob:.1f}%")
                    
                    st.subheader("📊 Kết quả chi tiết (10 dòng đầu)")
                    
                    id_col = None
                    for col in ['customerID', 'CustomerID', 'Account length', 'State']:
                        if col in df_clean.columns:
                            id_col = col
                            break
                    
                    display_cols = [id_col] if id_col else []
                    display_cols += ["Churn_Probability", "Churn_Prediction"]
                    
                    st.dataframe(
                        df_clean[display_cols].head(10).style.format({
                            "Churn_Probability": "{:.2%}"
                        }),
                        use_container_width=True
                    )
                    
                    high_risk = df_clean[df_clean["Churn_Probability"] > 0.75]
                    if len(high_risk) > 0:
                        st.warning(f"🚨 Có {len(high_risk)} khách hàng nguy cơ cao (>75%)")
                    
                    csv = df_clean.to_csv(index=False)
                    st.download_button(
                        label="💾 Tải kết quả đầy đủ (CSV)",
                        data=csv,
                        file_name=f"churn_predictions_{dataset_type}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Hoàn tất!")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
                    import traceback
                    with st.expander("🔍 Chi tiết lỗi (debug)"):
                        st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {str(e)}")
        st.info("💡 Đảm bảo file là CSV hợp lệ, encoding UTF-8")

st.divider()
st.caption("📡 Churn Prediction System | Multi-Dataset Support | Powered by Streamlit + MLflow")