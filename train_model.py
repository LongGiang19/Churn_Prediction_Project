import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from datasets_config import DATASET_CONFIGS

def train_model_for_dataset(dataset_key, config, file_path):
    print(f"\n🚀 Đang huấn luyện model cho: {config['name']}")
    
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        print(f"⏭️ Bỏ qua dataset này...\n")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"📄 Đã load {len(df)} rows từ {file_path}")
        
        # ==========================================
        # 1. FIX LỖI TÊN CỘT CHO DATASET CALL DETAILS
        # ==========================================
        if dataset_key == 'call_details':
            rename_map = {
                'AccountLength': 'Account length',
                'VMailMessage': 'Number vmail messages',
                'DayMins': 'Total day minutes',
                'EveMins': 'Total eve minutes',
                'NightMins': 'Total night minutes',
                'IntlMins': 'Total intl minutes',
                'CustServCalls': 'Customer service calls',
                'IntlPlan': 'International plan',
                'VMailPlan': 'Voice mail plan'
            }
            df = df.rename(columns=rename_map)
            
        # Kiểm tra columns có trong file
        available_cols = set(df.columns)
        required_cols = set(config['required_columns'])
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            print(f"❌ File thiếu các cột bắt buộc: {missing_cols}")
            return None
        
        # Làm sạch dữ liệu cơ bản
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
        # ==========================================
        # 2. XÓA CẢNH BÁO WARNING MÀU VÀNG CỦA MLFLOW
        # ==========================================
        # Ép tất cả các cột số nguyên sang số thực (float) để MLflow không phàn nàn
        for col in config['numeric_features']:
            if col in df.columns:
                df[col] = df[col].astype(float)
                
        # Xử lý target column
        target_col = 'Churn'
        if target_col in df.columns:
            df[target_col] = df[target_col].astype(str).str.lower().map({
                'yes': 1, 'no': 0, 'true.': 1, 'false.': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
            })
            df = df.dropna(subset=[target_col])
        else:
            print(f"⚠️ Không tìm thấy cột '{target_col}'")
            return None
        
        # Chuẩn bị features
        feature_cols = config['numeric_features'] + config['categorical_features']
        X = df[feature_cols]
        y = df[target_col]
        
        # Tạo pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, config['numeric_features']),
            ("cat", cat_transformer, config['categorical_features'])
        ])

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Cấu hình MLflow
        user_email = "lathienbao25@gmail.com" 
        experiment_path = f"/Users/{user_email}/churn_prediction_app"
        
        mlflow.set_experiment(experiment_path)
        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name=f"Train_{dataset_key}"):
            model.fit(X, y)
            
            model_dir = os.path.dirname(config['model_path'])
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, config['model_path'], compress=3)
            print(f"✅ Model đã lưu: {config['model_path']}")
        
        return model
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return None

def main():
    print("="*60)
    print(" BẮT ĐẦU HUẤN LUYỆN CÁC MODELS")
    print("="*60)
    
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # 1. Train Telco
    train_model_for_dataset('telco_ibm', DATASET_CONFIGS['telco_ibm'], 'data/Telco_customer_churn.csv')
    
    # 2. Train Call Details
    train_model_for_dataset('call_details', DATASET_CONFIGS['call_details'], 'data/Churn.csv')
    
    print("\n🎉 Hoàn tất!")

if __name__ == "__main__":
    main()