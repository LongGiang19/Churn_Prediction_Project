"""
Cấu hình cho các loại dataset churn khác nhau
"""

DATASET_CONFIGS = {
    'telco_ibm': {
        'name': 'Telco Customer Churn (IBM)',
        'required_columns': [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod', 'InternetService',
            'TechSupport', 'OnlineSecurity'
        ],
        'numeric_features': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical_features': ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity'],
        'model_path': 'models/churn_model_telco.pkl'
    },
    'call_details': {
        'name': 'Call Details Churn (BigML)',
        'required_columns': [
            'Account length', 'Total day minutes', 'Total eve minutes',
            'Total night minutes', 'Total intl minutes', 'Number vmail messages',
            'Customer service calls', 'International plan', 'Voice mail plan'
        ],
        'numeric_features': [
            'Account length', 'Total day minutes', 'Total eve minutes',
            'Total night minutes', 'Total intl minutes', 'Number vmail messages',
            'Customer service calls'
        ],
        'categorical_features': ['International plan', 'Voice mail plan'],
        'model_path': 'models/churn_model_calls.pkl'
    }
}

def detect_dataset_type(df):
    """
    Tự động phát hiện loại dataset dựa trên các cột có trong file
    """
    columns = set(df.columns)
    
    # 1. Kiểm tra Telco IBM
    if 'tenure' in columns and 'MonthlyCharges' in columns:
        return 'telco_ibm'
    
    # 2. Kiểm tra Call Details (BigML)
    if 'Account length' in columns and 'Total day minutes' in columns:
        return 'call_details'
            
    return None