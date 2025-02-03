from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


models = {
    'xgboost': joblib.load('models/saved_models/xgboost.joblib'),
    'random_forest': joblib.load('models/saved_models/random_forest.joblib')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_data = request.json
        model_name = input_data.pop('model_type', 'xgboost')
        model = models[model_name]


        all_features = [
            'year', 'month', 'user_account_id', 'user_lifetime', 'user_intake',
            'user_no_outgoing_activity_in_days', 'user_account_balance_last',
            'user_spendings', 'user_has_outgoing_calls', 'user_has_outgoing_sms',
            'user_use_gprs', 'user_does_reload', 'reloads_inactive_days',
            'reloads_count', 'reloads_sum', 'calls_outgoing_count',
            'calls_outgoing_spendings', 'calls_outgoing_duration',
            'calls_outgoing_spendings_max', 'calls_outgoing_duration_max',
            'calls_outgoing_inactive_days', 'calls_outgoing_to_onnet_count',
            'calls_outgoing_to_onnet_spendings', 'calls_outgoing_to_onnet_duration',
            'calls_outgoing_to_onnet_inactive_days', 'calls_outgoing_to_offnet_count',
            'calls_outgoing_to_offnet_spendings', 'calls_outgoing_to_offnet_duration',
            'calls_outgoing_to_offnet_inactive_days', 'calls_outgoing_to_abroad_count',
            'calls_outgoing_to_abroad_spendings', 'calls_outgoing_to_abroad_duration',
            'calls_outgoing_to_abroad_inactive_days', 'sms_outgoing_count',
            'sms_outgoing_spendings', 'sms_outgoing_spendings_max',
            'sms_outgoing_inactive_days', 'sms_outgoing_to_onnet_count',
            'sms_outgoing_to_onnet_spendings', 'sms_outgoing_to_onnet_inactive_days',
            'sms_outgoing_to_offnet_count', 'sms_outgoing_to_offnet_spendings',
            'sms_outgoing_to_offnet_inactive_days', 'sms_outgoing_to_abroad_count',
            'sms_outgoing_to_abroad_spendings', 'sms_outgoing_to_abroad_inactive_days',
            'sms_incoming_count', 'sms_incoming_spendings',
            'sms_incoming_from_abroad_count', 'sms_incoming_from_abroad_spendings',
            'gprs_session_count', 'gprs_usage', 'gprs_spendings',
            'gprs_inactive_days', 'last_100_reloads_count', 'last_100_reloads_sum',
            'last_100_calls_outgoing_duration',
            'last_100_calls_outgoing_to_onnet_duration',
            'last_100_calls_outgoing_to_offnet_duration',
            'last_100_calls_outgoing_to_abroad_duration',
            'last_100_sms_outgoing_count', 'last_100_sms_outgoing_to_onnet_count',
            'last_100_sms_outgoing_to_offnet_count',
            'last_100_sms_outgoing_to_abroad_count', 'last_100_gprs_usage'
        ]
        

        df = pd.DataFrame(columns=all_features)
        df.loc[0] = 0  


        for key, value in input_data.items():
            if key in all_features:
                df.loc[0, key] = float(value)


        prediction = model.predict_proba(df)[0]
        churn_probability = prediction[1]
        
        return jsonify({
            'model_used': model_name,
            'churn_probability': float(churn_probability),
            'will_churn': bool(churn_probability > 0.5)
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)