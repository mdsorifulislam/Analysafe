# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
app = Flask(__name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

# LSTM মডেলের জন্য ডেটা প্রসেসিং
def create_dataset(data, time_step=60):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
    return np.array(X)

# LSTM মডেল তৈরি এবং প্রশিক্ষণ
def train_lstm_model(symbol):
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['Time Series (5min)']).T
        df['4. close'] = df['4. close'].astype(float)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['4. close']])
        
        time_step = 60
        X, y = create_dataset(scaled_data, time_step), scaled_data[time_step:]

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=1, batch_size=32, verbose=0)
        
        return model, scaler, scaled_data
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None

@app.route('/analyze_trade', methods=['POST'])
def analyze_trade():
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    model, scaler, scaled_data = train_lstm_model(symbol)
    if model is None:
        return jsonify({"error": "Failed to analyze market"}), 500

    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    prediction = model.predict(last_60_days)
    
    predicted_price = scaler.inverse_transform(prediction)
    last_price = scaler.inverse_transform(scaled_data[-1].reshape(1, -1))
    
    recommendation = "BUY" if predicted_price[0][0] > last_price[0][0] else "SELL"

    return jsonify({
        "symbol": symbol,
        "recommendation": recommendation,
        "message": f"AI (LSTM) recommends to {recommendation} for {symbol}."
    })

@app.route('/get_symbols', methods=['GET'])
def get_symbols():
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX', 'NVDA']
    return jsonify({"symbols": symbols})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
