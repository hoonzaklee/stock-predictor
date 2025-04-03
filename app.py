from flask import Flask, request, jsonify
from flask_cors import CORS  # ← 추가
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # ← 이 줄 추가로 모든 요청 허용!

model = joblib.load('stock_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ma5 = data.get('ma5')
    volume = data.get('volume')
    
    input_data = np.array([[ma5, volume]])
    prediction = model.predict(input_data)[0]
    
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
