from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

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
