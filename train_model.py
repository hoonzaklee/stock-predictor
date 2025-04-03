import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

# 주가 데이터 다운로드 (애플)
data = yf.download('AAPL', start='2022-01-01', end='2024-01-01')

# 특징 생성
data['MA5'] = data['Close'].rolling(window=5).mean()
data['Volume'] = data['Volume']
data = data.dropna()

# 입력(X), 출력(y)
X = data[['MA5', 'Volume']]
y = data['Close']

# 모델 훈련
model = RandomForestRegressor()
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'stock_model.pkl')
