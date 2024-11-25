import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Data
data = {
    "Tahun": [i for i in range(1970, 2023)],
    "Produksi Padi": [
        18693649.00, 20483687.00, 19393933.00, 21490578.00,
        22476073.00, 22339455.00, 23300939.00, 23347132.00,
        25771570.00, 26282663.00, 29651905.00, 32774176.00,
        33583677.00, 35303106.00, 38136446.00, 39032945.00,
        39726761.00, 40078195.00, 41676170.00, 44725582.00,
        45178751.00, 44688247.00, 48240009.00, 48181087.00,
        46641524.00, 49744140.00, 51101506.00, 49377054.00,
        49236692.00, 50866387.00, 51898852.00, 50460782.00,
        51489694.00, 52137604.00, 54088468.00, 54151097.00,
        54454937.00, 57157435.00, 60325925.00, 64398890.00,
        66469394.00, 65756904.00, 69056126.00, 71279709.00,
        70846465.00, 75397841.00, 79354767.00, 81148617.00,
        59101577.84, 54604033.34, 54649202.24, 53802637.44,
        54338410.44
    ]
}

df = pd.DataFrame(data)

# Lagging Data
df['Lag'] = df['Produksi Padi'].shift(1)
df.dropna(inplace=True)

# Training and Testing Data
X = df[['Lag']]
y = df['Produksi Padi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Predicting for 2023 and 2024
last_year_production = df['Produksi Padi'].iloc[-1]
pred_2023 = model.predict([[last_year_production]])[0]
pred_2024 = model.predict([[pred_2023]])[0]

# Creating a result DataFrame
results = {
    "Tahun": [2023, 2024],
    "Prediksi Produksi Padi": [pred_2023, pred_2024]
}

df_results = pd.DataFrame(results)

# Saving to Excel
with pd.ExcelWriter('prediksi_produksi_padi.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Data Lag', index=False)
    df_results.to_excel(writer, sheet_name='Hasil Prediksi', index=False)
    
print('File Excel berhasil disimpan sebagai "prediksi_produksi_padi.xlsx".')