import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title("Prediksi Biaya Asuransi dan Saran Kesehatan")

data = pd.read_csv('Regression.csv')

data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

st.write("### Heatmap Korelasi")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

X = data[['age', 'bmi', 'children', 'smoker', 'sex']]
y = data['charges']

scaler = StandardScaler()
X[['age', 'bmi', 'children']] = scaler.fit_transform(X[['age', 'bmi', 'children']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

model_path = 'random_forest_model.pkl'
joblib.dump(model, model_path)
st.write(f"Model disimpan di: `{model_path}`")

st.write("### Pentingnya Fitur")
importance = model.feature_importances_
features = ['Usia', 'BMI', 'Jumlah Anak', 'Perokok', 'Jenis Kelamin']
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(features, importance, color='skyblue')
ax.set_title("Pentingnya Fitur")
ax.set_xlabel("Pentingnya")
st.pyplot(fig)

st.write("### Kalkulator BMI")
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, step=0.1)
height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, step=0.1)
if weight and height:
    bmi_calculated = weight / (height / 100) ** 2
    st.write(f"BMI Anda adalah: {bmi_calculated:.2f}")
    if bmi_calculated < 18.5:
        st.info("Anda termasuk kategori berat badan kurang.")
    elif 18.5 <= bmi_calculated <= 24.9:
        st.success("Anda termasuk kategori berat badan normal.")
    elif 25 <= bmi_calculated <= 29.9:
        st.warning("Anda termasuk kategori kelebihan berat badan.")
    else:
        st.error("Anda termasuk kategori obesitas.")

st.write("### Prediksi Biaya Asuransi dengan Simulasi")
sex = st.radio("Jenis Kelamin", options=["Pria", "Wanita"])
sex_value = 1 if sex == "Wanita" else 0
age = st.slider("Usia", min_value=18, max_value=100, value=30, step=1)
bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
children = st.slider("Jumlah Anak", min_value=0, max_value=10, value=1, step=1)
smoker = st.radio("Apakah Anda Perokok?", options=["Ya", "Tidak"])
smoker_value = 1 if smoker == "Ya" else 0

if st.button("Prediksi"):

    user_data = scaler.transform([[age, bmi, children]])
    user_data = np.hstack((user_data, [[smoker_value, sex_value]]))
    prediction = model.predict(user_data)[0]
    st.write(f"### Perkiraan Biaya Asuransi: $ {prediction:.2f}")

    st.write("#### Detail Input")
    st.write(f"Jenis Kelamin: {sex}")
    st.write(f"Usia: {age}")
    st.write(f"BMI: {bmi}")
    st.write(f"Jumlah Anak: {children}")
    st.write(f"Perokok: {smoker}")

    
    st.write("### Skor Risiko Kesehatan")
    risk_score = 0
    if smoker == "Ya":
        risk_score += 2
    if bmi > 25:
        risk_score += 1
    if age > 50:
        risk_score += 1

    if risk_score == 0:
        st.success("Risiko kesehatan Anda rendah! Pertahankan gaya hidup sehat.")
    elif risk_score == 1:
        st.warning("Risiko kesehatan Anda sedang. Perhatikan kesehatan Anda dan lakukan pemeriksaan rutin.")
    else:
        st.error("Risiko kesehatan Anda tinggi! Disarankan untuk berkonsultasi dengan dokter.")
    st.write("### Saran Kesehatan")
    if smoker == "Ya":
        st.warning("Pertimbangkan untuk berhenti merokok agar mengurangi risiko kesehatan dan biaya asuransi.")
    if bmi < 18.5:
        st.info("BMI Anda menunjukkan berat badan kurang. Pertimbangkan konsultasi dengan ahli gizi untuk mencapai berat badan sehat.")
    elif 18.5 <= bmi <= 24.9:
        st.success("BMI Anda berada dalam kategori normal. Pertahankan gaya hidup sehat!")
    elif 25 <= bmi <= 29.9:
        st.warning("BMI Anda menunjukkan kelebihan berat badan. Olahraga rutin dan pola makan sehat dapat membantu.")
    else:
        st.error("BMI Anda menunjukkan obesitas. Disarankan untuk berkonsultasi dengan dokter untuk saran lebih lanjut.")

    if age >= 50:
        st.info("Lakukan pemeriksaan kesehatan rutin untuk memantau kondisi kesehatan Anda.")

    variations = [
        (age + 5, bmi + 2, children, smoker_value, sex_value),
        (age - 5, bmi - 1.5, children + 1, smoker_value, sex_value),
        (age, bmi, children, 1 if smoker_value == 0 else 0, sex_value)
    ]
    labels = ["Input Pengguna", "Usia+5, BMI+2", "Usia-5, BMI-1.5, Anak+1", "Status Perokok Berlawanan"]
    all_predictions = [prediction]

    for a, b, c, s, sx in variations:
        var_data = scaler.transform([[a, b, c]])
        var_data = np.hstack((var_data, [[s, sx]]))
        all_predictions.append(model.predict(var_data)[0])
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, all_predictions, color="skyblue")
    ax.set_title("Perkiraan Biaya Asuransi untuk Variasi")
    ax.set_ylabel("Biaya (Rp)")
    st.pyplot(fig)

st.write("### Distribusi Data dalam Dataset")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data['bmi'], bins=30, kde=True, ax=ax, color="skyblue")
ax.set_title("Distribusi BMI")
ax.set_xlabel("BMI")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data['children'], palette="pastel", ax=ax)
ax.set_title("Distribusi Jumlah Anak")
ax.set_xlabel("Jumlah Anak")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data['smoker'], palette="Set2", ax=ax)
ax.set_title("Distribusi Status Perokok")
ax.set_xlabel("Perokok (1=Ya, 0=Tidak)")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)

if 'region' in data.columns:
    st.write("### Analisis Biaya Asuransi Berdasarkan Wilayah")
    region_avg = data.groupby('region')['charges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=region_avg['region'], y=region_avg['charges'], ax=ax, palette="viridis")
    ax.set_title("Rata-rata Biaya Asuransi Berdasarkan Wilayah")
    ax.set_xlabel("Wilayah")
    ax.set_ylabel("Rata-rata Biaya Asuransi")
    st.pyplot(fig)

st.write("### Simulasi Dampak Perubahan")
simulated_bmi = st.slider("Ubah BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
simulated_smoker = st.radio("Ubah Status Merokok", options=["Ya", "Tidak"])
simulated_smoker_value = 1 if simulated_smoker == "Ya" else 0

simulated_data = scaler.transform([[age, simulated_bmi, children]])
simulated_data = np.hstack((simulated_data, [[simulated_smoker_value, sex_value]]))
simulated_prediction = model.predict(simulated_data)[0]
st.write(f"Dampak Perubahan: Biaya Asuransi $ {simulated_prediction:.2f}")

st.write("### Tren Biaya Asuransi Berdasarkan Usia")
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=data['age'], y=data['charges'], ax=ax)
ax.set_title("Tren Biaya Asuransi Berdasarkan Usia")
ax.set_xlabel("Usia")
ax.set_ylabel("Biaya Asuransi")
st.pyplot(fig)

st.write("### Evaluasi Model")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R2 Score: {r2}")

st.write("### Perbandingan Model")
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

comparison_df = pd.DataFrame(results).T
st.dataframe(comparison_df)

if st.button("Unduh Hasil Prediksi"):
    results = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
    csv = results.to_csv(index=False)
    st.download_button(label="Unduh Hasil Prediksi sebagai CSV",
                       data=csv,
                       file_name='hasil_prediksi.csv',
                       mime='text/csv')
