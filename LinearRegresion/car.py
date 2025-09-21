import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset de precios de autos
print("Cargando el dataset de autos...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = [
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
    'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg',
    'price'
]
df = pd.read_csv(url, names=column_names, na_values='?')
print("Dataset cargado exitosamente.")

# 2. Preprocesamiento de los datos
print("\nVerificando y eliminando valores faltantes...")
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['make', 'fuel-type', 'aspiration', 'num-of-doors',
                                 'body-style', 'drive-wheels', 'engine-location',
                                 'engine-type', 'num-of-cylinders', 'fuel-system'])
print("Preprocesamiento completado.")

# 3. Preparar los datos para el modelo
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inicializar y entrenar el modelo de Regresión Lineal
print("\nEntrenando el modelo...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Modelo entrenado exitosamente.")

# 5. Hacer predicciones y evaluar el rendimiento
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResultados de la evaluación:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

# 6. Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reales (Precios de autos)")
plt.ylabel("Valores Predichos (Precios de autos)")
plt.title("Valores Reales vs. Valores Predichos por el Modelo de Regresión Lineal")
plt.show()

# 7. Visualizar coeficientes
coefficients = pd.Series(model.coef_, index=X.columns)
top_10_features = coefficients.abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(12, 6))
plt.bar(coefficients[top_10_features].index, coefficients[top_10_features])
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Coeficientes en el Modelo de Regresión Lineal")
plt.xlabel("Características")
plt.ylabel("Magnitud del Coeficiente")
plt.tight_layout()
plt.show()