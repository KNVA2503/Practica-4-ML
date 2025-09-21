import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset de resistencia del concreto
print("Cargando el dataset de Resistencia del Concreto...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compresive/Concrete_Data.xls"
df = pd.read_excel(url)
print("Dataset cargado exitosamente.")

# 2. Preparar los datos para el modelo
# La última columna es la variable objetivo (resistencia del concreto)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicializar y entrenar el modelo de Regresión Lineal
print("\nEntrenando el modelo...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Modelo entrenado exitosamente.")

# 4. Hacer predicciones y evaluar el rendimiento
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResultados de la evaluación:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

# 5. Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reales (Resistencia del Concreto)")
plt.ylabel("Valores Predichos (Resistencia del Concreto)")
plt.title("Valores Reales vs. Valores Predichos por el Modelo de Regresión Lineal")
plt.show()

# 6. Visualizar coeficientes
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