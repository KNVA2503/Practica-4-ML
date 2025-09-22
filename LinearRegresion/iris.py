import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# 1. Cargar el dataset de Iris
print("Cargando el dataset de Iris...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = X['sepal length (cm)'] # La variable a predecir es el largo del sépalo
X = X.drop('sepal length (cm)', axis=1)
print("Dataset cargado exitosamente.")

# 2. Preparar los datos para el modelo
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
plt.xlabel("Valores Reales (Largo del sépalo en cm)")
plt.ylabel("Valores Predichos (Largo del sépalo en cm)")
plt.title("Valores Reales vs. Valores Predichos por el Modelo de Regresión Lineal")
plt.show()

# 6. Visualizar coeficientes
coefficients = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(12, 6))
plt.bar(coefficients.index, coefficients)
plt.xticks(rotation=45, ha='right')
plt.title("Coeficientes en el Modelo de Regresión Lineal")
plt.xlabel("Características")
plt.ylabel("Magnitud del Coeficiente")
plt.tight_layout()
plt.show()