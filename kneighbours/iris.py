import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")


def iris_knn_regressor():
    """
    KNeighborsRegressor para el dataset Iris - PREDICCIÓN DE LONGITUD DE PÉTALO
    """
    print("🌸 KNEIGHBORSREGRESSOR - IRIS DATASET (REGRESIÓN)")
    print("=" * 60)

    # Cargar dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y  # Añadir la especie como columna

    print("📊 INFORMACIÓN DEL DATASET IRIS:")
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"Características: {iris.feature_names}")
    print(f"Especies: {iris.target_names}")

    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

    # Vamos a predecir la longitud del pétalo (petal length) usando las otras características
    # Esto convierte el problema en uno de REGRESIÓN SUPERVISADA

    # Separar características y target (longitud del pétalo)
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
    y = df['petal length (cm)']

    print(f"\n🎯 OBJETIVO DE REGRESIÓN: Predecir 'petal length (cm)'")
    print(f"Características usadas: {X.columns.tolist()}")
    print(f"Rango de longitud de pétalo: {y.min():.1f}cm - {y.max():.1f}cm")

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"\n📊 DIVISIÓN DE DATOS:")
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba: {X_test.shape[0]} muestras")

    # Crear pipeline con escalado y KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    # Búsqueda de hiperparámetros
    print("\n🔍 OPTIMIZANDO HIPERPARÁMETROS...")
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1: Manhattan, 2: Euclidean
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"✅ MEJORES PARÁMETROS:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param.split('__')[1]}: {value}")

    # Mejor modelo
    best_model = grid_search.best_estimator_

    # Predicciones
    y_pred = best_model.predict(X_test)

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 RESULTADOS DE REGRESIÓN:")
    print("-" * 40)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f} cm")
    print(f"MAE: {mae:.4f} cm")
    print(f"R² Score: {r2:.4f}")

    # Visualizaciones
    print("\n📊 VISUALIZACIÓN DE RESULTADOS")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Predicciones vs Valores reales
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, s=60, c='purple')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Longitud Real del Pétalo (cm)')
    axes[0, 0].set_ylabel('Longitud Predicha del Pétalo (cm)')
    axes[0, 0].set_title('Predicciones vs Valores Reales')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuales
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, s=60, c='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicciones (cm)')
    axes[0, 1].set_ylabel('Residuales (cm)')
    axes[0, 1].set_title('Análisis de Residuales')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribución de errores
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Error de Predicción (cm)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Errores')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Comparación por especies
    species_names = iris.target_names
    species_colors = ['red', 'green', 'blue']

    # Añadir especies al DataFrame de test
    test_df = X_test.copy()
    test_df['actual'] = y_test.values
    test_df['predicted'] = y_pred
    test_df['species'] = df.iloc[X_test.index]['species'].values

    for i, species_name in enumerate(species_names):
        species_data = test_df[test_df['species'] == i]
        axes[1, 1].scatter(species_data['actual'], species_data['predicted'],
                           alpha=0.7, s=60, label=species_name, color=species_colors[i])

    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axes[1, 1].set_xlabel('Longitud Real (cm)')
    axes[1, 1].set_ylabel('Longitud Predicha (cm)')
    axes[1, 1].set_title('Predicciones por Especies')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Análisis de performance por especie
    print("\n📋 ANÁLISIS POR ESPECIE:")
    print("-" * 40)

    for i, species_name in enumerate(species_names):
        species_mask = (df.iloc[X_test.index]['species'] == i)
        if species_mask.any():
            species_actual = y_test[species_mask]
            species_pred = y_pred[species_mask]
            species_rmse = np.sqrt(mean_squared_error(species_actual, species_pred))
            species_r2 = r2_score(species_actual, species_pred)

            print(f"{species_name}:")
            print(f"  • RMSE: {species_rmse:.4f} cm")
            print(f"  • R²: {species_r2:.4f}")
            print(f"  • Muestras: {len(species_actual)}")

    # Predicciones de ejemplo
    print("\n🎯 PREDICCIONES DE EJEMPLO:")
    print("-" * 40)

    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_value = y_test.iloc[idx]
        predicted_value = y_pred[idx]
        error = abs(actual_value - predicted_value)
        species_idx = df.iloc[X_test.index[idx]]['species']
        species_name = species_names[int(species_idx)]

        print(f"\nMuestra {i + 1} ({species_name}):")
        print(f"  Real: {actual_value:.2f} cm")
        print(f"  Predicho: {predicted_value:.2f} cm")
        print(f"  Error: {error:.2f} cm")
        print(f"  Características:")
        for feature, value in X_test.iloc[idx].items():
            print(f"    • {feature}: {value:.2f}")

    return best_model, X, y, df


def predict_iris_flower(model, features_dict, feature_columns):
    """
    Predecir longitud de pétalo para una nueva flor
    """
    print("\n🌺 PREDICCIÓN PARA NUEVA FLOR:")
    print("=" * 40)

    # Crear DataFrame con las características
    input_df = pd.DataFrame([features_dict])

    # Asegurarse de que todas las columnas estén presentes
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = np.mean([features_dict.get(k, 0) for k in feature_columns])

    # Reordenar columnas
    input_df = input_df[feature_columns]

    # Predecir
    predicted_length = model.predict(input_df)[0]

    print("Características de la flor:")
    for feature, value in features_dict.items():
        print(f"  • {feature}: {value:.2f}")
    print(f"\n📏 Longitud de pétalo predicha: {predicted_length:.2f} cm")

    # Interpretación basada en especies típicas
    if predicted_length < 2.5:
        likely_species = "setosa"
    elif predicted_length < 5.0:
        likely_species = "versicolor"
    else:
        likely_species = "virginica"

    print(f"🌿 Especie probable: {likely_species}")

    return predicted_length


# Ejecutar el algoritmo
if __name__ == "__main__":
    print("🌸 ALGORITMO SUPERVISADO: KNeighborsRegressor en Iris Dataset")
    print("🔍 Objetivo: Predecir longitud del pétalo (regresión)")
    print("=" * 70)

    # Entrenar el modelo
    model, X, y, df = iris_knn_regressor()

    # Ejemplo de predicción para nuevas flores
    if model is not None:
        print("\n" + "=" * 70)
        print("🎯 PREDICCIONES PARA NUEVAS FLORES")
        print("=" * 70)

        # Flores de ejemplo para predecir
        example_flowers = [
            {
                'sepal length (cm)': 5.1,
                'sepal width (cm)': 3.5,
                'petal width (cm)': 0.2
            },
            {
                'sepal length (cm)': 6.3,
                'sepal width (cm)': 2.8,
                'petal width (cm)': 1.5
            },
            {
                'sepal length (cm)': 7.2,
                'sepal width (cm)': 3.0,
                'petal width (cm)': 2.1
            }
        ]

        for i, flower in enumerate(example_flowers, 1):
            print(f"\n🌷 Flor ejemplo {i}:")
            predicted = predict_iris_flower(model, flower, X.columns.tolist())