import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def california_housing_knn():
    """
    KNeighborsRegressor para California Housing Dataset - PREDICCIÓN DE PRECIOS DE VIVIENDAS
    """
    print("🏠 KNEIGHBORSREGRESSOR - CALIFORNIA HOUSING DATASET")
    print("=" * 65)

    # Cargar dataset California Housing
    california = fetch_california_housing()
    X, y = california.data, california.target

    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(X, columns=california.feature_names)
    df['MedHouseVal'] = y  # Añadir el valor mediano de la vivienda

    print("📊 INFORMACIÓN DEL DATASET:")
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"Características: {california.feature_names}")
    print(f"Target: MedHouseVal (Valor mediano de viviendas en $100,000)")

    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

    # Separar características y target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal'] * 100000  # Convertir a dólares reales

    print(f"\n🎯 OBJETIVO DE REGRESIÓN: Predecir precio de viviendas")
    print(f"Características usadas: {X.columns.tolist()}")
    print(f"Rango de precios: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Precio promedio: ${y.mean():,.0f}")

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
        'knn__p': [1, 2],  # 1: Manhattan, 2: Euclidean
        'knn__leaf_size': [20, 30, 40]
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

    # Métricas en porcentaje del precio promedio
    avg_price = y.mean()
    rmse_percent = (rmse / avg_price) * 100
    mae_percent = (mae / avg_price) * 100

    print("\n📈 RESULTADOS DE REGRESIÓN:")
    print("-" * 50)
    print(f"MSE: ${mse:,.0f}")
    print(f"RMSE: ${rmse:,.0f} ({rmse_percent:.1f}% del promedio)")
    print(f"MAE: ${mae:,.0f} ({mae_percent:.1f}% del promedio)")
    print(f"R² Score: {r2:.4f}")

    # Visualizaciones
    print("\n📊 VISUALIZACIÓN DE RESULTADOS")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predicciones vs Valores reales
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Precio Real ($)')
    axes[0, 0].set_ylabel('Precio Predicho ($)')
    axes[0, 0].set_title('Predicciones vs Valores Reales')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='plain', axis='both')

    # 2. Residuales
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicciones ($)')
    axes[0, 1].set_ylabel('Residuales ($)')
    axes[0, 1].set_title('Análisis de Residuales')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='plain', axis='both')

    # 3. Distribución de errores
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Error de Predicción ($)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Errores')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='plain', axis='x')

    # 4. Comparación de precios (primeras 10 muestras)
    sample_indices = range(min(10, len(y_test)))
    x_pos = np.arange(len(sample_indices))
    width = 0.35

    real_bars = axes[1, 1].bar(x_pos - width / 2, y_test.values[sample_indices] / 1000, width,
                               label='Real', alpha=0.7, color='blue')
    pred_bars = axes[1, 1].bar(x_pos + width / 2, y_pred[sample_indices] / 1000, width,
                               label='Predicho', alpha=0.7, color='red')

    axes[1, 1].set_xlabel('Muestra')
    axes[1, 1].set_ylabel('Precio (miles de $)')
    axes[1, 1].set_title('Comparación: Precio Real vs Predicho')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Añadir valores en las barras
    for i, (real_bar, pred_bar) in enumerate(zip(real_bars, pred_bars)):
        height_real = real_bar.get_height()
        height_pred = pred_bar.get_height()
        axes[1, 1].text(real_bar.get_x() + real_bar.get_width() / 2., height_real,
                        f'{height_real:.0f}k', ha='center', va='bottom')
        axes[1, 1].text(pred_bar.get_x() + pred_bar.get_width() / 2., height_pred,
                        f'{height_pred:.0f}k', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Análisis de importancia de características
    print("\n📋 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS:")
    print("-" * 50)

    # Calcular importancia por permutación
    X_test_scaled = best_model.named_steps['scaler'].transform(X_test)

    perm_importance = permutation_importance(
        best_model.named_steps['knn'],
        X_test_scaled,
        y_test,
        n_repeats=10,
        random_state=42
    )

    # Crear DataFrame de importancia
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print(importance_df.head(10).to_string(index=False))

    # Visualizar importancia
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Importancia')
    plt.title('Importancia de Características para Predecir Precios de Viviendas')
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Predicciones de ejemplo
    print("\n🎯 PREDICCIONES DE EJEMPLO:")
    print("-" * 50)

    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_price = y_test.iloc[idx]
        predicted_price = y_pred[idx]
        error = abs(actual_price - predicted_price)
        error_percent = (error / actual_price) * 100

        print(f"\n🏡 Muestra {i + 1}:")
        print(f"  Precio Real: ${actual_price:,.0f}")
        print(f"  Precio Predicho: ${predicted_price:,.0f}")
        print(f"  Error: ${error:,.0f} ({error_percent:.1f}%)")

        # Mostrar características importantes
        sample_data = X_test.iloc[idx]
        print(f"  Características:")
        for feature in importance_df['feature'].head(3):
            value = sample_data[feature]
            print(f"    • {feature}: {value:.2f}")

    return best_model, X, y, df


def predict_california_house(model, features_dict, feature_columns):
    """
    Predecir precio para una nueva vivienda en California
    """
    print("\n🏠 PREDICCIÓN PARA NUEVA VIVIENDA:")
    print("=" * 50)

    # Crear DataFrame con las características
    input_df = pd.DataFrame([features_dict])

    # Asegurarse de que todas las columnas estén presentes
    for col in feature_columns:
        if col not in input_df.columns:
            # Usar valores promedio si falta alguna característica
            input_df[col] = np.mean([features_dict.get(k, 0) for k in feature_columns])

    # Reordenar columnas
    input_df = input_df[feature_columns]

    # Predecir
    predicted_price = model.predict(input_df)[0]

    print("Características de la vivienda:")
    for feature, value in features_dict.items():
        print(f"  • {feature}: {value:.2f}")
    print(f"\n💰 Precio predicho: ${predicted_price:,.0f}")

    # Interpretación del precio
    if predicted_price < 100000:
        price_category = "Muy económico"
    elif predicted_price < 200000:
        price_category = "Económico"
    elif predicted_price < 350000:
        price_category = "Medio"
    elif predicted_price < 500000:
        price_category = "Alto"
    else:
        price_category = "Muy alto"

    print(f"🏷️  Categoría de precio: {price_category}")

    return predicted_price


# Ejecutar el algoritmo
if __name__ == "__main__":
    print("🏠 ALGORITMO SUPERVISADO: KNeighborsRegressor en California Housing")
    print("🔍 Objetivo: Predecir precios de viviendas en California")
    print("=" * 70)

    # Entrenar el modelo
    model, X, y, df = california_housing_knn()

    # Ejemplo de predicción para nuevas viviendas
    if model is not None:
        print("\n" + "=" * 70)
        print("🎯 PREDICCIONES PARA NUEVAS VIVIENDAS")
        print("=" * 70)

        # Viviendas de ejemplo para predecir
        example_houses = [
            {
                'MedInc': 8.0,  # Ingreso mediano alto
                'HouseAge': 25,  # Casa relativamente nueva
                'AveRooms': 6.0,  # Muchas habitaciones
                'AveBedrms': 2.0,  # Suficientes dormitorios
                'Population': 1000,  # Población moderada
                'AveOccup': 2.5,  # Ocupación normal
                'Latitude': 34.0,  # Buena ubicación
                'Longitude': -118.0  # Área de Los Angeles
            },
            {
                'MedInc': 3.0,  # Ingreso mediano bajo
                'HouseAge': 45,  # Casa vieja
                'AveRooms': 3.0,  # Pocas habitaciones
                'AveBedrms': 1.0,  # Pocos dormitorios
                'Population': 2500,  # Alta población
                'AveOccup': 4.0,  # Alta ocupación
                'Latitude': 36.0,  # Ubicación regular
                'Longitude': -120.0  # Área central
            },
            {
                'MedInc': 15.0,  # Ingreso muy alto
                'HouseAge': 5,  # Casa muy nueva
                'AveRooms': 8.0,  # Muchas habitaciones
                'AveBedrms': 3.0,  # Muchos dormitorios
                'Population': 500,  # Población baja
                'AveOccup': 2.0,  # Baja ocupación
                'Latitude': 37.7,  # Ubicación premium (cerca de SF)
                'Longitude': -122.4  # Área de San Francisco
            }
        ]

        for i, house in enumerate(example_houses, 1):
            print(f"\n🏡 Vivienda ejemplo {i}:")
            predicted = predict_california_house(model, house, X.columns.tolist())