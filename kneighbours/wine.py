import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def load_wine_quality_data():
    """
    Cargar dataset Wine Quality desde UCI o usar datos de ejemplo
    """
    try:
        # Intentar cargar desde URL (dataset UCI Wine Quality)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        print("✅ Dataset Wine Quality cargado desde UCI")
    except:
        # Datos de ejemplo si falla la descarga
        print("⚠️  Usando datos de ejemplo...")
        np.random.seed(42)
        n_samples = 1000

        data = {
            'fixed acidity': np.random.uniform(4.0, 16.0, n_samples),
            'volatile acidity': np.random.uniform(0.1, 1.6, n_samples),
            'citric acid': np.random.uniform(0.0, 1.0, n_samples),
            'residual sugar': np.random.uniform(0.5, 15.0, n_samples),
            'chlorides': np.random.uniform(0.01, 0.2, n_samples),
            'free sulfur dioxide': np.random.uniform(1, 70, n_samples),
            'total sulfur dioxide': np.random.uniform(5, 250, n_samples),
            'density': np.random.uniform(0.98, 1.005, n_samples),
            'pH': np.random.uniform(2.8, 4.0, n_samples),
            'sulphates': np.random.uniform(0.3, 2.0, n_samples),
            'alcohol': np.random.uniform(8.0, 15.0, n_samples),
            'quality': np.random.randint(3, 9, n_samples)
        }
        df = pd.DataFrame(data)

    return df


def wine_quality_knn_regressor():
    """
    KNeighborsRegressor para Wine Quality Dataset - PREDICCIÓN DE CALIDAD DEL VINO
    """
    print("🍷 KNEIGHBORSREGRESSOR - WINE QUALITY DATASET")
    print("=" * 60)

    # Cargar datos
    df = load_wine_quality_data()

    print("📊 INFORMACIÓN DEL DATASET:")
    print(f"Muestras: {df.shape[0]}, Características: {df.shape[1] - 1}")
    print(f"Características: {df.columns.tolist()[:-1]}")
    print(f"Target: quality (calidad del vino)")

    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

    # Análisis de la distribución de calidad
    print(f"\n📊 DISTRIBUCIÓN DE CALIDAD:")
    quality_counts = df['quality'].value_counts().sort_index()
    for quality, count in quality_counts.items():
        print(f"  Calidad {quality}: {count} muestras ({count / len(df) * 100:.1f}%)")

    # Información sobre las características del vino
    print("\n🔍 INFORMACIÓN DE LAS CARACTERÍSTICAS:")
    feature_info = {
        'fixed acidity': 'Ácidos fijos (g/dm³) - contribuyen a la acidez',
        'volatile acidity': 'Acidez volátil (g/dm³) - relacionado con el sabor avinagrado',
        'citric acid': 'Ácido cítrico (g/dm³) - aporta frescura',
        'residual sugar': 'Azúcar residual (g/dm³) - dulzor del vino',
        'chlorides': 'Cloruros (g/dm³) - salinidad',
        'free sulfur dioxide': 'SO2 libre (mg/dm³) - antioxidante y antiséptico',
        'total sulfur dioxide': 'SO2 total (mg/dm³) - preservante',
        'density': 'Densidad (g/cm³) - relacionado con alcohol y azúcar',
        'pH': 'pH - acidez del vino',
        'sulphates': 'Sulfatos (g/dm³) - antioxidante',
        'alcohol': 'Alcohol (% vol) - cuerpo del vino'
    }

    for feature, desc in feature_info.items():
        if feature in df.columns:
            print(f"  • {feature}: {desc}")

    # Separar características y target
    X = df.drop('quality', axis=1)
    y = df['quality']

    print(f"\n🎯 OBJETIVO DE REGRESIÓN: Predecir calidad del vino (escala 0-10)")
    print(f"Rango de calidad: {y.min()} - {y.max()}")
    print(f"Calidad promedio: {y.mean():.2f}")
    print(f"Desviación estándar: {y.std():.2f}")

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
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
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

    # Redondear predicciones a medio puntos (mejor para calidad de vino)
    y_pred_rounded = np.round(y_pred * 2) / 2

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Métricas con predicciones redondeadas
    mse_rounded = mean_squared_error(y_test, y_pred_rounded)
    mae_rounded = mean_absolute_error(y_test, y_pred_rounded)

    print("\n📈 RESULTADOS DE REGRESIÓN:")
    print("-" * 50)
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f} puntos")
    print(f"MAE: {mae:.3f} puntos")
    print(f"R² Score: {r2:.4f}")

    print(f"\n📊 CON PREDICCIONES REDONDEADAS (0.5 puntos):")
    print(f"MAE: {mae_rounded:.3f} puntos")

    # Porcentaje de predicciones dentro de ±0.5 y ±1 punto
    within_0_5 = np.sum(np.abs(y_test - y_pred_rounded) <= 0.5) / len(y_test) * 100
    within_1_0 = np.sum(np.abs(y_test - y_pred_rounded) <= 1.0) / len(y_test) * 100

    print(f"Predicciones dentro de ±0.5 puntos: {within_0_5:.1f}%")
    print(f"Predicciones dentro de ±1.0 punto: {within_1_0:.1f}%")

    # Visualizaciones
    print("\n📊 VISUALIZACIÓN DE RESULTADOS")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predicciones vs Valores reales
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, s=60, color='darkred')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Calidad Real')
    axes[0, 0].set_ylabel('Calidad Predicha')
    axes[0, 0].set_title('Predicciones vs Valores Reales')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuales
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, s=60, color='darkblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicciones')
    axes[0, 1].set_ylabel('Residuales')
    axes[0, 1].set_title('Análisis de Residuales')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribución de errores
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Error de Predicción (puntos)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Errores')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Comparación de calidades
    quality_categories = sorted(y_test.unique())
    actual_means = [y_test[y_test == q].mean() for q in quality_categories]
    predicted_means = [y_pred[y_test == q].mean() for q in quality_categories]

    x_pos = np.arange(len(quality_categories))
    width = 0.35

    axes[1, 1].bar(x_pos - width / 2, actual_means, width, label='Real', alpha=0.7, color='green')
    axes[1, 1].bar(x_pos + width / 2, predicted_means, width, label='Predicho', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Categoría de Calidad')
    axes[1, 1].set_ylabel('Calidad Promedio')
    axes[1, 1].set_title('Calidad Real vs Predicha por Categoría')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(quality_categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

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
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Importancia')
    plt.title('Importancia de Características para Predecir Calidad del Vino')
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Análisis de correlaciones
    print("\n🔗 ANÁLISIS DE CORRELACIONES:")
    print("-" * 50)

    corr_with_quality = df.corr()['quality'].sort_values(ascending=False)
    print("Correlación con la calidad del vino:")
    for feature, corr in corr_with_quality.items():
        if feature != 'quality':
            significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"  • {feature}: {corr:.3f} {significance}")

    # Predicciones de ejemplo
    print("\n🎯 PREDICCIONES DE EJEMPLO:")
    print("-" * 50)

    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_quality = y_test.iloc[idx]
        predicted_quality = y_pred[idx]
        predicted_rounded = np.round(predicted_quality * 2) / 2
        error = abs(actual_quality - predicted_quality)

        print(f"\n🍷 Vino {i + 1}:")
        print(f"  Calidad Real: {actual_quality}")
        print(f"  Calidad Predicha: {predicted_quality:.2f} → {predicted_rounded} (redondeado)")
        print(f"  Error: {error:.2f} puntos")

        # Mostrar características importantes
        sample_data = X_test.iloc[idx]
        print(f"  Características clave:")
        for feature in importance_df['feature'].head(3):
            value = sample_data[feature]
            desc = feature_info.get(feature, feature)
            print(f"    • {desc.split(' - ')[0]}: {value:.2f}")

    return best_model, X, y, df, importance_df


def predict_wine_quality(model, features_dict, feature_columns):
    """
    Predecir calidad del vino para una nueva muestra
    """
    print("\n🍷 PREDICCIÓN PARA NUEVO VINO:")
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
    predicted_quality = model.predict(input_df)[0]
    predicted_rounded = np.round(predicted_quality * 2) / 2  # Redondear a medio punto

    print("Características del vino:")
    feature_info = {
        'fixed acidity': 'Ácidos fijos',
        'volatile acidity': 'Acidez volátil',
        'citric acid': 'Ácido cítrico',
        'residual sugar': 'Azúcar residual',
        'chlorides': 'Cloruros',
        'free sulfur dioxide': 'SO2 libre',
        'total sulfur dioxide': 'SO2 total',
        'density': 'Densidad',
        'pH': 'pH',
        'sulphates': 'Sulfatos',
        'alcohol': 'Alcohol'
    }

    for feature, value in features_dict.items():
        desc = feature_info.get(feature, feature)
        print(f"  • {desc}: {value:.2f}")

    print(f"\n⭐ Calidad predicha: {predicted_quality:.2f} → {predicted_rounded}")

    # Interpretación de la calidad
    quality_description = {
        3: "Muy pobre",
        4: "Pobre",
        5: "Aceptable",
        6: "Bueno",
        7: "Muy bueno",
        8: "Excelente",
        9: "Excepcional"
    }

    rounded_quality = int(round(predicted_rounded))
    description = quality_description.get(rounded_quality, "Fuera de rango")

    print(f"🏆 Evaluación: {description}")

    # Recomendaciones basadas en la calidad
    if predicted_rounded >= 7.0:
        print("💡 Este vino tiene potencial premium - considerar envejecimiento")
    elif predicted_rounded >= 6.0:
        print("💡 Vino de buena calidad - listo para consumo")
    else:
        print("💡 Vino de calidad estándar - mejor para consumo inmediato")

    return predicted_quality


# Ejecutar el algoritmo
if __name__ == "__main__":
    print("🍷 ALGORITMO SUPERVISADO: KNeighborsRegressor en Wine Quality")
    print("🔍 Objetivo: Predecir calidad del vino (escala 0-10)")
    print("=" * 70)

    # Entrenar el modelo
    model, X, y, df, importance_df = wine_quality_knn_regressor()

    # Ejemplo de predicción para nuevos vinos
    if model is not None:
        print("\n" + "=" * 70)
        print("🎯 PREDICCIONES PARA NUEVOS VINOS")
        print("=" * 70)

        # Vinos de ejemplo para predecir
        example_wines = [
            {
                'fixed acidity': 7.8,
                'volatile acidity': 0.35,
                'citric acid': 0.45,
                'residual sugar': 2.5,
                'chlorides': 0.045,
                'free sulfur dioxide': 25,
                'total sulfur dioxide': 120,
                'density': 0.995,
                'pH': 3.25,
                'sulphates': 0.65,
                'alcohol': 12.5
            },
            {
                'fixed acidity': 6.2,
                'volatile acidity': 0.75,
                'citric acid': 0.05,
                'residual sugar': 8.5,
                'chlorides': 0.085,
                'free sulfur dioxide': 15,
                'total sulfur dioxide': 85,
                'density': 0.998,
                'pH': 3.65,
                'sulphates': 0.45,
                'alcohol': 10.8
            },
            {
                'fixed acidity': 9.1,
                'volatile acidity': 0.25,
                'citric acid': 0.55,
                'residual sugar': 1.8,
                'chlorides': 0.035,
                'free sulfur dioxide': 35,
                'total sulfur dioxide': 150,
                'density': 0.992,
                'pH': 3.15,
                'sulphates': 0.85,
                'alcohol': 14.2
            }
        ]

        for i, wine in enumerate(example_wines, 1):
            print(f"\n🍷 Vino ejemplo {i}:")
            predicted = predict_wine_quality(model, wine, X.columns.tolist())