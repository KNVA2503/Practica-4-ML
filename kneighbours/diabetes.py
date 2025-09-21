import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def diabetes_knn_regressor():
    """
    KNeighborsRegressor para Diabetes Dataset - PREDICCIÓN DE PROGRESIÓN DE DIABETES
    """
    print("💊 KNEIGHBORSREGRESSOR - DIABETES DATASET")
    print("=" * 55)

    # Cargar dataset Diabetes
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(X, columns=diabetes.feature_names)
    df['target'] = y  # Añadir la variable objetivo

    print("📊 INFORMACIÓN DEL DATASET DIABETES:")
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"Características: {diabetes.feature_names}")
    print(f"Target: measure of disease progression one year after baseline")

    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

    # Información sobre las características
    print("\n🔍 INFORMACIÓN DE LAS CARACTERÍSTICAS:")
    feature_info = {
        'age': 'age in years',
        'sex': 'gender',
        'bmi': 'body mass index',
        'bp': 'average blood pressure',
        's1': 'total serum cholesterol',
        's2': 'low-density lipoproteins',
        's3': 'high-density lipoproteins',
        's4': 'total cholesterol / HDL ratio',
        's5': 'possibly log of serum triglycerides level',
        's6': 'blood sugar level'
    }

    for feature, desc in feature_info.items():
        print(f"  • {feature}: {desc}")

    # Separar características y target
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"\n🎯 OBJETIVO DE REGRESIÓN: Predecir progresión de diabetes")
    print(f"Características usadas: {X.columns.tolist()}")
    print(f"Rango del target: {y.min():.1f} - {y.max():.1f}")
    print(f"Valor promedio: {y.mean():.1f}")
    print(f"Desviación estándar: {y.std():.1f}")

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

    # Métricas en términos del target
    std_target = y.std()
    rmse_std = rmse / std_target
    mae_std = mae / std_target

    print("\n📈 RESULTADOS DE REGRESIÓN:")
    print("-" * 50)
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} ({rmse_std:.2f} desviaciones estándar)")
    print(f"MAE: {mae:.2f} ({mae_std:.2f} desviaciones estándar)")
    print(f"R² Score: {r2:.4f}")

    # Interpretación clínica
    print(f"\n💡 INTERPRETACIÓN CLÍNICA:")
    print(f"El modelo puede predecir la progresión de la diabetes")
    print(f"con un error promedio de {mae:.1f} puntos")
    print(f"(equivalente a {mae_std:.2f} desviaciones estándar)")

    # Visualizaciones
    print("\n📊 VISUALIZACIÓN DE RESULTADOS")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predicciones vs Valores reales
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, s=60, color='teal')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Progresión Real de Diabetes')
    axes[0, 0].set_ylabel('Progresión Predicha de Diabetes')
    axes[0, 0].set_title('Predicciones vs Valores Reales')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuales
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, s=60, color='coral')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicciones')
    axes[0, 1].set_ylabel('Residuales')
    axes[0, 1].set_title('Análisis de Residuales')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribución de errores
    axes[1, 0].hist(residuals, bins=25, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Error de Predicción')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Errores')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Comparación de valores reales vs predichos
    sample_indices = range(min(8, len(y_test)))
    x_pos = np.arange(len(sample_indices))
    width = 0.35

    real_bars = axes[1, 1].bar(x_pos - width / 2, y_test.values[sample_indices], width,
                               label='Real', alpha=0.7, color='blue')
    pred_bars = axes[1, 1].bar(x_pos + width / 2, y_pred[sample_indices], width,
                               label='Predicho', alpha=0.7, color='orange')

    axes[1, 1].set_xlabel('Paciente (Muestra)')
    axes[1, 1].set_ylabel('Progresión de Diabetes')
    axes[1, 1].set_title('Comparación: Valores Reales vs Predichos')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Añadir valores en las barras
    for i, (real_bar, pred_bar) in enumerate(zip(real_bars, pred_bars)):
        height_real = real_bar.get_height()
        height_pred = pred_bar.get_height()
        axes[1, 1].text(real_bar.get_x() + real_bar.get_width() / 2., height_real,
                        f'{height_real:.0f}', ha='center', va='bottom', fontsize=8)
        axes[1, 1].text(pred_bar.get_x() + pred_bar.get_width() / 2., height_pred,
                        f'{height_pred:.0f}', ha='center', va='bottom', fontsize=8)

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
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Importancia')
    plt.title('Importancia de Variables Médicas para Predecir Progresión de Diabetes')
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Análisis de correlaciones
    print("\n🔗 ANÁLISIS DE CORRELACIONES:")
    print("-" * 50)

    # Añadir target al DataFrame para análisis de correlación
    corr_df = X.copy()
    corr_df['target'] = y

    correlation = corr_df.corr()['target'].sort_values(ascending=False)
    print("Correlación con la progresión de diabetes:")
    for feature, corr in correlation.items():
        if feature != 'target':
            print(f"  • {feature}: {corr:.3f}")

    # Predicciones de ejemplo
    print("\n🎯 PREDICCIONES DE EJEMPLO:")
    print("-" * 50)

    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_value = y_test.iloc[idx]
        predicted_value = y_pred[idx]
        error = abs(actual_value - predicted_value)
        error_percent = (error / actual_value) * 100 if actual_value != 0 else 0

        print(f"\n👤 Paciente {i + 1}:")
        print(f"  Progresión Real: {actual_value:.1f}")
        print(f"  Progresión Predicha: {predicted_value:.1f}")
        print(f"  Error: {error:.1f} ({error_percent:.1f}%)")

        # Mostrar características médicas importantes
        sample_data = X_test.iloc[idx]
        print(f"  Características médicas:")
        for feature in importance_df['feature'].head(3):
            value = sample_data[feature]
            feature_desc = feature_info.get(feature, feature)
            print(f"    • {feature_desc}: {value:.2f}")

    return best_model, X, y, df, importance_df


def predict_diabetes_progression(model, features_dict, feature_columns):
    """
    Predecir progresión de diabetes para un nuevo paciente
    """
    print("\n👤 PREDICCIÓN PARA NUEVO PACIENTE:")
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
    predicted_progression = model.predict(input_df)[0]

    print("Características del paciente:")
    feature_info = {
        'age': 'Edad (años escalados)',
        'sex': 'Sexo (escalado)',
        'bmi': 'Índice de Masa Corporal (escalado)',
        'bp': 'Presión Arterial (escalada)',
        's1': 'Colesterol Total (escalado)',
        's2': 'LDL (escalado)',
        's3': 'HDL (escalado)',
        's4': 'Ratio Colesterol/HDL (escalado)',
        's5': 'Triglicéridos (escalado)',
        's6': 'Nivel de Azúcar (escalado)'
    }

    for feature, value in features_dict.items():
        desc = feature_info.get(feature, feature)
        print(f"  • {desc}: {value:.2f}")

    print(f"\n📈 Progresión predicha de diabetes: {predicted_progression:.1f}")

    # Interpretación clínica
    if predicted_progression < 100:
        risk_level = "Bajo riesgo"
    elif predicted_progression < 150:
        risk_level = "Riesgo moderado"
    elif predicted_progression < 200:
        risk_level = "Alto riesgo"
    else:
        risk_level = "Riesgo muy alto"

    print(f"⚠️  Nivel de riesgo: {risk_level}")

    # Recomendaciones basadas en el riesgo
    if predicted_progression > 150:
        print("💡 Recomendaciones: Control médico intensivo, dieta estricta, ejercicio regular")
    elif predicted_progression > 100:
        print("💡 Recomendaciones: Control médico regular, dieta balanceada, actividad física")
    else:
        print("💡 Recomendaciones: Mantener estilo de vida saludable, chequeos anuales")

    return predicted_progression


# Ejecutar el algoritmo
if __name__ == "__main__":
    print("💊 ALGORITMO SUPERVISADO: KNeighborsRegressor en Diabetes Dataset")
    print("🔍 Objetivo: Predecir progresión de diabetes (regresión médica)")
    print("=" * 70)

    # Entrenar el modelo
    model, X, y, df, importance_df = diabetes_knn_regressor()

    # Ejemplo de predicción para nuevos pacientes
    if model is not None:
        print("\n" + "=" * 70)
        print("🎯 PREDICCIONES PARA NUEVOS PACIENTES")
        print("=" * 70)

        # Pacientes de ejemplo para predecir
        example_patients = [
            {
                'age': 0.2,  # Paciente joven
                'sex': 0.0,  # Mujer
                'bmi': 0.1,  # BMI normal
                'bp': -0.1,  # Presión normal
                's1': 0.05,  # Colesterol normal
                's2': 0.03,  # LDL normal
                's3': 0.15,  # HDL bueno
                's4': -0.1,  # Ratio favorable
                's5': 0.08,  # Triglicéridos normales
                's6': 0.1  # Azúcar normal
            },
            {
                'age': 0.8,  # Paciente mayor
                'sex': 0.1,  # Hombre
                'bmi': 0.7,  # BMI alto (sobrepeso)
                'bp': 0.6,  # Presión alta
                's1': 0.9,  # Colesterol alto
                's2': 0.85,  # LDL alto
                's3': -0.3,  # HDL bajo
                's4': 1.2,  # Ratio desfavorable
                's5': 1.1,  # Triglicéridos altos
                's6': 0.9  # Azúcar alto
            },
            {
                'age': 0.5,  # Edad media
                'sex': 0.05,  #
                'bmi': 0.4,  # BMI moderado
                'bp': 0.3,  # Presión moderada
                's1': 0.4,  # Colesterol moderado
                's2': 0.35,  # LDL moderado
                's3': 0.0,  # HDL promedio
                's4': 0.5,  # Ratio moderado
                's5': 0.4,  # Triglicéridos moderados
                's6': 0.3  # Azúcar moderado
            }
        ]

        for i, patient in enumerate(example_patients, 1):
            print(f"\n👤 Paciente ejemplo {i}:")
            predicted = predict_diabetes_progression(model, patient, X.columns.tolist())