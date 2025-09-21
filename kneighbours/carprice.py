import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('default')
sns.set_palette("husl")


def create_realistic_car_dataset():
    """
    Crear dataset realista de precios de coches usando California Housing como base
    """
    print("🚗 CREANDO DATASET REALISTA DE PRECIOS DE COCHES")
    print("=" * 60)

    # Cargar California Housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target

    # Convertir a DataFrame
    df = pd.DataFrame(X, columns=california.feature_names)

    # Simular precios de coches realistas
    np.random.seed(42)

    # Precio base más influencia de cada característica
    base_price = 15000

    # Factores de influencia para cada característica
    price_factors = {
        'MedInc': 8000,  # Ingreso del dueño → mayor precio
        'HouseAge': -300,  # Edad del coche → menor precio
        'AveRooms': 2000,  # Nivel de comodidad → mayor precio
        'AveBedrms': 1500,  # Calidad interior → mayor precio
        'Population': 10,  # Popularidad → leve aumento
        'AveOccup': -500  # Uso/mantenimiento → menor precio
    }

    # Calcular precio base
    base_prices = base_price + sum(df[col] * factor for col, factor in price_factors.items())

    # Añadir variabilidad aleatoria
    noise = np.random.normal(0, 3000, len(df))
    car_prices = np.clip(base_prices + noise, 5000, 80000)

    # Añadir características categóricas realistas de coches
    brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Nissan', 'Chevrolet', 'Volkswagen', 'Hyundai']
    brand_probs = [0.15, 0.14, 0.13, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04]
    brand_probs = [p / sum(brand_probs) for p in brand_probs]  # Normalizar a sum=1

    brand_premiums = {
        'Toyota': 0, 'Honda': 500, 'Ford': 300, 'Nissan': 200, 'Chevrolet': 100,
        'Volkswagen': 1500, 'Hyundai': -200, 'BMW': 8000, 'Mercedes': 9000, 'Audi': 7500
    }

    df['brand'] = np.random.choice(brands, size=len(df), p=brand_probs)

    # Aplicar premium de marca
    brand_premium = np.array([brand_premiums[brand] for brand in df['brand']])
    car_prices += brand_premium

    # Tipo de combustible
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
    fuel_probs = [0.6, 0.2, 0.15, 0.05]
    fuel_effects = {'Petrol': 0, 'Diesel': 1000, 'Hybrid': 3000, 'Electric': 8000}
    df['fuel_type'] = np.random.choice(fuel_types, size=len(df), p=fuel_probs)
    fuel_effect = np.array([fuel_effects[ft] for ft in df['fuel_type']])
    car_prices += fuel_effect

    # Transmisión
    transmissions = ['Automatic', 'Manual', 'CVT']
    transmission_probs = [0.7, 0.2, 0.1]
    transmission_effects = {'Automatic': 1500, 'Manual': 0, 'CVT': 800}
    df['transmission'] = np.random.choice(transmissions, size=len(df), p=transmission_probs)
    transmission_effect = np.array([transmission_effects[t] for t in df['transmission']])
    car_prices += transmission_effect

    # Año de fabricación (2010-2023)
    df['year'] = np.random.randint(2010, 2024, size=len(df))
    year_effect = (df['year'] - 2010) * 800
    car_prices += year_effect

    # Kilometraje (5,000 - 150,000 km)
    df['mileage'] = np.random.randint(5000, 150000, size=len(df))
    mileage_effect = -df['mileage'] * 0.1
    car_prices += mileage_effect

    # Estado del coche (basado en AveOccup)
    condition_labels = ['Excellent', 'Good', 'Fair', 'Poor']
    df['condition'] = pd.cut(df['AveOccup'], bins=4, labels=condition_labels)
    condition_effects = {'Excellent': 2000, 'Good': 500, 'Fair': -1000, 'Poor': -3000}
    condition_effect = np.array([condition_effects[cond] for cond in df['condition']])
    car_prices += condition_effect

    # Renombrar columnas numéricas
    df = df.rename(columns={
        'MedInc': 'owner_income',
        'HouseAge': 'car_age',
        'AveRooms': 'comfort_level',
        'AveBedrms': 'interior_quality',
        'Population': 'popularity_index',
        'AveOccup': 'maintenance_score'
    })

    # Añadir el precio simulado
    df['price'] = car_prices

    # Limpiar precios extremos
    df = df[(df['price'] >= 5000) & (df['price'] <= 80000)]

    print(f"✅ Dataset realista creado: {df.shape}")
    print(f"📊 Precios: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"📈 Promedio: ${df['price'].mean():,.0f}")

    return df


def car_price_knn_realistic():
    """
    Algoritmo KNeighborsRegressor con dataset realista de coches
    """
    print("🚗 KNEIGHBORSREGRESSOR - PREDICCIÓN DE PRECIOS DE COCHES")
    print("=" * 70)

    # Crear dataset realista
    df = create_realistic_car_dataset()

    # Separar características y target
    X = df.drop('price', axis=1)
    y = df['price']

    # Identificar columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\n📋 CARACTERÍSTICAS:")
    print(f"Categóricas: {categorical_cols}")
    print(f"Numéricas: {numerical_cols}")

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 DIVISIÓN:")
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba: {X_test.shape[0]} muestras")

    # Crear pipeline simplificado
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance', p=2))
    ])

    # Entrenar modelo (sin grid search para mayor velocidad)
    print("\n🔍 ENTRENANDO MODELO...")
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    avg_price = y.mean()
    rmse_percent = (rmse / avg_price) * 100
    mae_percent = (mae / avg_price) * 100

    print("\n📈 RESULTADOS:")
    print("-" * 40)
    print(f"RMSE: ${rmse:,.0f} ({rmse_percent:.1f}% del promedio)")
    print(f"MAE: ${mae:,.0f} ({mae_percent:.1f}% del promedio)")
    print(f"R² Score: {r2:.4f}")

    # Visualizaciones simplificadas
    print("\n📊 VISUALIZANDO RESULTADOS...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Predicciones vs Reales
    axes[0].scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Precio Real ($)')
    axes[0].set_ylabel('Precio Predicho ($)')
    axes[0].set_title('Predicciones vs Valores Reales')
    axes[0].grid(True, alpha=0.3)

    # Distribución de errores
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Error de Predicción ($)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Errores')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Predicciones de ejemplo
    print("\n🎯 PREDICCIONES DE EJEMPLO:")
    print("-" * 40)

    sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_price = y_test.iloc[idx]
        predicted_price = y_pred[idx]
        error = abs(actual_price - predicted_price)
        error_percent = (error / actual_price) * 100

        print(f"\nSample {i + 1}:")
        print(f"  Real: ${actual_price:,.0f}")
        print(f"  Predicho: ${predicted_price:,.0f}")
        print(f"  Error: ${error:,.0f} ({error_percent:.1f}%)")

        # Mostrar algunas características
        sample_data = X_test.iloc[idx]
        print(f"  Características: {sample_data['brand']} {sample_data['year']}, {sample_data['mileage']}km")

    return pipeline, X, y, df


def predict_car_price_simple(model, feature_columns):
    """
    Función simple para predecir precios
    """
    print("\n🎯 PREDICCIÓN DE EJEMPLO:")
    print("=" * 40)

    # Coches de ejemplo
    example_cars = [
        {
            'brand': 'Toyota', 'fuel_type': 'Petrol', 'transmission': 'Automatic',
            'condition': 'Good', 'year': 2018, 'mileage': 45000,
            'owner_income': 4.0, 'car_age': 5, 'comfort_level': 5.0,
            'interior_quality': 1.0, 'popularity_index': 1500, 'maintenance_score': 3.0
        },
        {
            'brand': 'BMW', 'fuel_type': 'Petrol', 'transmission': 'Automatic',
            'condition': 'Excellent', 'year': 2020, 'mileage': 20000,
            'owner_income': 8.0, 'car_age': 3, 'comfort_level': 6.5,
            'interior_quality': 1.5, 'popularity_index': 2000, 'maintenance_score': 2.0
        },
        {
            'brand': 'Ford', 'fuel_type': 'Diesel', 'transmission': 'Manual',
            'condition': 'Fair', 'year': 2015, 'mileage': 80000,
            'owner_income': 3.0, 'car_age': 8, 'comfort_level': 4.0,
            'interior_quality': 0.8, 'popularity_index': 1000, 'maintenance_score': 4.0
        }
    ]

    for i, car in enumerate(example_cars, 1):
        # Crear DataFrame para predicción
        input_df = pd.DataFrame([car])[feature_columns]

        # Predecir
        predicted_price = model.predict(input_df)[0]

        print(f"\n🚗 Coche {i} ({car['brand']} {car['year']}):")
        print(f"   • Combustible: {car['fuel_type']}")
        print(f"   • Transmisión: {car['transmission']}")
        print(f"   • Kilometraje: {car['mileage']}km")
        print(f"   • Estado: {car['condition']}")
        print(f"   💰 Precio predicho: ${predicted_price:,.0f}")

    return predicted_price


# Ejecutar el algoritmo completo
if __name__ == "__main__":
    try:
        # Entrenar modelo con datos realistas
        model, X, y, df = car_price_knn_realistic()

        # Predicción de ejemplo
        if model is not None:
            predict_car_price_simple(model, X.columns.tolist())

            print("\n" + "=" * 70)
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("=" * 70)
            print("Modelo KNeighborsRegressor entrenado correctamente")
            print("para predecir precios de coches con datos simulados realistas.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Solución: Asegúrate de tener todas las librerías instaladas:")
        print("pip install scikit-learn pandas numpy matplotlib seaborn")
