import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_dataset(dataset_name):
    """
    Carga el dataset especificado y retorna X, y
    
    Args:
        dataset_name (str): Nombre del dataset a cargar
    
    Returns:
        tuple: (X, y) características y objetivo
    """
    if dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X, y = data.data, data.target
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'california_housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'car_price':
        # Placeholder para dataset Car Price
        print(f"Dataset: {dataset_name.upper()} (PLACEHOLDER)")
        print("Este es un placeholder. Reemplazar con datos reales.")
        # Generar datos sintéticos para demostración
        np.random.seed(42)
        X = np.random.rand(1000, 8)  # 8 características simuladas
        y = np.random.rand(1000) * 50000 + 10000  # Precios simulados
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'concrete_strength':
        # Placeholder para dataset Concrete Strength
        print(f"Dataset: {dataset_name.upper()} (PLACEHOLDER)")
        print("Este es un placeholder. Reemplazar con datos reales.")
        # Generar datos sintéticos para demostración
        np.random.seed(42)
        X = np.random.rand(500, 9)  # 9 características simuladas
        y = np.random.rand(500) * 80 + 10  # Resistencia simulada
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' no reconocido")
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepara los datos dividiendo en conjuntos de entrenamiento y prueba,
    y aplica escalado estándar a las características.
    
    Args:
        X (array): Características
        y (array): Variable objetivo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Aplicar escalado estándar (obligatorio)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Datos de entrenamiento: {X_train_scaled.shape[0]} muestras")
    print(f"Datos de prueba: {X_test_scaled.shape[0]} muestras")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def optimize_mlp_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Optimiza los hiperparámetros del MLPRegressor usando GridSearchCV.
    
    Args:
        X_train (array): Características de entrenamiento escaladas
        y_train (array): Variable objetivo de entrenamiento
        cv (int): Número de folds para validación cruzada
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        GridSearchCV: Objeto fitted con los mejores parámetros
    """
    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Crear el modelo base
    mlp = MLPRegressor(max_iter=1000, random_state=random_state)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    print("Optimizando hiperparámetros...")
    print(f"Combinaciones totales: {len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['solver']) * len(param_grid['alpha']) * len(param_grid['learning_rate'])}")
    
    # Entrenar el modelo
    grid_search.fit(X_train, y_train)
    
    return grid_search

def train_final_model(best_params, X_train, y_train, random_state=42):
    """
    Entrena el modelo final con los mejores hiperparámetros.
    
    Args:
        best_params (dict): Mejores hiperparámetros encontrados
        X_train (array): Características de entrenamiento escaladas
        y_train (array): Variable objetivo de entrenamiento
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        MLPRegressor: Modelo entrenado
    """
    # Crear y entrenar el modelo final
    final_model = MLPRegressor(**best_params, max_iter=1000, random_state=random_state)
    final_model.fit(X_train, y_train)
    
    return final_model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba y calcula métricas.
    
    Args:
        model: Modelo entrenado
        X_test (array): Características de prueba
        y_test (array): Variable objetivo de prueba
    
    Returns:
        tuple: (mse, r2)
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def print_results(dataset_name, best_params, mse, r2):
    """
    Imprime un resumen claro de los resultados.
    
    Args:
        dataset_name (str): Nombre del dataset
        best_params (dict): Mejores hiperparámetros
        mse (float): Error cuadrático medio
        r2 (float): Coeficiente de determinación
    """
    print(f"\n{'='*60}")
    print(f"RESULTADOS PARA {dataset_name.upper()}")
    print(f"{'='*60}")
    
    print("\nMEJORES HIPERPARÁMETROS:")
    for param, value in best_params.items():
        print(f"  • {param}: {value}")
    
    print(f"\nMÉTRICAS DE RENDIMIENTO:")
    print(f"  • Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"  • Coeficiente de Determinación (R²): {r2:.4f}")
    
    # Interpretación del R²
    if r2 >= 0.9:
        interpretation = "Excelente"
    elif r2 >= 0.8:
        interpretation = "Muy bueno"
    elif r2 >= 0.6:
        interpretation = "Bueno"
    elif r2 >= 0.4:
        interpretation = "Regular"
    else:
        interpretation = "Pobre"
    
    print(f"  • Interpretación del R²: {interpretation}")

def process_dataset(dataset_name):
    """
    Procesa un dataset completo siguiendo el pipeline especificado.
    
    Args:
        dataset_name (str): Nombre del dataset a procesar
    """
    try:
        print(f"\n{'#'*80}")
        print(f"PROCESANDO DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        # 1. Carga y Preparación
        X, y = load_dataset(dataset_name)
        
        # 2. Escalado de Datos y división
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
        
        # 3. Optimización de Hiperparámetros
        grid_search = optimize_mlp_hyperparameters(X_train_scaled, y_train)
        
        # 4. Entrenamiento del modelo final
        best_params = grid_search.best_params_
        final_model = train_final_model(best_params, X_train_scaled, y_train)
        
        # 5. Evaluación
        mse, r2 = evaluate_model(final_model, X_test_scaled, y_test)
        
        # 6. Reporte
        print_results(dataset_name, best_params, mse, r2)
        
        return {
            'dataset': dataset_name,
            'best_params': best_params,
            'mse': mse,
            'r2': r2,
            'model': final_model,
            'scaler': scaler
        }
        
    except Exception as e:
        print(f"Error procesando {dataset_name}: {str(e)}")
        return None

def main():
    """
    Función principal que ejecuta el pipeline completo para todos los datasets.
    """
    print("OPTIMIZACIÓN DE MLPRegressor EN MÚLTIPLES DATASETS")
    print("=" * 80)
    print("Implementación de pipeline completo con GridSearchCV")
    
    # Lista de datasets a procesar
    datasets = [
        'iris',
        'wine', 
        'diabetes',
        'california_housing',
        'car_price',
        'concrete_strength'
    ]
    
    # Almacenar resultados
    results = []
    
    # Procesar cada dataset
    for dataset in datasets:
        result = process_dataset(dataset)
        if result:
            results.append(result)
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN FINAL DE TODOS LOS DATASETS")
    print(f"{'='*80}")
    
    if results:
        print(f"{'Dataset':<20} {'R²':<10} {'MSE':<15} {'Mejor Activación':<15}")
        print("-" * 65)
        
        for result in results:
            dataset_name = result['dataset']
            r2 = result['r2']
            mse = result['mse']
            activation = result['best_params']['activation']
            
            print(f"{dataset_name.upper():<20} {r2:<10.4f} {mse:<15.4f} {activation:<15}")
    
    print(f"\n{'='*80}")
    print("PROCESO COMPLETADO")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
