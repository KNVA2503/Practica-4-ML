import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # <--- 1. IMPORTACIÓN AÑADIDA
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
        # Placeholder
        print(f"Dataset: {dataset_name.upper()} (PLACEHOLDER)")
        np.random.seed(42)
        X = np.random.rand(1000, 8)
        y = np.random.rand(1000) * 50000 + 10000
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    elif dataset_name == 'concrete_strength':
        # Placeholder
        print(f"Dataset: {dataset_name.upper()} (PLACEHOLDER)")
        np.random.seed(42)
        X = np.random.rand(500, 9)
        y = np.random.rand(500) * 80 + 10
        print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' no reconocido")
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepara los datos dividiendo y escalando.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Datos de entrenamiento: {X_train_scaled.shape[0]} muestras")
    print(f"Datos de prueba: {X_test_scaled.shape[0]} muestras")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def optimize_mlp_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Optimiza los hiperparámetros del MLPRegressor usando GridSearchCV.
    """
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }
    
    mlp = MLPRegressor(max_iter=1000, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    print("Optimizando hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    return grid_search

def train_final_model(best_params, X_train, y_train, random_state=42):
    """
    Entrena el modelo final con los mejores hiperparámetros.
    """
    final_model = MLPRegressor(**best_params, max_iter=1000, random_state=random_state)
    final_model.fit(X_train, y_train)
    
    return final_model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba y calcula métricas.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

def print_results(dataset_name, best_params, mse, r2):
    """
    Imprime un resumen claro de los resultados.
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
    
    if r2 >= 0.9: interpretation = "Excelente"
    elif r2 >= 0.8: interpretation = "Muy bueno"
    elif r2 >= 0.6: interpretation = "Bueno"
    elif r2 >= 0.4: interpretation = "Regular"
    else: interpretation = "Pobre"
    
    print(f"  • Interpretación del R²: {interpretation}")

# <--- 2. NUEVA FUNCIÓN PARA CREAR GRÁFICOS --->
def plot_results(model, y_test, y_pred, r2, dataset_name):
    """
    Genera los gráficos de resultados para un dataset.
    
    Args:
        model: Modelo MLPRegressor entrenado.
        y_test (array): Valores reales del conjunto de prueba.
        y_pred (array): Valores predichos por el modelo.
        r2 (float): Coeficiente de determinación R².
        dataset_name (str): Nombre del dataset.
    """
    # --- Gráfico 1: Predicción vs. Valores Reales ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    # Línea de predicción perfecta (y=x)
    perfect_line = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, 'r--', linewidth=2, label='Predicción Perfecta')
    plt.title(f'Predicción vs. Valores Reales - {dataset_name.upper()}\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Valores Predichos', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Gráfico 2: Curva de Pérdida (Loss Curve) ---
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, color='b', label='Pérdida de Entrenamiento')
    plt.title(f'Curva de Pérdida del Entrenamiento - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas (Epochs)', fontsize=12)
    plt.ylabel('Pérdida (Loss)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def process_dataset(dataset_name):
    """
    Procesa un dataset completo siguiendo el pipeline especificado.
    """
    try:
        print(f"\n{'#'*80}")
        print(f"PROCESANDO DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        X, y = load_dataset(dataset_name)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
        grid_search = optimize_mlp_hyperparameters(X_train_scaled, y_train)
        best_params = grid_search.best_params_
        final_model = train_final_model(best_params, X_train_scaled, y_train)
        
        # Se obtiene y_pred para usarlo en los gráficos
        mse, r2, y_pred = evaluate_model(final_model, X_test_scaled, y_test)
        
        print_results(dataset_name, best_params, mse, r2)

        # <--- 3. LLAMADA A LA NUEVA FUNCIÓN DE GRÁFICOS --->
        plot_results(final_model, y_test, y_pred, r2, dataset_name)
        
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
    Función principal que ejecuta el pipeline completo.
    """
    print("OPTIMIZACIÓN DE MLPRegressor EN MÚLTIPLES DATASETS")
    print("=" * 80)
    
    datasets = ['iris', 'wine', 'diabetes', 'california_housing', 'car_price', 'concrete_strength']
    results = []
    
    for dataset in datasets:
        result = process_dataset(dataset)
        if result:
            results.append(result)
    
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
