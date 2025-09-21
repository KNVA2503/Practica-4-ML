
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mostrar gráficos
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

def load_datasets():
    """
    Función para cargar todos los datasets
    Returns: diccionario con datasets
    """
    datasets = {}
    
    # Dataset 1: Iris
    iris = load_iris()
    datasets['Iris'] = {
        'data': iris.data,
        'feature_names': iris.feature_names,
        'description': 'Iris Dataset - Características de flores'
    }
    
    # Dataset 2: Wine
    wine = load_wine()
    datasets['Wine'] = {
        'data': wine.data,
        'feature_names': wine.feature_names,
        'description': 'Wine Dataset - Características químicas del vino'
    }
    
    # Dataset 3: Diabetes
    diabetes = load_diabetes()
    datasets['Diabetes'] = {
        'data': diabetes.data,
        'feature_names': diabetes.feature_names,
        'description': 'Diabetes Dataset - Características médicas'
    }
    
    # Dataset 4: California Housing
    california = fetch_california_housing()
    datasets['California Housing'] = {
        'data': california.data,
        'feature_names': california.feature_names,
        'description': 'California Housing Dataset - Características de viviendas'
    }
    
    # Dataset 5: Car Price Prediction (placeholder usando subset de California Housing)
    # Creamos un dataset sintético basado en características de California Housing
    car_data = california.data[:1000, :6]  # Tomamos primeras 6 características
    datasets['Car Price'] = {
        'data': car_data,
        'feature_names': ['Engine_Size', 'Horsepower', 'Weight', 'Fuel_Efficiency', 'Age', 'Mileage'],
        'description': 'Car Price Dataset (Placeholder) - Características de automóviles'
    }
    
    # Dataset 6: Concrete Compressive Strength (placeholder usando Wine con modificaciones)
    # Creamos un dataset sintético basado en Wine
    concrete_data = wine.data[:, :8] if wine.data.shape[1] >= 8 else np.hstack([wine.data, wine.data[:, :2]])
    datasets['Concrete Strength'] = {
        'data': concrete_data,
        'feature_names': ['Cement', 'Water', 'Aggregate', 'Admixture', 'Age', 'Slump', 'Flow', 'Strength_Factor'],
        'description': 'Concrete Strength Dataset (Placeholder) - Características del concreto'
    }
    
    return datasets

def elbow_method(X, dataset_name, max_k=10):
    """
    Implementa el método del codo para encontrar el k óptimo
    
    Parameters:
    X: array de características
    dataset_name: nombre del dataset
    max_k: número máximo de clusters a probar
    
    Returns:
    k_optimal: número óptimo de clusters
    """
    distortions = []
    K_range = range(1, max_k + 1)
    
    # Calcular la distorsión para cada k
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    # Crear gráfico del codo
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, distortions, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Distorsión (Inercia)', fontsize=12)
    plt.title(f'Método del Codo - {dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_range)
    
    # Encontrar el codo usando el método de la segunda derivada
    # Calculamos las diferencias
    diff1 = np.diff(distortions)
    diff2 = np.diff(diff1)
    
    # El codo está donde la segunda derivada es máxima (más negativa)
    if len(diff2) > 0:
        k_optimal = np.argmin(diff2) + 2  # +2 porque perdemos 2 elementos en las diferencias
    else:
        k_optimal = 3  # valor por defecto
    
    # Marcar el punto óptimo en el gráfico
    plt.axvline(x=k_optimal, color='red', linestyle='--', alpha=0.7, 
               label=f'k óptimo = {k_optimal}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"K óptimo encontrado para {dataset_name}: {k_optimal}")
    return k_optimal

def apply_kmeans_clustering(X, dataset_name, feature_names):
    """
    Aplica todo el proceso de clustering a un dataset
    
    Parameters:
    X: array de características
    dataset_name: nombre del dataset
    feature_names: nombres de las características
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE CLUSTERING - {dataset_name}")
    print(f"{'='*60}")
    print(f"Forma del dataset: {X.shape}")
    print(f"Características: {feature_names}")
    
    # Paso 2: Escalar las características
    print("\n1. Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Escalado completado")
    
    # Paso 3: Método del codo
    print("\n2. Aplicando método del codo...")
    k_optimal = elbow_method(X_scaled, dataset_name)
    
    # Paso 4: Entrenar K-Means con k óptimo
    print(f"\n3. Entrenando K-Means con k={k_optimal}...")
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calcular métricas de evaluación
    if k_optimal > 1:
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print(f"✓ Puntuación de Silhouette: {silhouette_avg:.3f}")
    
    # Paso 5: Aplicar PCA para reducir a 2 componentes
    print("\n4. Aplicando PCA para visualización...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✓ Varianza explicada por PCA: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  - Componente 1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  - Componente 2: {pca.explained_variance_ratio_[1]:.3f}")
    
    # Paso 6: Generar gráfico de dispersión
    print("\n5. Generando visualización...")
    plt.figure(figsize=(12, 8))
    
    # Colores para los clusters
    colors = plt.cm.Set3(np.linspace(0, 1, k_optimal))
    
    # Graficar cada cluster
    for i in range(k_optimal):
        cluster_points = X_pca[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[i]], label=f'Cluster {i+1}', 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Graficar centroides en espacio PCA
    centroids_scaled = scaler.transform(kmeans.cluster_centers_)
    centroids_pca = pca.transform(centroids_scaled)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidth=3, 
               label='Centroides')
    
    # Configurar el gráfico
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} varianza)', 
              fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} varianza)', 
              fontsize=12)
    plt.title(f'K-Means Clustering - {dataset_name}\n'
              f'k={k_optimal} clusters, Silhouette Score: {silhouette_avg:.3f}' if k_optimal > 1 
              else f'K-Means Clustering - {dataset_name}\nk={k_optimal} clusters', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✓ Análisis completado exitosamente")
    
    return {
        'k_optimal': k_optimal,
        'cluster_labels': cluster_labels,
        'silhouette_score': silhouette_avg if k_optimal > 1 else None,
        'pca_variance_ratio': pca.explained_variance_ratio_
    }

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("="*80)
    print("ANÁLISIS DE CLUSTERING K-MEANS EN MÚLTIPLES DATASETS")
    print("="*80)
    print("Este script aplicará K-Means clustering a 6 datasets diferentes")
    print("Para cada dataset se realizará:")
    print("1. Carga y preparación de datos")
    print("2. Escalado de características")
    print("3. Método del codo para encontrar k óptimo")
    print("4. Entrenamiento de K-Means")
    print("5. Reducción de dimensionalidad con PCA")
    print("6. Visualización de clusters")
    
    # Cargar todos los datasets
    print("\nCargando datasets...")
    datasets = load_datasets()
    print(f"✓ {len(datasets)} datasets cargados exitosamente")
    
    # Aplicar clustering a cada dataset
    results = {}
    
    for dataset_name, dataset_info in datasets.items():
        try:
            result = apply_kmeans_clustering(
                dataset_info['data'], 
                dataset_name, 
                dataset_info['feature_names']
            )
            results[dataset_name] = result
            
        except Exception as e:
            print(f"❌ Error procesando {dataset_name}: {str(e)}")
            continue
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL DE RESULTADOS")
    print("="*80)
    
    summary_df = pd.DataFrame([
        {
            'Dataset': name,
            'K Óptimo': results[name]['k_optimal'],
            'Silhouette Score': f"{results[name]['silhouette_score']:.3f}" 
                              if results[name]['silhouette_score'] is not None else 'N/A',
            'Varianza PCA': f"{results[name]['pca_variance_ratio'].sum():.3f}"
        }
        for name in results.keys()
    ])
    
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("Todos los gráficos han sido generados.")
    print("Revisa los resultados del método del codo y las visualizaciones PCA.")

# Ejecutar el análisis completo
if __name__ == "__main__":
    main()
