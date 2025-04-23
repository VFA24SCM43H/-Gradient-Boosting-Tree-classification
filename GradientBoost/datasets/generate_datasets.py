import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt

def generate_simple_classification_data(n_samples=500, random_state=42):
    """Generate a simple classification dataset"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=random_state,
                              n_clusters_per_class=1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data=X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    
    return df

def generate_nonlinear_classification_data(n_samples=500, dataset_type='moons', random_state=42):
    """Generate nonlinear classification datasets"""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state)
    else:
        raise ValueError("dataset_type must be 'moons' or 'circles'")
    
    # Convert to DataFrame
    df = pd.DataFrame(data=X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    
    return df

def generate_collinear_data(n_samples=500, random_state=42):
    """Generate dataset with highly collinear features"""
    np.random.seed(random_state)
    
    # Generate base features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    
    # Generate collinear features
    X3 = 0.8 * X1 + 0.2 * np.random.randn(n_samples)  # Collinear with X1
    X4 = 0.9 * X2 + 0.1 * np.random.randn(n_samples)  # Collinear with X2
    X5 = 0.7 * X1 - 0.2 * X2 + 0.1 * np.random.randn(n_samples)  # Collinear with both
    
    # Generate target (depends on X1, X2, but not on the collinear features)
    y = (X1 + X2 > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'feature_4': X4,
        'feature_5': X5,
        'target': y
    })
    
    return df

def save_datasets(base_path='.'):
    """Generate all datasets and save them to CSV files"""
    # Simple classification dataset
    simple_df = generate_simple_classification_data()
    simple_df.to_csv(f'{base_path}/simple_classification.csv', index=False)
    print(f"Simple classification dataset saved to {base_path}/simple_classification.csv")
    
    # Nonlinear datasets
    moons_df = generate_nonlinear_classification_data(dataset_type='moons')
    moons_df.to_csv(f'{base_path}/moons_classification.csv', index=False)
    print(f"Moons dataset saved to {base_path}/moons_classification.csv")
    
    circles_df = generate_nonlinear_classification_data(dataset_type='circles')
    circles_df.to_csv(f'{base_path}/circles_classification.csv', index=False)
    print(f"Circles dataset saved to {base_path}/circles_classification.csv")
    
    # Collinear dataset
    collinear_df = generate_collinear_data()
    collinear_df.to_csv(f'{base_path}/collinear_classification.csv', index=False)
    print(f"Collinear dataset saved to {base_path}/collinear_classification.csv")

def plot_dataset(df, title="Classification Dataset", figsize=(10, 6)):
    """Plot the dataset to visualize the classification problem"""
    plt.figure(figsize=figsize)
    
    # Extract features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # If more than 2 features, use PCA to visualize the first 2 components
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        plt.title(f"{title} (PCA projection)")
    else:
        plt.title(title)
    
    # Plot the data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.8, c='red', 
                marker='o', label='Class 0', edgecolor='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.8, c='blue', 
                marker='^', label='Class 1', edgecolor='k')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def visualize_all_datasets():
    """Generate and visualize all datasets"""
    # Generate datasets
    datasets = {
        'Simple Classification': generate_simple_classification_data(),
        'Moons Classification': generate_nonlinear_classification_data(dataset_type='moons'),
        'Circles Classification': generate_nonlinear_classification_data(dataset_type='circles'),
        'Collinear Features': generate_collinear_data()
    }
    
    # Plot each dataset
    for title, df in datasets.items():
        plt_obj = plot_dataset(df, title=title)
        plt_obj.savefig(f"{title.lower().replace(' ', '_')}_visualization.png", dpi=300)
        plt_obj.close()
        print(f"Visualization saved for {title}")

if __name__ == "__main__":
    # Generate and save all datasets
    save_datasets()
    
    # Generate visualizations
    visualize_all_datasets()
    
    print("All datasets generated and saved successfully!")