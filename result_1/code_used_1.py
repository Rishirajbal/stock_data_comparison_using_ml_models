import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import time
import os
import warnings
import psutil
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

# Set matplotlib style safely
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')  # Fallback style
sns.set_palette("husl")

# Memory optimization function
def reduce_mem_usage(df):
    """Iterate through columns and reduce memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df

def load_data(limit_rows=1000):
    """Load datasets with limited rows/columns"""
    try:
        print("\nLoading data...")
        start_time = time.time()
        
        train_path = r'C:\Users\KIIT\OneDrive\Desktop\mini_project\datasets\df_train.csv'
        test_path = r'C:\Users\KIIT\OneDrive\Desktop\mini_project\datasets\df_test.csv'
        validate_path = r'C:\Users\KIIT\OneDrive\Desktop\mini_project\datasets\df_validate.csv'
        
        # Read data with progress bar
        train = pd.read_csv(train_path, nrows=limit_rows)
        test = pd.read_csv(test_path, nrows=limit_rows)
        validate = pd.read_csv(validate_path, nrows=limit_rows)
        
        # Memory optimization
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)
        validate = reduce_mem_usage(validate)
        
        # Select OHLCV columns (case insensitive)
        cols = []
        possible_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'open', 'high', 'low', 'close', 'volume']
        
        for col in possible_cols:
            if col in train.columns:
                cols.append(col)
        
        if not cols:
            raise ValueError("No OHLCV columns found in the dataset")
        
        target_col = 'Close' if 'Close' in train.columns else 'close'
        
        X_train = train[cols].drop(target_col, axis=1)
        y_train = train[target_col]
        X_test = test[cols].drop(target_col, axis=1)
        y_test = test[target_col]
        X_val = validate[cols].drop(target_col, axis=1)
        y_val = validate[target_col]
        
        print(f"Data loaded in {time.time()-start_time:.2f} seconds")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Validation shape: {X_val.shape}")
        
        return X_train, y_train, X_test, y_test, X_val, y_val, target_col
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(X_train, X_test, X_val):
    """Scale data and ensure CPU execution"""
    print("\nPreprocessing data...")
    start_time = time.time()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Data preprocessed in {time.time()-start_time:.2f} seconds")
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler

def create_plots(model_name, y_true, y_pred, dataset_name, target_col, features=None, model=None):
    """Generate visualization plots for model evaluation"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'{model_name} - {dataset_name}\nActual vs Predicted')
    
    # Plot 2: Residuals
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution')
    
    # Plot 3: Time Series (first 100 samples)
    plt.subplot(2, 2, 3)
    plt.plot(y_true.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted (First 100 Samples)')
    plt.legend()
    
    # Plot 4: Feature Importance (if available)
    if hasattr(model, 'feature_importances_') and features is not None:
        plt.subplot(2, 2, 4)
        feat_importances = pd.Series(model.feature_importances_, index=features)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title('Feature Importance')
    
    plt.tight_layout()
    plot_filename = f"{model_name.replace(' ', '_')}_{dataset_name}_performance.png"
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_filename

def print_system_stats():
    """Print current system resource usage"""
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    print(f"\nSystem Stats - CPU: {cpu_percent}% | RAM: {mem.percent}% (Used: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")

def train_model_with_progress(model, model_name, X_train, y_train):
    """Train model with progress tracking"""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    # Special handling for models that support verbose output
    if hasattr(model, 'set_params'):
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            model.set_params(verbose=1)
        elif isinstance(model, MLPRegressor):
            model.set_params(verbose=True)
    
    # Add progress bar for models without built-in verbose
    with tqdm(total=100, desc=f"Training {model_name}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        if isinstance(model, (LinearRegression, SVR)):
            model.fit(X_train, y_train)
            pbar.update(100)
        else:
            model.fit(X_train, y_train)
            pbar.update(100)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print_system_stats()
    
    return model, training_time

def generate_html_report(models_results, plots_data, target_col):
    """Generate comprehensive HTML report"""
    print("\nGenerating HTML report...")
    start_time = time.time()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction Report - {target_col}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2e6c80; }}
            .model-section {{ margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .plot {{ width: 45%; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .progress-container {{ width: 100%; background-color: #f1f1f1; }}
            .progress-bar {{ width: 100%; height: 30px; background-color: #4CAF50; text-align: center; line-height: 30px; color: white; }}
        </style>
    </head>
    <body>
        <h1>Stock Price Prediction Report</h1>
        <h2>Target Variable: {target_col}</h2>
        <div class="progress-container">
            <div class="progress-bar">Report Generation Progress: 25%</div>
        </div>
    """
    
    # Sort models by test R2 score
    sorted_models = sorted(models_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_models, 1):
        progress = 25 + int(i/len(sorted_models)*50)
        html_content += f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <div class="progress-container">
                <div class="progress-bar" style="width:{progress}%">Processing: {model_name} ({i}/{len(sorted_models)})</div>
            </div>
            
            <h3>Performance Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Training</th>
                    <th>Test</th>
                    <th>Validation</th>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td>{results['train_mse']:.4f}</td>
                    <td>{results['test_mse']:.4f}</td>
                    <td>{results['val_mse']:.4f}</td>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td>{results['train_mae']:.4f}</td>
                    <td>{results['test_mae']:.4f}</td>
                    <td>{results['val_mae']:.4f}</td>
                </tr>
                <tr>
                    <td>R²</td>
                    <td>{results['train_r2']:.4f}</td>
                    <td>{results['test_r2']:.4f}</td>
                    <td>{results['val_r2']:.4f}</td>
                </tr>
                <tr>
                    <td>Training Time (s)</td>
                    <td colspan="3">{results['training_time']:.2f}</td>
                </tr>
            </table>
            
            <h3>Visualizations</h3>
            <div class="plot-container">
        """
        
        for dataset in ['train', 'test', 'val']:
            if f"{model_name}_{dataset}" in plots_data:
                html_content += f"""
                <div class="plot">
                    <h4>{dataset.capitalize()} Set</h4>
                    <img src="{plots_data[f"{model_name}_{dataset}"]}" alt="{model_name} {dataset} performance">
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
        <div class="progress-container">
            <div class="progress-bar" style="width:100%">Report Generation Complete!</div>
        </div>
    </body>
    </html>
    """
    
    report_filename = f"stock_prediction_report_{target_col}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated in {time.time()-start_time:.2f} seconds")
    return report_filename

def main(limit_rows=1000):
    """Main execution function"""
    try:
        print("Starting stock prediction pipeline...")
        print_system_stats()
        
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_val, y_val, target_col = load_data(limit_rows)
        X_train_scaled, X_test_scaled, X_val_scaled, scaler = preprocess_data(X_train, X_test, X_val)
        
        # Initialize models with optimized parameters
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=10,
                n_jobs=1,  # Single core to avoid memory issues
                random_state=42,
                verbose=1
            ),
            'Support Vector Regressor': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                max_iter=1000  # Limit iterations
            ),
            'XGBoost': XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,  # Single core
                verbosity=1
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(50, 25),  # Reduced network size
                max_iter=200,  # Reduced iterations
                random_state=42,
                verbose=True
            )
        }
        
        # Train and evaluate models
        models_results = {}
        plots_data = {}
        
        for name, model in models.items():
            # Train model with progress tracking
            trained_model, training_time = train_model_with_progress(
                model, name, 
                X_train_scaled, y_train
            )
            
            # Predictions
            print(f"\nMaking predictions with {name}...")
            y_train_pred = trained_model.predict(X_train_scaled)
            y_test_pred = trained_model.predict(X_test_scaled)
            y_val_pred = trained_model.predict(X_val_scaled)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_mse': mean_squared_error(y_test, y_test_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'val_mse': mean_squared_error(y_val, y_val_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'training_time': training_time
            }
            models_results[name] = metrics
            
            # Create plots
            print(f"Generating plots for {name}...")
            plot_files = {}
            for dataset, y_true, y_pred in zip(
                ['train', 'test', 'val'],
                [y_train, y_test, y_val],
                [y_train_pred, y_test_pred, y_val_pred]
            ):
                plot_file = create_plots(
                    name, y_true, y_pred, dataset, target_col,
                    features=X_train.columns, model=trained_model
                )
                plot_files[f"{name}_{dataset}"] = plot_file
                print(f"  - {dataset} set plot saved: {plot_file}")
            
            plots_data.update(plot_files)
            print(f"Completed {name}\n{'='*50}")
        
        # Generate HTML report
        report_file = generate_html_report(models_results, plots_data, target_col)
        print(f"\nFinal Report: {report_file}")
        print_system_stats()
        
        return models_results, plots_data, report_file
    
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print_system_stats()
        raise

if __name__ == "__main__":
    # Run with limited rows and show full progress
    results, plots, report = main(limit_rows=1000)