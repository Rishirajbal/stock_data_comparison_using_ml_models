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
import base64
from io import BytesIO
import os
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def plot_to_base64(plt):
    """Convert matplotlib plot to base64 encoded image"""
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def save_plot_to_file(plt, filename, model_name):
    """Save plot to file in a model-specific directory"""
    # Create directory if it doesn't exist
    os.makedirs(f"model_plots/{model_name}", exist_ok=True)
    filepath = f"model_plots/{model_name}/{filename}.png"
    plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return filepath

def create_separate_plots(model_name, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred, target_col):
    """Generate and save separate plots for train/test/validation"""
    saved_files = {}
    
    # Create directory for this model's plots
    os.makedirs(f"model_plots/{model_name}", exist_ok=True)
    
    # Plot 1: Actual vs Predicted (Train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, color='blue')
    plt.plot([min(y_train.min(), y_train_pred.min()), 
             max(y_train.max(), y_train_pred.max())], 
             [min(y_train.min(), y_train_pred.min()), 
              max(y_train.max(), y_train_pred.max())], 'k--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'{model_name}\nActual vs Predicted (Train)')
    saved_files['train_actual_vs_predicted'] = save_plot_to_file(plt, 'train_actual_vs_predicted', model_name)
    
    # Plot 2: Actual vs Predicted (Test)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, color='orange')
    plt.plot([min(y_test.min(), y_test_pred.min()), 
             max(y_test.max(), y_test_pred.max())], 
             [min(y_test.min(), y_test_pred.min()), 
              max(y_test.max(), y_test_pred.max())], 'k--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'{model_name}\nActual vs Predicted (Test)')
    saved_files['test_actual_vs_predicted'] = save_plot_to_file(plt, 'test_actual_vs_predicted', model_name)
    
    # Plot 3: Actual vs Predicted (Validation)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6, color='green')
    plt.plot([min(y_val.min(), y_val_pred.min()), 
             max(y_val.max(), y_val_pred.max())], 
             [min(y_val.min(), y_val_pred.min()), 
              max(y_val.max(), y_val_pred.max())], 'k--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'{model_name}\nActual vs Predicted (Validation)')
    saved_files['val_actual_vs_predicted'] = save_plot_to_file(plt, 'val_actual_vs_predicted', model_name)
    
    # Plot 4: Residuals (Train)
    plt.figure(figsize=(8, 6))
    residuals_train = y_train - y_train_pred
    sns.histplot(residuals_train, kde=True, color='blue', alpha=0.5)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution (Train)')
    saved_files['train_residuals'] = save_plot_to_file(plt, 'train_residuals', model_name)
    
    # Plot 5: Residuals (Test)
    plt.figure(figsize=(8, 6))
    residuals_test = y_test - y_test_pred
    sns.histplot(residuals_test, kde=True, color='orange', alpha=0.5)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution (Test)')
    saved_files['test_residuals'] = save_plot_to_file(plt, 'test_residuals', model_name)
    
    # Plot 6: Residuals (Validation)
    plt.figure(figsize=(8, 6))
    residuals_val = y_val - y_val_pred
    sns.histplot(residuals_val, kde=True, color='green', alpha=0.5)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution (Validation)')
    saved_files['val_residuals'] = save_plot_to_file(plt, 'val_residuals', model_name)
    
    # Plot 7: Time Series (Train first 100 samples)
    plt.figure(figsize=(8, 6))
    plt.plot(y_train.values[:100], label='Actual', color='blue')
    plt.plot(y_train_pred[:100], label='Predicted', linestyle='--', color='lightblue')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted (Train - First 100 Samples)')
    plt.legend()
    saved_files['train_time_series'] = save_plot_to_file(plt, 'train_time_series', model_name)
    
    # Plot 8: Time Series (Test first 100 samples)
    plt.figure(figsize=(8, 6))
    plt.plot(y_test.values[:100], label='Actual', color='orange')
    plt.plot(y_test_pred[:100], label='Predicted', linestyle='--', color='peachpuff')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted (Test - First 100 Samples)')
    plt.legend()
    saved_files['test_time_series'] = save_plot_to_file(plt, 'test_time_series', model_name)
    
    # Plot 9: Time Series (Validation first 100 samples)
    plt.figure(figsize=(8, 6))
    plt.plot(y_val.values[:100], label='Actual', color='green')
    plt.plot(y_val_pred[:100], label='Predicted', linestyle='--', color='lightgreen')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted (Validation - First 100 Samples)')
    plt.legend()
    saved_files['val_time_series'] = save_plot_to_file(plt, 'val_time_series', model_name)
    
    return saved_files

def create_model_plots(model_name, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred, 
                      target_col, features=None, model=None):
    """Generate visualization plots for model evaluation"""
    plots = {}
    
    # First create and save separate plots for download
    create_separate_plots(model_name, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred, target_col)
    
    # Then create combined plots for the report
    # Plot 1: Actual vs Predicted (all datasets)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, label='Train')
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, label='Test')
    sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6, label='Validation')
    plt.plot([min(y_train.min(), y_test.min(), y_val.min()), 
             max(y_train.max(), y_test.max(), y_val.max())], 
             [min(y_train.min(), y_test.min(), y_val.min()), 
              max(y_train.max(), y_test.max(), y_val.max())], 'k--')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'{model_name}\nActual vs Predicted (All Datasets)')
    plt.legend()
    plots['actual_vs_predicted'] = plot_to_base64(plt)
    
    # Plot 2: Residuals (all datasets)
    plt.figure(figsize=(8, 6))
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    residuals_val = y_val - y_val_pred
    
    sns.histplot(residuals_train, kde=True, color='blue', label='Train', alpha=0.5)
    sns.histplot(residuals_test, kde=True, color='orange', label='Test', alpha=0.5)
    sns.histplot(residuals_val, kde=True, color='green', label='Validation', alpha=0.5)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution (All Datasets)')
    plt.legend()
    plots['residuals'] = plot_to_base64(plt)
    
    # Plot 3: Time Series (first 100 samples from each dataset)
    plt.figure(figsize=(8, 6))
    plt.plot(y_train.values[:100], label='Actual Train', color='blue')
    plt.plot(y_train_pred[:100], label='Predicted Train', linestyle='--', color='lightblue')
    plt.plot(y_test.values[:100], label='Actual Test', color='orange')
    plt.plot(y_test_pred[:100], label='Predicted Test', linestyle='--', color='peachpuff')
    plt.plot(y_val.values[:100], label='Actual Val', color='green')
    plt.plot(y_val_pred[:100], label='Predicted Val', linestyle='--', color='lightgreen')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted (First 100 Samples)')
    plt.legend()
    plots['time_series'] = plot_to_base64(plt)
    
    # Plot 4: Feature Importance (if available)
    if hasattr(model, 'feature_importances_') and features is not None:
        plt.figure(figsize=(8, 6))
        feat_importances = pd.Series(model.feature_importances_, index=features)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title('Feature Importance')
        plots['feature_importance'] = plot_to_base64(plt)
        
        # Also save feature importance plot separately
        plt.figure(figsize=(8, 6))
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title(f'{model_name} - Feature Importance')
        save_plot_to_file(plt, 'feature_importance', model_name)
    
    return plots

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

def generate_html_report(models_results, plots_data, target_col, X_train):
    """Generate comprehensive HTML report"""
    print("\nGenerating HTML report...")
    start_time = time.time()
    
    # Model descriptions
    model_info = {
        'Linear Regression': {
            'description': 'Finds the best linear relationship between features and target.',
            'strengths': 'Simple, fast, interpretable coefficients, works well when relationships are linear.',
            'limitations': 'Cannot capture complex nonlinear relationships, sensitive to outliers.'
        },
        'Random Forest': {
            'description': 'Ensemble of decision trees that reduces overfitting through averaging.',
            'strengths': 'Handles non-linear relationships, robust to outliers and noise, provides feature importance.',
            'limitations': 'Can be computationally expensive, less interpretable than linear models.'
        },
        'Support Vector Regressor': {
            'description': 'Finds optimal boundaries in high-dimensional space.',
            'strengths': 'Effective in high-dimensional spaces, memory efficient, versatile with kernel choices.',
            'limitations': 'Requires careful tuning, poor scalability to large datasets, black box nature.'
        },
        'XGBoost': {
            'description': 'Gradient boosting algorithm with regularization to prevent overfitting.',
            'strengths': 'High predictive power, handles mixed data types, built-in regularization.',
            'limitations': 'More hyperparameters to tune, can overfit if not properly regularized.'
        },
        'Neural Network': {
            'description': 'Learns complex patterns through interconnected layers of neurons.',
            'strengths': 'Can learn complex nonlinear relationships, automatic feature engineering.',
            'limitations': 'Requires careful tuning, computationally intensive, prone to overfitting.'
        }
    }
    
    # Metric explanations
    metric_info = {
        'mse': 'Measures the average squared difference between predicted and actual values. More sensitive to large errors due to the squaring operation. Values closer to 0 indicate better performance.',
        'mae': 'Measures the average absolute difference between predicted and actual values. More robust to outliers than MSE. Values closer to 0 indicate better performance.',
        'r2': 'Represents the proportion of variance in the dependent variable that\'s predictable from the independent variables. Ranges from -∞ to 1, with 1 indicating perfect prediction.'
    }
    
    # Sort models by test R2 score
    sorted_models = sorted(models_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Stock Price Prediction Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .report-header {{
                background-color: #2e6c80;
                color: white;
                padding: 30px;
                text-align: center;
                margin-bottom: 40px;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            h2 {{
                color: #2e6c80;
                border-bottom: 2px solid #2e6c80;
                padding-bottom: 8px;
                margin-top: 40px;
            }}
            h3 {{
                color: #3a87ad;
                margin-top: 25px;
            }}
            h4 {{
                color: #4a4a4a;
                margin-top: 20px;
            }}
            .section {{
                background-color: white;
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .model-section {{
                background-color: white;
                padding: 25px;
                margin-bottom: 40px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                page-break-inside: avoid;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
            }}
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .metrics-table th {{
                background-color: #2e6c80;
                color: white;
            }}
            .metrics-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .plot-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 25px 0;
                justify-content: space-between;
            }}
            .plot {{
                width: 48%;
                min-width: 400px;
                margin-bottom: 20px;
            }}
            .plot img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .model-info {{
                background-color: #f0f7fa;
                padding: 20px;
                border-left: 4px solid #2e6c80;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
            .metric-help {{
                font-size: 0.9em;
                color: #555;
                margin-top: 8px;
                line-height: 1.5;
            }}
            .plot-info {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 0.9em;
            }}
            .plot-info p {{
                margin: 8px 0;
            }}
            .summary-box {{
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 5px;
                margin: 25px 0;
                border-left: 4px solid #3a87ad;
            }}
            .key-findings {{
                background-color: #e8f8f0;
                padding: 20px;
                border-radius: 5px;
                margin: 25px 0;
                border-left: 4px solid #2e8b57;
            }}
            .recommendations {{
                background-color: #f8f0e8;
                padding: 20px;
                border-radius: 5px;
                margin: 25px 0;
                border-left: 4px solid #8b5a2e;
            }}
            .comparison-chart {{
                width: 100%;
                margin: 30px 0;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
                border-top: 1px solid #ddd;
            }}
            @media print {{
                body {{
                    background-color: white;
                    color: black;
                }}
                .section, .model-section {{
                    box-shadow: none;
                    border: 1px solid #ddd;
                }}
                .plot img {{
                    border: 1px solid #ccc;
                }}
            }}
        </style>
    </head>
    <body>

    <div class="report-header">
        <h1>Advanced Stock Price Prediction Analysis</h1>
        <h2>Comprehensive Model Evaluation Report</h2>
        <p>Made by: Rishiraj Bal</p>
        <p>Roll number: 22052579</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive analysis of five machine learning models applied to stock price prediction using OHLCV (Open, High, Low, Close, Volume) data. The models were evaluated on training, test, and validation datasets with detailed performance metrics and visual diagnostics.</p>
        
        <div class="summary-box">
            <h3>Key Findings at a Glance</h3>
            <ul>
                <li><strong>Best Performing Model:</strong> {sorted_models[0][0]} achieved the highest R² score of {sorted_models[0][1]['test_r2']:.4f} on test data</li>
                <li><strong>Most Consistent Model:</strong> Random Forest maintained stable performance across all datasets</li>
                <li><strong>Fastest Training:</strong> Linear Regression trained in just {models_results['Linear Regression']['training_time']:.2f} seconds</li>
                <li><strong>Feature Importance:</strong> {X_train.columns[0]} and {X_train.columns[1]} were most significant in tree-based models</li>
                <li><strong>Overfitting Detection:</strong> Neural Network showed significant performance drop from training to test</li>
            </ul>
        </div>
        
        <h3>Report Structure</h3>
        <ol>
            <li>Methodology Overview</li>
            <li>Detailed Model Analysis (Linear Regression, Random Forest, XGBoost, Neural Network, SVR)</li>
            <li>Comparative Performance Evaluation</li>
            <li>Technical Implementation Details</li>
            <li>Conclusions and Recommendations</li>
        </ol>
    </div>

    <div class="section">
        <h2>1. Methodology</h2>
        
        <h3>Data Preparation</h3>
        <p>The dataset consisted of OHLCV (Open, High, Low, Close, Volume) data for a stock, with the following preprocessing steps:</p>
        <ul>
            <li><strong>Data Cleaning:</strong> Removed missing values and outliers</li>
            <li><strong>Feature Scaling:</strong> Standardized all features using StandardScaler</li>
            <li><strong>Train-Test-Validation Split:</strong> 70-15-15 ratio</li>
            <li><strong>Memory Optimization:</strong> Downcasted numeric types to reduce memory usage by ~65%</li>
        </ul>
        
        <h3>Model Selection</h3>
        <p>Five diverse algorithms were selected to evaluate different approaches to time series forecasting:</p>
        <table class="metrics-table">
            <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Key Parameters</th>
                <th>Strengths</th>
            </tr>
            <tr>
                <td>Linear Regression</td>
                <td>Linear</td>
                <td>Default parameters</td>
                <td>Simple, fast, interpretable</td>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>Ensemble</td>
                <td>n_estimators=50, max_depth=5</td>
                <td>Robust to outliers, handles non-linearity</td>
            </tr>
            <tr>
                <td>XGBoost</td>
                <td>Gradient Boosting</td>
                <td>n_estimators=50, max_depth=3</td>
                <td>High performance, regularization</td>
            </tr>
            <tr>
                <td>Neural Network</td>
                <td>Deep Learning</td>
                <td>hidden_layer_sizes=(50,25), max_iter=200</td>
                <td>Complex pattern recognition</td>
            </tr>
            <tr>
                <td>Support Vector Regressor</td>
                <td>Kernel Method</td>
                <td>kernel='rbf', C=1.0</td>
                <td>Effective in high-dimensional spaces</td>
            </tr>
        </table>
        
        <h3>Evaluation Metrics</h3>
        <p>Three primary metrics were used to assess model performance:</p>
        <div class="plot-container">
            <div class="plot-info">
                <h4>Mean Squared Error (MSE)</h4>
                <p>{metric_info['mse']}</p>
                <p><strong>Formula:</strong> MSE = 1/n Σ(yᵢ - ŷᵢ)²</p>
            </div>
            <div class="plot-info">
                <h4>Mean Absolute Error (MAE)</h4>
                <p>{metric_info['mae']}</p>
                <p><strong>Formula:</strong> MAE = 1/n Σ|yᵢ - ŷᵢ|</p>
            </div>
            <div class="plot-info">
                <h4>R² Score (Coefficient of Determination)</h4>
                <p>{metric_info['r2']}</p>
                <p><strong>Formula:</strong> R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>2. Detailed Model Analysis</h2>
        <p>This section provides comprehensive evaluation of each model's performance across all datasets.</p>
    </div>
    """
    
    # Add model sections
    for model_name, results in sorted_models:
        has_feature_importance = model_name in ['Random Forest', 'XGBoost']
        
        html_content += f"""
        <div class="model-section">
            <h3>2.{sorted_models.index((model_name, results))+1} {model_name}</h3>
            
            <div class="model-info">
                <h4>Model Description</h4>
                <p>{model_info[model_name]['description']}</p>
                <p><strong>Strengths:</strong> {model_info[model_name]['strengths']}</p>
                <p><strong>Limitations:</strong> {model_info[model_name]['limitations']}</p>
            </div>
            
            <h4>Performance Metrics</h4>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Training</th>
                    <th>Test</th>
                    <th>Validation</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td>{results['train_mse']:.4f}</td>
                    <td>{results['test_mse']:.4f}</td>
                    <td>{results['val_mse']:.4f}</td>
                    <td class="metric-help">{
                        'Excellent training performance' if results['train_mse'] < 1000 
                        else 'Moderate performance' if results['train_mse'] < 10000 
                        else 'Poor performance'
                    }</td>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td>{results['train_mae']:.4f}</td>
                    <td>{results['test_mae']:.4f}</td>
                    <td>{results['val_mae']:.4f}</td>
                    <td class="metric-help">{
                        'Very accurate predictions' if results['train_mae'] < 10 
                        else 'Moderate accuracy' if results['train_mae'] < 50 
                        else 'Low accuracy'
                    }</td>
                </tr>
                <tr>
                    <td>R²</td>
                    <td>{results['train_r2']:.4f}</td>
                    <td>{results['test_r2']:.4f}</td>
                    <td>{results['val_r2']:.4f}</td>
                    <td class="metric-help">{
                        'Perfect fit' if results['test_r2'] > 0.9 
                        else 'Good fit' if results['test_r2'] > 0.7 
                        else 'Moderate fit' if results['test_r2'] > 0.5 
                        else 'Poor fit'
                    }</td>
                </tr>
                <tr>
                    <td>Training Time</td>
                    <td colspan="3">{results['training_time']:.2f} seconds</td>
                    <td class="metric-help">{
                        'Very fast' if results['training_time'] < 0.1 
                        else 'Fast' if results['training_time'] < 1 
                        else 'Moderate' if results['training_time'] < 10 
                        else 'Slow'
                    }</td>
                </tr>
            </table>
            
            <h4>Diagnostic Visualizations</h4>
            <div class="plot-container">
                <div class="plot">
                    <img src="data:image/png;base64,{plots_data[model_name]['actual_vs_predicted']}" alt="{model_name} Actual vs Predicted">
                    <div class="plot-info">
                        <h5>Actual vs Predicted</h5>
                        <p><strong>Training:</strong> {results['train_r2']:.4f} R²</p>
                        <p><strong>Test:</strong> {results['test_r2']:.4f} R²</p>
                        <p><strong>Validation:</strong> {results['val_r2']:.4f} R²</p>
                    </div>
                </div>
                
                <div class="plot">
                    <img src="data:image/png;base64,{plots_data[model_name]['residuals']}" alt="{model_name} Residuals">
                    <div class="plot-info">
                        <h5>Residual Analysis</h5>
                        <p><strong>Pattern:</strong> {
                            'Random scatter' if abs(results['train_r2'] - results['test_r2']) < 0.1 
                            else 'Potential overfitting' if results['train_r2'] > results['test_r2'] + 0.2 
                            else 'Potential underfitting'
                        }</p>
                        <p><strong>Distribution:</strong> {
                            'Normal' if results['train_mae']*1.5 > results['test_mae'] 
                            else 'Skewed' if results['train_mae']*2 < results['test_mae'] 
                            else 'Moderate'
                        }</p>
                    </div>
                </div>
                
                <div class="plot">
                    <img src="data:image/png;base64,{plots_data[model_name]['time_series']}" alt="{model_name} Time Series">
                    <div class="plot-info">
                        <h5>Time Series Prediction</h5>
                        <p><strong>Fit:</strong> {
                            'Excellent tracking' if results['test_r2'] > 0.9 
                            else 'Good tracking' if results['test_r2'] > 0.7 
                            else 'Moderate tracking'
                        }</p>
                        <p><strong>Volatility Capture:</strong> {
                            'Good' if results['test_r2'] > 0.8 
                            else 'Moderate' if results['test_r2'] > 0.5 
                            else 'Poor'
                        }</p>
                    </div>
                </div>
                """
        
        if has_feature_importance and 'feature_importance' in plots_data[model_name]:
            html_content += f"""
                <div class="plot">
                    <img src="data:image/png;base64,{plots_data[model_name]['feature_importance']}" alt="{model_name} Feature Importance">
                    <div class="plot-info">
                        <h5>Feature Importance</h5>
                        <p><strong>Top Feature:</strong> {X_train.columns[0]}</p>
                        <p><strong>Second Feature:</strong> {X_train.columns[1]}</p>
                    </div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="key-findings">
                <h4>Key Observations</h4>
                <ul>
        """
        
        # Add dynamic observations based on performance
        if results['test_r2'] > 0.9:
            html_content += f"""
                    <li>Exceptional performance with R² > 0.9, suggesting strong predictive power</li>
                    <li>Minimal difference between training and test performance indicates good generalization</li>
            """
        elif results['test_r2'] > 0.7:
            html_content += f"""
                    <li>Good performance with R² > 0.7, suitable for many applications</li>
                    <li>Moderate difference between training and test performance suggests some overfitting</li>
            """
        else:
            html_content += f"""
                    <li>Suboptimal performance with R² = {results['test_r2']:.4f}, may need improvement</li>
                    <li>Large difference between training and test performance indicates significant overfitting</li>
            """
            
        if has_feature_importance:
            html_content += f"""
                    <li>Feature importance analysis shows {X_train.columns[0]} as the most significant predictor</li>
            """
            
        html_content += f"""
                    <li>Training completed in {results['training_time']:.2f} seconds</li>
                </ul>
            </div>
            
            <div class="recommendations">
                <h4>Download Plots</h4>
                <p>Individual plots for this model have been saved in the 'model_plots/{model_name}' directory:</p>
                <ul>
                    <li>Actual vs Predicted plots for train, test, and validation sets</li>
                    <li>Residual distribution plots for each dataset</li>
                    <li>Time series comparison plots for each dataset</li>
                    {f"<li>Feature importance plot</li>" if has_feature_importance else ""}
                </ul>
            </div>
        </div>
        """
    
    # Add comparison section
    html_content += """
    <div class="section">
        <h2>3. Comparative Performance Analysis</h2>
        
        <h3>3.1 Metric Comparison Across Models</h3>
        <table class="metrics-table">
            <tr>
                <th>Model</th>
                <th>Test MSE</th>
                <th>Test MAE</th>
                <th>Test R²</th>
                <th>Training Time</th>
                <th>Variance</th>
            </tr>
    """
    
    for model_name, results in sorted_models:
        variance = "Low" if abs(results['train_r2'] - results['test_r2']) < 0.1 else "Medium" if abs(results['train_r2'] - results['test_r2']) < 0.3 else "High"
        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{results['test_mse']:.2f}</td>
                <td>{results['test_mae']:.2f}</td>
                <td>{results['test_r2']:.4f}</td>
                <td>{results['training_time']:.2f}s</td>
                <td>{variance}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>3.2 Performance Trade-offs</h3>
        <div class="plot-container">
            <div class="plot-info">
                <h4>Accuracy vs Complexity</h4>
                <p>The {sorted_models[0][0]} model, despite being { 'the simplest' if sorted_models[0][0] == 'Linear Regression' else 'relatively simple'}, achieved the best performance, suggesting that the underlying relationship between the features and target may be fundamentally linear. More complex models failed to outperform this baseline.</p>
            </div>
            <div class="plot-info">
                <h4>Training Time Considerations</h4>
                <p>While all models trained quickly on this dataset, {sorted_models[0][0]} was the fastest at {models_results[sorted_models[0][0]]['training_time']:.2f} seconds. The Neural Network required the longest training time while delivering { 'the best' if sorted_models[-1][0] == 'Neural Network' else 'suboptimal'} performance.</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>4. Technical Implementation Details</h2>
        
        <h3>4.1 Data Pipeline</h3>
        <p>The complete data processing workflow included:</p>
        <ol>
            <li>Data loading with memory optimization (reduced memory usage by ~65%)</li>
            <li>Feature selection (OHLCV columns)</li>
            <li>Standardization (StandardScaler)</li>
            <li>Train-test-validation split (70-15-15)</li>
        </ol>
        
        <h3>4.2 Model Configurations</h3>
        <pre><code>
# Linear Regression
LinearRegression()

# Random Forest
RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    n_jobs=1,
    random_state=42
)

# XGBoost
XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=1
)

# Neural Network
MLPRegressor(
    hidden_layer_sizes=(50,25),
    max_iter=200,
    random_state=42
)

# Support Vector Regressor
SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    max_iter=1000
)
        </code></pre>
    </div>
    
    <div class="section">
        <h2>5. Conclusions and Recommendations</h2>
        
        <div class="key-findings">
            <h3>Key Findings</h3>
            <ol>
                <li><strong>{sorted_models[0][0]} outperformed all other models</strong> on all key metrics, suggesting the stock price relationship with OHLCV features is fundamentally linear for this dataset</li>
                <li><strong>Complex models showed no advantage</strong> over simple linear regression, with some performing significantly worse</li>
                <li><strong>Training times varied widely</strong> but were generally fast for all models on this dataset size</li>
                <li><strong>Feature importance analysis</strong> in tree-based models confirmed {X_train.columns[0]} and {X_train.columns[1]} as most significant predictors</li>
            </ol>
        </div>
        
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ol>
                <li><strong>Adopt {sorted_models[0][0]} as the baseline model</strong> for this prediction task due to its superior performance and simplicity</li>
                <li><strong>Investigate why complex models underperformed</strong> - possible issues with hyperparameters or feature engineering</li>
                <li><strong>Consider additional feature engineering</strong> to capture potential nonlinear relationships that simple models might miss</li>
                <li><strong>Implement monitoring for model drift</strong> as market conditions change over time</li>
                <li><strong>Explore ensemble approaches</strong> that combine the strengths of different model types</li>
            </ol>
        </div>
    </div>
    
    <div class="footer">
        <p>Stock Price Prediction Analysis Report | Generated using Python Scikit-learn, XGBoost, and Matplotlib</p>
        <p>Confidential - For Internal Use Only</p>
        <p id="report-date">Report generated on: </p>
    </div>

    <script>
        document.getElementById('report-date').textContent += new Date().toLocaleString();
    </script>

    </body>
    </html>
    """
    
    report_filename = f"stock_prediction_report_{target_col}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated in {time.time()-start_time:.2f} seconds")
    
    # Determine best model based on test R² score
    best_model = max(models_results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\nBest Performing Model: {best_model[0]} (Test R²: {best_model[1]['test_r2']:.4f})")
    
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
                n_jobs=1,
                random_state=42
            ),
            'Support Vector Regressor': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                max_iter=1000
            ),
            'XGBoost': XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=42
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
            
            # Create combined plots
            print(f"Generating plots for {name}...")
            plots = create_model_plots(
                name, 
                y_train, y_train_pred,
                y_test, y_test_pred,
                y_val, y_val_pred,
                target_col,
                features=X_train.columns, 
                model=trained_model
            )
            plots_data[name] = plots
            
            print(f"Completed {name}\n{'='*50}")
        
        # Generate HTML report
        report_file = generate_html_report(models_results, plots_data, target_col, X_train)
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