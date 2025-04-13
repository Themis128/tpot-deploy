import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
from sklearn.inspection import permutation_importance

# ================================
# Data Preparation
# ================================
def load_and_prepare_data(df, date_col='time', target_col='wine_quality_score'):
    """
    Cleans and returns features (X), target (y), and date index from dataframe.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'.")
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col)
        dates = df[date_col]
    else:
        print(f"⚠️ '{date_col}' not found. Using row index for time axis.")
        dates = pd.Series(range(len(df)), name="index")
    
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    
    drop_cols = [target_col, date_col, 'id', 'region', 'wine_type']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    if 'wine_type' in df.columns and 'wine_type' not in X.columns:
        X['wine_type'] = df['wine_type'].astype('category').cat.codes
    
    return X, y, dates

# ================================
# Plotting
# ================================
def plot_predictions(y_true, y_pred, dates, region_name, output_dir):
    """
    Saves a line plot comparing predicted vs actual wine quality scores.
    """
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=dates, y=y_true, label='Actual', color='blue')
    sns.lineplot(x=dates, y=y_pred, label='Predicted', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Wine Quality Score')
    plt.title(f'Predicted vs Actual – {region_name}')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{region_name}_pred_vs_actual.png')
    plt.savefig(save_path)
    plt.close()

# ================================
# Feature Importance Summary
# ================================
def summarize_top_features(models_dir="models", top_k=10):
    """
    Summarizes top K features from all trained models (.pkl files) in the given directory.
    Returns a DataFrame with region, model, feature, importance.
    """
    summary = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".pkl"):
            continue
        try:
            model_path = os.path.join(models_dir, fname)
            model = joblib.load(model_path)
            if not hasattr(model, "feature_importances_"):
                continue
            region = fname.split("_")[-1].replace(".pkl", "")
            model_name = fname.split("_")[0]
            importances = model.feature_importances_
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                continue
            top_indices = importances.argsort()[-top_k:][::-1]
            for idx in top_indices:
                summary.append({
                    "region": region,
                    "model": model_name,
                    "feature": feature_names[idx],
                    "importance": importances[idx]
                })
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return pd.DataFrame(summary).sort_values(["region", "model", "importance"], ascending=[True, True, False])

# ================================
# Save Feature Importances (JSON)
# ================================
def save_feature_importance(model, feature_names, region, model_name, output_dir="models/features"):
    """
    Save feature importances for a single model to JSON.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️ Model {model_name} for {region} has no feature_importances_")
        return
    importances = dict(zip(feature_names, model.feature_importances_))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{region}.json")
    with open(out_path, "w") as f:
        json.dump({
            "region": region,
            "model": model_name,
            "importances": importances
        }, f, indent=2)
    print(f"✅   Saved feature importances: {out_path}")

# ================================
# Versioned Pipeline Saver
# ================================
def save_pipeline_with_version(pipeline, region, output_dir="src/pipelines_clean"):
    """
    Saves pipeline with version tag (datetime-based) for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{region}_pipeline_{timestamp}.pkl"
    path = os.path.join(output_dir, filename)
    joblib.dump(pipeline, path)

    # Also save metadata
    metadata = {
        "region": region,
        "timestamp": timestamp,
        "model_type": type(pipeline).__name__
    }
    with open(path.replace(".pkl", ".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅   Saved versioned pipeline and metadata: {path}")
    return path

# ================================
# Pipeline Validation
# ================================
def validate_pipeline(pipeline, X_sample):
    """
    Validates if the pipeline can generate predictions from X_sample.
    """
    try:
        _ = pipeline.predict(X_sample.head(1))
        return True
    except Exception as e:
        print(f"❌ Pipeline validation failed: {e}")
        return False
