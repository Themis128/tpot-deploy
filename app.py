# Wine Quality Forecast Dashboard (Streamlit + TPOT + Prophet)
import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from src.utils import load_and_prepare_data, validate_pipeline
from tpot import TPOTRegressor
from prophet import Prophet
import numpy as np

# Configuration
st.set_page_config(page_title="Wine Quality Forecast", layout="wide")
st.sidebar.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
VERSION = "1.0.1"
PIPELINE_DIR = "src/pipelines_clean"
DATA_DIR = "data/processed_new/split_regions"

# Utility Functions
def count_files_and_dirs(path):
    total_files, total_dirs = 0, 0
    for _, dirs, files in os.walk(path):
        total_dirs += len(dirs)
        total_files += len(files)
    return total_dirs, total_files

def get_git_info():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        return branch, commit
    except:
        return None, None

# Sidebar Configuration
st.sidebar.title("Wine Quality Forecast")
st.sidebar.markdown(f"**App Version:** `{VERSION}`")

region = st.sidebar.selectbox("Select Region", sorted([
    f.replace("merged_", "").replace(".csv", "")
    for f in os.listdir(DATA_DIR) if f.endswith(".csv")
]))
retrain = st.sidebar.button("Retrain Model")
uploaded_model = st.sidebar.file_uploader("Upload Custom Model", type=["pkl"])

# Load Data and Preprocess
data_path = os.path.join(DATA_DIR, f"merged_{region}.csv")
df = pd.read_csv(data_path)
X, y, dates = load_and_prepare_data(df)

# Filter to 2024 onward
start_date = pd.Timestamp("2024-01-01")
dates = pd.to_datetime(dates)
mask = dates >= start_date
X, y, dates = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), dates[mask].reset_index(drop=True)

# Optional Wine Type Filter
wine_type_filter = None
if 'wine_type' in df.columns:
    wine_type_filter = st.sidebar.selectbox("Select Wine Type", sorted(df['wine_type'].dropna().unique().tolist()) + ["All"])
    if wine_type_filter != "All":
        wine_mask = df['wine_type'] == wine_type_filter
        X = X[wine_mask].reset_index(drop=True)
        y = y[wine_mask].reset_index(drop=True)
        dates = dates[wine_mask].reset_index(drop=True)
if 'wine_type' in X.columns:
    X = X.drop(columns=['wine_type'])

# Date and score range filtering
years = sorted(dates.dt.year.unique())
year_range = st.sidebar.slider("Select Year Range", min_value=int(min(years)), max_value=int(max(years)), value=(2024, max(years)))
score_range = st.sidebar.slider("Wine Quality Score", min_value=int(y.min()), max_value=int(y.max()), value=(int(y.min()), int(y.max())))
mask = dates.dt.year.between(*year_range) & y.between(*score_range)
X, y, dates = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), dates[mask].reset_index(drop=True)

# Display summary
st.title("Wine Quality Prediction Dashboard")
st.write(f"**Samples:** {len(df)} | **Filtered Rows:** {len(y)} | **Features:** {X.shape[1]}")
st.dataframe(X.head())

# Retrain the pipeline using TPOT
if retrain:
    st.info("Retraining model using TPOT... please wait")
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1)
    tpot.fit(X, y)
    best_model = tpot.fitted_pipeline_
    joblib.dump(best_model, os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl"))
    st.success("✅ Model retrained and saved.")

# Load pipeline
pipeline = joblib.load(uploaded_model if uploaded_model else os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl"))
if not validate_pipeline(pipeline, X):
    st.error("❌ Invalid pipeline")
    st.stop()

# Prediction & Metrics
y_pred = pipeline.predict(X)
residuals = y - y_pred
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

# Metrics
st.subheader("Model Performance")
st.markdown(f"""
- **R²:** `{r2:.3f}`
- **MAE:** `{mae:.3f}`
- **RMSE:** `{rmse:.3f}`
""")

# 1. Time Series Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=y, mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines+markers', name='Predicted'))
fig.update_layout(title="1. Wine Quality Over Time", xaxis_title="Date", yaxis_title="Score")
st.plotly_chart(fig, use_container_width=True)

# 2. Bar Chart
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=dates, y=y, name='Actual'))
fig_bar.add_trace(go.Bar(x=dates, y=y_pred, name='Predicted'))
fig_bar.update_layout(title="2. Predicted vs Actual", barmode='group')
st.plotly_chart(fig_bar, use_container_width=True)

# 3. Residuals Over Time
fig_res = go.Figure()
fig_res.add_trace(go.Scatter(x=dates, y=residuals, mode='lines+markers'))
fig_res.update_layout(title="3. Residuals Over Time")
st.plotly_chart(fig_res, use_container_width=True)

# 4. Error Distribution
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=residuals))
fig_hist.update_layout(title="4. Error Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

# 5. Prophet Forecast
prophet_df = pd.DataFrame({"ds": dates, "y": y})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=730)
forecast = model.predict(future)
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', line=dict(dash='dot')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', line=dict(dash='dot')))
fig_forecast.update_layout(title="5. Prophet Forecast")
st.plotly_chart(fig_forecast, use_container_width=True)

# 6. Anomaly Detection
threshold = 2 * np.std(residuals)
anomalies = residuals[abs(residuals) > threshold]
fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=dates, y=residuals, name='Residuals'))
fig_anom.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode='markers', name='Anomalies', marker=dict(color='red')))
fig_anom.update_layout(title="6. Anomaly Detection (±2σ)")
st.plotly_chart(fig_anom, use_container_width=True)

# 7. Rolling MAE
rolling_mae = residuals.abs().rolling(window=30, min_periods=1).mean()
fig_mae = go.Figure()
fig_mae.add_trace(go.Scatter(x=dates, y=rolling_mae, name='Rolling MAE'))
fig_mae.update_layout(title="7. Rolling MAE (30-day window)")
st.plotly_chart(fig_mae, use_container_width=True)

# 8. Feature Importance Treemap
if hasattr(pipeline, "feature_importances_"):
    importances = pipeline.feature_importances_
    names = pipeline.feature_names_in_
    fig_tree = go.Figure(go.Treemap(labels=names, parents=[""] * len(names), values=importances))
    fig_tree.update_layout(title="8. Feature Importance Treemap")
    st.plotly_chart(fig_tree, use_container_width=True)

# Downloads
results_df = pd.DataFrame({"Date": dates, "Actual": y, "Predicted": y_pred})
metrics_txt = f"Region: {region}\nR²: {r2:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}"
file_label = f"{region}_{wine_type_filter or 'all'}_{year_range[0]}-{year_range[1]}"
st.download_button("Download Predictions CSV", results_df.to_csv(index=False), file_name=f"{file_label}.csv")
st.download_button("Download Metrics TXT", metrics_txt, file_name=f"{file_label}_metrics.txt")
st.download_button("Download Forecast CSV", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False), file_name=f"{file_label}_forecast.csv")

# Footer
with st.sidebar.expander("Project Info"):
    dirs, files = count_files_and_dirs(".")
    st.markdown(f"**Structure:** `{dirs}` dirs, `{files}` files")
    branch, commit = get_git_info()
    if branch:
        st.markdown(f"**Git Branch:** `{branch}`")
        st.markdown(f"**Commit:** `{commit}`")
    st.markdown("**Author:** Baltzakis Themistoklis")
