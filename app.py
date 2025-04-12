import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from src.utils import load_and_prepare_data
import seaborn as sns 



# Config
st.set_page_config(page_title="Wine Quality Forecast", layout="wide")
VERSION = "1.0.1"
PIPELINE_DIR = "src/pipelines_clean"
DATA_DIR = "data/processed_new/split_regions"

# Utility: File + Git
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

# Sidebar
st.sidebar.title("Wine Quality Forecast")
st.sidebar.markdown(f"**App Version:** `{VERSION}`")

region = st.sidebar.selectbox("Select Region", sorted([
    f.replace("merged_", "").replace(".csv", "")
    for f in os.listdir(DATA_DIR) if f.endswith(".csv")
]))

retrain = st.sidebar.button("Retrain Model")
uploaded_model = st.sidebar.file_uploader("Upload Custom Model", type=["pkl"])

# Load Data
data_path = os.path.join(DATA_DIR, f"merged_{region}.csv")
df = pd.read_csv(data_path)
X, y, dates = load_and_prepare_data(df)

if 'wine_type' in X.columns:
    X = X.drop(columns=['wine_type'])

# Show sample
st.title("Wine Quality Prediction Dashboard")
st.write(f"**Samples:** {len(df)} | **Features:** {X.shape[1]}")
st.dataframe(X.head())

# Optional retraining
if retrain:
    st.info("Retraining model using TPOT... please wait (~1 min)")
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1)
    tpot.fit(X, y)
    best_model = tpot.fitted_pipeline_
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl"))
    st.success("✅ Retrained and saved model.")

# Load pipeline
try:
    if uploaded_model:
        pipeline = joblib.load(uploaded_model)
        st.success("✅ Custom model loaded.")
    else:
        pipeline_path = os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl")
        pipeline = joblib.load(pipeline_path)

    y_pred = pipeline.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    st.subheader("Model Performance")
    st.markdown(f"- **R²:** `{r2:.3f}`")
    st.markdown(f"- **MAE:** `{mae:.3f}`")
    st.markdown(f"- **RMSE:** `{rmse:.3f}`")

    # Prediction plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title=f"Predicted vs Actual – {region}", xaxis_title="Date", yaxis_title="Wine Quality Score")
    st.plotly_chart(fig, use_container_width=True)

    # Download button
    results_df = pd.DataFrame({"Date": dates, "Actual": y, "Predicted": y_pred})
    st.download_button("Download Predictions as CSV",
        data=results_df.to_csv(index=False),
        file_name=f"{region}_predictions_{datetime.now().date()}.csv",
        mime="text/csv")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Footer
with st.sidebar.expander("Project Info"):
    dirs, files = count_files_and_dirs(".")
    st.markdown(f"**Structure:** `{dirs}` directories, `{files}` files")
    branch, commit = get_git_info()
    if branch and commit:
        st.markdown(f"**Git Branch:** `{branch}`")
        st.markdown(f"**Commit:** `{commit}`")
    else:
        st.markdown("**Git:** Not a repository")
    st.markdown("---")
    st.markdown("**Author:** Baltzakis Themistoklis")
