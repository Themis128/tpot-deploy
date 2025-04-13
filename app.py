import streamlit as st import pandas as pd import joblib import os import subprocess from datetime import datetime from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error import plotly.graph_objects as go from src.utils import load_and_prepare_data, validate_pipeline from tpot import TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1) tpot.fit(X, y) best_model = tpot.fitted_pipeline_ os.makedirs(PIPELINE_DIR, exist_ok=True) joblib.dump(best_model, os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl")) st.success("✅  Retrained and saved model.")

Load pipeline

try: if uploaded_model: pipeline = joblib.load(uploaded_model) st.success("✅  Custom model loaded.") else: pipeline_path = os.path.join(PIPELINE_DIR, f"{region}_pipeline.pkl") pipeline = joblib.load(pipeline_path)

# Validate pipeline
if not validate_pipeline(pipeline, X):
    raise ValueError("Invalid pipeline: failed to make predictions.")

y_pred = pipeline.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
st.subheader("Model Performance")
st.markdown(f"- **R²:** `{r2:.3f}`")
st.markdown(f"- **MAE:** `{mae:.3f}`")
st.markdown(f"- **RMSE:** `{rmse:.3f}`")

# Time-series plot
st.subheader("Wine Quality Over Time")
trend_fig = go.Figure()
trend_fig.add_trace(go.Scatter(x=dates, y=y, mode='lines+markers', name='Actual Quality'))
trend_fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted Quality'))
trend_fig.update_layout(title=f"Wine Quality Trend – {region}",
                        xaxis_title="Date",
                        yaxis_title="Wine Quality Score",
                        height=400)
st.plotly_chart(trend_fig, use_container_width=True)

st.markdown("""This chart shows how wine quality has evolved over time in the selected region.

You can observe patterns such as seasonal changes, production cycles, or outliers.""")

# Prediction plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted'))
fig.update_layout(title=f"Predicted vs Actual – {region}",
                  xaxis_title="Date", yaxis_title="Wine Quality Score")
st.plotly_chart(fig, use_container_width=True)

# Feature importance
if hasattr(pipeline, "named_steps"):
    for name, step in pipeline.named_steps.items():
        if hasattr(step, "feature_importances_"):
            st.subheader("Top Features")
            feature_names = getattr(step, "feature_names_in_", X.columns)
            importances = step.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)
            st.bar_chart(importance_df.set_index("Feature"))
            break

# Download button
results_df = pd.DataFrame({"Date": dates, "Actual": y, "Predicted": y_pred})
st.download_button("Download Predictions as CSV",
                   data=results_df.to_csv(index=False),
                   file_name=f"{region}_predictions_{datetime.now().date()}.csv",
                   mime="text/csv")

except Exception as e: st.error(f"❌  Error loading model: {e}")

Footer

with st.sidebar.expander("Project Info"): dirs, files = count_files_and_dirs(".") st.markdown(f"Structure: {dirs} directories, {files} files") branch, commit = get_git_info() if branch and commit: st.markdown(f"Git Branch: {branch}") st.markdown(f"Commit: {commit}") else: st.markdown("Git: Not a repository") st.markdown("---") st.markdown("Author: Baltzakis Themistoklis")


