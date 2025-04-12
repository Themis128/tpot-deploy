Here’s a complete README.md for your project, tailored to your current structure and Streamlit dashboard setup:


---

Viticulture Wine Quality Forecast Dashboard

Author: Baltzakis Themistoklis
Project Type: AutoML Forecasting for Wine Quality
Frontend: Streamlit Web App
Backend: TPOT (Genetic Programming for AutoML)


---

Overview

This project uses automated machine learning (AutoML) to train regression models that predict wine quality scores across 17 viticulture regions in Greece. It leverages TPOT (Tree-based Pipeline Optimization Tool) to find optimal pipelines and visualizes predictions using an interactive Streamlit dashboard.


---

Features

Predicts wine quality score based on chemical properties and weather data

Region-specific models and plots (forecast & residuals)

Feature importance summaries

Supports model versioning and retraining

Fast UI with Plotly and Altair for interactive charts

Downloadable predictions

Lightweight and deployable on Streamlit Community Cloud



---

Project Structure

tpot-project/
│
├── app.py                      ← Streamlit dashboard
├── src/                        ← Source code and pipeline scripts
│   ├── utils.py                ← Utility functions (loading, plotting, saving)
│   └── pipelines_clean/        ← Final trained TPOT pipelines (.pkl)
│
├── data/
│   ├── raw_new/                ← Raw daily/hourly CSVs per region
│   ├── preprocessed/           ← Cleaned and feature-engineered datasets
│   └── processed_new/
│       └── split_regions/      ← Region-specific merged wine datasets
│
├── results/
│   └── plots/
│       ├── forecasts/          ← Forecast visualizations per region
│       └── residuals/          ← Residuals (errors) per region
│
├── requirements.txt
└── README.md


---

How to Run

Local Run

# Step 1: Clone the repo
git clone https://github.com/<your-repo>/tpot-project.git
cd tpot-project

# Step 2: Create environment
python3 -m venv tpot-venv
source tpot-venv/bin/activate
pip install -r requirements.txt

# Step 3: Launch the dashboard
streamlit run app.py

Streamlit Cloud

Upload all necessary files to your repo

Include app.py, requirements.txt, src/, and data files if needed

Connect repo to Streamlit Cloud



---

Algorithm

This project uses TPOT to automate machine learning pipeline design via genetic programming.
TPOT evolves models over generations, testing and mutating pipelines to improve performance.

Typical Chosen Pipeline

Most regions use:

ExtraTreesRegressor(
    RBFSampler(
        ExtraTreesRegressor(
            MaxAbsScaler(input_matrix),
            bootstrap=True,
            max_features=0.65,
            min_samples_leaf=5,
            min_samples_split=8,
            n_estimators=100
        ),
        gamma=0.5
    ),
    bootstrap=True,
    max_features=0.75,
    min_samples_leaf=1,
    min_samples_split=9,
    n_estimators=100
)

Evaluation Metrics

R² (coefficient of determination)

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)



---

Contributing

Pull requests and suggestions are welcome.
For issues or improvements, please open an issue.


---

License

MIT License — free to use, modify, and distribute.


---

Would you like this also as a file you can directly use (README.md)?


