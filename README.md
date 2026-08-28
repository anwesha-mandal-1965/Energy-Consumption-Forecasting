# 🔋 Energy Consumption Forecasting using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline for predictive energy demand forecasting. This project demonstrates comprehensive data preprocessing, exploratory data analysis (EDA), multi-model training, and robust performance benchmarking to support smart grid resource allocation and sustainable energy management.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Sample Visualizations](#-sample-visualizations)
- [Future Scope](#-future-scope)
- [Author](#-author)
- [License](#-license)

---

## 📖 Overview

Reliable electricity demand forecasting is critical for modern energy markets, power grid management, and sustainability planning. This repository provides a modular, production-ready machine learning framework to forecast hourly energy consumption based on historical load profiles, meteorological indicators, and cyclical temporal features.

The project covers every stage of the applied data science lifecycle:
1. **Data Engineering & Synthesis**: Simulating and structuring realistic time-series demand patterns.
2. **Exploratory Data Analysis**: Uncovering seasonal, diurnal, and temperature-driven load behaviors.
3. **Feature Engineering**: Transforming calendar and weather variables into predictive ML features.
4. **Model Development & Evaluation**: Training, cross-evaluating, and comparing baseline linear and non-linear ensemble models.

---

## 🎯 Problem Statement

Power grid operators and utility providers must continuously balance electricity generation with real-time consumer demand. Under-generation risks blackouts and grid instability, while over-generation results in significant energy waste, financial losses, and unnecessary greenhouse gas emissions.

Traditional statistical models often struggle to capture complex non-linear patterns, multi-level seasonality (daily, weekly, annual), and abrupt weather-driven variations. Applying modern machine learning techniques enables accurate, data-driven load forecasting that empowers utilities to optimize dispatch schedules, integrate renewable resources effectively, and minimize operational costs.

---

## 🏆 Objectives

1. **Historical Pattern Analysis**: Perform in-depth time-series analysis to identify underlying demand baselines.
2. **Trend & Seasonality Identification**: Isolate diurnal fluctuations, weekday/weekend variations, and seasonal demand shifts.
3. **Model Development**: Train and tune multiple regression algorithms, including Linear Regression, Random Forest, and Gradient Boosting.
4. **Performance Benchmarking**: Rigorously compare models using standardized statistical metrics ($\text{RMSE}$, $\text{MAE}$, and $R^2$).
5. **Actionable Energy Insights**: Translate empirical findings and feature importances into strategic recommendations for energy planners.

---

## 💻 Technologies Used

- **Language & Runtime**: Python 3.11+
- **Interactive Development**: VS Code Interactive Python / Jupyter Notebook environment
- **Data Manipulation & Mathematics**:
  - `pandas` – High-performance time-series and tabular data manipulation
  - `numpy` – Numerical computation and vectorized array operations
- **Data Visualization**:
  - `matplotlib` – Core plotting and customized chart styling
  - `seaborn` – Statistical distributions and correlation heatmaps
- **Machine Learning**:
  - `scikit-learn` – Model architectures, preprocessing pipelines, cross-validation, and metrics:
    - *Linear Regression*
    - *Random Forest Regressor*
    - *Gradient Boosting Regressor*

---

## 📂 Project Structure

```text
energy-consumption-forecasting/
├── data/
│   ├── generate_data.py       # Script to simulate realistic energy & weather data
│   └── energy_consumption.csv # Dataset containing time-series energy records
├── results/
│   └── (saved plots)          # Generated figures, charts, and evaluation plots
├── models/                    # Serialized models and training checkpoints
├── energy_forecasting.py      # Core interactive script (preprocessing, EDA, modeling)
├── README.md                  # Project documentation
└── requirements.txt           # Python environment dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system (Python 3.11+ recommended).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/energy-consumption-forecasting.git
cd energy-consumption-forecasting
pip install -r requirements.txt
```

### 3. Generate Dataset
If you are generating the dataset locally:
```bash
py data/generate_data.py
```
*This will create the structured `energy_consumption.csv` file inside the `data/` directory.*

### 4. Run the Pipeline
Open `energy_forecasting.py` in **VS Code**:
- The script is organized with `# %%` cell markers, enabling VS Code's interactive notebook mode.
- Execute cells individually using <kbd>Ctrl</kbd> + <kbd>Enter</kbd> (or <kbd>Shift</kbd> + <kbd>Enter</kbd>) to view plots and metrics interactively in the VS Code Python Interactive window.
- Alternatively, run the full script directly from the terminal:
  ```bash
  py energy_forecasting.py
  ```

---

## 🔬 Methodology

```mermaid
flowchart LR
    A[Data Ingestion] --> B[Data Cleaning & Prep]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Strategic Insights]
```

1. **Data Collection & Ingestion**: Loading chronological time-series records containing timestamps, weather metrics (temperature, humidity), and electricity consumption readings.
2. **Data Cleaning & Preprocessing**: Handling null values, validating timestamp sequence continuity, and setting up time-series indexing.
3. **Exploratory Data Analysis (EDA)**: Visualizing distribution densities, hourly demand cycles, weekly load curves, and correlation matrices.
4. **Feature Selection & Transformation**: Extracting calendar features (hour, day of week, month, weekend indicators), encoding cyclic features, and scaling inputs.
5. **Model Training & Tuning**: Fitting linear baselines alongside ensemble architectures (Random Forest, Gradient Boosting) with cross-validation.
6. **Model Evaluation**: Quantifying performance across training and holdout test sets using:
   - **Root Mean Squared Error (RMSE)**
   - **Mean Absolute Error (MAE)**
   - **Coefficient of Determination ($R^2$ Score)**
7. **Insight Generation & Interpretation**: Analyzing feature importance rankings and formulating data-backed energy policy insights.

---

## 📊 Key Results

| Model | MAE | RMSE | $R^2$ Score | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | Baseline | Baseline | Moderate | Captured linear macro trends; struggled with peak demand |
| **Random Forest Regressor** | Low | Low | High | Excellent capture of non-linear interactions |
| **Gradient Boosting Regressor** | **Lowest** | **Lowest** | **Highest** | **Best overall accuracy and generalization** |

### Critical Findings:
- **Tree-Based Superiority**: Non-linear ensemble models (Random Forest and Gradient Boosting) substantially outperformed Linear Regression across all metrics.
- **Dominant Drivers**: **Ambient Temperature** and **Time-of-Day (Hour)** exhibited the highest feature importance in predicting energy demand.
- **U-Shaped Thermal Demand**: Identified a distinct U-shaped curve with respect to temperature — energy consumption surges during extreme cold (electric heating) and extreme heat (HVAC cooling), with a temperate baseline around 18°C–22°C.
- **Bimodal Daily Peaks**: Clear demand peaks occur at **10:00 AM** (industrial/commercial ramp-up) and **8:00 PM** (residential peak).
- **Day-Type Variation**: Weekend consumption is significantly lower and flatter than weekday profiles due to reduced commercial activity.

---

## 📈 Sample Visualizations

All visual artifacts are saved automatically to the `results/` directory during execution:
- `actual_vs_predicted.png` – Multi-model forecasted load vs. ground truth time-series.
- `diurnal_consumption_pattern.png` – Average hourly load curves comparing weekdays vs. weekends.
- `temperature_vs_demand_curve.png` – Scatter and regression fit highlighting the non-linear thermal effect.
- `feature_importance_ranking.png` – Relative importance scores from ensemble models.
- `correlation_heatmap.png` – Cross-variable correlation matrix across meteorological and temporal features.

---

## 🔮 Future Scope

- [ ] **Real-Time Weather Integration**: Connecting live OpenWeatherMap / NOAA REST APIs for rolling dynamic predictions.
- [ ] **Deep Learning Time-Series Models**: Implementing Recurrent Neural Networks (LSTM, GRU) and Temporal Fusion Transformers (TFT).
- [ ] **Interactive Dashboard**: Building and deploying a Streamlit / Dash web interface for real-time scenario simulation.
- [ ] **Market Price & Tariff Integration**: Incorporating dynamic electricity pricing signals to model demand-response behavior.

---

## 👩‍💻 Author

**Anwesha Mandal**  
*B.Tech in Chemical Engineering*  
**Delhi Technological University (DTU)** | *CGPA: 9.6*  
📧 Email: [23ch012@dtu.ac.in](mailto:23ch012@dtu.ac.in)  
🔗 GitHub: [@anweshamandal](https://github.com/)  

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
