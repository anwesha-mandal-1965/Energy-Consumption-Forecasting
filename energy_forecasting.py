# %% [markdown]
# # 🔋 Energy Consumption Forecasting using Machine Learning
#
# **Author:** Anwesha Mandal
# **Institution:** Delhi Technological University (DTU) | B.Tech Chemical Engineering | CGPA: 9.6
#
# ## Project Overview
#
# Energy consumption forecasting is critical for efficient grid management, cost optimization,
# and sustainability planning. In this project, we analyze historical energy consumption data
# along with weather and calendar features to build predictive models.
#
# **Approach:**
# 1. Exploratory Data Analysis (EDA) to understand consumption patterns
# 2. Feature Engineering to capture cyclical and non-linear relationships
# 3. Machine Learning models — Linear Regression, Random Forest, and Gradient Boosting
# 4. Model comparison and evaluation using RMSE, MAE, and R² metrics
#
# **Key Question:** Can we accurately predict energy consumption using weather and temporal features?

# %%
# ============================================================
# 📦 SECTION 1: Import Libraries
# ============================================================
# We import all the libraries we'll need upfront so we can
# reference them throughout the notebook without interruption.

import pandas as pd                  # Data manipulation and analysis
import numpy as np                   # Numerical computing
import matplotlib.pyplot as plt      # Core plotting library
import seaborn as sns                # Statistical visualizations (built on matplotlib)
import warnings                      # To suppress non-critical warnings
import os                            # For file/directory operations

# Scikit-learn: the go-to library for classical machine learning in Python
from sklearn.model_selection import train_test_split    # Split data into train/test sets
from sklearn.preprocessing import StandardScaler        # Normalize features to zero mean, unit variance
from sklearn.linear_model import LinearRegression       # Simple linear model (baseline)
from sklearn.ensemble import (
    RandomForestRegressor,          # Ensemble of decision trees (bagging)
    GradientBoostingRegressor       # Sequential ensemble (boosting)
)
from sklearn.metrics import (
    mean_squared_error,             # MSE — average squared error
    mean_absolute_error,            # MAE — average absolute error
    r2_score                        # R² — proportion of variance explained
)

# ---------- Global plot settings ----------
# 'seaborn-v0_8-whitegrid' gives us a clean, professional look with gridlines
plt.style.use('seaborn-v0_8-whitegrid')

# Set default figure size so every plot is large enough to read easily
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Suppress warnings (e.g., future-deprecation notices) to keep output clean
warnings.filterwarnings('ignore')

# Professional color palette used throughout the notebook
COLORS = {
    'blue':   '#2196F3',
    'green':  '#4CAF50',
    'orange': '#FF9800',
    'pink':   '#E91E63',
    'purple': '#9C27B0',
    'cyan':   '#00BCD4',
}

print("✅ All libraries imported successfully!")
print(f"   • pandas  {pd.__version__}")
print(f"   • numpy   {np.__version__}")
print(f"   • sklearn  (scikit-learn)")

# %%
# ============================================================
# 📁 SECTION 2: Create Results Directory
# ============================================================
# We'll save all our plots to a 'results/' folder so the
# analysis is fully reproducible and easy to share.

os.makedirs('results', exist_ok=True)   # exist_ok=True prevents error if folder exists
print("✅ 'results/' directory is ready for saving plots.")

# %%
# ============================================================
# 📋 SECTION 3: Load & Explore the Dataset
# ============================================================
# The dataset contains hourly energy consumption readings along
# with weather conditions and calendar information.

# Read the CSV file; parse the 'datetime' column as a proper datetime object
df = pd.read_csv('data/energy_consumption.csv', parse_dates=['datetime'])

print("=" * 60)
print("📊 DATASET OVERVIEW")
print("=" * 60)
print(f"\n🔹 Shape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns\n")

# Show column names and their data types
print("🔹 Column Data Types:")
print("-" * 40)
print(df.dtypes.to_string())

print("\n🔹 First 5 Rows:")
print("-" * 40)
df.head()

# %%
# Print the first 5 rows in a readable format
print(df.head().to_string())

# %%
# Statistical summary — gives min, max, mean, std, quartiles for each numeric column
print("=" * 60)
print("📊 STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe().round(2).to_string())

# %% [markdown]
# ## 🔍 Missing Values Analysis
#
# Before building any model, we must check for missing data.
# Missing values can bias results and cause errors during training.

# %%
# ============================================================
# 🔍 SECTION 4: Missing Values Analysis
# ============================================================

# Count missing values per column
missing_count = df.isnull().sum()
# Calculate percentage of missing values
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

# Combine into a neat summary table
missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %': missing_pct
})

print("=" * 60)
print("🔍 MISSING VALUES ANALYSIS")
print("=" * 60)
print(missing_summary.to_string())

# Total rows that have at least one missing value
rows_with_missing = df.isnull().any(axis=1).sum()
print(f"\n📋 Total rows with at least one missing value: {rows_with_missing:,} "
      f"({rows_with_missing / len(df) * 100:.2f}%)")

# %%
# ============================================================
# 🧹 SECTION 5: Data Cleaning
# ============================================================
# Strategy: Use forward-fill (propagate the last valid value forward)
# followed by backward-fill (fill any remaining NaNs at the start).
# This is appropriate for time-series data where values change gradually.

print("🧹 Cleaning missing values...")
print(f"   Before — temperature_c missing: {df['temperature_c'].isnull().sum()}")
print(f"   Before — humidity_pct missing:  {df['humidity_pct'].isnull().sum()}")

# Forward fill: if a value is NaN, copy the previous row's value
# Backward fill: handles NaNs at the very beginning of the series
df['temperature_c'] = df['temperature_c'].ffill().bfill()
df['humidity_pct'] = df['humidity_pct'].ffill().bfill()

print(f"\n   After  — temperature_c missing: {df['temperature_c'].isnull().sum()}")
print(f"   After  — humidity_pct missing:  {df['humidity_pct'].isnull().sum()}")

# Final verification — make sure there are absolutely no NaNs left
total_missing_after = df.isnull().sum().sum()
print(f"\n✅ Total missing values remaining: {total_missing_after}")

# %% [markdown]
# ## 📊 Exploratory Data Analysis (EDA)
#
# EDA is the most important step — it helps us **understand the data**
# before we throw it into a model. We'll create 6 visualizations to
# uncover patterns in energy consumption.

# %%
# ============================================================
# 📊 EDA 1: Daily Average Energy Consumption Over Time
# ============================================================
# Aggregating hourly data to daily averages smooths out noise
# and reveals the underlying trend.

# Group by date (not datetime) and compute the mean consumption per day
daily_avg = df.groupby(df['datetime'].dt.date)['energy_consumption_kwh'].mean()

# 30-day rolling average further smooths seasonal fluctuations
rolling_avg = daily_avg.rolling(window=30, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(daily_avg.index, daily_avg.values,
        alpha=0.3, color=COLORS['blue'], linewidth=0.8, label='Daily Average')
ax.plot(rolling_avg.index, rolling_avg.values,
        color=COLORS['pink'], linewidth=2.5, label='30-Day Rolling Average')

ax.set_title('Daily Average Energy Consumption Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Energy Consumption (kWh)', fontsize=13)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('results/01_daily_energy_trend.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Energy consumption shows clear seasonal variation — higher in summer")
print("   (cooling) and winter (heating), with dips in mild spring/autumn months.")

# %%
# ============================================================
# 📊 EDA 2: Monthly Average Energy Consumption (Bar Chart)
# ============================================================
# Bar charts are ideal for comparing discrete categories like months.

monthly_avg = df.groupby('month')['energy_consumption_kwh'].mean()

# Month names for readable x-axis labels
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(range(1, 13), monthly_avg.values,
              color=COLORS['cyan'], edgecolor='white', linewidth=1.2)

# Highlight the month with highest consumption
max_month_idx = monthly_avg.values.argmax()
bars[max_month_idx].set_color(COLORS['pink'])

ax.set_title('Average Energy Consumption by Month', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=13)
ax.set_ylabel('Average Energy Consumption (kWh)', fontsize=13)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)

# Add value labels on top of each bar
for bar_item in bars:
    height = bar_item.get_height()
    ax.text(bar_item.get_x() + bar_item.get_width() / 2., height + 0.5,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/02_monthly_energy.png', dpi=150, bbox_inches='tight')
plt.show()

peak_month = month_names[max_month_idx]
print(f"💡 Insight: {peak_month} has the highest average energy consumption.")
print("   The U-shaped monthly pattern confirms heating & cooling demand drives consumption.")

# %%
# ============================================================
# 📊 EDA 3: Hourly Consumption Patterns — Weekday vs Weekend
# ============================================================
# Energy usage follows a daily rhythm tied to human activity.
# We compare weekdays vs weekends to see if patterns differ.

# Separate weekday and weekend data
weekday_hourly = df[df['is_weekend'] == 0].groupby('hour')['energy_consumption_kwh'].mean()
weekend_hourly = df[df['is_weekend'] == 1].groupby('hour')['energy_consumption_kwh'].mean()
overall_hourly = df.groupby('hour')['energy_consumption_kwh'].mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left panel: Overall hourly pattern with shaded area ---
ax1 = axes[0]
ax1.plot(overall_hourly.index, overall_hourly.values,
         color=COLORS['blue'], linewidth=2.5, marker='o', markersize=4)
ax1.fill_between(overall_hourly.index, overall_hourly.values,
                  alpha=0.15, color=COLORS['blue'])
ax1.set_title('Overall Hourly Energy Pattern', fontsize=14, fontweight='bold')
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Avg Energy Consumption (kWh)', fontsize=12)
ax1.set_xticks(range(0, 24, 2))

# --- Right panel: Weekday vs Weekend comparison ---
ax2 = axes[1]
ax2.plot(weekday_hourly.index, weekday_hourly.values,
         color=COLORS['blue'], linewidth=2.5, marker='o', markersize=4, label='Weekday')
ax2.plot(weekend_hourly.index, weekend_hourly.values,
         color=COLORS['orange'], linewidth=2.5, marker='s', markersize=4, label='Weekend')
ax2.set_title('Weekday vs Weekend Hourly Pattern', fontsize=14, fontweight='bold')
ax2.set_xlabel('Hour of Day', fontsize=12)
ax2.set_ylabel('Avg Energy Consumption (kWh)', fontsize=12)
ax2.set_xticks(range(0, 24, 2))
ax2.legend(fontsize=12)

plt.suptitle('Hourly Energy Consumption Patterns', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/03_hourly_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Energy peaks during morning (8–10 AM) and evening (6–9 PM) hours.")
print("   Weekend consumption is generally lower, with a later morning ramp-up.")

# %%
# ============================================================
# 📊 EDA 4: Temperature vs Energy Consumption (U-Shape)
# ============================================================
# We expect a U-shaped relationship: energy rises at both
# low temperatures (heating) and high temperatures (cooling).

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Scatter plot of raw data ---
ax1 = axes[0]
# Sample 5000 points to avoid over-plotting
sample_idx = np.random.RandomState(42).choice(len(df), size=min(5000, len(df)), replace=False)
ax1.scatter(df.iloc[sample_idx]['temperature_c'],
            df.iloc[sample_idx]['energy_consumption_kwh'],
            alpha=0.15, s=10, color=COLORS['blue'])
ax1.set_title('Temperature vs Energy (Scatter)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Temperature (°C)', fontsize=12)
ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)

# --- Right: Binned average showing the U-shape clearly ---
ax2 = axes[1]
# Cut temperature into bins and compute mean energy for each bin
df['temp_bin'] = pd.cut(df['temperature_c'], bins=20)
temp_binned = df.groupby('temp_bin', observed=True)['energy_consumption_kwh'].mean()

ax2.bar(range(len(temp_binned)), temp_binned.values,
        color=COLORS['green'], edgecolor='white', linewidth=0.8)
ax2.set_title('Avg Energy by Temperature Bins (U-Shape)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Temperature Bin', fontsize=12)
ax2.set_ylabel('Avg Energy Consumption (kWh)', fontsize=12)
# Show every other label to prevent overlap
tick_labels = [str(interval.mid)[:4] for interval in temp_binned.index]
ax2.set_xticks(range(0, len(temp_binned), 2))
ax2.set_xticklabels(tick_labels[::2], rotation=45, fontsize=9)

# Clean up the temporary column
df.drop(columns=['temp_bin'], inplace=True)

plt.suptitle('Temperature–Energy Relationship', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/04_temperature_vs_energy.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Clear U-shaped relationship confirms that extreme temperatures")
print("   (both hot and cold) drive higher energy consumption — a key feature for modeling.")

# %%
# ============================================================
# 📊 EDA 5: Correlation Heatmap (Lower Triangle)
# ============================================================
# Correlation tells us how strongly each pair of variables
# moves together. Values near ±1 indicate strong relationships.

# Select only numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

# Create a mask for the upper triangle (we only show the lower half)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            annot_kws={'size': 9},
            cbar_kws={'label': 'Correlation Coefficient'})

ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: 'hour' and 'temperature_c' show notable correlations with energy")
print("   consumption. Multi-collinear features (e.g., month ↔ temperature) should be")
print("   handled carefully — tree-based models are robust to this, linear models less so.")

# %%
# ============================================================
# 📊 EDA 6: Distribution & Monthly Box Plot
# ============================================================
# Understanding the distribution shape (skewness, outliers)
# helps choose appropriate models and preprocessing.

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Histogram with mean & median ---
ax1 = axes[0]
ax1.hist(df['energy_consumption_kwh'], bins=50, color=COLORS['cyan'],
         edgecolor='white', alpha=0.85, linewidth=0.8)

mean_val = df['energy_consumption_kwh'].mean()
median_val = df['energy_consumption_kwh'].median()

ax1.axvline(mean_val, color=COLORS['pink'], linewidth=2, linestyle='--', label=f'Mean: {mean_val:.1f}')
ax1.axvline(median_val, color=COLORS['purple'], linewidth=2, linestyle='-', label=f'Median: {median_val:.1f}')

ax1.set_title('Distribution of Energy Consumption', fontsize=14, fontweight='bold')
ax1.set_xlabel('Energy Consumption (kWh)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(fontsize=11)

# --- Right: Box plot by month ---
ax2 = axes[1]
month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Use seaborn boxplot for a cleaner look
sns.boxplot(x='month', y='energy_consumption_kwh', data=df,
            ax=ax2, palette='coolwarm', linewidth=1.2,
            fliersize=2)

ax2.set_title('Energy Consumption by Month (Box Plot)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax2.set_xticklabels(month_names_short)

plt.suptitle('Energy Consumption Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/06_distribution_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"💡 Insight: Mean ({mean_val:.1f} kWh) ≈ Median ({median_val:.1f} kWh), indicating a")
print("   roughly symmetric distribution. Monthly box plots confirm higher variability")
print("   in extreme-weather months (Jan, Jul, Aug).")

# %% [markdown]
# ## ⚙️ Feature Engineering
#
# Raw features like `hour` (0–23) and `month` (1–12) are **cyclical** — hour 23 is close
# to hour 0, but a model sees them as far apart numerically. We fix this with **sine/cosine
# encoding**, which maps cyclical values onto a circle.
#
# We also add a **quadratic temperature term** to capture the U-shaped relationship and
# a **temperature × humidity interaction** because humid heat feels different from dry heat.

# %%
# ============================================================
# ⚙️ SECTION 7: Feature Engineering
# ============================================================
# Creating new features that help the model capture patterns
# it wouldn't discover from the raw columns alone.

print("=" * 60)
print("⚙️  FEATURE ENGINEERING")
print("=" * 60)

# --- Cyclical encoding for hour (period = 24) ---
# sin and cos together preserve the circular relationship
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# --- Cyclical encoding for month (period = 12) ---
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- Squared temperature (captures the U-shape) ---
# If temp matters quadratically, the model can learn the U-curve
df['temp_squared'] = df['temperature_c'] ** 2

# --- Temperature × Humidity interaction ---
# High temp + high humidity → more A/C usage (feels hotter)
df['temp_humidity'] = df['temperature_c'] * df['humidity_pct']

new_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'temp_squared', 'temp_humidity']

print("\n✅ New features created:")
for feat in new_features:
    print(f"   • {feat}")

print(f"\n📊 Dataset now has {df.shape[1]} columns (was 10, added {len(new_features)})")

# %%
# ============================================================
# 📋 SECTION 8: Prepare Data for Modeling
# ============================================================
# We select our feature columns, split into train/test sets,
# and standardize features for the linear model.

# Define the 14 features we'll use for prediction
feature_columns = [
    'temperature_c', 'humidity_pct', 'wind_speed_kmh',     # Weather features
    'hour', 'day_of_week', 'month',                         # Calendar features
    'is_weekend', 'is_holiday',                              # Binary flags
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',       # Cyclical encodings
    'temp_squared', 'temp_humidity'                          # Engineered features
]

target_column = 'energy_consumption_kwh'

print("=" * 60)
print("📋 PREPARING DATA FOR MODELING")
print("=" * 60)
print(f"\n🔹 Features ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {col}")

# --- Separate features (X) and target (y) ---
X = df[feature_columns]
y = df[target_column]

# --- Train/Test split: 80% train, 20% test ---
# random_state=42 ensures reproducibility (same split every run)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Standardize features (zero mean, unit variance) ---
# Fit on training data ONLY to prevent data leakage from the test set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)          # only transform on test

print(f"\n✅ Data split complete:")
print(f"   • Training set: {X_train.shape[0]:,} samples")
print(f"   • Test set:     {X_test.shape[0]:,} samples")
print(f"   • Features:     {X_train.shape[1]}")

# %% [markdown]
# ## 🤖 Model Training
#
# We train three models of increasing complexity:
# 1. **Linear Regression** — simple baseline; assumes a straight-line relationship
# 2. **Random Forest** — ensemble of decision trees; captures non-linear patterns
# 3. **Gradient Boosting** — builds trees sequentially, each correcting the previous one

# %%
# ============================================================
# 🤖 MODEL 1: Linear Regression
# ============================================================
# Linear Regression fits a straight line (hyperplane in high dimensions)
# through the data.  It's fast and interpretable, but assumes that
# the relationship between features and target is LINEAR.
#
# We use SCALED data here because Linear Regression is sensitive
# to feature magnitudes — larger values would dominate the fit.

print("=" * 60)
print("🤖 MODEL 1: Linear Regression")
print("=" * 60)
print("\n📖 How it works: Finds the best-fit line by minimizing the")
print("   sum of squared differences between predicted and actual values.")
print("   Formula: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b\n")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Show learned coefficients — how much each feature influences the prediction
print("📊 Learned Coefficients (scaled features):")
print("-" * 45)
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for _, row in coef_df.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"   {direction} {row['Feature']:20s}  {row['Coefficient']:+.4f}")

print(f"\n   Intercept (bias): {lr_model.intercept_:.4f}")
print("\n✅ Linear Regression trained successfully!")

# %%
# ============================================================
# 🤖 MODEL 2: Random Forest Regressor
# ============================================================
# Random Forest builds many decision trees, each on a random subset
# of the data and features.  The final prediction is the AVERAGE of
# all trees — this reduces overfitting compared to a single tree.
#
# We use UNSCALED data because tree-based models split on thresholds
# and are NOT affected by feature scale.

print("=" * 60)
print("🤖 MODEL 2: Random Forest Regressor")
print("=" * 60)
print("\n📖 How it works: Trains 200 decision trees in parallel, each on")
print("   a random subset of data. The final prediction is the average")
print("   of all trees — this 'wisdom of the crowd' approach reduces noise.\n")

rf_model = RandomForestRegressor(
    n_estimators=200,       # Number of trees in the forest
    max_depth=15,           # Maximum depth of each tree (prevents overfitting)
    min_samples_split=10,   # Minimum samples needed to split an internal node
    min_samples_leaf=5,     # Minimum samples in a leaf node
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores for speed
)
rf_model.fit(X_train, y_train)   # Unscaled data — trees don't need scaling

print("✅ Random Forest trained successfully!")
print(f"   • Trees: {rf_model.n_estimators}")
print(f"   • Max depth: {rf_model.max_depth}")

# %%
# ============================================================
# 🤖 MODEL 3: Gradient Boosting Regressor
# ============================================================
# Gradient Boosting builds trees SEQUENTIALLY — each new tree tries
# to correct the errors (residuals) of the previous ensemble.
# It's often the most accurate classical ML method, but slower to train.
#
# We use UNSCALED data (same reason as Random Forest).

print("=" * 60)
print("🤖 MODEL 3: Gradient Boosting Regressor")
print("=" * 60)
print("\n📖 How it works: Builds trees one at a time. Each new tree focuses")
print("   on the mistakes the previous trees made. The learning_rate (0.1)")
print("   controls how much each tree contributes — smaller = more robust.\n")

gb_model = GradientBoostingRegressor(
    n_estimators=200,       # Number of boosting stages
    max_depth=5,            # Shallower trees — boosting prefers weak learners
    learning_rate=0.1,      # Step size shrinkage (regularization)
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=5,     # Minimum samples in leaf
    random_state=42         # Reproducibility
)
gb_model.fit(X_train, y_train)   # Unscaled data

print("✅ Gradient Boosting trained successfully!")
print(f"   • Trees: {gb_model.n_estimators}")
print(f"   • Max depth: {gb_model.max_depth}")
print(f"   • Learning rate: {gb_model.learning_rate}")

# %% [markdown]
# ## 📊 Model Evaluation
#
# We evaluate all three models on the **test set** (data the models have never seen).
#
# **Metrics explained:**
# - **RMSE** (Root Mean Squared Error): Average error in the same units as the target (kWh). Lower = better.
# - **MAE** (Mean Absolute Error): Average absolute error. Less sensitive to outliers than RMSE. Lower = better.
# - **R²** (R-squared): Proportion of variance explained. 1.0 = perfect. Higher = better.

# %%
# ============================================================
# 📊 SECTION 10: Model Evaluation
# ============================================================

def evaluate_model(model, X_test_data, y_test_data, model_name):
    """
    Evaluate a trained model and return a dictionary of metrics.

    Parameters:
        model         : trained sklearn model
        X_test_data   : test features (scaled or unscaled as appropriate)
        y_test_data   : true target values
        model_name    : string name for display

    Returns:
        dict with 'Model', 'RMSE', 'MAE', 'R2', and 'Predictions'
    """
    predictions = model.predict(X_test_data)
    rmse = np.sqrt(mean_squared_error(y_test_data, predictions))
    mae = mean_absolute_error(y_test_data, predictions)
    r2 = r2_score(y_test_data, predictions)

    return {
        'Model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'Predictions': predictions    # Keep predictions for plotting later
    }


# Evaluate each model with the correct data (scaled for LR, unscaled for trees)
lr_results = evaluate_model(lr_model, X_test_scaled, y_test, 'Linear Regression')
rf_results = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
gb_results = evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')

all_results = [lr_results, rf_results, gb_results]

# %%
# --- Build comparison table ---
results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'Predictions'}
    for r in all_results
])

print("=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# Identify the best model by R²
best_result = max(all_results, key=lambda x: x['R2'])
print(f"\n🏆 Best Model: {best_result['Model']}")
print(f"   R² = {best_result['R2']:.4f}  |  RMSE = {best_result['RMSE']:.4f}  |  MAE = {best_result['MAE']:.4f}")

# %%
# --- Bar chart comparing metrics across models ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = [r['Model'] for r in all_results]
bar_colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]

# RMSE comparison
axes[0].bar(model_names, [r['RMSE'] for r in all_results], color=bar_colors, edgecolor='white')
axes[0].set_title('RMSE (Lower is Better)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RMSE (kWh)', fontsize=12)
for i, v in enumerate([r['RMSE'] for r in all_results]):
    axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

# MAE comparison
axes[1].bar(model_names, [r['MAE'] for r in all_results], color=bar_colors, edgecolor='white')
axes[1].set_title('MAE (Lower is Better)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('MAE (kWh)', fontsize=12)
for i, v in enumerate([r['MAE'] for r in all_results]):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

# R² comparison
axes[2].bar(model_names, [r['R2'] for r in all_results], color=bar_colors, edgecolor='white')
axes[2].set_title('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_ylim(0, 1.1)
for i, v in enumerate([r['R2'] for r in all_results]):
    axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Ensemble methods (Random Forest, Gradient Boosting) outperform")
print("   Linear Regression because energy consumption has non-linear relationships.")

# %%
# --- Actual vs Predicted scatter plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, result in enumerate(all_results):
    ax = axes[idx]
    preds = result['Predictions']

    ax.scatter(y_test, preds, alpha=0.15, s=8, color=bar_colors[idx])

    # Perfect prediction line (y = x)
    plot_min = min(y_test.min(), preds.min())
    plot_max = max(y_test.max(), preds.max())
    ax.plot([plot_min, plot_max], [plot_min, plot_max],
            'r--', linewidth=2, label='Perfect Prediction')

    ax.set_title(f'{result["Model"]}\nR² = {result["R2"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual (kWh)', fontsize=11)
    ax.set_ylabel('Predicted (kWh)', fontsize=11)
    ax.legend(fontsize=10)

plt.suptitle('Actual vs Predicted Energy Consumption', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/08_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Points closer to the red dashed line indicate better predictions.")
print("   Notice how ensemble models cluster more tightly around the line.")

# %%
# --- Residual plots (errors = actual - predicted) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, result in enumerate(all_results):
    ax = axes[idx]
    preds = result['Predictions']
    residuals = y_test.values - preds   # Positive = model under-predicted

    ax.scatter(preds, residuals, alpha=0.15, s=8, color=bar_colors[idx])
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')

    ax.set_title(f'{result["Model"]}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted (kWh)', fontsize=11)
    ax.set_ylabel('Residual (kWh)', fontsize=11)

plt.suptitle('Residual Analysis (Actual − Predicted)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/09_residual_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Insight: Ideally, residuals should scatter randomly around zero with no pattern.")
print("   A funnel shape would indicate heteroscedasticity (varying error magnitude).")

# %% [markdown]
# ## 🌲 Feature Importance Analysis
#
# Tree-based models can tell us which features matter most for prediction.
# This is valuable for understanding **why** the model makes certain predictions
# and for guiding future data collection.

# %%
# ============================================================
# 🌲 SECTION 11: Feature Importance
# ============================================================

# Extract importance scores from both tree-based models
rf_importance = rf_model.feature_importances_
gb_importance = gb_model.feature_importances_

# Build a DataFrame for easy sorting and plotting
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Random Forest': rf_importance,
    'Gradient Boosting': gb_importance
}).sort_values('Gradient Boosting', ascending=True)   # Sort for horizontal bar chart

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Random Forest feature importance ---
rf_sorted = importance_df.sort_values('Random Forest', ascending=True)
axes[0].barh(rf_sorted['Feature'], rf_sorted['Random Forest'],
             color=COLORS['green'], edgecolor='white')
axes[0].set_title('Random Forest\nFeature Importance', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance Score', fontsize=12)

# --- Gradient Boosting feature importance ---
gb_sorted = importance_df.sort_values('Gradient Boosting', ascending=True)
axes[1].barh(gb_sorted['Feature'], gb_sorted['Gradient Boosting'],
             color=COLORS['orange'], edgecolor='white')
axes[1].set_title('Gradient Boosting\nFeature Importance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance Score', fontsize=12)

plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Print top 5 features for each model
print("=" * 60)
print("🏆 TOP 5 MOST IMPORTANT FEATURES")
print("=" * 60)

rf_top5 = importance_df.sort_values('Random Forest', ascending=False).head(5)
gb_top5 = importance_df.sort_values('Gradient Boosting', ascending=False).head(5)

print("\n🌲 Random Forest:")
for rank, (_, row) in enumerate(rf_top5.iterrows(), 1):
    print(f"   {rank}. {row['Feature']:20s}  (importance: {row['Random Forest']:.4f})")

print("\n🚀 Gradient Boosting:")
for rank, (_, row) in enumerate(gb_top5.iterrows(), 1):
    print(f"   {rank}. {row['Feature']:20s}  (importance: {row['Gradient Boosting']:.4f})")

# %% [markdown]
# ## 📝 Conclusions
#
# ### Key Findings
#
# 1. **Model Performance:** Ensemble models (Random Forest, Gradient Boosting) significantly
#    outperform Linear Regression, confirming that energy consumption has **non-linear**
#    relationships with weather and temporal features.
#
# 2. **Seasonal Patterns:** Energy consumption follows a clear **U-shaped seasonal curve** —
#    peaking in summer (cooling demand) and winter (heating demand), with valleys in
#    spring and autumn.
#
# 3. **Daily Peaks:** Consumption peaks twice daily — during **morning (8–10 AM)** and
#    **evening (6–9 PM)** hours, corresponding to work start and return-home activity.
#
# 4. **Weekend Effect:** Weekend consumption is lower overall and ramps up later in the
#    morning, reflecting differences in residential vs. commercial activity.
#
# 5. **Feature Importance:** Temperature-related features (`temperature_c`, `temp_squared`,
#    `temp_humidity`) and time-of-day (`hour`) are the strongest predictors, validating
#    our domain knowledge about HVAC-driven consumption.
#
# ### Practical Applications
#
# - **Energy Providers:** Optimize power generation schedules and reduce peak-hour strain
# - **Building Management:** Implement smart HVAC scheduling based on predicted demand
# - **Sustainability:** Identify energy waste patterns and recommend efficiency improvements
# - **Cost Optimization:** Shift flexible loads to off-peak hours using consumption forecasts
#
# ### Future Scope
#
# - Incorporate **real-time weather API data** for live forecasting
# - Experiment with **ARIMA / LSTM** models that explicitly model time-series dependencies
# - Build a **web dashboard** (e.g., Streamlit / Flask) for interactive exploration

# %%
# ============================================================
# 🏁 SECTION 13: Final Summary
# ============================================================

print("=" * 60)
print("🏁 FINAL SUMMARY")
print("=" * 60)

best = max(all_results, key=lambda x: x['R2'])

print(f"""
📊 Dataset:
   • Total samples:     {len(df):,}
   • Features used:     {len(feature_columns)}
   • Train/Test split:  80% / 20%

🤖 Models Trained:      3
   • Linear Regression
   • Random Forest (200 trees)
   • Gradient Boosting (200 stages)

🏆 Best Model:          {best['Model']}
   • R² Score:          {best['R2']:.4f}
   • RMSE:              {best['RMSE']:.4f} kWh
   • MAE:               {best['MAE']:.4f} kWh

📁 Plots saved to:      results/
""")

print("=" * 60)
print("✅ Analysis complete! All results saved to the 'results/' folder.")
print("=" * 60)
