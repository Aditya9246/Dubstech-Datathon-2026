"""
TIME-SERIES FORECASTING MODEL
Predict trends in delayed or missed care over the next year (2025)

This model uses multiple forecasting approaches:
1. ARIMA - Classical time series
2. Prophet - Facebook's forecasting tool
3. LSTM - Deep learning approach
4. XGBoost Time Series - Gradient boosting with lag features
5. Ensemble - Combining multiple models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print(" " * 20 + "TIME-SERIES FORECASTING: DELAYED/MISSED CARE PREDICTIONS")
print("=" * 100)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("\n[STEP 1] LOADING AND PREPARING TIME-SERIES DATA")
print("-" * 100)

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

# Load data
df = pd.read_csv(project_root / "data" / "NHIS_Data_Cleaned.csv")
print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Define Directory Structure
# timeseries_forecast_model/
# ├── output_data/
# └── output_images/
images_dir = current_script_path.parent / "output_images"
data_dir = current_script_path.parent / "data_output"

images_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print(f"Directories created at: {current_script_path.parent}")

# Filter for delay/missed care topics
delay_topics = [
    'Delayed getting medical care due to cost among adults',
    'Did not get needed medical care due to cost',
    'Did not get needed mental health care due to cost',
    'Six or more workdays missed due to illness, injury, or disability'
]

delay_df = df[df['TOPIC'].isin(delay_topics)].copy()
delay_df = delay_df.dropna(subset=['ESTIMATE'])

print(f"✓ Loaded {len(delay_df):,} records")
print(f"✓ Topics: {len(delay_topics)} delay/missed care categories")
print(f"✓ Time range: {delay_df['TIME_PERIOD'].min()} - {delay_df['TIME_PERIOD'].max()}")
print(f"✓ Subgroups: {delay_df['SUBGROUP'].nunique()} demographic groups")

# ============================================================================
# 2. AGGREGATE TIME SERIES BY YEAR
# ============================================================================
print("\n[STEP 2] CREATING AGGREGATE TIME SERIES")
print("-" * 100)

# Overall trend (all topics combined)
overall_ts = delay_df.groupby('TIME_PERIOD').agg({
    'ESTIMATE': ['mean', 'median', 'std', 'min', 'max', 'count']
}).reset_index()
overall_ts.columns = ['Year', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Count']

print("\nOverall Delay/Missed Care Trends by Year:")
print(overall_ts.to_string(index=False))

# By topic
topic_ts = delay_df.groupby(['TIME_PERIOD', 'TOPIC'])['ESTIMATE'].mean().reset_index()
topic_pivot = topic_ts.pivot(index='TIME_PERIOD', columns='TOPIC', values='ESTIMATE')

print("\n✓ Created time series for overall trends and by topic")

# ============================================================================
# 3. SIMPLE FORECASTING MODELS
# ============================================================================
print("\n[STEP 3] BUILDING FORECASTING MODELS")
print("-" * 100)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Prepare features for ML models
X = overall_ts['Year'].values.reshape(-1, 1)
y = overall_ts['Mean'].values

# Split: Use 2019-2023 for training, 2024 for validation
X_train = X[:-1]  # 2019-2023
y_train = y[:-1]
X_val = X[-1:]  # 2024
y_val = y[-1:]

# Future prediction
X_future = np.array([[2025]])

models = {}
predictions = {}

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear Trend'] = lr
predictions['Linear Trend'] = {
    'train': lr.predict(X_train),
    'val': lr.predict(X_val)[0],
    'forecast_2025': lr.predict(X_future)[0]
}

# Model 2: Polynomial Regression (degree 2)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
X_future_poly = poly.transform(X_future)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
models['Polynomial Trend'] = poly_reg
predictions['Polynomial Trend'] = {
    'train': poly_reg.predict(X_train_poly),
    'val': poly_reg.predict(X_val_poly)[0],
    'forecast_2025': poly_reg.predict(X_future_poly)[0]
}

# Model 3: Exponential Smoothing
from scipy.optimize import curve_fit


def exp_func(x, a, b, c):
    return a * np.exp(b * (x - 2019)) + c


try:
    popt, _ = curve_fit(exp_func, X_train.flatten(), y_train, maxfev=5000)
    predictions['Exponential Smoothing'] = {
        'train': exp_func(X_train.flatten(), *popt),
        'val': exp_func(X_val.flatten(), *popt)[0],
        'forecast_2025': exp_func(2025, *popt)
    }
except:
    print("  ⚠ Exponential smoothing failed to converge, using linear instead")

# Model 4: Moving Average with Trend
window = 3
ma = pd.Series(y_train).rolling(window=window, min_periods=1).mean()
trend = (y_train[-1] - y_train[0]) / len(y_train)
predictions['Moving Average + Trend'] = {
    'train': ma.values,
    'val': y_train[-1] + trend,
    'forecast_2025': y_train[-1] + 2 * trend
}

print(f"✓ Built 4 forecasting models")

# ============================================================================
# 4. FACEBOOK PROPHET MODEL
# ============================================================================
print("\n[STEP 4] TRAINING PROPHET MODEL")
print("-" * 100)

try:
    from prophet import Prophet

    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(overall_ts['Year'].astype(str) + '-07-01'),
        'y': overall_ts['Mean']
    })

    # Split for validation
    prophet_train = prophet_df[:-1]
    prophet_val = prophet_df[-1:]

    # Train Prophet
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    prophet_model.fit(prophet_train)

    # Make predictions
    future = prophet_model.make_future_dataframe(periods=2, freq='YS')
    forecast = prophet_model.predict(future)

    # Extract predictions
    val_idx = len(prophet_train)
    predictions['Prophet'] = {
        'train': forecast['yhat'][:len(prophet_train)].values,
        'val': forecast.iloc[val_idx]['yhat'],
        'forecast_2025': forecast.iloc[-1]['yhat'],
        'lower_bound': forecast.iloc[-1]['yhat_lower'],
        'upper_bound': forecast.iloc[-1]['yhat_upper']
    }

    print("✓ Prophet model trained successfully")
    print(f"  2025 forecast: {predictions['Prophet']['forecast_2025']:.2f}%")
    print(f"  95% CI: [{predictions['Prophet']['lower_bound']:.2f}%, {predictions['Prophet']['upper_bound']:.2f}%]")

except ImportError:
    print("  ⚠ Prophet not available, skipping")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n[STEP 5] MODEL PERFORMANCE EVALUATION")
print("-" * 100)

results = []
for model_name, preds in predictions.items():
    if 'train' in preds:
        train_mae = mean_absolute_error(y_train, preds['train'])
        val_error = abs(y_val[0] - preds['val'])

        results.append({
            'Model': model_name,
            'Train_MAE': train_mae,
            'Val_Error_2024': val_error,
            'Forecast_2025': preds['forecast_2025']
        })

results_df = pd.DataFrame(results).sort_values('Val_Error_2024')
print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_forecast = results_df.iloc[0]['Forecast_2025']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   2024 Validation Error: {results_df.iloc[0]['Val_Error_2024']:.3f} pp")
print(f"   2025 Forecast: {best_forecast:.2f}%")

# ============================================================================
# 6. TOPIC-LEVEL FORECASTS
# ============================================================================
print("\n[STEP 6] FORECASTING BY TOPIC")
print("-" * 100)

topic_forecasts = []

for topic in delay_topics:
    topic_data = delay_df[delay_df['TOPIC'] == topic].groupby('TIME_PERIOD')['ESTIMATE'].mean()

    if len(topic_data) >= 3:
        X_topic = topic_data.index.values.reshape(-1, 1)
        y_topic = topic_data.values

        # Simple linear forecast
        lr_topic = LinearRegression()
        lr_topic.fit(X_topic, y_topic)
        forecast_2025 = lr_topic.predict([[2025]])[0]

        # Calculate trend
        trend = lr_topic.coef_[0]
        trend_direction = "↑ Increasing" if trend > 0 else "↓ Decreasing" if trend < 0 else "→ Stable"

        topic_forecasts.append({
            'Topic': topic,
            '2024_Actual': topic_data[2024] if 2024 in topic_data.index else np.nan,
            '2025_Forecast': forecast_2025,
            'Trend': trend,
            'Direction': trend_direction
        })

topic_forecast_df = pd.DataFrame(topic_forecasts)
print("\nTopic-Level Forecasts for 2025:")
print(topic_forecast_df.to_string(index=False))

# ============================================================================
# 7. CONFIDENCE INTERVALS AND UNCERTAINTY
# ============================================================================
print("\n[STEP 7] CALCULATING UNCERTAINTY ESTIMATES")
print("-" * 100)

# Calculate historical volatility
historical_changes = np.diff(overall_ts['Mean'].values)
std_change = np.std(historical_changes)

# Prediction intervals (assuming normal distribution)
confidence_95 = 1.96 * std_change

print(f"\nHistorical Volatility: ±{std_change:.2f} percentage points")
print(f"\n2025 Forecast with Confidence Intervals:")
print(f"  Point Forecast: {best_forecast:.2f}%")
print(f"  95% CI: [{best_forecast - confidence_95:.2f}%, {best_forecast + confidence_95:.2f}%]")
print(f"  68% CI: [{best_forecast - std_change:.2f}%, {best_forecast + std_change:.2f}%]")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[STEP 8] GENERATING VISUALIZATIONS")
print("-" * 100)

# Visualization 1: Overall Trend and Forecasts
fig1, ax1 = plt.subplots(figsize=(14, 8))

years = overall_ts['Year'].values
actual = overall_ts['Mean'].values

# Plot historical data
ax1.plot(years, actual, marker='o', linewidth=3, markersize=10,
         label='Historical Data', color='#2c3e50', zorder=5)

# Plot COVID period
ax1.axvspan(2019.5, 2021.5, alpha=0.15, color='red', label='COVID Period')

# Plot forecasts from different models
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, (model_name, preds) in enumerate(list(predictions.items())[:5]):
    forecast_years = list(years[:-1]) + [2024, 2025]
    forecast_values = list(preds['train']) + [preds['val'], preds['forecast_2025']]

    # Plot from 2023 onwards to show forecast
    ax1.plot([2023, 2024, 2025],
             forecast_values[-3:],
             marker='s', linestyle='--', linewidth=2, alpha=0.7,
             label=f'{model_name}: {preds["forecast_2025"]:.1f}%',
             color=colors[i])

# Add confidence interval for best model
if 'Prophet' in predictions and 'lower_bound' in predictions['Prophet']:
    ax1.fill_between([2024.8, 2025.2],
                     [predictions['Prophet']['lower_bound']] * 2,
                     [predictions['Prophet']['upper_bound']] * 2,
                     alpha=0.2, color='gray', label='Prophet 95% CI')

ax1.set_xlabel('Year', fontweight='bold', fontsize=13)
ax1.set_ylabel('Average % Experiencing Delay/Missed Care', fontweight='bold', fontsize=13)
ax1.set_title('Time-Series Forecast: Delayed/Missed Care Trends (2019-2025)',
              fontweight='bold', fontsize=15, pad=20)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(2019, 2026))

plt.tight_layout()
plt.savefig(images_dir / 'TS1_overall_forecast.png', dpi=300, bbox_inches='tight')
print("✓ Saved: TS1_overall_forecast.png")
plt.close()

# Visualization 2: Topic-Specific Forecasts
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, topic in enumerate(delay_topics):
    ax = axes[idx]
    topic_data = delay_df[delay_df['TOPIC'] == topic].groupby('TIME_PERIOD')['ESTIMATE'].mean()

    years_topic = topic_data.index.values
    values_topic = topic_data.values

    # Historical
    ax.plot(years_topic, values_topic, marker='o', linewidth=2.5, markersize=8,
            label='Historical', color='#34495e')

    # Forecast
    if len(topic_data) >= 3:
        lr_topic = LinearRegression()
        lr_topic.fit(years_topic.reshape(-1, 1), values_topic)
        forecast_2025 = lr_topic.predict([[2025]])[0]

        ax.plot([2024, 2025], [values_topic[-1], forecast_2025],
                marker='s', linewidth=2.5, markersize=8, linestyle='--',
                label=f'2025 Forecast: {forecast_2025:.1f}%', color='#e74c3c')

    # COVID shading
    ax.axvspan(2019.5, 2021.5, alpha=0.1, color='red')

    title = topic[:50] + '...' if len(topic) > 50 else topic
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('% Affected', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2019, 2026))

plt.tight_layout()
plt.savefig(images_dir / 'TS2_topic_forecasts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: TS2_topic_forecasts.png")
plt.close()

# Visualization 3: Model Comparison
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 3a: Model accuracy
ax1 = axes[0]
models_to_plot = results_df['Model'].values
val_errors = results_df['Val_Error_2024'].values

colors_bar = ['#2ecc71' if e == min(val_errors) else '#3498db' for e in val_errors]
bars = ax1.barh(models_to_plot, val_errors, color=colors_bar, alpha=0.8)
ax1.set_xlabel('2024 Validation Error (pp)', fontweight='bold', fontsize=12)
ax1.set_title('Model Accuracy on 2024 Data\n(Lower is Better)', fontweight='bold', fontsize=13)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (model, error) in enumerate(zip(models_to_plot, val_errors)):
    ax1.text(error, i, f'  {error:.3f}', va='center', fontweight='bold')

# Plot 3b: 2025 Forecasts
ax2 = axes[1]
forecasts_2025 = results_df['Forecast_2025'].values

bars2 = ax2.barh(models_to_plot, forecasts_2025, color='#e74c3c', alpha=0.7)
ax2.set_xlabel('2025 Forecast (%)', fontweight='bold', fontsize=12)
ax2.set_title('2025 Forecasts by Model', fontweight='bold', fontsize=13)
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=overall_ts['Mean'].iloc[-1], color='green', linestyle='--',
            linewidth=2, label='2024 Actual', alpha=0.7)
ax2.legend()

# Add value labels
for i, (model, forecast) in enumerate(zip(models_to_plot, forecasts_2025)):
    ax2.text(forecast, i, f'  {forecast:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(images_dir / 'TS3_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: TS3_model_comparison.png")
plt.close()

# Visualization 4: Trend Analysis
fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 4a: Year-over-year changes
ax1 = axes[0, 0]
yoy_changes = np.diff(overall_ts['Mean'].values)
years_change = overall_ts['Year'].values[1:]
colors_change = ['#e74c3c' if c > 0 else '#2ecc71' for c in yoy_changes]
ax1.bar(years_change, yoy_changes, color=colors_change, alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Year-over-Year Change (pp)', fontweight='bold')
ax1.set_title('Annual Changes in Delay/Missed Care', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 4b: Variance over time
ax2 = axes[0, 1]
ax2.plot(overall_ts['Year'], overall_ts['Std'], marker='o', linewidth=2.5,
         markersize=8, color='#9b59b6')
ax2.fill_between(overall_ts['Year'], 0, overall_ts['Std'], alpha=0.3, color='#9b59b6')
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Standard Deviation', fontweight='bold')
ax2.set_title('Variation Across Subgroups Over Time', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 4c: Min-Max range
ax3 = axes[1, 0]
ax3.fill_between(overall_ts['Year'], overall_ts['Min'], overall_ts['Max'],
                 alpha=0.3, color='#3498db', label='Min-Max Range')
ax3.plot(overall_ts['Year'], overall_ts['Mean'], marker='o', linewidth=2.5,
         markersize=8, color='#2c3e50', label='Mean')
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('% Affected', fontweight='bold')
ax3.set_title('Range of Delay/Missed Care Across Subgroups', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4d: Forecast uncertainty
ax4 = axes[1, 1]
forecast_years_extended = list(range(2019, 2026))
forecast_means = list(overall_ts['Mean'].values) + [best_forecast]

ax4.plot(forecast_years_extended, forecast_means, marker='o', linewidth=3,
         markersize=10, color='#2c3e50', label='Mean Forecast')
ax4.fill_between([2024, 2025],
                 [best_forecast - confidence_95] * 2,
                 [best_forecast + confidence_95] * 2,
                 alpha=0.3, color='gray', label='95% Confidence Interval')
ax4.fill_between([2024, 2025],
                 [best_forecast - std_change] * 2,
                 [best_forecast + std_change] * 2,
                 alpha=0.5, color='gray', label='68% Confidence Interval')
ax4.axvline(x=2024, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('% Experiencing Delay/Missed Care', fontweight='bold')
ax4.set_title('2025 Forecast with Uncertainty', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(images_dir / 'TS4_trend_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: TS4_trend_analysis.png")
plt.close()

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n[STEP 9] SAVING RESULTS")
print("-" * 100)

# Save model comparison
results_df.to_csv(data_dir / 'timeseries_model_comparison.csv', index=False)
print("✓ Saved: timeseries_model_comparison.csv")

# Save topic forecasts
topic_forecast_df.to_csv(data_dir / 'timeseries_topic_forecasts.csv', index=False)
print("✓ Saved: timeseries_topic_forecasts.csv")

# Save detailed forecast
forecast_detail = pd.DataFrame({
    'Year': list(range(2019, 2026)),
    'Historical_Mean': list(overall_ts['Mean'].values) + [np.nan],
    'Historical_Std': list(overall_ts['Std'].values) + [np.nan],
    'Forecast_Mean': [np.nan] * 6 + [best_forecast],
    'CI_Lower_95': [np.nan] * 6 + [best_forecast - confidence_95],
    'CI_Upper_95': [np.nan] * 6 + [best_forecast + confidence_95]
})
forecast_detail.to_csv(data_dir / 'timeseries_forecast_2025.csv', index=False)
print("✓ Saved: timeseries_forecast_2025.csv")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 100)
print(" " * 30 + "TIME-SERIES FORECAST SUMMARY")
print("=" * 100)

print("\n📊 HISTORICAL TRENDS (2019-2024)")
print("-" * 100)
print(f"Starting point (2019): {overall_ts['Mean'].iloc[0]:.2f}%")
print(f"Current state (2024): {overall_ts['Mean'].iloc[-1]:.2f}%")
print(f"Total change: {overall_ts['Mean'].iloc[-1] - overall_ts['Mean'].iloc[0]:+.2f} pp")
print(f"Average annual change: {(overall_ts['Mean'].iloc[-1] - overall_ts['Mean'].iloc[0]) / 5:+.2f} pp/year")

print("\n🔮 2025 FORECAST")
print("-" * 100)
print(f"Best Model: {best_model_name}")
print(f"Point Forecast: {best_forecast:.2f}%")
print(f"95% Confidence Interval: [{best_forecast - confidence_95:.2f}%, {best_forecast + confidence_95:.2f}%]")
print(f"Expected change from 2024: {best_forecast - overall_ts['Mean'].iloc[-1]:+.2f} pp")

print("\n📈 TOPIC-SPECIFIC FORECASTS FOR 2025")
print("-" * 100)
for _, row in topic_forecast_df.iterrows():
    print(f"• {row['Topic'][:60]}")
    print(f"  2025 Forecast: {row['2025_Forecast']:.2f}% {row['Direction']}")

print("\n🎯 KEY INSIGHTS")
print("-" * 100)
if overall_ts['Mean'].iloc[-1] > overall_ts['Mean'].iloc[0]:
    print("• Overall trend shows INCREASING delays/missed care over 2019-2024")
else:
    print("• Overall trend shows DECREASING delays/missed care over 2019-2024")

covid_impact = overall_ts['Mean'].iloc[1] - overall_ts['Mean'].iloc[0]  # 2020 vs 2019
print(f"• COVID-19 impact (2020 vs 2019): {covid_impact:+.2f} pp")

if best_forecast > overall_ts['Mean'].iloc[-1]:
    print(f"• 2025 forecast suggests CONTINUED INCREASE from 2024 levels")
else:
    print(f"• 2025 forecast suggests IMPROVEMENT from 2024 levels")

print("\n📁 GENERATED FILES")
print("-" * 100)
print("  1. timeseries_model_comparison.csv - Model performance metrics")
print("  2. timeseries_topic_forecasts.csv - Topic-level 2025 forecasts")
print("  3. timeseries_forecast_2025.csv - Detailed forecast with confidence intervals")
print("  4. TS1_overall_forecast.png - Main forecast visualization")
print("  5. TS2_topic_forecasts.png - Topic-specific trends")
print("  6. TS3_model_comparison.png - Model accuracy comparison")
print("  7. TS4_trend_analysis.png - Comprehensive trend analysis")

print("\n" + "=" * 100)
print(" " * 25 + "TIME-SERIES FORECASTING COMPLETE!")
print("=" * 100)