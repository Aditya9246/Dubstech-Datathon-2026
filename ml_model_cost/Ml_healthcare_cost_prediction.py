"""
COMPREHENSIVE ML MODEL: Healthcare Cost Barrier Prediction
Predicts which subgroups are affected by healthcare cost barriers
Using multiple ML algorithms with 70-30 train-test split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')

# Set style for presentations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("=" * 100)
print(" " * 25 + "HEALTHCARE COST BARRIER ML PREDICTION MODEL")
print("=" * 100)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[STEP 1] LOADING AND PREPROCESSING DATA")
print("-" * 100)

# Load data
df = pd.read_csv('../data/NHIS_Data_Cleaned.csv')
print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Filter for cost-related topics
cost_topics = [
    'Delayed getting medical care due to cost among adults',
    'Did not get needed medical care due to cost',
    'Did not get needed mental health care due to cost'
]

cost_df = df[df['TOPIC'].isin(cost_topics)].copy()
print(f"✓ Filtered to cost barriers: {len(cost_df):,} rows")

# Remove rows with missing estimates
cost_df = cost_df.dropna(subset=['ESTIMATE'])
print(f"✓ After removing nulls: {len(cost_df):,} rows")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] FEATURE ENGINEERING")
print("-" * 100)

# Create features from existing columns
features_df = cost_df.copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['TOPIC', 'CLASSIFICATION', 'GROUP', 'SUBGROUP']

for col in categorical_cols:
    le = LabelEncoder()
    features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} unique categories")

# Create temporal features
features_df['is_covid_period'] = (features_df['TIME_PERIOD'].isin([2020, 2021])).astype(int)
features_df['is_post_covid'] = (features_df['TIME_PERIOD'] >= 2022).astype(int)
features_df['years_since_2019'] = features_df['TIME_PERIOD'] - 2019

# Calculate confidence interval width (measure of uncertainty)
features_df['ci_width'] = features_df['ESTIMATE_UCI'] - features_df['ESTIMATE_LCI']

# Group statistics - average barrier rate by subgroup over time
subgroup_avg = features_df.groupby('SUBGROUP')['ESTIMATE'].mean()
features_df['subgroup_avg_barrier'] = features_df['SUBGROUP'].map(subgroup_avg)

# Year statistics - overall trend
year_avg = features_df.groupby('TIME_PERIOD')['ESTIMATE'].mean()
features_df['year_avg_barrier'] = features_df['TIME_PERIOD'].map(year_avg)

print(f"✓ Created temporal features (COVID indicators, time trends)")
print(f"✓ Created statistical features (subgroup averages, CI width)")

# ============================================================================
# 3. PREPARE FEATURE MATRIX AND TARGET
# ============================================================================
print("\n[STEP 3] PREPARING FEATURE MATRIX")
print("-" * 100)

# Select features for modeling
feature_columns = [
    'TOPIC_encoded',
    'CLASSIFICATION_encoded',
    'GROUP_encoded',
    'SUBGROUP_encoded',
    'TIME_PERIOD',
    'years_since_2019',
    'is_covid_period',
    'is_post_covid',
    'ci_width',
    'subgroup_avg_barrier',
    'year_avg_barrier'
]

X = features_df[feature_columns].copy()
y = features_df['ESTIMATE'].copy()

# Handle any NaN values
print(f"✓ Checking for missing values...")
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    print(f"  Found {nan_counts.sum()} missing values, filling with median...")
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
else:
    print(f"  No missing values found!")

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target variable shape: {y.shape}")
print(f"\nFeatures being used:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 4. TRAIN-TEST SPLIT (70-30)
# ============================================================================
print("\n[STEP 4] SPLITTING DATA (70% TRAIN / 30% TEST)")
print("-" * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)

print(f"✓ Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"✓ Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")

# ============================================================================
# 5. BUILD AND TRAIN MULTIPLE ML MODELS
# ============================================================================
print("\n[STEP 5] TRAINING MACHINE LEARNING MODELS")
print("-" * 100)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1, max_iter=5000),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...", end=" ")

    # Use scaled data for linear models, original for tree-based
    if 'Regression' in name:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }

    print(f"✓ Done!")
    print(f"     Train MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f} | Test R²: {test_r2:.3f}")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n[STEP 6] MODEL PERFORMANCE COMPARISON")
print("-" * 100)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train_MAE': [r['train_mae'] for r in results.values()],
    'Test_MAE': [r['test_mae'] for r in results.values()],
    'Train_RMSE': [r['train_rmse'] for r in results.values()],
    'Test_RMSE': [r['test_rmse'] for r in results.values()],
    'Train_R2': [r['train_r2'] for r in results.values()],
    'Test_R2': [r['test_r2'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('Test_MAE')
print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test MAE:  {comparison_df.iloc[0]['Test_MAE']:.3f} percentage points")
print(f"   Test R²:   {comparison_df.iloc[0]['Test_R2']:.3f}")

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[STEP 7] FEATURE IMPORTANCE ANALYSIS")
print("-" * 100)

# Get feature importance from Random Forest
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 8. CREATE PRESENTATION-READY VISUALIZATIONS
# ============================================================================
print("\n[STEP 8] GENERATING PRESENTATION VISUALIZATIONS")
print("-" * 100)

# VISUALIZATION 1: Model Performance Comparison
fig1, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1a: MAE Comparison
ax1 = axes[0]
x_pos = np.arange(len(comparison_df))
width = 0.35
ax1.bar(x_pos - width / 2, comparison_df['Train_MAE'], width, label='Train MAE', alpha=0.8, color='#3498db')
ax1.bar(x_pos + width / 2, comparison_df['Test_MAE'], width, label='Test MAE', alpha=0.8, color='#e74c3c')
ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
ax1.set_ylabel('Mean Absolute Error (pp)', fontweight='bold', fontsize=12)
ax1.set_title('Model Comparison: MAE (Lower is Better)', fontweight='bold', fontsize=13)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 1b: R² Comparison
ax2 = axes[1]
ax2.bar(x_pos - width / 2, comparison_df['Train_R2'], width, label='Train R²', alpha=0.8, color='#2ecc71')
ax2.bar(x_pos + width / 2, comparison_df['Test_R2'], width, label='Test R²', alpha=0.8, color='#f39c12')
ax2.set_xlabel('Model', fontweight='bold', fontsize=12)
ax2.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax2.set_title('Model Comparison: R² (Higher is Better)', fontweight='bold', fontsize=13)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 1c: Overfitting Check
ax3 = axes[2]
overfit = comparison_df['Train_MAE'] - comparison_df['Test_MAE']
colors = ['#2ecc71' if x > -1 else '#e74c3c' for x in overfit]
ax3.barh(comparison_df['Model'], overfit, color=colors, alpha=0.8)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Train MAE - Test MAE', fontweight='bold', fontsize=12)
ax3.set_title('Overfitting Analysis\n(Negative = Good Generalization)', fontweight='bold', fontsize=13)
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('1_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 1_model_performance_comparison.png")
plt.close()

# VISUALIZATION 2: Best Model - Actual vs Predicted
fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 2a: Scatter plot
ax1 = axes[0]
scatter = ax1.scatter(y_test, best_predictions, alpha=0.5, s=50, c=y_test, cmap='viridis')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Cost Barrier (%)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Predicted Cost Barrier (%)', fontweight='bold', fontsize=12)
ax1.set_title(f'Best Model: {best_model_name}\nActual vs Predicted Values', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Actual Value')

# Add metrics text
textstr = f'MAE: {comparison_df.iloc[0]["Test_MAE"]:.3f}\nRMSE: {comparison_df.iloc[0]["Test_RMSE"]:.3f}\nR²: {comparison_df.iloc[0]["Test_R2"]:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# Plot 2b: Residuals
ax2 = axes[1]
residuals = y_test - best_predictions
ax2.scatter(best_predictions, residuals, alpha=0.5, s=50, c=residuals, cmap='RdYlGn_r')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Cost Barrier (%)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold', fontsize=12)
ax2.set_title('Residual Plot\n(Points should be randomly scattered around 0)', fontweight='bold', fontsize=13)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('2_best_model_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_best_model_predictions.png")
plt.close()

# VISUALIZATION 3: Feature Importance
fig3, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(10)
colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors_grad)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Feature Importance Score', fontweight='bold', fontsize=12)
ax.set_title('Top 10 Most Important Features\n(Random Forest Model)', fontweight='bold', fontsize=14)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['Importance'], i, f' {row["Importance"]:.4f}',
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_feature_importance.png")
plt.close()

# VISUALIZATION 4: Prediction Error Distribution
fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 4a: Error histogram
ax1 = axes[0, 0]
errors = y_test - best_predictions
ax1.hist(errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='r', linestyle='--', lw=2, label=f'Mean Error: {errors.mean():.3f}')
ax1.set_xlabel('Prediction Error (Actual - Predicted)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax1.set_title('Distribution of Prediction Errors', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 4b: Error by actual value
ax2 = axes[0, 1]
ax2.scatter(y_test, np.abs(errors), alpha=0.5, s=40, c='#e74c3c')
ax2.set_xlabel('Actual Cost Barrier (%)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Absolute Error', fontweight='bold', fontsize=11)
ax2.set_title('Absolute Error vs Actual Value', fontweight='bold', fontsize=12)
ax2.grid(alpha=0.3)

# Plot 4c: Predictions by time period
ax3 = axes[1, 0]
test_data = features_df.loc[y_test.index].copy()
test_data['prediction'] = best_predictions
test_data['actual'] = y_test

time_actual = test_data.groupby('TIME_PERIOD')['actual'].mean()
time_pred = test_data.groupby('TIME_PERIOD')['prediction'].mean()

x_time = time_actual.index
ax3.plot(x_time, time_actual.values, marker='o', linewidth=3, markersize=10,
         label='Actual', color='#2ecc71')
ax3.plot(x_time, time_pred.values, marker='s', linewidth=3, markersize=10,
         label='Predicted', color='#e74c3c', linestyle='--')
ax3.fill_between(x_time, time_actual.values, time_pred.values, alpha=0.3, color='gray')
ax3.axvspan(2019.5, 2021.5, alpha=0.15, color='red', label='COVID Period')
ax3.set_xlabel('Year', fontweight='bold', fontsize=11)
ax3.set_ylabel('Average Cost Barrier (%)', fontweight='bold', fontsize=11)
ax3.set_title('Model Performance Over Time', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4d: Top/Bottom predictions
ax4 = axes[1, 1]
test_data['abs_error'] = np.abs(test_data['actual'] - test_data['prediction'])
best_preds = test_data.nsmallest(5, 'abs_error')[['SUBGROUP', 'actual', 'prediction', 'abs_error']]
worst_preds = test_data.nlargest(5, 'abs_error')[['SUBGROUP', 'actual', 'prediction', 'abs_error']]

y_pos = np.arange(10)
combined = pd.concat([best_preds, worst_preds])
colors_bar = ['#2ecc71'] * 5 + ['#e74c3c'] * 5

ax4.barh(y_pos, combined['abs_error'], color=colors_bar, alpha=0.7)
ax4.set_yticks(y_pos)
labels = [s[:30] + '...' if len(s) > 30 else s for s in combined['SUBGROUP'].values]
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('Absolute Prediction Error', fontweight='bold', fontsize=11)
ax4.set_title('Best & Worst Predictions\n(Green=Best, Red=Worst)', fontweight='bold', fontsize=12)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('4_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 4_prediction_error_analysis.png")
plt.close()

# VISUALIZATION 5: Subgroup-level predictions
fig5, ax = plt.subplots(figsize=(14, 10))

# Get predictions for each subgroup (average across test set)
subgroup_results = test_data.groupby('SUBGROUP').agg({
    'actual': 'mean',
    'prediction': 'mean',
    'abs_error': 'mean'
}).reset_index()

subgroup_results = subgroup_results.sort_values('prediction', ascending=False).head(20)

y_pos = np.arange(len(subgroup_results))
width = 0.35

bars1 = ax.barh(y_pos - width / 2, subgroup_results['actual'], width,
                label='Actual', alpha=0.8, color='#3498db')
bars2 = ax.barh(y_pos + width / 2, subgroup_results['prediction'], width,
                label='Predicted', alpha=0.8, color='#e74c3c')

ax.set_yticks(y_pos)
labels = [s[:45] + '...' if len(s) > 45 else s for s in subgroup_results['SUBGROUP'].values]
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Average Cost Barrier (%)', fontweight='bold', fontsize=12)
ax.set_title('Top 20 Subgroups: Actual vs Predicted Cost Barriers\n(Test Set)',
             fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('5_subgroup_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 5_subgroup_predictions.png")
plt.close()

# ============================================================================
# 9. SAVE DETAILED RESULTS
# ============================================================================
print("\n[STEP 9] SAVING DETAILED RESULTS")
print("-" * 100)

# Save model comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("✓ Saved: model_comparison.csv")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# Save test set predictions with actual values
test_results = features_df.loc[y_test.index].copy()
test_results['Actual_Barrier'] = y_test.values
test_results['Predicted_Barrier'] = best_predictions
test_results['Absolute_Error'] = np.abs(y_test.values - best_predictions)
test_results['Percentage_Error'] = (test_results['Absolute_Error'] / test_results['Actual_Barrier'] * 100)

output_cols = ['TOPIC', 'SUBGROUP', 'TIME_PERIOD', 'Actual_Barrier',
               'Predicted_Barrier', 'Absolute_Error', 'Percentage_Error']
test_results[output_cols].to_csv('test_set_predictions.csv', index=False)
print("✓ Saved: test_set_predictions.csv")

# ============================================================================
# 10. GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 100)
print(" " * 35 + "FINAL SUMMARY REPORT")
print("=" * 100)

print("\n📊 DATASET STATISTICS")
print("-" * 100)
print(f"Total samples:           {len(X):,}")
print(f"Training samples:        {len(X_train):,} (70%)")
print(f"Test samples:            {len(X_test):,} (30%)")
print(f"Number of features:      {X.shape[1]}")
print(f"Time period:             {features_df['TIME_PERIOD'].min()} - {features_df['TIME_PERIOD'].max()}")
print(f"Unique subgroups:        {features_df['SUBGROUP'].nunique()}")

print("\n🏆 BEST MODEL PERFORMANCE")
print("-" * 100)
print(f"Model:                   {best_model_name}")
print(f"Test MAE:                {comparison_df.iloc[0]['Test_MAE']:.3f} percentage points")
print(f"Test RMSE:               {comparison_df.iloc[0]['Test_RMSE']:.3f} percentage points")
print(f"Test R²:                 {comparison_df.iloc[0]['Test_R2']:.3f}")
print(f"\nInterpretation:")
print(f"  • The model predictions are off by an average of {comparison_df.iloc[0]['Test_MAE']:.2f} percentage points")
print(f"  • The model explains {comparison_df.iloc[0]['Test_R2'] * 100:.1f}% of the variance in cost barriers")

print("\n📈 TOP 5 MOST IMPORTANT FEATURES")
print("-" * 100)
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']:30s} → {row['Importance']:.4f}")

print("\n🎯 KEY INSIGHTS")
print("-" * 100)
print(f"  • Average actual cost barrier in test set: {y_test.mean():.2f}%")
print(f"  • Average predicted cost barrier: {best_predictions.mean():.2f}%")
print(f"  • Minimum barrier observed: {y_test.min():.2f}%")
print(f"  • Maximum barrier observed: {y_test.max():.2f}%")
print(f"  • Standard deviation of errors: {np.std(errors):.3f} pp")

print("\n📁 GENERATED FILES FOR DATATHON PRESENTATION")
print("-" * 100)
print("  1. model_comparison.csv               - Performance metrics for all models")
print("  2. feature_importance.csv             - Feature importance rankings")
print("  3. test_set_predictions.csv           - Detailed predictions on test set")
print("  4. 1_model_performance_comparison.png - Model comparison charts")
print("  5. 2_best_model_predictions.png       - Actual vs predicted scatter plots")
print("  6. 3_feature_importance.png           - Feature importance visualization")
print("  7. 4_prediction_error_analysis.png    - Error distribution and analysis")
print("  8. 5_subgroup_predictions.png         - Subgroup-level predictions")

print("\n" + "=" * 100)
print(" " * 30 + "ANALYSIS COMPLETE - READY FOR PRESENTATION!")
print("=" * 100)