"""
TASK 2: CLUSTERING & ANOMALY DETECTION
Identify subgroups most at risk of falling through the cracks

This model uses multiple techniques:
1. K-Means Clustering - Group similar subgroups
2. Hierarchical Clustering - Dendrogram-based grouping
3. DBSCAN - Density-based clustering for outliers
4. Isolation Forest - Anomaly detection
5. Local Outlier Factor - Anomaly detection based on local density
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print(" " * 15 + "CLUSTERING & ANOMALY DETECTION: AT-RISK SUBGROUP IDENTIFICATION")
print("=" * 100)

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 1] LOADING AND ENGINEERING FEATURES FOR CLUSTERING")
print("-" * 100)

df = pd.read_csv('../data/NHIS_Data_Cleaned.csv')

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
print(f"✓ Unique subgroups: {delay_df['SUBGROUP'].nunique()}")

# ============================================================================
# 2. CREATE FEATURE MATRIX FOR EACH SUBGROUP
# ============================================================================
print("\n[STEP 2] CREATING FEATURE MATRIX")
print("-" * 100)

# Aggregate features for each subgroup
subgroup_features = []

for subgroup in delay_df['SUBGROUP'].unique():
    sg_data = delay_df[delay_df['SUBGROUP'] == subgroup]

    # Overall statistics
    mean_barrier = sg_data['ESTIMATE'].mean()
    max_barrier = sg_data['ESTIMATE'].max()
    min_barrier = sg_data['ESTIMATE'].min()
    std_barrier = sg_data['ESTIMATE'].std()

    # Temporal features
    if 2024 in sg_data['TIME_PERIOD'].values:
        current_2024 = sg_data[sg_data['TIME_PERIOD'] == 2024]['ESTIMATE'].mean()
    else:
        current_2024 = mean_barrier

    if 2019 in sg_data['TIME_PERIOD'].values:
        baseline_2019 = sg_data[sg_data['TIME_PERIOD'] == 2019]['ESTIMATE'].mean()
    else:
        baseline_2019 = mean_barrier

    change_2019_2024 = current_2024 - baseline_2019

    # COVID impact
    covid_years = sg_data[sg_data['TIME_PERIOD'].isin([2020, 2021])]
    covid_avg = covid_years['ESTIMATE'].mean() if len(covid_years) > 0 else mean_barrier
    covid_impact = covid_avg - baseline_2019

    # Trend (simple linear)
    years = sg_data['TIME_PERIOD'].values
    estimates = sg_data['ESTIMATE'].values
    if len(years) > 1:
        trend = np.polyfit(years, estimates, 1)[0]
    else:
        trend = 0

    # Volatility
    volatility = np.std(np.diff(sorted(estimates))) if len(estimates) > 1 else 0

    # Recovery (post-COVID)
    post_covid = sg_data[sg_data['TIME_PERIOD'].isin([2022, 2023, 2024])]
    post_covid_avg = post_covid['ESTIMATE'].mean() if len(post_covid) > 0 else current_2024
    recovery = covid_avg - post_covid_avg  # Positive = improved

    # Number of topics affected
    num_topics = sg_data['TOPIC'].nunique()

    # Confidence interval width (uncertainty)
    avg_ci_width = (sg_data['ESTIMATE_UCI'] - sg_data['ESTIMATE_LCI']).mean()

    subgroup_features.append({
        'Subgroup': subgroup,
        'Mean_Barrier': mean_barrier,
        'Current_2024': current_2024,
        'Baseline_2019': baseline_2019,
        'Change_2019_2024': change_2019_2024,
        'Max_Barrier': max_barrier,
        'Min_Barrier': min_barrier,
        'Std_Barrier': std_barrier,
        'COVID_Impact': covid_impact,
        'Trend': trend,
        'Volatility': volatility,
        'Recovery': recovery,
        'Num_Topics': num_topics,
        'CI_Width': avg_ci_width
    })

features_df = pd.DataFrame(subgroup_features)
print(f"✓ Created feature matrix: {features_df.shape[0]} subgroups × {features_df.shape[1] - 1} features")

# Display feature summary
print("\nFeature Statistics:")
print(features_df.describe().round(2).to_string())

# ============================================================================
# 3. FEATURE SCALING AND DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[STEP 3] SCALING FEATURES AND DIMENSIONALITY REDUCTION")
print("-" * 100)

# Prepare feature matrix
feature_cols = [col for col in features_df.columns if col != 'Subgroup']
X = features_df[feature_cols].values
subgroup_names = features_df['Subgroup'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Scaled {len(feature_cols)} features")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(
    f"✓ PCA variance explained: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {pca.explained_variance_ratio_.sum():.1%}")

# ============================================================================
# 4. K-MEANS CLUSTERING
# ============================================================================
print("\n[STEP 4] K-MEANS CLUSTERING")
print("-" * 100)

# Find optimal k using elbow method and silhouette score
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose optimal k (highest silhouette score)
optimal_k = k_range[np.argmax(silhouettes)]
print(f"✓ Optimal number of clusters (by silhouette): {optimal_k}")

# Fit final K-Means
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

features_df['KMeans_Cluster'] = kmeans_labels

print(f"✓ K-Means clustering complete")
print(f"  Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"  Davies-Bouldin Score: {davies_bouldin_score(X_scaled, kmeans_labels):.3f}")

# Cluster summary
print("\nCluster Distribution:")
for i in range(optimal_k):
    cluster_size = (kmeans_labels == i).sum()
    cluster_mean_barrier = features_df[kmeans_labels == i]['Mean_Barrier'].mean()
    print(f"  Cluster {i}: {cluster_size} subgroups, Avg barrier: {cluster_mean_barrier:.2f}%")

# ============================================================================
# 5. HIERARCHICAL CLUSTERING
# ============================================================================
print("\n[STEP 5] HIERARCHICAL CLUSTERING")
print("-" * 100)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

features_df['Hierarchical_Cluster'] = hierarchical_labels

print(f"✓ Hierarchical clustering complete")
print(f"  Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.3f}")

# ============================================================================
# 6. DBSCAN - DENSITY-BASED CLUSTERING
# ============================================================================
print("\n[STEP 6] DBSCAN CLUSTERING")
print("-" * 100)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_scaled)

features_df['DBSCAN_Cluster'] = dbscan_labels

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers_dbscan = list(dbscan_labels).count(-1)

print(f"✓ DBSCAN clustering complete")
print(f"  Number of clusters: {n_clusters_dbscan}")
print(f"  Number of outliers: {n_outliers_dbscan}")

# ============================================================================
# 7. ANOMALY DETECTION - ISOLATION FOREST
# ============================================================================
print("\n[STEP 7] ANOMALY DETECTION - ISOLATION FOREST")
print("-" * 100)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.15, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.score_samples(X_scaled)

features_df['IsoForest_Anomaly'] = iso_predictions  # -1 = anomaly, 1 = normal
features_df['IsoForest_Score'] = iso_scores

n_anomalies_iso = (iso_predictions == -1).sum()
print(f"✓ Isolation Forest complete")
print(f"  Anomalies detected: {n_anomalies_iso} ({n_anomalies_iso / len(features_df) * 100:.1f}%)")

# ============================================================================
# 8. ANOMALY DETECTION - LOCAL OUTLIER FACTOR
# ============================================================================
print("\n[STEP 8] ANOMALY DETECTION - LOCAL OUTLIER FACTOR")
print("-" * 100)

# Local Outlier Factor
lof = LocalOutlierFactor(contamination=0.15, novelty=False)
lof_predictions = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

features_df['LOF_Anomaly'] = lof_predictions  # -1 = anomaly, 1 = normal
features_df['LOF_Score'] = lof_scores

n_anomalies_lof = (lof_predictions == -1).sum()
print(f"✓ Local Outlier Factor complete")
print(f"  Anomalies detected: {n_anomalies_lof} ({n_anomalies_lof / len(features_df) * 100:.1f}%)")

# ============================================================================
# 9. IDENTIFY AT-RISK SUBGROUPS
# ============================================================================
print("\n[STEP 9] IDENTIFYING AT-RISK SUBGROUPS")
print("-" * 100)

# Combine anomaly detection results
features_df['Anomaly_Count'] = (
        (features_df['IsoForest_Anomaly'] == -1).astype(int) +
        (features_df['LOF_Anomaly'] == -1).astype(int) +
        (features_df['DBSCAN_Cluster'] == -1).astype(int)
)

# Define risk score (composite metric)
features_df['Risk_Score'] = (
        0.3 * features_df['Mean_Barrier'] +
        0.2 * features_df['Current_2024'] +
        0.2 * features_df['Change_2019_2024'] +
        0.1 * features_df['COVID_Impact'] +
        0.1 * -features_df['Recovery'] +  # Negative recovery = worse
        0.1 * features_df['Volatility']
)

# Normalize risk score to 0-100
features_df['Risk_Score_Normalized'] = (
        (features_df['Risk_Score'] - features_df['Risk_Score'].min()) /
        (features_df['Risk_Score'].max() - features_df['Risk_Score'].min()) * 100
)

# Identify high-risk groups
high_risk_threshold = features_df['Risk_Score_Normalized'].quantile(0.75)
features_df['High_Risk'] = features_df['Risk_Score_Normalized'] > high_risk_threshold

# Top at-risk subgroups
top_at_risk = features_df.nlargest(20, 'Risk_Score_Normalized')[
    ['Subgroup', 'Mean_Barrier', 'Current_2024', 'Risk_Score_Normalized',
     'Anomaly_Count', 'KMeans_Cluster']
]

print("\nTop 20 At-Risk Subgroups:")
print(top_at_risk.to_string(index=False))

# Anomaly summary
anomalies = features_df[features_df['Anomaly_Count'] >= 2]
print(f"\n✓ High-confidence anomalies (detected by 2+ methods): {len(anomalies)}")

if len(anomalies) > 0:
    print("\nHigh-Confidence Anomalies:")
    anomaly_display = anomalies[['Subgroup', 'Mean_Barrier', 'Current_2024',
                                 'Anomaly_Count', 'Risk_Score_Normalized']].sort_values(
        'Risk_Score_Normalized', ascending=False)
    print(anomaly_display.head(10).to_string(index=False))

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n[STEP 10] GENERATING VISUALIZATIONS")
print("-" * 100)

# Visualization 1: PCA Clustering
fig1, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1a: K-Means clusters
ax1 = axes[0, 0]
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels,
                       cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax1.set_xlabel('First Principal Component', fontweight='bold', fontsize=11)
ax1.set_ylabel('Second Principal Component', fontweight='bold', fontsize=11)
ax1.set_title(f'K-Means Clustering (k={optimal_k})\nSilhouette: {silhouette_score(X_scaled, kmeans_labels):.3f}',
              fontweight='bold', fontsize=12)
plt.colorbar(scatter1, ax=ax1, label='Cluster')
ax1.grid(True, alpha=0.3)

# Plot 1b: Hierarchical clusters
ax2 = axes[0, 1]
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels,
                       cmap='plasma', s=100, alpha=0.6, edgecolors='black')
ax2.set_xlabel('First Principal Component', fontweight='bold', fontsize=11)
ax2.set_ylabel('Second Principal Component', fontweight='bold', fontsize=11)
ax2.set_title(
    f'Hierarchical Clustering (k={optimal_k})\nSilhouette: {silhouette_score(X_scaled, hierarchical_labels):.3f}',
    fontweight='bold', fontsize=12)
plt.colorbar(scatter2, ax=ax2, label='Cluster')
ax2.grid(True, alpha=0.3)

# Plot 1c: DBSCAN
ax3 = axes[1, 0]
colors_dbscan = ['red' if l == -1 else 'blue' for l in dbscan_labels]
scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_dbscan,
                       s=100, alpha=0.6, edgecolors='black')
ax3.set_xlabel('First Principal Component', fontweight='bold', fontsize=11)
ax3.set_ylabel('Second Principal Component', fontweight='bold', fontsize=11)
ax3.set_title(f'DBSCAN Clustering\nOutliers (red): {n_outliers_dbscan}',
              fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Plot 1d: Anomaly Detection
ax4 = axes[1, 1]
colors_anomaly = ['red' if a >= 2 else 'blue' for a in features_df['Anomaly_Count']]
scatter4 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_anomaly,
                       s=100, alpha=0.6, edgecolors='black')
ax4.set_xlabel('First Principal Component', fontweight='bold', fontsize=11)
ax4.set_ylabel('Second Principal Component', fontweight='bold', fontsize=11)
ax4.set_title(f'Anomaly Detection (Consensus)\nAnomalies (red): {(features_df["Anomaly_Count"] >= 2).sum()}',
              fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('CLUST1_clustering_pca.png', dpi=300, bbox_inches='tight')
print("✓ Saved: CLUST1_clustering_pca.png")
plt.close()

# Visualization 2: Risk Scores
fig2, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 2a: Risk score distribution
ax1 = axes[0, 0]
ax1.hist(features_df['Risk_Score_Normalized'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.axvline(high_risk_threshold, color='red', linestyle='--', linewidth=2,
            label=f'High Risk Threshold: {high_risk_threshold:.1f}')
ax1.set_xlabel('Risk Score (0-100)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Number of Subgroups', fontweight='bold', fontsize=11)
ax1.set_title('Distribution of Risk Scores', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2b: Top 15 at-risk subgroups
ax2 = axes[0, 1]
top_15 = features_df.nlargest(15, 'Risk_Score_Normalized')
y_pos = np.arange(len(top_15))
colors_risk = plt.cm.RdYlGn_r(top_15['Risk_Score_Normalized'] / 100)
ax2.barh(y_pos, top_15['Risk_Score_Normalized'], color=colors_risk, alpha=0.8)
ax2.set_yticks(y_pos)
labels = [s[:40] + '...' if len(s) > 40 else s for s in top_15['Subgroup']]
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel('Risk Score', fontweight='bold', fontsize=11)
ax2.set_title('Top 15 At-Risk Subgroups', fontweight='bold', fontsize=12)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Plot 2c: Risk vs Current Barrier
ax3 = axes[1, 0]
scatter_risk = ax3.scatter(features_df['Current_2024'], features_df['Risk_Score_Normalized'],
                           c=features_df['Anomaly_Count'], cmap='Reds', s=100, alpha=0.6,
                           edgecolors='black')
ax3.set_xlabel('Current 2024 Barrier (%)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Risk Score', fontweight='bold', fontsize=11)
ax3.set_title('Risk Score vs Current Barrier Level', fontweight='bold', fontsize=12)
plt.colorbar(scatter_risk, ax=ax3, label='Anomaly Count')
ax3.grid(True, alpha=0.3)

# Plot 2d: Anomaly detection comparison
ax4 = axes[1, 1]
anomaly_methods = ['Isolation\nForest', 'Local Outlier\nFactor', 'DBSCAN', 'Consensus\n(2+ methods)']
anomaly_counts = [
    (features_df['IsoForest_Anomaly'] == -1).sum(),
    (features_df['LOF_Anomaly'] == -1).sum(),
    (features_df['DBSCAN_Cluster'] == -1).sum(),
    (features_df['Anomaly_Count'] >= 2).sum()
]
colors_methods = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
bars = ax4.bar(anomaly_methods, anomaly_counts, color=colors_methods, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Number of Anomalies Detected', fontweight='bold', fontsize=11)
ax4.set_title('Anomaly Detection Method Comparison', fontweight='bold', fontsize=12)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, count in zip(bars, anomaly_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(count)}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('CLUST2_risk_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: CLUST2_risk_analysis.png")
plt.close()

# Visualization 3: Cluster Characteristics
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 3a: Cluster size
ax1 = axes[0, 0]
cluster_sizes = [sum(kmeans_labels == i) for i in range(optimal_k)]
bars1 = ax1.bar(range(optimal_k), cluster_sizes, color='#3498db', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Cluster ID', fontweight='bold', fontsize=11)
ax1.set_ylabel('Number of Subgroups', fontweight='bold', fontsize=11)
ax1.set_title('K-Means Cluster Sizes', fontweight='bold', fontsize=12)
ax1.set_xticks(range(optimal_k))
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, size in zip(bars1, cluster_sizes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(size)}',
             ha='center', va='bottom', fontweight='bold')

# Plot 3b: Average barrier by cluster
ax2 = axes[0, 1]
cluster_barriers = [features_df[kmeans_labels == i]['Mean_Barrier'].mean() for i in range(optimal_k)]
colors_barriers = plt.cm.RdYlGn_r(np.array(cluster_barriers) / max(cluster_barriers))
bars2 = ax2.bar(range(optimal_k), cluster_barriers, color=colors_barriers, alpha=0.8, edgecolor='black')
ax2.set_xlabel('Cluster ID', fontweight='bold', fontsize=11)
ax2.set_ylabel('Average Barrier (%)', fontweight='bold', fontsize=11)
ax2.set_title('Average Barrier Level by Cluster', fontweight='bold', fontsize=12)
ax2.set_xticks(range(optimal_k))
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, barrier in zip(bars2, cluster_barriers):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{barrier:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# Plot 3c: Feature importance heatmap for clusters
ax3 = axes[1, 0]
cluster_feature_means = []
feature_names_short = ['Mean', 'Current', '2019', 'Change', 'Max', 'Min',
                       'Std', 'COVID', 'Trend', 'Volatility', 'Recovery', 'Topics', 'CI']
for i in range(optimal_k):
    cluster_data = features_df[kmeans_labels == i][feature_cols]
    cluster_feature_means.append(cluster_data.mean().values)

cluster_feature_means = np.array(cluster_feature_means)
# Normalize for better visualization
cluster_feature_norm = (cluster_feature_means - cluster_feature_means.mean(axis=0)) / (
            cluster_feature_means.std(axis=0) + 1e-10)

im = ax3.imshow(cluster_feature_norm, cmap='RdBu_r', aspect='auto')
ax3.set_xticks(range(len(feature_names_short)))
ax3.set_xticklabels(feature_names_short, rotation=45, ha='right', fontsize=9)
ax3.set_yticks(range(optimal_k))
ax3.set_yticklabels([f'Cluster {i}' for i in range(optimal_k)])
ax3.set_title('Cluster Characteristics Heatmap\n(Standardized)', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax3, label='Std. Dev from Mean')

# Plot 3d: Elbow curve
ax4 = axes[1, 1]
ax4.plot(k_range, inertias, marker='o', linewidth=2, markersize=8,
         color='#e74c3c', label='Inertia')
ax4.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Inertia', fontweight='bold', fontsize=11, color='#e74c3c')
ax4.tick_params(axis='y', labelcolor='#e74c3c')
ax4.set_title('Elbow Method for Optimal K', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Add silhouette score on secondary axis
ax4_twin = ax4.twinx()
ax4_twin.plot(k_range, silhouettes, marker='s', linewidth=2, markersize=8,
              color='#3498db', label='Silhouette')
ax4_twin.set_ylabel('Silhouette Score', fontweight='bold', fontsize=11, color='#3498db')
ax4_twin.tick_params(axis='y', labelcolor='#3498db')
ax4_twin.axvline(optimal_k, color='green', linestyle='--', linewidth=2, alpha=0.5)

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('CLUST3_cluster_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: CLUST3_cluster_analysis.png")
plt.close()

# Visualization 4: Hierarchical Dendrogram
fig4, ax = plt.subplots(figsize=(16, 10))

# Create linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram - only show top 30 for readability
dendrogram(linkage_matrix, ax=ax,
           truncate_mode='lastp',
           p=30,
           leaf_font_size=10,
           show_contracted=True)

ax.set_xlabel('Subgroup Index (or Cluster Size)', fontweight='bold', fontsize=12)
ax.set_ylabel('Distance', fontweight='bold', fontsize=12)
ax.set_title('Hierarchical Clustering Dendrogram\n(Top 30 Branches)', fontweight='bold', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('CLUST4_dendrogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: CLUST4_dendrogram.png")
plt.close()

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n[STEP 11] SAVING RESULTS")
print("-" * 100)

# Save full feature matrix with cluster assignments
features_df.to_csv('clustering_full_results.csv', index=False)
print("✓ Saved: clustering_full_results.csv")

# Save at-risk subgroups
at_risk_df = features_df[features_df['High_Risk']][
    ['Subgroup', 'Mean_Barrier', 'Current_2024', 'Change_2019_2024',
     'COVID_Impact', 'Recovery', 'Risk_Score_Normalized', 'KMeans_Cluster',
     'Anomaly_Count']
].sort_values('Risk_Score_Normalized', ascending=False)

at_risk_df.to_csv('clustering_at_risk_subgroups.csv', index=False)
print("✓ Saved: clustering_at_risk_subgroups.csv")

# Save cluster summaries
cluster_summary = []
for i in range(optimal_k):
    cluster_data = features_df[kmeans_labels == i]
    cluster_summary.append({
        'Cluster_ID': i,
        'Size': len(cluster_data),
        'Avg_Mean_Barrier': cluster_data['Mean_Barrier'].mean(),
        'Avg_Current_2024': cluster_data['Current_2024'].mean(),
        'Avg_Change_2019_2024': cluster_data['Change_2019_2024'].mean(),
        'Avg_COVID_Impact': cluster_data['COVID_Impact'].mean(),
        'Avg_Recovery': cluster_data['Recovery'].mean(),
        'Avg_Risk_Score': cluster_data['Risk_Score_Normalized'].mean(),
        'Num_High_Risk': cluster_data['High_Risk'].sum()
    })

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv('clustering_cluster_summary.csv', index=False)
print("✓ Saved: clustering_cluster_summary.csv")

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 100)
print(" " * 25 + "CLUSTERING & ANOMALY DETECTION SUMMARY")
print("=" * 100)

print("\n📊 CLUSTERING RESULTS")
print("-" * 100)
print(f"Optimal number of clusters: {optimal_k}")
print(f"K-Means silhouette score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"Hierarchical silhouette score: {silhouette_score(X_scaled, hierarchical_labels):.3f}")

print("\n🚨 ANOMALY DETECTION RESULTS")
print("-" * 100)
print(f"Isolation Forest anomalies: {n_anomalies_iso} ({n_anomalies_iso / len(features_df) * 100:.1f}%)")
print(f"Local Outlier Factor anomalies: {n_anomalies_lof} ({n_anomalies_lof / len(features_df) * 100:.1f}%)")
print(f"DBSCAN outliers: {n_outliers_dbscan} ({n_outliers_dbscan / len(features_df) * 100:.1f}%)")
print(f"High-confidence anomalies (2+ methods): {len(anomalies)} ({len(anomalies) / len(features_df) * 100:.1f}%)")

print("\n🎯 AT-RISK SUBGROUPS")
print("-" * 100)
print(f"Total at-risk subgroups (top 25%): {features_df['High_Risk'].sum()}")
print(f"Average risk score (at-risk): {features_df[features_df['High_Risk']]['Risk_Score_Normalized'].mean():.1f}/100")
print(f"Average current barrier (at-risk): {features_df[features_df['High_Risk']]['Current_2024'].mean():.1f}%")

print("\nTop 10 At-Risk Subgroups:")
top_10_risk = features_df.nlargest(10, 'Risk_Score_Normalized')[
    ['Subgroup', 'Current_2024', 'Risk_Score_Normalized', 'Anomaly_Count']
]
for idx, row in top_10_risk.iterrows():
    print(f"  • {row['Subgroup'][:60]}")
    print(f"    Current barrier: {row['Current_2024']:.1f}%, Risk score: {row['Risk_Score_Normalized']:.1f}/100")

print("\n💡 KEY INSIGHTS")
print("-" * 100)
# Find cluster with highest average barrier
high_risk_cluster = cluster_summary_df.loc[cluster_summary_df['Avg_Mean_Barrier'].idxmax()]
print(f"• Highest-risk cluster: Cluster {int(high_risk_cluster['Cluster_ID'])}")
print(f"  - {int(high_risk_cluster['Size'])} subgroups")
print(f"  - Average barrier: {high_risk_cluster['Avg_Mean_Barrier']:.1f}%")

# COVID impact
avg_covid_impact = features_df['COVID_Impact'].mean()
print(f"\n• Average COVID impact across subgroups: {avg_covid_impact:+.1f} percentage points")

# Recovery
avg_recovery = features_df['Recovery'].mean()
if avg_recovery > 0:
    print(f"• Average post-COVID recovery: {avg_recovery:.1f} pp improvement")
else:
    print(f"• Average post-COVID trend: {abs(avg_recovery):.1f} pp worsening")

print("\n📁 GENERATED FILES")
print("-" * 100)
print("  1. clustering_full_results.csv - Complete feature matrix with cluster assignments")
print("  2. clustering_at_risk_subgroups.csv - High-risk subgroups (top 25%)")
print("  3. clustering_cluster_summary.csv - Cluster characteristics summary")
print("  4. CLUST1_clustering_pca.png - PCA visualization of clustering methods")
print("  5. CLUST2_risk_analysis.png - Risk scores and anomaly detection")
print("  6. CLUST3_cluster_analysis.png - Cluster characteristics analysis")
print("  7. CLUST4_dendrogram.png - Hierarchical clustering dendrogram")

print("\n" + "=" * 100)
print(" " * 20 + "CLUSTERING & ANOMALY DETECTION COMPLETE!")
print("=" * 100)