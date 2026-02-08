# 🏥 Complete Healthcare ML Analysis 
## DubsTech Datathon 2026 - Three Comprehensive Models

---

## 🎯 Project Overview

This submission contains **THREE complete machine learning analyses** for healthcare access barriers:

1. **Predictive Model** - Predicts cost barriers by demographic subgroup (93.7% accuracy)
2. **Time-Series Forecasting** - Forecasts 2025 trends in delayed/missed care
3. **Clustering & Anomaly Detection** - Identifies at-risk subgroups

---
## 🚀 Quick Start Guide

### Run All Three Models:

```bash
# 1. Create the virtual environment (named 'venv')
python -m venv venv

# 2. Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Upgrade pip to the latest version
python -m pip install --upgrade pip

# 4. Install dependencies from the file
pip install -r requirements.txt

# 5. Run Model 1: Predictive Model
python .\cost_prediction_model\healthcare_cost_prediction.py

# 6. Run Model 2: Time-Series Forecasting
python .\timeseries_forecast_model\timeseries_forecast.py

# 7. Run Model 3: Clustering & Anomaly Detection
python .\clustering_anomaly_model\clustering_anomaly.py
```

**Total Runtime:** ~90 seconds for all three models

---

## 📦 Complete File Inventory

### 🔵 MODEL 1: Predictive ML Model (Subgroup Cost Barriers)

**Python Script:**
- `healthcare_cost_prediction.py` - Main analysis (6 algorithms, 70-30 split between train-test data)

**Key Result:** 93.7% accuracy (R²), 0.52 pp average error

---

### 🟢 MODEL 2: Time-Series Forecasting (Future Trends)

**Python Script:**
- `timeseries_forecast.py` - Multi-model forecasting pipeline

**Key Result:** 9.15% predicted average barrier for 2025 (95% CI: 7.93-10.36%)

---

### 🟠 MODEL 3: Clustering & Anomaly Detection (At-Risk Groups)

**Python Script:**
- `clustering_anomaly.py` - Multi-method clustering and anomaly detection

**Key Result:** Identified 11 high-confidence anomalies, 19 at-risk subgroups

---

### 📚 Documentation

- `README.md` - This comprehensive guide
- `requirements.txt` - This is the list of all dependencies
- `healthcare_cost_prediction.py` - Model 1
- `clustering_anomaly.py` - Model 2 
- `timeseries_forecast.py` - Model 3 

---



## 💡 Key Insights Across All Three Models

### 🔴 Critical Findings

**1. Most Vulnerable Populations (Consistent Across Models)**
- **Bisexual individuals:** 23.5% current barrier, highest actual rate
- **Uninsured adults:** 19.0% barrier, predictable high risk
- **People with disabilities:** 16.3% barrier, chronic high risk
- **Native Hawaiian/Pacific Islander:** 14.8% barrier, statistical outlier
- **LGBTQ+ individuals:** Consistently above average

**2. COVID-19 Impact & Recovery**
- **Immediate Impact (2020):** Paradoxical decrease (-0.75 pp)
  - Likely due to expanded coverage, telehealth
- **Post-COVID (2022-2024):** Barriers increasing (+1.0 pp worsening)
  - Policy rollbacks, economic pressures
- **2025 Forecast:** Continued increase to 9.15%

**3. Topic-Specific Divergence**
- **Good News:** Medical and delayed care barriers decreasing
- **Bad News:** Mental health care barriers increasing
- **Major Concern:** Missed workdays increasing sharply (16.67% forecast)

**4. Prediction Confidence**
- **Predictive Model:** 93.7% accuracy - very reliable
- **Time-Series:** 95% CI ±1.2 pp - moderately confident
- **Clustering:** 0.815 silhouette - excellent separation

### 🟢 Policy Recommendations

**Immediate Actions:**
1. **Target LGBTQ+ populations** - Highest barriers, often overlooked
2. **Mental health access** - Only category with increasing trend
3. **Disability accommodations** - Persistently high barriers
4. **Insurance expansion** - Uninsured show dramatic barriers

**Monitoring Systems:**
5. **Deploy predictive model** - Real-time risk scoring
6. **Track workday losses** - Sharp upward trend needs intervention
7. **Anomaly alerts** - Flag new at-risk groups early

**Resource Allocation:**
8. **Prioritize top 25% risk groups** - 19 subgroups need focus
9. **Cultural competency** - Multiple racial/ethnic outliers
10. **Income-based programs** - <100% FPL consistently at-risk

---

## 📊 Summary Statistics

| Metric                          | Value               |
|---------------------------------|---------------------|
| **Total Code Lines**            | ~2,000 lines        |
| **Total Runtime**               | ~90 seconds         |
| **Models Built**                | 13 (6 + 4 + 3)      |
| **Visualizations**              | 15 high-res images  |
| **CSV Outputs**                 | 10 detailed files   |
| **Demographic Groups Analyzed** | 75 subgroups        |
| **Time Period Coverage**        | 2019-2025 (7 years) |
| **Data Points Processed**       | 1,800+ records      |

---

## ❓ Anticipated Questions & Answers

**Q: Why three separate models instead of one?**

A: Each addresses a different research question from the datathon brief. Model 1 predicts by subgroup, Model 2 forecasts overall trends, Model 3 finds hidden at-risk groups.

**Q: Which model is most important?**

A: All three together. Model 1 shows what's predictable, Model 2 shows where we're heading, Model 3 shows who's falling through the cracks.

**Q: How accurate are these predictions?**

A: Model 1: 93.7% (excellent). Model 2: ±1.2 pp at 95% confidence (good). Model 3: Unsupervised, but 0.815 silhouette score (excellent separation).

**Q: Can these be deployed in production?**

A: Yes! All models are production-ready. Model 1 can score new subgroups in real-time. Model 2 updates with new yearly data. Model 3 recomputes risk scores quarterly.

**Q: What's the most surprising finding?**

A: The COVID paradox - barriers initially decreased in 2020 but are now increasing above pre-COVID levels. Also, mental health care is the only category getting worse while others improve.

**Q: How can impact be achieved using this project?

A: The broadest manner to achieve impact in healthcare is through policy-making. We hope that policymakers can gain insights on
which underserved sub-groups need to be uplifted in order to receive the healthcare they deserve.

**Q: Who should policymakers focus on?**

A: Top priorities: (1) Bisexual individuals (23.5% barrier), (2) Uninsured (19%), (3) People with disabilities (16.3%), (4) Native Hawaiian/Pacific Islander (14.8%).

---

## 🎓 Technical Stack

**Languages:**
- Python 3.8+

**Core Libraries:**
- pandas, numpy (data manipulation)
- scikit-learn (ML algorithms)
- matplotlib, seaborn (visualization)
- scipy (statistical functions)

**ML Algorithms Used:**
- Supervised: Linear/Ridge/Lasso Regression, Decision Trees, Random Forest, Gradient Boosting
- Time-Series: ARIMA-style, Exponential Smoothing, Moving Average, Polynomial Regression
- Unsupervised: K-Means, Hierarchical, DBSCAN, Isolation Forest, Local Outlier Factor
- Dimensionality Reduction: PCA

**Total Unique Algorithms:** 13 different ML methods

---

## 💻 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- Any OS (Windows/Mac/Linux)

**Recommended:**
- Python 3.10+
- 8GB RAM
- Multi-core CPU for faster processing

---

**Created for DubsTech Datathon 2026**  
**Topic: Access to Care - Healthcare Barriers Analysis**  
**Total Time Investment:** ~16 hours of analysis, coding, and documentation

**Date: February 2026**

**Three Models. Comprehensive Analysis. Actionable Insights.** 🚀

**Thank you for reviewing this submission!** 🙏