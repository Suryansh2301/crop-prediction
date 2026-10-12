# Paper skeleton (6-8 pages)

**Title:** Crop Yield and Recommendation Prediction for Indian Agriculture Using Ensemble Regression and Machine Learning Algorithms

1. **Abstract** – problem, two modules, datasets, best models + headline numbers (grouped-CV R², recommendation robust score), SHAP/permutation findings.
2. **Introduction** – role of agriculture in India; yield variability; need for decision support; contributions (leakage-safe protocol, robustness-focused recommendation evaluation, explainability, deployable app).
3. **Related work** – ML for yield prediction (RF/GBM/XGBoost, deep learning), crop recommendation systems, explainable AI in agriculture; gap: random-split evaluation and saturated benchmarks.
4. **Datasets** – tables of both datasets; cleaning steps; EDA figures (`eda_*.png`).
5. **Methodology** – preprocessing, log target, models, ensembles, tuning, evaluation protocols (temporal + grouped CV), metrics, significance testing. Pipeline diagram (draw this one by hand).
6. **Results – Yield** – `yield_results.csv` table; `yield_model_comparison.png`; `yield_pred_vs_actual.png`; significance table.
7. **Results – Recommendation** – `reco_results.csv`; confusion matrix; confused pairs; noise curve; range-extrapolation table.
8. **Explainability** – importance plots; per-crop heatmap; agronomic interpretation.
9. **Deployment** – Streamlit screenshots.
10. **Limitations & future work** – see README.
11. **Conclusion**
12. **References**

Figures to include: pipeline diagram, `eda_yield_by_crop`, `yield_model_comparison`, `yield_shap_summary` (or permutation), `reco_confusion_matrix`, `reco_noise_robustness`, app screenshot.
