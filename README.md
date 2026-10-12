# Crop Yield & Recommendation Prediction for Indian Agriculture
Ensemble regression + machine learning, with leakage-safe evaluation, significance tests, explainability and a Streamlit demo.

## Two modules
| Module | Task | Data | Target |
|---|---|---|---|
| 1. Yield | Regression | Kaggle *Crop Production in India* (district-wise, 1997-2015) | `log1p(Production / Area)` |
| 2. Recommendation | 22-class classification | Kaggle *Crop Recommendation Dataset* | crop label |

## Quick start
```bash
pip install -r requirements.txt

# A) Smoke test, no downloads (SYNTHETIC data - not for the paper!)
python run_all.py --synthetic --fast

# B) Real run
#   1. Put crop_production.csv and Crop_recommendation.csv in data/raw/
#   2. (optional) add data/raw/state_rainfall.csv  with columns: State,Year,rainfall_mm
python run_all.py            # add --tune for Optuna tuning of RF & GB
streamlit run app/streamlit_app.py
```
Outputs: `reports/*.csv` (all result tables), `reports/figures/*.png`, `models/*.joblib`.

## Datasets
- Crop Production in India: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india  → `data/raw/crop_production.csv`
  (columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production)
- Crop Recommendation: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset → `data/raw/Crop_recommendation.csv`
  Context:
        This dataset was build by augmenting datasets of rainfall, climate and fertilizer data available for India.
  Data fields:
              N - ratio of Nitrogen content in soil
              P - ratio of Phosphorous content in soil
              K - ratio of Potassium content in soil
              temperature - temperature in degree Celsius
              humidity - relative humidity in %
              ph - ph value of the soil
              rainfall - rainfall in mm
             (columns: N, P, K, temperature, humidity, ph, rainfall, label)
- Weather enrichment (strongly recommended for a publishable yield model): IMD state/district rainfall
  (imdpune.gov.in), data.gov.in, ICRISAT district database. Save as `state_rainfall.csv`; the pipeline merges it automatically.

## Project layout
`src/*.ipynb` are Jupyter notebooks. `run_all.py`, `app/streamlit_app.py` and `tests/` are plain scripts and import the
notebooks through the small loader in `src/__init__.py` (it executes each notebook's code cells on import).
You can also open any notebook in Jupyter and run it cell by cell (open Jupyter from the project root or `src/`).
```
run_all.py               one-command pipeline
src/config.ipynb            paths, seeds, split year, outlier rules
src/data_prep.ipynb         cleaning (name fixes, yield = production/area, outliers) + EDA figures
src/train_yield.ipynb       baselines, ensembles, tuning, temporal + grouped CV, significance tests
src/train_recommend.ipynb   classifiers + noise robustness + range-extrapolation tests
src/explain.ipynb           permutation importance (+ SHAP if installed), per-crop importance
src/predict.ipynb           inference helpers
src/synthetic_data.ipynb    fake data for smoke tests
app/streamlit_app.py     two-tab demo
tests/test_smoke.py      leakage / split / inference checks
reports/paper_outline.md paper skeleton
```

## Methodology decisions (cite these in the paper)
1. **No leakage**: `Production = Area × Yield`, so Production/Yield are never features (checked in `tests/`).
2. **Two evaluation protocols**: temporal hold-out (train ≤ 2012, test 2013-15) and **district-grouped** K-fold, so a district never appears in both train and test.
3. **Target**: `log1p(yield)` because yield is heavily right-skewed. Both log-scale R² and original-scale R²/RMSE/MAE are reported.
4. **Cleaning**: Coconut dropped (reported in nuts, not tonnes), rare crops dropped, per-crop 99.5th-percentile yield cap.
5. **Models** – baselines: Ridge, Decision Tree, KNN. Ensembles: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost (auto-detected), plus Voting and Stacking (Ridge meta-learner).
6. **Tuning**: Optuna TPE (falls back to random search) with grouped CV on the *training* years only.
7. **Significance**: paired t-test + Wilcoxon over CV folds, best model vs. every other (`*_significance.csv`). With 5 folds the smallest possible Wilcoxon p is 0.0625 – set `CV_FOLDS = 10` in `config.ipynb` for the final run.
8. **Recommendation**: the public dataset is near-perfectly separable, so besides accuracy/macro-F1 the pipeline reports
   noise robustness (Gaussian noise on test features), *range extrapolation* (train without each crop's top-15% of temperature / humidity / rainfall, test on them), most-confused crop pairs, and picks the deployed model by a **robust score** rather than clean accuracy.
9. **Explainability**: permutation importance always; SHAP summary/bar plots when `shap` is installed.

## Known limitations (state them honestly in the paper)
- The Kaggle yield data has no weather/soil covariates; without enrichment, district/crop/season/year dominate.
- The recommendation dataset is not linked to the yield dataset (different geography/provenance); the two modules are complementary, not a single joint model.
- Stacking's inner CV is a plain K-fold (not grouped), which can slightly optimise meta-learner weights; the outer evaluation remains grouped.
- Years after 2015 are extrapolation; the app warns about this.

## Suggested extensions
Lag/trailing-mean yield features (computed from training years only), district-level soil data, spatial cross-validation by state, quantile/conformal prediction intervals, FastAPI deployment.
