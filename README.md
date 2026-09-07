# Kidney Stone Prediction Tool — Complete Version

Publication-oriented Streamlit application using the **actual fitted `kidney_stone_stacking_model.pkl` pipeline** supplied for the NHANES kidney-stone manuscript.

## What changed in this complete version

- All 35 selected features are presented with **plain-language labels**.
- Technical NHANES-only fields are moved to optional sections instead of cluttering the main form.
- Unknown optional values are handled by the **fitted median imputer stored inside the model**.
- Alcohol can be entered directly as g/day or estimated from drinks/week using a transparent arithmetic conversion.
- The app shows three simple engineered summaries (sodium:magnesium, calcium:magnesium, waist:upper-leg ratio) **for display only**. They are intentionally not fed into the classifier because the published model was not retrained with them.
- A dedicated **How it works** tab explains preprocessing, feature engineering, encoding, imputation, scaling, L1 feature selection and stacking in user-friendly language.
- A **Feature guide** maps all 35 model codes to readable names and reported global SHAP importance.
- Results show the fitted stacking probability, each base learner, the number of fields explicitly supplied vs median-imputed, and manuscript SHAP-pattern context.

## Scientific integrity

The displayed percentage is the fitted classifier's `predict_proba()` output. It must not be presented as a clinically validated absolute-risk percentage. The model was internally evaluated with cross-validation and has not been externally validated.

The app does **not** add new engineered ratios to the prediction model after training. Doing so would silently change the published model. Derived ratios are visualization aids only.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Upload the contents of this folder to a GitHub repository.
2. In Streamlit Community Cloud choose **Create app**.
3. Select the repository and set the main file to `app.py`.
4. Deploy.

`requirements.txt` pins **scikit-learn 1.9.0**, matching the version used to serialize the supplied model.

## Docker

```bash
docker build -t kidney-stone-tool .
docker run -p 8501:8501 kidney-stone-tool
```
