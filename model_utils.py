from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("kidney_stone_stacking_model.pkl")


def load_model():
    """Load the exact fitted scikit-learn pipeline supplied with the study."""
    return joblib.load(MODEL_PATH)


def _kept_feature_names(model):
    columns = np.asarray(model.feature_names_in_, dtype=object)
    stats = np.asarray(model.named_steps["imputer"].statistics_, dtype=float)
    return columns[~np.isnan(stats)]


def selected_feature_names(model):
    """Recover the 35 raw feature names retained by the fitted selector."""
    kept = _kept_feature_names(model)
    support = model.named_steps["feature_select"].get_support()
    return list(kept[support])


def median_baseline(model):
    """Return a raw-input row populated with fitted training medians where available."""
    columns = list(model.feature_names_in_)
    stats = np.asarray(model.named_steps["imputer"].statistics_, dtype=float)
    row = {}
    for col, stat in zip(columns, stats):
        row[col] = np.nan if np.isnan(stat) else float(stat)
    return row


def selected_medians(model):
    """Return fitted median values for the 35 selected raw features."""
    kept = _kept_feature_names(model)
    stats = np.asarray(model.named_steps["imputer"].statistics_, dtype=float)
    kept_stats = stats[~np.isnan(stats)]
    support = model.named_steps["feature_select"].get_support()
    return dict(zip(kept[support], kept_stats[support]))


def build_input_frame(model, user_values):
    """Create a complete raw model row; None values are left missing for fitted imputation."""
    row = median_baseline(model)
    for key, value in user_values.items():
        if key not in row:
            raise KeyError(f"Unknown model feature: {key}")
        row[key] = np.nan if value is None else float(value)
    return pd.DataFrame([row], columns=list(model.feature_names_in_))


def selected_imputed_values(model, user_values):
    """Show the 35 selected feature values after the fitted median imputer."""
    X = build_input_frame(model, user_values)
    imp = model.named_steps["imputer"]
    Xi = imp.transform(X)
    kept = _kept_feature_names(model)
    support = model.named_steps["feature_select"].get_support()
    vals = Xi[0][support]
    return dict(zip(kept[support], vals))


def predict(model, user_values):
    """Return final probability/class plus fitted base-learner probabilities."""
    X = build_input_frame(model, user_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_probability = float(model.predict_proba(X)[0, 1])
        predicted_class = int(model.predict(X)[0])

        Xt = model.named_steps["imputer"].transform(X)
        Xt = model.named_steps["scaler"].transform(Xt)
        Xt = model.named_steps["feature_select"].transform(Xt)
        clf = model.named_steps["clf"]
        base_outputs = {}
        for (name, _), fitted in zip(clf.estimators, clf.estimators_):
            if hasattr(fitted, "predict_proba"):
                base_outputs[name] = float(fitted.predict_proba(Xt)[0, 1])
    return final_probability, predicted_class, base_outputs
