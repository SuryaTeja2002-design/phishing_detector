"""
train_model.py
--------------
Trains FOUR ML models and an ensemble, with:
  - Model comparison (RF, SVM, MLP Neural Net, Gradient Boosting)
  - Permutation-based feature importance (SHAP-equivalent)
  - Confidence scoring with uncertainty estimates
  - Online learning buffer for real-time retraining
  - Full evaluation suite

Usage:
    python train_model.py
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from feature_extractor import features_to_vector, get_feature_names
from online_learner import OnlineLearner

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com", "https://www.github.com/login",
    "https://stackoverflow.com/questions", "https://www.wikipedia.org/wiki/Python",
    "https://www.amazon.com/products", "https://docs.python.org/3/library",
    "https://www.youtube.com/watch?v=abc123", "https://www.linkedin.com/in/johndoe",
    "https://www.reddit.com/r/programming", "https://www.netflix.com/browse",
    "https://mail.google.com/mail/u/0", "https://www.apple.com/iphone",
    "https://www.microsoft.com/en-us", "https://www.twitter.com/home",
    "https://www.dropbox.com/home", "https://www.coursera.org/courses",
    "https://medium.com/python-tutorials", "https://www.bbc.com/news/technology",
    "https://www.nytimes.com/section/technology", "https://www.stripe.com/docs",
    "https://www.cloudflare.com/products", "https://www.heroku.com/home",
    "https://www.mongodb.com/atlas", "https://www.postgresql.org/docs",
    "https://www.docker.com/products", "https://kubernetes.io/docs/home",
    "https://www.elastic.co/elasticsearch", "https://www.tensorflow.org/tutorials",
    "https://pytorch.org/docs/stable", "https://www.kaggle.com/competitions",
]

PHISHING_URLS = [
    "http://secure-paypal-login.xyz/account/verify?id=382910",
    "http://192.168.1.105/amazon/login.php",
    "http://www.google.com.phishing-site.ru/login",
    "http://bit.ly/3xAm2Zq",
    "http://update-your-account-now.com/banking/confirm",
    "http://apple-id-suspended.support/verify?user=abc",
    "http://login-microsoft-account.malicious.net/signin",
    "http://paypal-update.secure-account.tk/confirm",
    "http://amazon-security-alert.com/account?suspended=true",
    "https://faceb00k-login.com/verify/credentials",
    "http://ebay-account-update.scamsite.pw/login",
    "http://netflix-billing-update.xyz/account/payment",
    "http://secure-login-bankofamerica.com/auth?token=xyz",
    "http://64.182.23.11/phish/paypal.html",
    "http://chase-bank-secure.tk/login?redirect=account",
    "http://verify-your-paypal.com/wp-login.php",
    "http://support-apple-id-locked.com/unlock?user=1",
    "http://www.amazon-account-suspend.online/verify",
    "http://microsoft-security-update.com/credential-reset",
    "http://signin-google-account-support.com/login",
    "http://wellsfargo-secure-login.xyz/auth",
    "http://instagram-verify.tk/confirm?id=12345",
    "http://dropbox-file-share.malicious.pw/download",
    "http://netflix-account-suspended.xyz/reactivate",
    "http://icloud-locked.support/unlock?device=iphone",
    "http://paypal.com.secure-verify.xyz/login",
    "http://amazon.com.account-update.tk/verify",
    "http://apple.com-id-support.xyz/reset",
    "http://10.0.0.1/admin/login.php",
    "http://172.16.0.5/paypal/signin.html",
]


def generate_synthetic(n_legit=600, n_phish=600):
    random.seed(SEED)
    urls, labels = [], []

    legit_domains = ["google.com", "github.com", "stackoverflow.com", "amazon.com",
                     "wikipedia.org", "youtube.com", "linkedin.com", "reddit.com",
                     "apple.com", "microsoft.com", "netflix.com", "dropbox.com",
                     "stripe.com", "cloudflare.com", "mongodb.com", "kaggle.com"]
    legit_paths = ["/home", "/login", "/products", "/about", "/docs",
                   "/help", "/search", "/profile", "/dashboard", "/settings",
                   "/api/v1/users", "/blog/post", "/tutorials", "/download"]

    for _ in range(n_legit):
        domain = random.choice(legit_domains)
        path = random.choice(legit_paths)
        query = f"?id={random.randint(1,999)}" if random.random() > 0.5 else ""
        urls.append(f"https://www.{domain}{path}{query}")
        labels.append(0)

    bad_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".pw", ".ru", ".online", ".top"]
    phish_kw = ["secure", "login", "account", "verify", "update", "confirm", "billing"]
    brands = ["paypal", "amazon", "apple", "microsoft", "google",
              "netflix", "ebay", "instagram", "chase", "wellsfargo"]

    for _ in range(n_phish):
        brand = random.choice(brands)
        kw = random.choice(phish_kw)
        tld = random.choice(bad_tlds)
        rand_n = random.randint(10000, 99999)
        rand_tok = ''.join(random.choices('abcdef0123456789', k=16))

        style = random.randint(0, 4)
        if style == 0:
            url = f"http://{brand}-{kw}{tld}/{kw}?token={rand_tok}"
        elif style == 1:
            url = f"http://{kw}-{brand}-secure{tld}/verify?user={rand_n}"
        elif style == 2:
            ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
            url = f"http://{ip}/{brand}/login.php?token={rand_n}"
        elif style == 3:
            url = f"http://www.{brand}.com.{kw}{tld}/login?id={rand_n}&session={rand_tok}"
        else:
            url = f"http://{brand}.{kw}.support/account/{kw}?suspended=true&user={rand_n}"
        urls.append(url)
        labels.append(1)

    return urls, labels


def build_dataset():
    urls = LEGITIMATE_URLS + PHISHING_URLS
    labels = [0] * len(LEGITIMATE_URLS) + [1] * len(PHISHING_URLS)
    su, sl = generate_synthetic(600, 600)
    urls += su
    labels += sl
    print(f"Dataset: {len(urls)} URLs  ({labels.count(0)} legit / {labels.count(1)} phishing)")
    return urls, labels


# ── Feature importance (SHAP-equivalent) ─────────────────────────────────────

def compute_permutation_importance(model, X_val, y_val, feature_names, n_repeats=10):
    """
    Permutation importance: for each feature, randomly shuffle it and measure
    how much model accuracy drops. Larger drop = more important feature.
    This is equivalent to what SHAP does but uses only sklearn.
    """
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=SEED,
        scoring="roc_auc"
    )
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return importance_df


# ── Per-prediction explanation ────────────────────────────────────────────────

def explain_prediction(url: str, model, scaler, feature_names: list,
                       global_importances: pd.DataFrame) -> dict:
    """
    For a single URL, compute local feature contribution scores.
    Method: compare each feature value vs the dataset mean, weighted by
    global permutation importance. This gives a SHAP-like local explanation.
    """
    from feature_extractor import extract_features
    feats = extract_features(url)
    feat_vals = list(feats.values())

    # Use global importance as weights
    imp_map = dict(zip(global_importances["feature"], global_importances["importance_mean"]))
    contributions = []
    for name, val in zip(feature_names, feat_vals):
        imp = imp_map.get(name, 0)
        # Flag: is this feature pointing toward phishing?
        phish_signal = _feature_is_phishing_signal(name, val)
        contributions.append({
            "feature": name,
            "value": round(val, 4),
            "importance": round(imp, 4),
            "phishing_signal": phish_signal,
            "contribution": round(imp * phish_signal, 4),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:12]  # top 12


def _feature_is_phishing_signal(name: str, val: float) -> float:
    """Return +1 if value points toward phishing, -1 if toward legitimate."""
    phish_positive = {
        "has_ip_address", "has_at_symbol", "suspicious_keyword_count",
        "is_url_shortener", "suspicious_tld", "brand_in_subdomain",
        "subdomain_count", "has_hyphen_domain", "url_length", "domain_length",
        "hyphen_count", "digit_count_domain", "path_length", "query_length",
        "special_char_count", "has_encoded_chars", "consecutive_digits",
        "url_entropy", "tld_in_path", "has_double_slash", "digit_ratio_url",
        "query_param_count", "long_token", "domain_entropy",
    }
    phish_negative = {"has_https", "trusted_tld", "domain_vowel_ratio"}

    if name in phish_positive:
        return 1.0 if val > 0 else -0.5
    elif name in phish_negative:
        return -1.0 if val > 0 else 0.5
    return 0.0


# ── Model confidence / uncertainty ────────────────────────────────────────────

def compute_uncertainty(models: dict, X: np.ndarray) -> dict:
    """
    Compute prediction uncertainty via disagreement across models.
    High variance = model is unsure = treat with more caution.
    """
    probas = []
    for name, model in models.items():
        try:
            p = model.predict_proba(X)[0][1]
            probas.append(p)
        except Exception:
            pass

    if not probas:
        return {"mean_prob": 0.5, "std": 0.5, "confidence": "LOW", "model_probas": {}}

    mean_p = float(np.mean(probas))
    std_p = float(np.std(probas))

    if std_p < 0.05:
        confidence_level = "HIGH"
    elif std_p < 0.15:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "mean_prob": round(mean_p, 4),
        "std": round(std_p, 4),
        "confidence": confidence_level,
        "model_probas": {name: round(p, 4) for name, p in zip(models.keys(), probas)},
    }


# ── Online learning ───────────────────────────────────────────────────────────

# ── Main training ─────────────────────────────────────────────────────────────

def train():
    print("=" * 65)
    print("  Phishing Detector — Multi-Model Training Suite")
    print("=" * 65)

    urls, labels = build_dataset()
    feature_names = get_feature_names()
    X = np.array([features_to_vector(u) for u in urls])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Shared scaler for SVM and MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Define models ─────────────────────────────────────────────────────────
    models_raw = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=4,
            random_state=SEED, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=SEED
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, random_state=SEED, class_weight="balanced"))
        ]),
        "Neural Net (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", solver="adam",
                max_iter=500, random_state=SEED,
                early_stopping=True, validation_fraction=0.1,
                learning_rate_init=0.001
            ))
        ]),
    }

    # ── Train & evaluate each model ───────────────────────────────────────────
    print(f"\n{'Model':<22} {'Accuracy':>9} {'F1':>7} {'AUC':>7} {'Precision':>10} {'Recall':>8}")
    print("-" * 65)

    model_metrics = {}
    trained_models = {}

    for name, model in models_raw.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = (y_pred == y_test).mean()
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)

        model_metrics[name] = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

        cv = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
        model_metrics[name]["cv_auc_mean"] = round(cv.mean(), 4)
        model_metrics[name]["cv_auc_std"]  = round(cv.std(), 4)

        print(f"  {name:<20} {acc:>9.4f} {f1:>7.4f} {auc:>7.4f} {prec:>10.4f} {rec:>8.4f}")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\nTraining final ensemble (Soft Voting)...")
    ensemble = VotingClassifier(
        estimators=list(models_raw.items()),
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_prob = ensemble.predict_proba(X_test)[:, 1]
    ens_auc  = roc_auc_score(y_test, ens_prob)
    ens_f1   = f1_score(y_test, ens_pred)
    ens_acc  = (ens_pred == y_test).mean()
    print(f"  {'Ensemble':<20} {ens_acc:>9.4f} {ens_f1:>7.4f} {ens_auc:>7.4f}")

    model_metrics["Ensemble"] = {
        "accuracy": round(ens_acc, 4),
        "f1": round(ens_f1, 4),
        "auc": round(ens_auc, 4),
        "precision": round(precision_score(y_test, ens_pred), 4),
        "recall": round(recall_score(y_test, ens_pred), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, ens_pred)
    print(f"\n── Ensemble Confusion Matrix ──────────────")
    print(f"  True Legit:  {cm[0,0]:4d}  |  False Phish: {cm[0,1]:4d}")
    print(f"  False Legit: {cm[1,0]:4d}  |  True Phish:  {cm[1,1]:4d}")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\nComputing permutation importance (SHAP-equivalent)...")
    rf_model = trained_models["Random Forest"]
    perm_imp = compute_permutation_importance(rf_model, X_test, y_test, feature_names)

    print(f"\n── Top 12 Features (Permutation Importance) ──────────────────")
    for _, row in perm_imp.head(12).iterrows():
        bar = "█" * max(0, int(row["importance_mean"] * 200))
        print(f"  {row['feature']:<30} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}  {bar}")

    # ── Online learner ────────────────────────────────────────────────────────
    print("\nTraining online learner (SGD)...")
    online_learner = OnlineLearner(feature_names)
    online_learner.initial_fit(X_train, y_train)
    ol_pred = [online_learner.model.predict(online_learner.scaler.transform([features_to_vector(u)]))[0] for u in urls[:len(y_test)]]

    # ── Save everything ───────────────────────────────────────────────────────
    model_data = {
        "ensemble": ensemble,
        "individual_models": trained_models,
        "online_learner": online_learner,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_metrics": model_metrics,
        "permutation_importance": perm_imp.to_dict(orient="records"),
        "ensemble_auc": ens_auc,
        "training_size": len(urls),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n✓ All models saved to model.pkl")
    print(f"  Ensemble AUC: {ens_auc:.4f}")
    print("=" * 65)
    return model_data


if __name__ == "__main__":
    train()