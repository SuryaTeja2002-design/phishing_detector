"""
online_learner.py
-----------------
SGDClassifier-based online learner for incremental URL retraining.
Kept in its own module so it can be correctly pickled/unpickled.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42


class OnlineLearner:
    """
    SGD-based online learner that updates incrementally with new labeled samples.
    Uses partial_fit() — no full retraining required.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss", random_state=SEED,
            learning_rate="optimal", class_weight="balanced"
        )
        self.buffer_X = []
        self.buffer_y = []
        self.fitted = False
        self.n_updates = 0

    def initial_fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def update(self, url: str, label: int):
        from feature_extractor import features_to_vector
        vec = features_to_vector(url)
        self.buffer_X.append(vec)
        self.buffer_y.append(label)
        self.n_updates += 1
        if self.fitted:
            X_new = self.scaler.transform([vec])
            self.model.partial_fit(X_new, [label], classes=[0, 1])
        return {"status": "updated", "total_updates": self.n_updates}

    def predict(self, url: str) -> dict:
        if not self.fitted:
            return {"error": "Not yet fitted"}
        from feature_extractor import features_to_vector
        vec = features_to_vector(url)
        X_scaled = self.scaler.transform([vec])
        pred = int(self.model.predict(X_scaled)[0])
        prob = float(self.model.predict_proba(X_scaled)[0][1])
        return {
            "prediction": "phishing" if pred == 1 else "legitimate",
            "phishing_prob": round(prob, 4),
            "n_updates": self.n_updates,
        }

    def get_stats(self) -> dict:
        return {
            "fitted": self.fitted,
            "n_updates": self.n_updates,
            "buffer_size": len(self.buffer_X),
        }