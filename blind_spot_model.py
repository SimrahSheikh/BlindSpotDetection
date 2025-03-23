import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

class BlindSpotDetectionModel:
    def __init__(self):
        """Initialize optimized Random Forest & XGBoost models."""
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=2, min_samples_split=10, class_weight="balanced", random_state=42
        )
        self.xgb = XGBClassifier(
            n_estimators=150, learning_rate=0.02, max_depth=2,
            reg_lambda=3.0, reg_alpha=2.0, subsample=0.7, colsample_bytree=0.7,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)  # Reduce dimensionality for better learning
        self.trained = False

    def train_models(self, X_train, y_train):
        """Scale and apply PCA, then train models with cross-validation."""
        X_train = self.scaler.fit_transform(X_train)
        X_train = self.pca.fit_transform(X_train)

        rf_cv_score = np.mean(cross_val_score(self.rf, X_train, y_train, cv=5))
        xgb_cv_score = np.mean(cross_val_score(self.xgb, X_train, y_train, cv=5))

        print(f"Random Forest CV Acc: {rf_cv_score:.2f}")
        print(f"XGBoost CV Acc: {xgb_cv_score:.2f}")

        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        self.trained = True

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """Evaluate model accuracy after training."""
        X_train = self.scaler.transform(X_train)
        X_train = self.pca.transform(X_train)

        X_test = self.scaler.transform(X_test)
        X_test = self.pca.transform(X_test)

        rf_train_acc = accuracy_score(y_train, self.rf.predict(X_train))
        rf_test_acc = accuracy_score(y_test, self.rf.predict(X_test))
        
        xgb_train_acc = accuracy_score(y_train, self.xgb.predict(X_train))
        xgb_test_acc = accuracy_score(y_test, self.xgb.predict(X_test))

        print(f"Random Forest - Train Acc: {rf_train_acc:.2f}, Test Acc: {rf_test_acc:.2f}")
        print(f"XGBoost - Train Acc: {xgb_train_acc:.2f}, Test Acc: {xgb_test_acc:.2f}")

    def save_models(self):
        """Save trained models."""
        joblib.dump(self.rf, "rf_model.pkl")
        joblib.dump(self.xgb, "xgb_model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        joblib.dump(self.pca, "pca.pkl")

    def load_models(self):
        """Load saved models."""
        self.rf = joblib.load("rf_model.pkl")
        self.xgb = joblib.load("xgb_model.pkl")
        self.scaler = joblib.load("scaler.pkl")
        self.pca = joblib.load("pca.pkl")
        self.trained = True

    def predict(self, sensor_input):
        """Predict the risk level using both models."""
        if not self.trained:
            raise ValueError("Models have not been trained yet! Load or train the models first.")

        sensor_input = self.scaler.transform([sensor_input])
        sensor_input = self.pca.transform(sensor_input)

        rf_probs = self.rf.predict_proba(sensor_input)[0]
        xgb_probs = self.xgb.predict_proba(sensor_input)[0]

        combined_probs = (rf_probs + xgb_probs) / 2
        return np.argmax(xgb_probs)

# Example usage:
# model = BlindSpotDetectionModel()
# model.train_models(X_train, y_train)
# model.evaluate_models(X_train, y_train, X_test, y_test)
# model.save_models()
