# Nomor 2
# Arieldhipta Tarliman - 2702234636

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from preprocessor import HotelPreprocessor

class HotelBookingModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.x = None
        self.y = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Data loaded successfully.")

    def preprocess(self):
        processor = HotelPreprocessor(self.df)
        df = processor.preprocess(is_training=True)
        self.df = df
        self.x = df.drop('booking_status', axis=1)
        self.y = df['booking_status']
        print("Preprocessing done.")

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print("Train-test split done.")

    def feature_selection_and_tuning(self):
        # Initial RF for feature importance
        rf_selection = RandomForestClassifier(random_state=42)
        rf_selection.fit(self.x_train, self.y_train)

        # Get top 10 features
        feature_importances = pd.DataFrame({
            'Feature': self.x_train.columns,
            'Importance': rf_selection.feature_importances_
        }).sort_values('Importance', ascending=False)

        self.selected_features = feature_importances['Feature'].head(10).tolist()
        self.x_train_selected = self.x_train[self.selected_features]
        self.x_test_selected = self.x_test[self.selected_features]

        print("Selected Features:", self.selected_features)

        # Randomized Search CV
        param_grid = {
            'n_estimators': [100, 150, 200, 250, 300, 500],
            'max_depth': [None, 10, 50, 70, 100, 150, 200],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.5],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced']
        }

        rf_model = RandomForestClassifier(random_state=42)
        random_search_rf = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=100,
            cv=3,
            scoring='f1',
            verbose=2,
            n_jobs=-1
        )
        random_search_rf.fit(self.x_train_selected, self.y_train)

        self.model = random_search_rf.best_estimator_
        print("Best Parameters:", random_search_rf.best_params_)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test_selected)
        print("Evaluation Report:")
        print(classification_report(self.y_test, y_pred))

    def save_model(self, filename='hotel_booking.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': self.selected_features # for inference
            }, f)
        print(f"Model and selected features saved as {filename}.")



# Step 1 & 2: Initialize the model and load data
model = HotelBookingModel("Dataset_B_hotel.csv")  # Ganti dengan path yang benar kalau di folder lain
model.load_data()

# Step 3: Preprocess the data
model.preprocess()

# Step 4: Split the data
model.split_data()

# Step 5: Perform feature selection and hyperparameter tuning
model.feature_selection_and_tuning()

# Step 6: Evaluate the model
model.evaluate_model()

# Step 7: Optionally save the model
model.save_model("hotel_booking_rf.pkl")
