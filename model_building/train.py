# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# XGBClassifier used because ProdTaken is binary (0 or 1) — classification task
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Use environment variable for HF token (do not hardcode credentials)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train/test data from Hugging Face dataset space (inside data/ folder)
Xtrain_path = "hf://datasets/SrikanthKontham/tourism_project/data/Xtrain.csv"
Xtest_path  = "hf://datasets/SrikanthKontham/tourism_project/data/Xtest.csv"
ytrain_path = "hf://datasets/SrikanthKontham/tourism_project/data/ytrain.csv"
ytest_path  = "hf://datasets/SrikanthKontham/tourism_project/data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()   # convert single-column DataFrame to Series
ytest  = pd.read_csv(ytest_path).squeeze()

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")

# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# Preprocessor: scale numeric, one-hot encode categorical
# Raw string values in Xtrain/Xtest are handled directly here
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# XGBoost Classifier (binary classification: ProdTaken = 0 or 1)
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search with ROC-AUC scoring (appropriate for binary classification)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations as nested MLflow runs
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)
    y_prob_test  = best_model.predict_proba(Xtest)[:, 1]

    # Classification Metrics
    train_acc    = accuracy_score(ytrain, y_pred_train)
    test_acc     = accuracy_score(ytest,  y_pred_test)
    train_f1     = f1_score(ytrain, y_pred_train)
    test_f1      = f1_score(ytest,  y_pred_test)
    test_roc_auc = roc_auc_score(ytest, y_prob_test)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_Accuracy": train_acc,
        "test_Accuracy":  test_acc,
        "train_F1":       train_f1,
        "test_F1":        test_f1,
        "test_ROC_AUC":   test_roc_auc
    })

    print(f"Best Params: {grid_search.best_params_}")
    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Train F1:       {train_f1:.4f} | Test F1:       {test_f1:.4f}")
    print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(ytest, y_pred_test))

    # Save the best model locally
    model_path = "toursim_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact: {model_path}")

    # Upload best model to Hugging Face model hub
    repo_id   = "SrikanthKontham/tourism_project"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj="toursim_model_v1.joblib",
        path_in_repo="toursim_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model uploaded to Hugging Face model hub successfully.")
