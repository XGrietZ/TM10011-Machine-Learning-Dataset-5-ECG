#%%
# -----------------------------------
# 0. Imports
# -----------------------------------

# Custom
from ecg.load_data import load_data

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Utilities
from tqdm import tqdm

# Scikit-learn: model selection
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    ParameterGrid,
    learning_curve,
    LearningCurveDisplay
)

# Scikit-learn: preprocessing
from sklearn.preprocessing import FunctionTransformer, RobustScaler

# Scikit-learn: models
from sklearn.ensemble import RandomForestClassifier

# Scikit-learn: metrics
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Scikit-learn: utilities
from sklearn.base import clone

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#%%
#-----------------------------------
# 1. Load data
#-----------------------------------

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

df = pd.DataFrame(data)

# Remove missing value column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Check if the expected 'label' column exists in the DataFrame, if not raise an error
if "label" not in df.columns:
    raise ValueError("Expected a column named 'label' in the CSV.")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"].astype(int)

print("\nDataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

#%%
#-----------------------------------
# 2. Train/Test split (hold-out)
#-----------------------------------   
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, 
    y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("\nTrain_full label distribution:\n", y_train_full.value_counts())
print("Final test label distribution:\n", y_test_final.value_counts())

#%%
#-----------------------------------
# 3. Nested CV setup
#-----------------------------------
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#%%
#-----------------------------------
# 4. Pipeline + Hyperparameters
#-----------------------------------
# Imbalance-aware pipeline, because SMOTE upsamples imbalanced data inside the CV loop
pipeline = ImbPipeline([
    ('log_transform', 'passthrough'),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42,n_jobs=-1))
])
# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'log_transform': ['passthrough', FunctionTransformer(np.log1p, validate=False)],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [2, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4, 8]
}

# Generate a list of all hyperparameter combinations and print total number of models and fits 
param_list = list(ParameterGrid(param_grid))
print(f"\nTotal models: {len(param_list)}")
print(f"Total fits (with {inner_cv.get_n_splits()}-fold CV): {len(param_list) * inner_cv.get_n_splits()}")


#%%
#-----------------------------------
# 5. Nested cross-validation with live tqdm info
#-----------------------------------
outer_results = []
outer_best_params = []

start_total_time = time.time()

# Splitting the full training set into outer CV folds
outer_splits = list(outer_cv.split(X_train_full, y_train_full))

# Progress bar for outer CV
pbar = tqdm(outer_splits, desc="Outer CV progress", unit="fold")

# Loop through each fold of the outer CV
for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(pbar, start=1):
    fold_start_time = time.time()

    # Create the training and validation sets for current fold 
    X_outer_train = X_train_full.iloc[outer_train_idx]
    X_outer_val = X_train_full.iloc[outer_val_idx]
    y_outer_train = y_train_full.iloc[outer_train_idx]
    y_outer_val = y_train_full.iloc[outer_val_idx]

    # Set up GridSearchCV for the inner CV loop
    grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )

    # Fit best model found in inner CV of current fold on the complete outer training set
    grid.fit(X_outer_train, y_outer_train)

    # Best grid combination on the outer validation set 
    best_model_fold = grid.best_estimator_
    # Use the best model to predict classes on the outer validation set
    y_outer_pred = best_model_fold.predict(X_outer_val)
    # Use the best model to predict probabilities on the outer validation set
    y_outer_proba = best_model_fold.predict_proba(X_outer_val)[:, 1]

    fold_time = time.time() - fold_start_time
    
    # Store results for current fold
    fold_result = {
        "fold": fold_idx,
        "best_params": grid.best_params_,
        "inner_best_roc_auc": grid.best_score_,
        "outer_roc_auc": roc_auc_score(y_outer_val, y_outer_proba),
        "outer_f1": f1_score(y_outer_val, y_outer_pred),
        "outer_bal_acc": balanced_accuracy_score(y_outer_val, y_outer_pred),
        "outer_ap": average_precision_score(y_outer_val, y_outer_proba),
        "fold_time_sec": fold_time
    }

    # Store best parameters for current fold
    fold_result_param = {
        "fold": fold_idx,
        "best_params": grid.best_params_
    }

    # Fill in results and best parameters for current fold
    outer_results.append(fold_result)
    outer_best_params.append(fold_result_param)

    # Update progress bar with fold time
    pbar.set_postfix({
        "fold_time_s": f"{fold_time:.1f}",
        "outer_auc": f"{fold_result['outer_roc_auc']:.3f}"
    })

# Print total time taken for nested CV
total_time = time.time() - start_total_time
print(f"\nNested CV took {total_time:.2f} seconds")

# Convert results and best parameters to DataFrames
nested_results_df = pd.DataFrame(outer_results)
nested_results_df_params = pd.DataFrame(outer_best_params)

# Print results per fold
print("\nNested CV results per outer fold:")
print(nested_results_df[[
    "fold", "inner_best_roc_auc", "outer_roc_auc",
    "outer_f1", "outer_bal_acc", "outer_ap", "fold_time_sec"
]])

# Print best parameters per fold
print("\nNested CV best parameters per outer fold:")
print(nested_results_df_params[['fold', 'best_params']])

# Print mean performance across folds
print("\nMean nested CV performance:")
print("Mean outer ROC AUC:", nested_results_df["outer_roc_auc"].mean())
print("Std outer ROC AUC:", nested_results_df["outer_roc_auc"].std())
print("Mean outer F1:", nested_results_df["outer_f1"].mean())
print("Mean outer Balanced Accuracy:", nested_results_df["outer_bal_acc"].mean())
print("Mean outer Average Precision:", nested_results_df["outer_ap"].mean())
print("Mean outer fold runtime (sec):", nested_results_df["fold_time_sec"].mean())

#%%
#-----------------------------------
# 6. Refit/tune best model on full training set
#-----------------------------------
# Get (hyper)parameters from fold with the best outer ROC AUC 
best_idx = nested_results_df["outer_roc_auc"].idxmax()
best_fold = nested_results_df.loc[best_idx]
final_best_params = best_fold["best_params"]

# Refit best model on full training set using  best hyperparameters found in nested CV
best_model_final = clone(pipeline)
best_model_final.set_params(**final_best_params)
best_model_final.fit(X_train_full, y_train_full)

print("\nBest hyperparameters from nested CV:")
for param, value in final_best_params.items():
    print(f"  {param}: {value}")

# Print performance of best parameters on full training set 
print("\nPerformance of best parameters on full training set:")
print("ROC AUC:", best_fold["outer_roc_auc"])
print("F1:", best_fold["outer_f1"])
print("Balanced Accuracy:", best_fold["outer_bal_acc"])
print("Average Precision:", best_fold["outer_ap"])

#%%
#-----------------------------------
# 6b. Learning curve on full training set
#-----------------------------------
# Set up a StratifiedKFold for the learning curve
cv_learning = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a figure for the learning curve
fig, ax = plt.subplots(figsize=(8, 5))

# Define common parameters for the learning curve display
common_params = {
    "X": X_train_full,
    "y": y_train_full,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": cv_learning,
    "score_type": "both",
    "n_jobs": -1,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "ROC AUC",
}

# Generate the learning curve
LearningCurveDisplay.from_estimator(
    best_model_final,
    **common_params,
    ax=ax
)

# Plot learning curve
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["Training ROC AUC", "Validation ROC AUC"])
ax.set_title("Learning Curve - Logistic Regression")
ax.set_xlabel("Training set size")
ax.set_ylabel("ROC AUC")
ax.grid(True)
plt.tight_layout()
plt.show()

#%%
#-----------------------------------
# 7. Final evaluation on untouched test set
#-----------------------------------
# Use best model to predict classes and probabilities on the untouched test set
y_pred_final = best_model_final.predict(X_test_final)
y_proba_final = best_model_final.predict_proba(X_test_final)[:, 1]

# Print final performance on the untouched test set
print("\nFinal test set performance")
print(classification_report(y_test_final, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test_final, y_pred_final))
print("Final ROC AUC:", roc_auc_score(y_test_final, y_proba_final))
print("Final F1:", f1_score(y_test_final, y_pred_final))
print("Final Balanced Accuracy:", balanced_accuracy_score(y_test_final, y_pred_final))
print("Final Average Precision:", average_precision_score(y_test_final, y_proba_final))

#%%
#-----------------------------------
# 8. Plots for final model
#-----------------------------------
# Create confusion matrix
cm = confusion_matrix(y_test_final, y_pred_final)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Normal', 'Abnormal'],
    yticklabels=['Normal', 'Abnormal']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Test Set) - Logistic Regression')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_final, y_proba_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Final Test Set) - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_final, y_proba_final)
ap = average_precision_score(y_test_final, y_proba_final)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Final Test Set) - Logistic Regression')
plt.legend(loc='lower left')
plt.show()

#%%
