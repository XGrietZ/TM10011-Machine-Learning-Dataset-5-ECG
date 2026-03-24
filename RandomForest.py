#%%
#-----------------------------------
# 0. Imports
#-----------------------------------
from ecg.load_data import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
    cross_val_score,
    ParameterGrid
)

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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.base import clone
from matplotlib.patches import Rectangle

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

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "label" not in df.columns:
    raise ValueError("Expected a column named 'label' in the CSV.")

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
pipeline = ImbPipeline([
    ('log_transform', 'passthrough'),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42,n_jobs=-1))
])

param_grid = {
    'log_transform': ['passthrough', FunctionTransformer(np.log1p, validate=False)],
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

param_list = list(ParameterGrid(param_grid))
print(f"Total models: {len(param_list)}")
print(f"Total fits (with {inner_cv.get_n_splits()}-fold CV): {len(param_list) * inner_cv.get_n_splits()}")


#%%
#-----------------------------------
# 5. Nested cross-validation with live tqdm info
#-----------------------------------
outer_results = []
outer_best_params = []

start_total_time = time.time()
outer_splits = list(outer_cv.split(X_train_full, y_train_full))

pbar = tqdm(outer_splits, desc="Outer CV progress", unit="fold")

for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(pbar, start=1):
    fold_start_time = time.time()

    X_outer_train = X_train_full.iloc[outer_train_idx]
    X_outer_val = X_train_full.iloc[outer_val_idx]
    y_outer_train = y_train_full.iloc[outer_train_idx]
    y_outer_val = y_train_full.iloc[outer_val_idx]

    grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_outer_train, y_outer_train)

    best_model_fold = grid.best_estimator_
    y_outer_pred = best_model_fold.predict(X_outer_val)
    y_outer_proba = best_model_fold.predict_proba(X_outer_val)[:, 1]

    fold_time = time.time() - fold_start_time

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

    outer_results.append(fold_result)
    outer_best_params.append(grid.best_params_)

    pbar.set_postfix({
        "fold_time_s": f"{fold_time:.1f}",
        "outer_auc": f"{fold_result['outer_roc_auc']:.3f}"
    })

total_time = time.time() - start_total_time
print(f"\nNested CV took {total_time:.2f} seconds")

nested_results_df = pd.DataFrame(outer_results)

print("\nNested CV results per outer fold:")
print(nested_results_df[[
    "fold", "inner_best_roc_auc", "outer_roc_auc",
    "outer_f1", "outer_bal_acc", "outer_ap", "fold_time_sec"
]])

print("\nMean nested CV performance:")
print("Mean outer ROC AUC:", nested_results_df["outer_roc_auc"].mean())
print("Std outer ROC AUC:", nested_results_df["outer_roc_auc"].std())
print("Mean outer F1:", nested_results_df["outer_f1"].mean())
print("Mean outer Balanced Accuracy:", nested_results_df["outer_bal_acc"].mean())
print("Mean outer Average Precision:", nested_results_df["outer_ap"].mean())
print("Mean outer fold runtime (sec):", nested_results_df["fold_time_sec"].mean())

#%%
#-----------------------------------
# 6. Refit best model on full training set
#-----------------------------------
final_grid = GridSearchCV(
    estimator=clone(pipeline),
    param_grid=param_grid,
    cv=inner_cv,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True
)

final_grid.fit(X_train_full, y_train_full)

final_model = final_grid.best_estimator_

print("\nBest final params from full training set:")
print(final_grid.best_params_)
print("Best inner CV ROC AUC on full training set:", final_grid.best_score_)

#%%
#-----------------------------------
# 7. Final evaluation on untouched test set
#-----------------------------------
y_pred_final = final_model.predict(X_test_final)
y_proba_final = final_model.predict_proba(X_test_final)[:, 1]

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
cm = confusion_matrix(y_test_final, y_pred_final)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Test Set)')
plt.show()

fpr, tpr, _ = roc_curve(y_test_final, y_proba_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Final Test Set)')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test_final, y_proba_final)
ap = average_precision_score(y_test_final, y_proba_final)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Final Test Set)')
plt.legend(loc='lower left')
plt.show()

#%%
#-----------------------------------
# 9. Learning curve on full training set
#-----------------------------------
train_sizes, train_scores, val_scores = learning_curve(
    estimator=final_model,
    X=X_train_full,
    y=y_train_full,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training ROC AUC')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation ROC AUC')
plt.xlabel('Training examples')
plt.ylabel('ROC AUC')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
