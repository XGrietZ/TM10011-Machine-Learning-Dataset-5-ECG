#%%
# -----------------------------------
# 0. Imports
# -----------------------------------
from ecg.load_data import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from tqdm import tqdm

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc
)

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

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
    X, y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("\nTrain label distribution:\n", y_train_full.value_counts())
print("Test  label distribution:\n", y_test_final.value_counts())

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
    ("scaler", RobustScaler()),
    ("features", "passthrough"),
    ('smote', SMOTE(random_state=42)),
    ("classifier", SVC(kernel = 'poly', probability=True))   
])

param_grid = [
    {
        'scaler': [ "passthrough", RobustScaler(), RobustScaler(unit_variance = True)],
        "features": [PCA()],
        "features__n_components": [0.94,0.97],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto']
    },
    {
        'scaler': [ "passthrough", RobustScaler(), RobustScaler(unit_variance = True)],
        "features": [SelectKBest()],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto'],
        "features__k": [20, 50, 100]
    }
    ]

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

#%%
# Inner-loop grid: modelselectie + hyperparameters
param_grid = [
    {
        'scaler': [preprocessing.RobustScaler(), 
                   "passthrough", 
                   preprocessing.RobustScaler(unit_variance = False), 
                   preprocessing.RobustScaler(unit_variance = True)],
        "features": [pca],
        "features__n_components": [0.94,0.97],
        "classifier": [svc],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto']
    },
    {
        "features": [kbest],
        "classifier": [svc],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto'],
        "features__k": [20, 50, 100]
    }
    ]


#%% OUD!!!!!

scaler = preprocessing.RobustScaler()
scaler.fit(X_train, y_train)
X_scaled = scaler.transform(X_train)

sm = SMOTE(sampling_strategy=1, random_state=None, k_neighbors=5)
X_balanced, y_balanced = sm.fit_resample(X_train, y_train)

#%% OUD EN NU AAN HET RUNNEN
# pipeline = ImbPipeline('scalar', scalar, 'balancing', sm, 'feature reduction', PCA(), 'svm', svm.SVC(kernel="poly"))

# sm = SMOTE(sampling_strategy=1, random_state=None, k_neighbors=5)
# X_balanced, y_balanced = sm.fit_resample(X_train, y_train)

# # classifications
# svc = svm.SVC(kernel='linear')
# scaler = preprocessing.RobustScaler()

# pca = PCA()
# rfe = feature_selection.RFE(estimator=svc)
# kbest = feature_selection.SelectKBest()

# pipeline = Pipeline([( "scaler" , scaler),
#                      ("features","passthrough"),
#                        ("classifier",svc)])
# # import Grid Search class
# # make lists of different parameters to check
# ParameterGrid = {
#   'features':[pca,rfe,kbest]
#   }
# # initialize
# grid_pipeline = GridSearchCV(pipeline,ParameterGrid,cv=5)
# # fit
# grid_pipeline.fit(X_balanced,y_balanced)
# grid_pipeline.best_params_
# print("Beste feature extractor:", grid_pipeline.best_params_)
# print("Beste score:", grid_pipeline.best_score_)



# %% 


#%%
# soft_voting = VotingClassifier(
#     estimators=[ ('svc', svc)],
#     voting='soft'
# )

# soft_voting.fit(X_train, y_train)
# y_pred_soft = soft_voting.predict(X_test_final)
# print(f"Soft Voting Accuracy: {accuracy_score(y_test_final, y_pred_soft):.2f}")

#%%
# Modellen
svc = svm.SVC(kernel = 'poly', probability=True, class_weight='balanced')

# Feature extractors
pca = PCA(n_components=0.94)
kbest = SelectKBest()

outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("scaler", preprocessing.RobustScaler()),
    ("features", "passthrough"),
    ('smote', SMOTE(random_state=42)),
    ("classifier", svc)   
])

# Inner-loop grid: modelselectie + hyperparameters
param_grid = [
    {
        'scaler': [preprocessing.RobustScaler(), 
                   "passthrough", 
                   preprocessing.RobustScaler(unit_variance = False), 
                   preprocessing.RobustScaler(unit_variance = True)],
        "features": [pca],
        "features__n_components": [0.94,0.97],
        "classifier": [svc],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto']
    },
    {
        "features": [kbest],
        "classifier": [svc],
        "classifier__C": [0.001,0.1,1,10],
        "classifier__gamma": ['scale','auto'],
        "features__k": [20, 50, 100]
    }
    ]

# Inner CV
grid = GridSearchCV(pipeline, 
                    param_grid, 
                    cv=inner_cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=3
)


# Outer CV
nested_scores = cross_val_score(grid, X_train, y_train, cv=outer_cv)

print("Nested CV scores per fold:", nested_scores)
print("Gemiddelde nested CV score:", np.mean(nested_scores))

# Fit op volledige training data om beste model te inspecteren
grid.fit(X_train, y_train)
print("Beste parameters:", grid.best_params_)
print("Beste inner-loop score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_final)
y_proba = best_model.predict_proba(X_test_final)[:, 1]

print("\nFinal test set performance")
print("Confusion Matrix:\n", confusion_matrix(y_test_final, y_pred))
print("Final ROC AUC:", roc_auc_score(y_test_final, y_proba))
print("Final F1:", f1_score(y_test_final, y_pred))
print("Final Balanced Accuracy:", balanced_accuracy_score(y_test_final, y_pred))
print("Final Average Precision:", average_precision_score(y_test_final, y_proba))

cm = confusion_matrix(y_test_final, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Test Set)')
plt.show()

fpr, tpr, _ = roc_curve(y_test_final, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Final Test Set)')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test_final, y_proba)
ap = average_precision_score(y_test_final, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Final Test Set)')
plt.legend(loc='lower left')
plt.show()

train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
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
plt.grid()
plt.show()
# %%
