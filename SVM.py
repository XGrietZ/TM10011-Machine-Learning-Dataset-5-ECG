#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score, ParameterGrid
from sklearn.metrics import (
    auc,  confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score, roc_curve, precision_recall_curve, average_precision_score)
from sklearn import svm
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import svm

from imblearn.over_sampling import SMOTE

#%%
#-----------------------------------
# 1. Load data
#-----------------------------------

from ecg.load_data import load_data
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg/ecg_data/ecg_data.csv',index_col=0)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

print("\nDataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

#%%
#-----------------------------------
# 2. Train/Test split (hold-out)
#-----------------------------------   
X_train, X_test_final, y_train, y_test_final = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("\nTrain label distribution:\n", y_train.value_counts())
print("Test  label distribution:\n", y_test_final.value_counts())

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
from sklearn import svm, preprocessing, feature_selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


#%%
# soft_voting = VotingClassifier(
#     estimators=[ ('svc', svc_rbf)],
#     voting='soft'
# )

# soft_voting.fit(X_train, y_train)
# y_pred_soft = soft_voting.predict(X_test_final)
# print(f"Soft Voting Accuracy: {accuracy_score(y_test_final, y_pred_soft):.2f}")

#%%
# Modellen
svc_rbf = svm.SVC(kernel = 'rbf', probability=True, class_weight='balanced')

# Feature extractors
pca = PCA(n_components=0.94)
kbest = SelectKBest()

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("scaler", preprocessing.RobustScaler()),
    ("features", "passthrough"),
    ('smote', SMOTE(random_state=42)),
    ("classifier", svc_rbf)   # placeholder
])

# Inner-loop grid: modelselectie + hyperparameters
param_grid = [
    {
        'scaler': [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.MinMaxScaler()],
        "features": [pca],
        "classifier": [svc_rbf],
        "classifier__C": np.logspace(-2,3,10),
        "classifier__gamma": np.logspace(-4,1,10)
    },
    {
        "features": [kbest],
        "classifier": [svc_rbf],
        "classifier__C": np.logspace(-2,3,10),
        "classifier__gamma": np.logspace(-4,1,10),
        "features__k": [5, 10, 20]
    }
    ]

# Inner CV
grid = GridSearchCV(pipeline, 
                    param_grid, 
                    cv=inner_cv, 
                    scoring='f1',
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
