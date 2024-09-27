import pickle
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

dataset, labels = pickle.load(open("HW 3 Data and Source Codes/data/part2_dataset2.data", "rb"))

C_values=[0.1, 1, 10]
kernel_values=['linear', 'rbf', 'poly']

n_repeats=5
n_splits=10
cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
svm=SVC()

results = []

for C in C_values:
    for kernel in kernel_values:
        param_grid={'C': [C], 'kernel': [kernel]}
        grid_search=GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy')
        
        parameter_results=[]
        for train_index, test_index in cv.split(dataset, labels):
            X_train,X_test=dataset[train_index], dataset[test_index]
            y_train,y_test=labels[train_index], labels[test_index]
            scaler=StandardScaler()
            X_train_scaled=scaler.fit_transform(X_train)
            grid_search.fit(X_train_scaled, y_train)
            best_parameter=grid_search.best_params_
            best_score=grid_search.best_score_

            parameter_results.append({'params': best_parameter, 'score': best_score})
        
        results.extend(parameter_results)

best_score=max(results, key=lambda x: x['score'])
best_hyperparameters = best_score['params']

print(f"The best hyperparameter overall is: {best_hyperparameters}")

param_scores={}
for result in results:
    params=tuple(result['params'].items())
    param_scores[params]=param_scores.get(params,[])+[result['score']]

confidence_intervals={}
for param, scores in param_scores.items():
    mean_score=np.mean(scores)
    std_score=np.std(scores)
    confidence_intervals[param]={'mean':mean_score,'std':std_score}

print("Confidence intervals for each hyperparameter:")
for param, values in confidence_intervals.items():
    print(f"Hyperparameters: {param}, Mean: {values['mean']:.3f}, Std: {values['std']:.3f}")