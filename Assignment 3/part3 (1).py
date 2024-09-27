import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from DataLoader import DataLoader

data_path = "HW 3 Data and Source Codes/data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

random.seed(11)
np.random.seed(17)

knn_parameters = {
    'n_neighbors': [2, 3],
    'metric': ['cosine','euclidean','manhattan']
}

svm_parameters = {
    'C': [0.1, 1],
    'kernel': ['poly','rbf']
}

dectree_parameters = {
    'max_depth': [10, 20],
    'criterion': ['gini','entropy']
}

randomforest_parameters = {
    'n_estimators': [50],
    'max_depth': [10],
    'criterion': ['gini']
}

algorithms = {
    'SVM': (SVC(), svm_parameters),
    'KNN': (KNeighborsClassifier(),knn_parameters),
    'RandomForest': (RandomForestClassifier(),randomforest_parameters),
    'DecisionTree': (DecisionTreeClassifier(),dectree_parameters)
}

scaler=MinMaxScaler(feature_range=(-1, 1))
data_normalized=scaler.fit_transform(dataset)

outer_cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=np.random.randint(1, 1000))
inner_cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000))

results={}
for algo_name,(model,params) in algorithms.items():
    print(f"Running {algo_name}...")
    
    outer_scores_acc=[]
    outer_scores_f1=[]
    
    for train_index,test_index in outer_cv.split(data_normalized, labels):
        X_train,X_test=data_normalized[train_index],data_normalized[test_index]
        y_train,y_test=labels[train_index],labels[test_index]
        
        if algo_name=='RandomForest':
            inner_scores_acc=[]
            inner_scores_f1=[]
            for _ in range(5):
                inner_search=GridSearchCV(model, params, cv=inner_cv, scoring='accuracy', n_jobs=-1)
                inner_search.fit(X_train, y_train)
                
                best_model = inner_search.best_estimator_
                best_model.fit(X_train, y_train)
                
                y_pred=best_model.predict(X_test)
                acc=accuracy_score(y_test, y_pred)
                f1=f1_score(y_test, y_pred)
                
                inner_scores_acc.append(acc)
                inner_scores_f1.append(f1)
            
            acc = np.mean(inner_scores_acc)
            f1 = np.mean(inner_scores_f1)
            
        else:
            inner_search=GridSearchCV(model, params, cv=inner_cv, scoring='accuracy')
            inner_search.fit(X_train, y_train)
            
            best_model=inner_search.best_estimator_
            best_model.fit(X_train, y_train)
            
            y_pred=best_model.predict(X_test)
            acc=accuracy_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)
        
        outer_scores_acc.append(acc)
        outer_scores_f1.append(f1)

    results[algo_name] = {
        'Accuracy': np.mean(outer_scores_acc),
        'Accuracy_CI': 1.96 * np.std(outer_scores_acc),
        'F1 Score': np.mean(outer_scores_f1),
        'F1 Score_CI': 1.96 * np.std(outer_scores_f1)
    }

for algo_name, metrics in results.items():
    print(f"{algo_name} Results:")
    print(f"  Accuracy: {metrics['Accuracy']:.3f} +/- {metrics['Accuracy_CI']:.3f}")
    print(f"  F1 Score: {metrics['F1 Score']:.3f} +/- {metrics['F1 Score_CI']:.3f}")

#not implementing second last and last point as dont know how to