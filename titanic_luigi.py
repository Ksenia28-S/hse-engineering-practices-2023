import luigi
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class PrepareDataTask(luigi.Task):

    predict_var = luigi.Parameter()
    cache_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.cache_dir + f'prepared_data_{self.predict_var}.csv')

    def run(self):
        
        data = pd.read_csv('train.csv')
        prepared_data = self.prepare_data(data)
        prepared_data.to_csv(self.output().path, index=False)

    def prepare_data(self, data):
        
        data.drop(data[pd.isnull(data['Embarked'])].index, inplace=True)
        
        data["Title_raw"] = data["Name"].str.extract('([A-Za-z]+)\.')

        title_mapping = {
            "Ms": "Miss",
            "Mlle": "Miss",
            "Miss": "Miss",
            "Mme": "Mrs",
            "Mrs": "Mrs",
            "Mr": "Mr",
            "Master": "Master"
        }

        data["Title"] = data["Title_raw"].map(title_mapping).fillna("Other")
        
        means = data.groupby('Title_raw')['Age'].mean()
        map_means = means.to_dict()
        idx_nan_age = data['Age'].isnull()
        data.loc[idx_nan_age, 'Age'] = data['Title_raw'].map(map_means)

        data['Relatives'] = data['SibSp'] + data['Parch']
        
        data.drop(columns=['Cabin', 'PassengerId', 'Ticket', 'Name', 'Title_raw'], inplace=True)

        
        data['Sex'] = pd.Categorical(data['Sex'])
        data['Embarked'] = pd.Categorical(data['Embarked'])
        data['Title'] = pd.Categorical(data['Title'])
        if self.predict_var == 'Survived':
            data['Pclass'] = pd.Categorical(data['Pclass'])
        data = pd.get_dummies(data, drop_first=True, dtype=int)

        return data

class SplitTrainTestTask(luigi.Task):

    predict_var = luigi.Parameter()
    cache_dir = luigi.Parameter()

    def requires(self):
        return PrepareDataTask(predict_var=self.predict_var, cache_dir = self.cache_dir)

    def output(self):
        return (luigi.LocalTarget(self.cache_dir + f'X_train_{self.predict_var}.csv'),
                luigi.LocalTarget(self.cache_dir + f'Y_train_{self.predict_var}.csv'),
                luigi.LocalTarget(self.cache_dir + f'X_test_{self.predict_var}.csv'),
                luigi.LocalTarget(self.cache_dir + f'Y_test_{self.predict_var}.csv'))

    def run(self):
        
        with self.input().open('r') as f:
            prepared_data = pd.read_csv(f)
        X_train, X_test, Y_train, Y_test = self.split_train_test(prepared_data)

        
        X_train.to_csv(self.output()[0].path, index=False)
        Y_train.to_csv(self.output()[1].path, index=False, header=True)
        X_test.to_csv(self.output()[2].path, index=False)
        Y_test.to_csv(self.output()[3].path, index=False, header=True)

    def split_train_test(self, data):
        X = data[[col for col in data.columns if col not in [self.predict_var]]]
        y = data[self.predict_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
        return X_train, X_test, y_train, y_test

class TrainModelsTask(luigi.Task):

    predict_var = luigi.Parameter()
    cache_dir = luigi.Parameter()
    classifiers = luigi.DictParameter()
    param_grids = luigi.DictParameter()

    def requires(self):
        return SplitTrainTestTask(predict_var=self.predict_var, cache_dir=self.cache_dir)

    def output(self):
        return luigi.LocalTarget(self.cache_dir + f'train_models_{self.predict_var}.csv')

    def run(self):
        
        X_train = pd.read_csv(self.input()[0].path)
        Y_train = pd.read_csv(self.input()[1].path)
        X_test = pd.read_csv(self.input()[2].path)
        Y_test = pd.read_csv(self.input()[3].path)

        classifiers = eval(self.classifiers)
        param_grids = eval(self.param_grids)

        results = self.train_models(X_train, X_test, Y_train, Y_test, classifiers, param_grids)
        results.to_csv(self.output().path, index=False)

    def train_models(self, X_train, X_test, Y_train, Y_test, classifiers, param_grids):
        results = pd.DataFrame(columns=["Classifier", "Best Params", "Train Accuracy", "Test Accuracy"])
        for name, clf in classifiers.items():
            print(f"Train {name} model")

            model = clf
            param_grid = param_grids[name]

            start = time.time()
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, Y_train)
            end = time.time()
            print('Training time:', end - start)

            print("Best Hyperparameters:", grid_search.best_params_)
            best_model = grid_search.best_estimator_

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            accuracy_train = accuracy_score(Y_train, y_pred_train)
            accuracy_test = accuracy_score(Y_test, y_pred_test)

            print("Accuracy on the train set:", accuracy_train)
            print("Accuracy on the test set:", accuracy_test)

            results.loc[len(results)] = [name,
                                        grid_search.best_params_,
                                        accuracy_train,
                                        accuracy_test]

            print("-----------------------------------------------------\n")

        results = results.sort_values("Test Accuracy", ascending=False)

        results = results.reset_index()
        results = results.drop(columns = ['index'])

        return results

if __name__ == '__main__':

    classifiers = '''
{
        "KNN": KNeighborsClassifier(),
        "LR": LogisticRegression(max_iter=10000),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "SVM": SVC(),
        "LGBM": LGBMClassifier()
}
'''

    param_grids = '''
{
        "KNN": {'n_neighbors': [9, 15, 20, 30 , 40],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
        "LR": {'C': np.logspace(-4, 4, 9),
               'tol': [0.1, 0.01, 0.001, 0.0001],
                "solver": ['lbfgs', 'liblinear']
            },
        "DT": {'criterion': ['gini', 'entropy'],
               'max_depth': [2, 5, 10, 20, 30],
               'min_samples_split': [5, 10, 20],
               'min_samples_leaf': [4, 5, 10]
            },
        "RF": {'n_estimators': [50, 100, 200],
               'max_depth': [2, 5, 10, 20, 30],
               'min_samples_split': [5, 10, 20],
               'min_samples_leaf': [5, 10, 20],
               'bootstrap': [True, False]
            },
        "SVM": {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
        "LGBM": {'num_leaves': [30, 50, 100],
                 'learning_rate': [0.01, 0.05, 0.1],
                 'n_estimators': [50, 100, 200],
                 'max_depth': [10, 20],
                 'min_child_samples': [20, 40, 60],
                 'reg_alpha': [0.0, 0.1],
                 'reg_lambda': [0.0, 0.1]
            }
}
'''

    # Predict 'Pclass'
    pclass_task = TrainModelsTask(predict_var='Pclass', cache_dir='luigi_cache/', classifiers=classifiers, param_grids=param_grids)
    luigi.build([pclass_task], local_scheduler="--scheduler-host localhost")

    # Predict 'Survived'
    #survived_task = TrainModelsTask(data_path=data_path, cache_dir='luigi_cache/', predict_var='Survived', classifiers=classifiers, param_grids=param_grids)
    #luigi.build([survived_task], local_scheduler="--scheduler-host localhost")
