'''
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

import data_utilities as du

# from Approaching Almost Any Machine Learning Problem, pg157
class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        '''Custom wrapper for univariate feature selection taken from page 157 of Approaching Almost Any Machine Learning Problem by Abishek Thakur.
        Input: n_features = int representing number of features to select for training
            problem_type = string for either 'classification' or 'regression' problem to model
            scoring = string as method to evaluate features
        Output: a scikit-learn selection model
        '''
        self.n_features = n_features
        self.problem_type = problem_type
        self.scoring = scoring
        # f_classif for ANOVA, chi2 for chi-squared,
        # mutual info for dependence between two variables, 0 if independent
        # f_regression for Pearson's Correlation Coefficient
        if problem_type == 'classification':
            valid_scoring = {
                'f_classif': f_classif,
                'chi2': chi2,
                'mutual_info_classif': mutual_info_classif
            }
        else:
            valid_scoring = {
                'f_regression': f_regression,
                'mutual_info_regression': mutual_info_regression
            }

        if scoring not in valid_scoring:
            raise Exception('Invalid scoring function')

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")

    def fit(self, X, y):
        return self.selection.fit(X,y)

    def transform(self, X):
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        return self.selection.fit_transform(X,y)

#
def get_UFS_features(X, y, n_features, problem_type, scoring):
    '''Use Univariate Feature Selection to get the n_features best training features.
    Input: X = dataframe with all data features
        y = array of labels
        n_features = integer representing number of desired features
        problem_type = string for either regression or classification
        scoring = string representing the scoring function desired
    Output: X = dataframe with n_features selected
        named_scores = dictionary of feature names and associated scores
    '''
    X,y = du.check_dataset_nulls(X,y)

    feats = X.columns

    # Select the best n_features after evaluating feature statistical correlation scores
    ufs = UnivariateFeatureSelection(n_features=n_features,
                                     problem_type=problem_type,
                                     scoring=scoring)
    X = ufs.fit_transform(X,y)
    feature_scores = ufs.selection.scores_

    num_index = 0
    named_scores = []
    for score in feature_scores:
        name = feats[num_index]
        named_scores.append({'feature':name, 'score':score})
        num_index+=1


    #print(named_scores)
    return X, named_scores[:n_features]

#
def get_RFE_features(X, y, n_features, problem_type):
    '''Use Recursive Feature Elimination to get the best features for training.
    Input: X = dataframe with all data features
        y = array of labels
        n_features = integer representing number of desired features
        problem_type = string for either regression or classification
    Output: X = dataframe with n_features selected
        feature_names = names of the subset n_features selected
    '''
    X,y = du.check_dataset_nulls(X,y)
    if problem_type == 'regression':
        est = LinearRegression()
    else:
        est = DecisionTreeClassifier()
    rfe = RFE(estimator=est, n_features_to_select=n_features, step=1)
    rfe.fit(X,y)
    ranks = rfe.ranking_
    print(ranks)

    feats = X.columns
    num_index = 0
    feature_names = []
    for rank in ranks:
        if rank == 1:
            name = feats[num_index]
            #print("keep", name)
            feature_names.append(name)
        else:
            name = feats[num_index]
            #print("drop", name)
            X = X.drop([name], axis=1)
        num_index+=1


    return X, feature_names

#
# get model datasets
def get_train_test_data(model_data, train_cols, target_col,
                        problem_type, scoring,
                        n_features=None, selection=None,
                        test_size=0.30, random_state=17):
    '''
    Split dataset into training and testing datasets for model training
    Parameters: model_data = dataframe,
        train_cols = list of strings for column names/features to be evaluated for training
        target_col = string representing the y-value to be predicted
        problem_type = string representing problem to be solved,
        ['classification', 'regression']
        scoring = string representing selection scoring criterion,
        ['f_regression','mutual_info_regression', 'chi2', 'f_classif',
        'mutual_info_classif']
        n_features = (optional) int representing number of features to be selected
        selection = (optional) string representing the type of feature selection to use, ['stat','RFE']
        test-size = float represeting percentage of data to be held back for testing, default to 30%
        random_state = int representing the random seed for reproducibility, default to 17
    Returns: X-train = set of training columns and rows
        X_test = set of testing columns and rows
        y_train = target column to be learned in training
        y_test = target column to be tested and scored against predictions
        features_used = array of column names, either selected or original
    '''
    y = model_data[target_col]
    X = model_data[list(train_cols)]

    #check for null values
    X, y = du.check_dataset_nulls(X, y)

    if selection == 'stat':
        # Select the best n_features after evaluating Pearsons correlation with f_regression
        X, named_scores = get_UFS_features(X, y, n_features, problem_type, scoring)
        #Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, named_scores
    elif selection == 'RFE':
        # Select the best features using Recursive Feature Elimination
        X, feature_names = get_RFE_features(X, y, n_features, problem_type)
        #Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, feature_names
    else:
        print("No feature selection used.")
        #Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, [list(train_cols)]

#
def get_selected_feature_names(selection, n_features, features_used):
    '''Get the names of the features selected if selection process was used.
    Input: selection = string used in selection process, ['stat', 'RFE']
        features_used = dictionary or list returned from selection process
    Output: return list of strings of names of features selected
    '''
    if selection == 'stat':
        selected_columns = []
        for i in range(0, n_features):
            feat = features_used[i]['feature']
            selected_columns.append(feat)
        return selected_columns
    elif selection == 'RFE':
        return features_used[:n_features]
    else:
        return features_used[0]

#
def train_and_score_model(model, X_train, y_train, X_test, y_test, problem_type='regression', md='model desc', ):
    '''
    Convert categorical values to one-hot encodings.
    Parameters: model = scikit-learn model object,
        md = string describing the model being trained, default to model-desc
    Returns: metrics_dict = dictionary holding the results of the training and testing runs
    '''
    # Train
    model.fit(X_train, y_train)

    #Predict
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    #Score the model
    if problem_type == 'regression':
        train_r2_score = r2_score(y_train, y_train_preds)
        test_r2_score = r2_score(y_test, y_test_preds)
        train_mse = mean_squared_error(y_train, y_train_preds)
        test_mse = mean_squared_error(y_test, y_test_preds)
        train_mae = mean_absolute_error(y_train, y_train_preds)
        test_mae = mean_absolute_error(y_test, y_test_preds)
        print(md)
        print("R2 train: %.3f, test: %.3f" % (train_r2_score, test_r2_score))
        print("MSE train: %.3f, test: %.3f" % (train_mse, test_mse))
        print("MAE train: %.3f, test: %.3f" % (train_mae, test_mae))
        metrics_dict = {'trainR2' : train_r2_score, 'testR2': test_r2_score,
                       'trainMSE': train_mse, 'testMSE': test_mse,
                       'trainMAE' : train_mae, 'testMAE': test_mae}
    else:
        train_acc = accuracy_score(y_train, y_train_preds)
        test_acc = accuracy_score(y_test, y_test_preds)
        train_prec = precision_score(y_train, y_train_preds)
        test_prec = precision_score(y_test, y_test_preds)
        train_recall = recall_score(y_train, y_train_preds)
        test_recall = recall_score(y_test, y_test_preds)
        print(md)
        print("Accuracy train: %.3f, test: %.3f" % (train_acc, test_acc))
        print("Precision train: %.3f, test: %.3f" % (train_prec, test_prec))
        print("Recall train: %.3f, test: %.3f" % (train_recall, test_recall))
        metrics_dict = {'train_acc' : train_acc, 'test_acc': test_acc,
                       'train_prec': train_prec, 'test_prec': test_prec,
                       'train_prec' : train_recall, 'test_recall': test_recall}

    return model, metrics_dict

#
def visualize_feature_importance(model, X_train, features_used, image_path, selection=None, model_type='tree'):
    '''Create a graphic of the selected features and their importance to model outcome.
    Input: model = trained estimator
        X_train = training dataset used for estimator
        features_used = array containing the column names used in training
        selection = string representing type of feature selection performed, ['RFE', 'stat', None]
        model_type = string representing whether linear estimator or tree-based, ['tree', 'linear']
        image_path = string for path/filename to be saved
    Output: a horizontal bar chart for feature importance.
    '''
    n_features = X_train.shape[1]
    features = get_selected_feature_names(selection, n_features, features_used)
    n_outputs = model.n_outputs_
    if model_type == 'linear':
        impts = model.coef_
    else:
        impts = model.feature_importances_
    plt.figure(figsize=(10,10))
    plt.barh(range(n_features), impts, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    # 'RF_'+selection+'_'+str(n_outputs)+'_10x10.png'
    plt.savefig(image_path)
    plt.show()
