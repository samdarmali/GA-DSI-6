import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class classification_models(object):
    
    def __init__(self, name='classification_models'):
        self.name = name

    def logistic_regression(X_train, X_test, y_train, y_test, log_params, sampling=None, cv=10):
        '''
        Function for building a logistic regression classifier. Uses GridsearchCV to find the best logistic regression model. 
        Stores:
        - Best logistic regression model
        - Confusion matrix
        - Dictionary of results:
            - Model name
            - Best penalty
            - Best C
            - Number of coefficients
            - Cross-validation accuracy mean
            - Cross-validation accuracy standard deviation
            - Test Accuracy
            - Test Recall
            - Test Precision
        - Classification report
        '''
        # Model name
        name = 'Logistic Regression Model ({})'.format(sampling)
        
        # Fit grid searcher to find best model
        log_gridsearcher = GridSearchCV(LogisticRegression(), param_grid=log_params, cv=cv, verbose=1, iid=True)
        log_gridsearcher.fit(X_train, y_train)

        # Best estimator and its mean cross-validated accuracy score
        best_log = log_gridsearcher.best_estimator_
        best_score = log_gridsearcher.best_score_
        print('best estimator:', best_log)
        print('mean cv acc:', best_score)

        # Number of coefficients
        coefs = len(best_log.coef_[0])

        # Make predictions
        y_pred = best_log.predict(X_test)

        # Evaluate performance 
        conf_mat = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred)),
                                index=['is_not_abnormal', 'is_abnormal'],
                                columns=['predicted_not_abnormal','predicted_abnormal'])

        # Accuracy - proportion of correct guesses made by model, regardless of class
        # Recall - the percent of times the model correctly predicted 1 when the label was in fact 1
        # Precision - the percent of times the model was correct when it was predicting the true (1) class
        # Classification report - shows precision, recall and f1 for each class. 
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        class_report = metrics.classification_report(y_test, y_pred)

        results_dict = {'Model': name,
                        'Best penalty': log_gridsearcher.best_params_['penalty'], 
                        'Best C': log_gridsearcher.best_params_['C'], 
                        'No. of coefs:': coefs, 
                        'Best CV acc mean': best_score,
                        'Best CV acc std': log_gridsearcher.cv_results_['std_test_score'][log_gridsearcher.best_index_],
                        'Test Accuracy': accuracy,
                        'Test Recall': recall,
                        'Test Precision': precision}
        return best_log, conf_mat, results_dict, class_report
    
    
    def knn(X_train, X_test, y_train, y_test, knn_params, sampling=None, cv=5):
        '''
        Function for building a k-nearest-neighbors classifier. Uses GridsearchCV to find the best knn model. 
        Stores:
        - Best knn model
        - Confusion matrix
        - Dictionary of results:
            - Model name
            - Best metric 
            - Best n-neighbors
            - Cross-validation accuracy mean
            - Cross-validation accuracy standard deviation
            - Test Accuracy
            - Test Recall
            - Test Precision
        - Classification report
        '''
        # Model name
        name = 'K-Nearest Neighbors ({})'.format(sampling)
        
        # Fit grid searcher to find best model
        knn_gridsearcher = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=cv, verbose=1, iid=True)
        knn_gridsearcher.fit(X_train, y_train)

        # Best estimator and its mean cross-validated accuracy score
        best_knn = knn_gridsearcher.best_estimator_
        best_score = knn_gridsearcher.best_score_
        print('best estimator:', best_knn)
        print('mean cv acc:', best_score)

        # Make predictions with best knn estimator
        y_pred = best_knn.predict(X_test)

        # Evaluate performance 
        conf_mat = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred)),
                                index=['is_not_abnormal', 'is_abnormal'],
                                columns=['predicted_not_abnormal','predicted_abnormal'])

        # Accuracy - proportion of correct guesses made by model, regardless of class
        # Recall - the percent of times the model correctly predicted 1 when the label was in fact 1
        # Precision - the percent of times the model was correct when it was predicting the true (1) class
        # Classification report - shows precision, recall and f1 for each class. 
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        class_report = metrics.classification_report(y_test, y_pred)

        results_dict = {'Model': name,
                        'Metric': knn_gridsearcher.best_params_['metric'],
                        'N-neighbors': knn_gridsearcher.best_params_['n_neighbors'], 
                        'Best CV acc mean': best_score,
                        'Best CV acc std': knn_gridsearcher.cv_results_['std_test_score'][knn_gridsearcher.best_index_],
                        'Test Accuracy': accuracy,
                        'Test Recall': recall,
                        'Test Precision': precision}
        return best_knn, conf_mat, results_dict, class_report
    
    
    def rfc(X_train, X_test, y_train, y_test, rfc_params, sampling=None, cv=5):
        '''
        Function for building a random forest classifier. Uses GridsearchCV to find the best random forest model. 
        Stores:
        - Best random forest model
        - Confusion matrix
        - Dictionary of results:
            - Model name
            - Bootstrap
            - Max depth
            - Max features
            - Min samples leaf
            - Min samples split
            - Number of estimators
            - Cross-validation accuracy mean
            - Cross-validation accuracy standard deviation
            - Test Accuracy
            - Test Recall
            - Test Precision
        - Classification report
        '''
        # Model name
        name = 'Random Forest Classifier ({})'.format(sampling)
        
        # Fit grid searcher to find best model
        rfc_gridsearcher = GridSearchCV(RandomForestClassifier(), param_grid=rfc_params, cv=cv, verbose=1, iid=True)
        rfc_gridsearcher.fit(X_train, y_train)

        # Best estimator and its mean cross-validated accuracy score
        best_rfc = rfc_gridsearcher.best_estimator_
        best_score = rfc_gridsearcher.best_score_
        print('best estimator:', best_rfc)
        print('mean cv acc:', best_score)

        # Make predictions with best knn estimator
        y_pred = best_rfc.predict(X_test)

        # Evaluate performance 
        conf_mat = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred)),
                                index=['is_not_abnormal', 'is_abnormal'],
                                columns=['predicted_not_abnormal','predicted_abnormal'])

        # Accuracy - proportion of correct guesses made by model, regardless of class
        # Recall - the percent of times the model correctly predicted 1 when the label was in fact 1
        # Precision - the percent of times the model was correct when it was predicting the true (1) class
        # Classification report - shows precision, recall and f1 for each class. 
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        class_report = metrics.classification_report(y_test, y_pred)

        results_dict = {'Model': name,
                        'Bootstrap': rfc_gridsearcher.best_params_['bootstrap'],
                        'Max depth': rfc_gridsearcher.best_params_['max_depth'],
                        'Max features': rfc_gridsearcher.best_params_['max_features'],
                        'Min samples split': rfc_gridsearcher.best_params_['min_samples_split'], 
                        'Number of estimators': rfc_gridsearcher.best_params_['n_estimators'],
                        'Best CV acc mean': best_score,
                        'Best CV acc std': rfc_gridsearcher.cv_results_['std_test_score'][rfc_gridsearcher.best_index_],
                        'Test Accuracy': accuracy,
                        'Test Recall': recall,
                        'Test Precision': precision}
        return best_rfc, conf_mat, results_dict, class_report
    
    
    
    
    
    