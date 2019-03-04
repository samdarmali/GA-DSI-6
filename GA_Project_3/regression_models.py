from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn import metrics
import numpy as np
import scipy.stats as stats
import pandas as pd


class regression_models(object):
    
    def __init__(self, name='regression_models'):
        self.name = name
    
    def linear_reg(X_train, X_test, y_train, y_test, poly_name=False, features=None, cv=10):
        '''
        Function for building a simple linear regression model. 
        Stores:
        - Model name
        - Number of coefficients
        - R-squared train score
        - Cross-validation r-squared score
        - R-squared test score
        - Root mean squared error (RMSE)
        '''
        # Model name
        if poly_name == False:
            name = 'Linear Regression Model ({})'.format(str(features) + ' features')
        else:
            name = 'Polynomial Linear Regression Model ({})'.format(str(features) + ' features')
        
        # Cross-validate model on training data
        lin_reg = LinearRegression()
        cv_scores = cross_val_score(lin_reg, X_train, y_train, cv=cv)
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)
        
        # fit model
        lin_reg.fit(X_train, y_train)

        # number of coefficients
        coefs = len(lin_reg.coef_[0])

        # make predictions
        y_pred = lin_reg.predict(X_test)

        # evaluate performance on train and test data 
        r2_train = lin_reg.score(X_train, y_train)
        r2_test = lin_reg.score(X_test, y_test)
        rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # save results in dictionary
        results_dict = {'Model': name,
                        'No. of coefs:': coefs,
                        'R^2 train': r2_train,
                        'Cross-Validation R^2 scores': cv_scores,
                        'Cross-Validation R^2 mean': cv_r2_mean,
                        'Cross-Validation R^2 std': cv_r2_std,
                        'R^2 test': r2_test,
                        'RMSE': rsme}
        return lin_reg, results_dict
    
        
    def ridge_reg(X_train, X_test, y_train, y_test, poly_name=False, features=None, cv=10):
        '''
        Function for building a ridge regression model. 
        Stores:
        - Model name
        - Number of coefficients
        - Best alpha parameter
        - R-squared train score
        - Cross-validation r-squared score
        - R-squared test score
        - Root mean squared error (RMSE)
        '''
        # model name
        if poly_name == False:
            name = 'Ridge Regression Model ({})'.format(str(features) + ' features')
        else:
            name = 'Polynomial Ridge Regression Model ({})'.format(str(features) + ' features')

        # Create an array of alpha values
        alpha_range = np.logspace(0, 5, 100)
        
        # Cross-validate model
        ridge_reg = RidgeCV(alphas=alpha_range)
        cv_scores = cross_val_score(ridge_reg, X_train, y_train, cv=cv)
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)

        # fit model
        ridge_reg.fit(X_train, y_train)

        # number of coefficients
        coefs = len(ridge_reg.coef_[0])

        # best alpha
        best_alpha = ridge_reg.alpha_

        # make predictions
        y_pred = ridge_reg.predict(X_test)

        # evaluate performance on train and test data 
        r2_train = ridge_reg.score(X_train, y_train)
        r2_test = ridge_reg.score(X_test, y_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # save results in dictionary
        results_dict = {'Model': name,
                        'No. of coefs:': coefs,
                        'Alpha': best_alpha,     
                        'R^2 train': r2_train,
                        'Cross-Validation R^2 scores': cv_scores,
                        'Cross-Validation R^2 mean': cv_r2_mean,
                        'Cross-Validation R^2 std': cv_r2_std,
                        'R^2 test': r2_test,
                        'RMSE': rmse}
        return ridge_reg, results_dict

    
    def lasso_reg(X_train, X_test, y_train, y_test, poly_name=False, features=None, cv=10):
        '''
        Function for building a lasso regression model. 
        Stores:
        - Model name
        - Number of coefficients
        - Best alpha parameter
        - R-squared train score
        - Cross-validation r-squared score
        - R-squared test score
        - Root mean squared error (RMSE)
        '''
        # model name
        if poly_name == False:
            name = 'Lasso Regression Model ({})'.format(str(features) + ' features')
        else:
            name = 'Polynomial Lasso Regression Model ({})'.format(str(features) + ' features')
        
        # Cross-validate model
        lasso_reg = LassoCV(n_alphas=200, max_iter=1000000, tol=0.001, random_state=1, cv=3)
        cv_scores = cross_val_score(lasso_reg, X_train, y_train, cv=cv)
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)

        # fit model
        lasso_reg.fit(X_train, y_train)

        # number of coefficients
        coefs = len(lasso_reg.coef_)

        # best alpha
        best_alpha = lasso_reg.alpha_

        # make predictions
        y_pred = lasso_reg.predict(X_test)

        # evaluate performance on train and test data 
        r2_train = lasso_reg.score(X_train, y_train)
        r2_test = lasso_reg.score(X_test, y_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # save results in dictionary
        results_dict = {'Model': name,
                        'Alpha': best_alpha,
                        'No. of coefs:': coefs,              
                        'R^2 train': r2_train,
                        'Cross-Validation R^2 scores': cv_scores,
                        'Cross-Validation R^2 mean': cv_r2_mean,
                        'Cross-Validation R^2 std': cv_r2_std,
                        'R^2 test': r2_test,
                        'RMSE': rmse}
        return lasso_reg, results_dict
    
    
    def elastnet_reg(X_train, X_test, y_train, y_test, poly_name=False, features=None, cv=10):
        '''
        Function for building an elastic net regression model. 
        Stores:
        - Model name
        - Number of coefficients
        - Best alpha parameter
        - Best l1 parameter
        - R-squared train score
        - Cross-validation r-squared score
        - R-squared test score
        - Root mean squared error (RMSE)
        '''
        # model name
        if poly_name == False:
            name = 'Elastic Net Regression Model ({})'.format(str(features) + ' features')
        else:
            name = 'Polynomial Elastic Net Regression Model ({})'.format(str(features) + ' features')
        
        # Cross-validate model
        elast_net = ElasticNetCV(n_alphas=100, l1_ratio=np.linspace(0.01,1,20), max_iter=1000000, random_state=1, cv=3)
        cv_scores = cross_val_score(elast_net, X_train, y_train, cv=cv)
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)

        # fit model
        elast_net.fit(X_train, y_train)

        # number of coefficients
        coefs = len(elast_net.coef_)

        # best alpha and l1 ratio
        best_alpha = elast_net.alpha_
        best_l1 = elast_net.l1_ratio_

        # make predictions
        y_pred = elast_net.predict(X_test)

        # evaluate performance on train and test data 
        r2_train = elast_net.score(X_train, y_train)
        r2_test = elast_net.score(X_test, y_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # save results in dictionary
        results_dict = {'Model': name,
                        'Alpha': best_alpha,
                        'L1': best_l1,
                        'No. of coefs:': coefs,              
                        'R^2 train': r2_train,
                        'Cross-Validation R^2 scores': cv_scores,
                        'Cross-Validation R^2 mean': cv_r2_mean,
                        'Cross-Validation R^2 std': cv_r2_std,
                        'R^2 test': r2_test,
                        'RMSE': rmse}
        return elast_net, results_dict
