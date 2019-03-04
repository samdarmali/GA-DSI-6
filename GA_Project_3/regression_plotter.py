import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from regression_models import regression_models
from feature_scaler import scaler


class reg_plotter(object):
    
    def __init__(self, name='reg_plotter'):
        self.name = name
    
    def plot_linreg_scores(feat_index, feat_corr_df, temp_splits, poly=False):
        '''
        Plots performance of linear regression models against number of features used per model. 
        Scores plotted:
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        
        Additionally, saves performance of each model in a dataframe.
        Scores saved:
        - Number of features
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        '''
        # Unscaled X values
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = temp_splits

        # Save values that we will use for plotting in a dictionary
        plot_dict = {'Features': [],
                     'R^2 train': [],
                     'Cross-Validation R^2 mean': [],
                     'Cross-Validation R^2 std': [],
                     'R^2 test': [],
                     'RMSE': []}

        for i in feat_index:
            # Correlations
            i_corr = feat_corr_df.head(i)

            # Scale X
            X_train, X_test, y_train, y_test = scaler.feat_scaler(X_train_temp[i_corr.index], 
                                                                  X_test_temp[i_corr.index], 
                                                                  y_train_temp, 
                                                                  y_test_temp, 
                                                                  poly=poly)

            # Train, test and score model based on above preprocessing
            model, results = regression_models.linear_reg(X_train, X_test, y_train, y_test, poly_name=False, features=i)

            # Update plot_dict with values we are interested in
            plot_dict['Features'].append(i)
            plot_dict['R^2 train'].append(results['R^2 train'])
            plot_dict['Cross-Validation R^2 mean'].append(results['Cross-Validation R^2 mean'])
            plot_dict['Cross-Validation R^2 std'].append(results['Cross-Validation R^2 std'])
            plot_dict['R^2 test'].append(results['R^2 test'])
            plot_dict['RMSE'].append(results['RMSE'])

        scores_df = pd.DataFrame(plot_dict)

        # Plot results
        plt.figure(figsize=(10,7));
        plt.plot(plot_dict['Features'], plot_dict['R^2 train'], label='R^2 train');
        plt.plot(plot_dict['Features'], plot_dict['Cross-Validation R^2 mean'], label='CV R^2 mean');
        plt.plot(plot_dict['Features'], plot_dict['R^2 test'], label='R^2 test');
        plt.title('R^2 Scores by Number of Features', fontsize=15, pad=20)
        plt.xlabel('Number of Features', fontsize=15, labelpad=20)
        plt.ylabel('R^2', fontsize=15, labelpad=20)
        plt.legend()
        return scores_df
    
    
    def plot_ridgereg_scores(feat_index, feat_corr_df, temp_splits, poly=False):
        '''
        Plots performance of ridge regression models against number of features used per model. 
        Scores plotted:
        - R-squared train
        - Cross-validation r-squared train
        - R-squared test
        
        Additionally, saves performance of each model in a dataframe.
        Scores saved:
        - Number of features
        - Best alpha parameter
        - R-squared train
        - Cross-validation r-squared train
        - R-squared test
        '''
        # Unscaled X values
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = temp_splits

        # Save values that we will use for plotting in a dictionary
        plot_dict = {'Features': [],
                     'Alpha': [],
                     'R^2 train': [],
                     'Cross-Validation R^2 mean': [],
                     'Cross-Validation R^2 std': [],
                     'R^2 test': [],
                     'RMSE': []}

        for i in feat_index:
            # Correlations
            i_sp_corr = feat_corr_df.head(i)

            # Scale X
            X_train, X_test, y_train, y_test = scaler.feat_scaler(X_train_temp[i_sp_corr.index], 
                                                           X_test_temp[i_sp_corr.index], 
                                                           y_train_temp, 
                                                           y_test_temp, 
                                                           poly=poly)

            # Train, test and score model based on above preprocessing
            model, results = regression_models.ridge_reg(X_train, X_test, y_train, y_test, poly_name=False, features=i)

            # Update plot_dict with values we are interested in
            plot_dict['Features'].append(i)
            plot_dict['Alpha'].append(results['Alpha'])
            plot_dict['R^2 train'].append(results['R^2 train'])
            plot_dict['Cross-Validation R^2 mean'].append(results['Cross-Validation R^2 mean'])
            plot_dict['Cross-Validation R^2 std'].append(results['Cross-Validation R^2 std'])
            plot_dict['R^2 test'].append(results['R^2 test'])
            plot_dict['RMSE'].append(results['RMSE'])

        scores_df = pd.DataFrame(plot_dict)

        # Plot results
        plt.figure(figsize=(10,7));
        plt.plot(plot_dict['Features'], plot_dict['R^2 train'], label='R^2 train');
        plt.plot(plot_dict['Features'], plot_dict['Cross-Validation R^2 mean'], label='CV R^2 mean');
        plt.plot(plot_dict['Features'], plot_dict['R^2 test'], label='R^2 test');
        plt.title('R^2 Scores by Number of Features', fontsize=15, pad=20)
        plt.xlabel('Number of Features', fontsize=15, labelpad=20)
        plt.ylabel('R^2', fontsize=15, labelpad=20)
        plt.legend()
        return scores_df
    
    
    def plot_lassoreg_scores(feat_index, feat_corr_df, temp_splits, poly=False):
        '''
        Plots performance of lasso regression models against number of features used per model. 
        Scores plotted:
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        
        Additionally, saves performance of each model in a dataframe.
        Scores saved:
        - Number of features
        - Best alpha parameter
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        '''
        # Unscaled X values
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = temp_splits

        # Save values that we will use for plotting in a dictionary
        plot_dict = {'Features': [],
                     'Alpha': [],
                     'R^2 train': [],
                     'Cross-Validation R^2 mean': [],
                     'Cross-Validation R^2 std': [],
                     'R^2 test': [],
                     'RMSE': []}

        for i in feat_index:
            # Correlations
            i_sp_corr = feat_corr_df.head(i)

            # Scale X
            X_train, X_test, y_train, y_test = scaler.feat_scaler(X_train_temp[i_sp_corr.index], 
                                                           X_test_temp[i_sp_corr.index], 
                                                           y_train_temp['SalePrice'].values, 
                                                           y_test_temp['SalePrice'].values, 
                                                           poly=poly)

            # Train, test and score model based on above preprocessing
            model, results = regression_models.lasso_reg(X_train, X_test, y_train, y_test, poly_name=False, features=i)

            # Update plot_dict with values we are interested in
            plot_dict['Features'].append(i)
            plot_dict['Alpha'].append(results['Alpha'])
            plot_dict['R^2 train'].append(results['R^2 train'])
            plot_dict['Cross-Validation R^2 mean'].append(results['Cross-Validation R^2 mean'])
            plot_dict['Cross-Validation R^2 std'].append(results['Cross-Validation R^2 std'])
            plot_dict['R^2 test'].append(results['R^2 test'])
            plot_dict['RMSE'].append(results['RMSE'])

        scores_df = pd.DataFrame(plot_dict)

        # Plot results
        plt.figure(figsize=(10,7));
        plt.plot(plot_dict['Features'], plot_dict['R^2 train'], label='R^2 train');
        plt.plot(plot_dict['Features'], plot_dict['Cross-Validation R^2 mean'], label='CV R^2 mean');
        plt.plot(plot_dict['Features'], plot_dict['R^2 test'], label='R^2 test');
        plt.title('R^2 Scores by Number of Features', fontsize=15, pad=20)
        plt.xlabel('Number of Features', fontsize=15, labelpad=20)
        plt.ylabel('R^2', fontsize=15, labelpad=20)
        plt.legend()
        return scores_df
    
    
    def plot_elastnet_scores(feat_index, feat_corr_df, temp_splits, poly=False):
        '''
        Plots performance of lasso regression models against number of features used per model. 
        Scores plotted:
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        
        Additionally, saves performance of each model in a dataframe.
        Scores saved:
        - Number of features
        - Best alpha parameter
        - Best L1 parameter
        - R-squared train
        - Cross-validation r-squared mean
        - R-squared test
        '''
        # Unscaled X values
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = temp_splits

        # Save values that we will use for plotting in a dictionary
        plot_dict = {'Features': [],
                     'Alpha': [],
                     'L1': [],
                     'R^2 train': [],
                     'Cross-Validation R^2 mean': [],
                     'Cross-Validation R^2 std': [],
                     'R^2 test': [],
                     'RMSE': []}

        for i in feat_index:
            # Correlations
            i_sp_corr = feat_corr_df.head(i)

            # Scale X
            X_train, X_test, y_train, y_test = scaler.feat_scaler(X_train_temp[i_sp_corr.index], 
                                                           X_test_temp[i_sp_corr.index], 
                                                           y_train_temp['SalePrice'].values, 
                                                           y_test_temp['SalePrice'].values, 
                                                           poly=poly)

            # Train, test and score model based on above preprocessing
            model, results = regression_models.elastnet_reg(X_train, X_test, y_train, y_test, poly_name=False, features=i)

            # Update plot_dict with values we are interested in
            plot_dict['Features'].append(i)
            plot_dict['Alpha'].append(results['Alpha'])
            plot_dict['L1'].append(results['L1'])
            plot_dict['R^2 train'].append(results['R^2 train'])
            plot_dict['Cross-Validation R^2 mean'].append(results['Cross-Validation R^2 mean'])
            plot_dict['Cross-Validation R^2 std'].append(results['Cross-Validation R^2 std'])
            plot_dict['R^2 test'].append(results['R^2 test'])
            plot_dict['RMSE'].append(results['RMSE'])

        scores_df = pd.DataFrame(plot_dict)

        # Plot results
        plt.figure(figsize=(10,7));
        plt.plot(plot_dict['Features'], plot_dict['R^2 train'], label='R^2 train');
        plt.plot(plot_dict['Features'], plot_dict['Cross-Validation R^2 mean'], label='CV R^2 mean');
        plt.plot(plot_dict['Features'], plot_dict['R^2 test'], label='R^2 test');
        plt.title('R^2 Scores by Number of Features', fontsize=15, pad=20)
        plt.xlabel('Number of Features', fontsize=15, labelpad=20)
        plt.ylabel('R^2', fontsize=15, labelpad=20)
        plt.legend()
        return scores_df


