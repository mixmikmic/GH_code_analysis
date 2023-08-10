import pandas as pd
import math
import csv
import os
import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from itertools import combinations
from merf.merf import MERF
lme = importr('lme4')
nlme = importr('nlme')
arm = importr('arm')
pandas2ri.activate()
get_ipython().run_line_magic('matplotlib', 'inline')

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

class EnsembleModel():
    """ Class definition for an Ensemble Model using a weighted
    average of LME and MERF predictions
    """
        
    def __init__(self, training_sets, validation_sets, testing_sets, model_info, feature_info, output_info):
        """ 
        Parameters:
        ----------
        *training_sets: list of dataframes of training data
        *testing_sets: list of dataframes of testing data
        *validation_sets: list of dataframes of validation data
        *model_info = model information {'modelType':type, 'weight':#}
        *feature_info: contains information about what features to use {'random_effect_features':[], 'fixed_effect_features':[], 'intercept':True/False, 'clusterBy':feature}
        *output_info: {'featureName': name, 'logOutput': True/False}
        """
        self.model_info = model_info
        self.feature_info = feature_info
        self.output_info = output_info
        if self.output_info['log_output']:
            self.fullOutputName = 'Log:'+self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        else:
            self.fullOutputName = self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        self.merfModel = None
        self.lmeModel = None
        self.training_sets = training_sets
        self.testing_sets = testing_sets
        self.validation_sets = validation_sets
        self.results = None
    
    def RMSE(self):
        """Gets RMSE for results"""
        return math.sqrt(mean_squared_error(self.results[self.fullOutputName], self.results['prediction']))

    def MAE(self):
        """Gets MAE for results"""
        return np.mean(abs(self.results[self.fullOutputName]-self.results['prediction']))
    
    def get_results(self, training_sets, testing_sets, validation_sets):
        """Fits an Ensemble Model to every training set, makes 
        predictions on its test split, and aggregates results
        
         Parameters:
        ----------
        *training_sets: list of dataframes of training data
        *testing_sets: list of dataframes of testing data
        *validation_sets: list of dataframes of validation data
        
        Return:
        ------
        Dataframe of testing data with predictions
        """
        
        if 'weight' not in self.model_info or not self.feature_info:
            print('Specify features and weights!')
            return 
        
        result_df = []
        for i in range(len(training_sets)):
            train = training_sets[i].copy()
            validate = validation_sets[i].copy()
            test = testing_sets[i].copy()
            
            feature_info = {'intercept': self.feature_info['intercept'], 'clusterBy':self.feature_info['clusterBy'], 'fixed_effect_features': self.feature_info['fixed_effect_features'], 'random_effect_features': self.feature_info['random_effect_features']}

            #LME Model
            LMEmodel = LinearMixedEffectsModel(self.model_info, feature_info, self.output_info)
            LMEresults = LMEmodel.get_results([train],[test],[validate])
            self.lmeModel = LMEmodel
            
            #MERF Model
            MERFmodel = MERFModel(self.model_info, feature_info, self.output_info)
            MERFresults = MERFmodel.get_results([train],[test],[validate])
            self.merfModel = MERFmodel
            
            y_predMERF = MERFresults['prediction']
            y_predLME = LMEresults['prediction']
            averaged_predictions = []
            for i in range(len(y_predMERF)):
                averaged_predictions.append((self.model_info['weight']*y_predMERF[i]+(2-self.model_info['weight'])*y_predLME[i])/2.0)
            
            test.loc[:,'prediction'] = averaged_predictions
            result_df.append(test)
        final_result = pd.concat(result_df).reset_index(drop=True)
        self.results = final_result
        return final_result
    
    def pickBestModel(self,featuresToTry):
        """ Function that uses cross-validation to choose the fixed/random
        features to use and the weight to use that minimizes RMSE on the 
        validation set. Then the best model is evaluated on the testing set.
        
        For features, every combination of size 1-len(featuresToTry) for the
        fixed features and every combination of size 0-len(featuresToTry) for
        the random features is tried. For the weights, all values from 0
        to 2 with a step size of 0.1 are tried.
        
        Parameters:
        ----------
        *featuresToTry: list of features to use in the model, each possible
        combination of these features for the random and fixed effects is tried
        
        Return:
        ------
        Dataframe of testing data with predictions
        """
        if len(self.training_sets) > 1:
            print("Only able to pick best model on Predict Second Half")
            return
        train = self.training_sets[0]
        validate = self.validation_sets[0]
        test = self.testing_sets[0]
        
        featWeightcombos = []
        for n1 in range(1,len(featuresToTry)+1):
            for fixedfeat in combinations(featuresToTry, n1): #try all combinations of size 1-len(featuresToTry) for fixed features
                for n2 in range(len(featuresToTry)+1):
                    for randomfeat in combinations(featuresToTry, n2): #try all combinations of size 0-len(featuresToTry) for random features
        
                        feature_info = {'intercept': self.feature_info['intercept'], 'clusterBy':self.feature_info['clusterBy'] , 'fixed_effect_features': fixedfeat, 'random_effect_features': randomfeat}

                        #LME Model
                        LMEmodel = LinearMixedEffectsModel(self.model_info, feature_info, self.output_info)
                        LMEresults = LMEmodel.get_results([train],[validate])
                        
                        #MERF Model
                        MERFmodel = MERFModel(self.model_info, feature_info, self.output_info)
                        MERFresults = MERFmodel.get_results([train],[validate])
                        
                        y_predMERF = MERFresults['prediction']
                        y_predLME = LMEresults['prediction']
                        
                        #pick weights
                        errors = []
                        for weight in list(np.arange(0,2.1,0.1)): #weigh
                            averaged_predictions = []
                            for i in range(len(y_predMERF)):
                                averaged_predictions.append((weight*y_predMERF[i]+(2-weight)*y_predLME[i])/2.0)
                            averageRMSE = math.sqrt(mean_squared_error(validate['PVTMeanInverseRT(t)'], averaged_predictions))
                            errors.append([averageRMSE,weight])
                            
                        best_opt = sorted(errors, key=lambda x: x[0])[0]
                        featWeightcombos.append([fixedfeat, randomfeat, best_opt])
        best_model = sorted(featWeightcombos, key=lambda x: x[2][0])[0]
        print('Best Model Fixed Effect Feature:',best_model[0])
        print('Best Model Random Effect Feature:',best_model[1])
        print('Best Model MERF Weight:',best_model[2][1])
        
        self.model_info['weight'] = best_model[2][1]
        self.feature_info['fixed_effect_features'] = best_model[0]
        self.feature_info['random_effect_features'] = best_model[1]
        return self.get_results(self.training_sets, self.testing_sets, self.validation_sets)        
    
class LinearMixedEffectsModel():
    """Class definition for fitting and evaluating results for
    a Linear Mixed Effect Model"""
    
    def __init__(self, model_info, feature_info, output_info):
        """Initializes Linear Mixed Effects Model (uses rpy2)
        
        Parameters:
        ----------
        *model_info = model information {'modelType':type}
        *feature_info: contains information about what features to use {'random_effect_features':[], 'fixed_effect_features':[], 'intercept':True/False, 'clusterBy':feature}
        *output_info: {'featureName': name, 'logOutput': True/False}
        """
        self.model_info = model_info
        self.feature_info = feature_info
        self.output_info = output_info
        if self.output_info['log_output']:
            self.fullOutputName = 'Log:'+self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        else:
            self.fullOutputName = self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        self.formula = None
        self.model_summary = None
        self.results = None
        
    def Rify_Names(self, name):
        """Function that R-ifies column names in dataset (as R does not allow
        parentheses, dashes or underscores which we use in our pandas dfs) """
        name = name.replace("(","")
        name = name.replace(")","")
        name = name.replace("-","")
        name = name.replace("_","")
        return name
    
    def fitLME(self, train_sets, test_sets):
        """Fits a Linear Mixed Effects Model to the given
        training set and tests on the test set"""
        train = train_sets.copy()
        test = test_sets.copy()

        #get training data
        y_train = train[self.fullOutputName]
        X_train = train.drop([self.fullOutputName], axis = 1)

        #get testing data
        y_test = test[self.fullOutputName]
        X_test = test.drop([self.fullOutputName], axis = 1)

        df = X_train.assign(y=y_train)
        df.columns = [self.Rify_Names(string) for string in df.columns]

        formula = 'y~('+'+'.join([self.Rify_Names(i) for i in self.feature_info['fixed_effect_features']])+")"
        if self.feature_info['intercept'] and self.feature_info['random_effect_features'] != [] and self.feature_info['random_effect_features'] != ():
            formula += '+ (1+'+'+'.join([self.Rify_Names(i) for i in self.feature_info['random_effect_features']])+'|'+self.feature_info['clusterBy']+')'
        elif self.feature_info['intercept'] and (self.feature_info['random_effect_features'] == [] or self.feature_info['random_effect_features'] == ()):
            formula += '+ (1|'+self.feature_info['clusterBy']+')'
        else:
            formula += '+ ('+'+'.join([self.Rify_Names(i) for i in self.feature_info['random_effect_features']])+'|'+self.feature_info['clusterBy']+')'
        self.formula = formula
        
        r_dataframe = pandas2ri.py2ri(df)
        robjects.r('''
                f <- function(train,stringFormula) {
                    library(lme4)
                    fitted_model <- lmer(stringFormula, data = train, REML = FALSE)
                    return(fitted_model)
                }
                ''')
        r_f = robjects.r['f']
        fitted_model = r_f(r_dataframe,formula)

        ##get coefficients
        robjects.r('''
                f <- function(train,stringFormula) {
                    library(lme4)
                    fitted_model <- lmer(stringFormula, data = train, REML = FALSE)
                    return(summary(fitted_model))
                }
                ''')

        r_f = robjects.r['f']
        model_summary = r_f(r_dataframe,formula)
        self.model_summary = model_summary
        ###########

        df = X_test
        df.columns = [self.Rify_Names(string) for string in df.columns]

        r_dataframe = pandas2ri.py2ri(df)
        robjects.r('''
                f <- function(fitted_model, test) {
                    library(lme4)
                    pred <- predict(fitted_model, test, allow.new.levels = TRUE)
                }
                ''')
        r_f = robjects.r['f']
        response = r_f(fitted_model,r_dataframe)
        y_pred=pandas2ri.ri2py(response)
        return y_pred
        
    def get_results(self, training_sets, testing_sets, validation_sets=None):
        """Fits a Linear Mixed Effects Model to every training set, makes 
        predictions on its test split, and aggregates results"""
        result_df = []
        for i in range(len(training_sets)):
            train = training_sets[i].copy()
            test = testing_sets[i].copy()
            
            if validation_sets:
                validate = validation_sets[i].copy()
                all_training = pd.concat([train, validate]).reset_index(drop=True)
                ypred = self.fitLME(all_training, test)
            else:
                ypred = self.fitLME(train, test)
            test.loc[:,'prediction'] = ypred
            result_df.append(test.copy())
            
        if len(result_df) == 1:
            return result_df[0]
        final_result = pd.concat(result_df).reset_index(drop=True)
        self.results = final_result
        return final_result
            
    def RMSE(self):
        """Gets RMSE for results"""
        return math.sqrt(mean_squared_error(self.results[self.fullOutputName], self.results['prediction']))

    def MAE(self):
        """Gets MAE for results"""
        return np.mean(abs(self.results[self.fullOutputName]-self.results['prediction']))
    
    def getModelCoefficients(self):
        """Print summary of lme4 model"""
        print(self.model_summary)
        

class MERFModel():
    """Class definition for fitting and evaluating results for
    a Mixed Effect Random Forest Model"""
    
    def __init__(self, model_info, feature_info, output_info):
        """Initializes MERF model
        
        Parameters:
        ----------
        *model_info = model information {'modelType':type}
        *feature_info: contains information about what features to use {'random_effect_features':[], 'fixed_effect_features':[], 'intercept':True/False, 'clusterBy':feature}
        *output_info: {'featureName': name, 'logOutput': True/False}
        """
        self.model_info = model_info
        self.feature_info = feature_info
        self.output_info = output_info
        if self.output_info['log_output']:
            self.fullOutputName = 'Log:'+self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        else:
            self.fullOutputName = self.output_info['outputDataType']+self.output_info['outputFeature']+"(t)"
        self.model_summary = None
        self.randomFeatureList = None
        self.fixedFeatureList = None
        self.participantOrder = None
        self.results = None
    
    def fitMERF(self, train_sets, test_sets):
        """Fits a MERF Model to the given
        training set and tests on the test set"""
        train = train_sets.copy()
        test = test_sets.copy()

        #get training data
        y_train = train[self.fullOutputName]
        X_train = train.loc[:,self.feature_info['fixed_effect_features']]
        Z_train = train.loc[:,self.feature_info['random_effect_features']]

        #get testing data
        y_test = test[self.fullOutputName]
        X_test = test.loc[:,self.feature_info['fixed_effect_features']]
        Z_test = test.loc[:,self.feature_info['random_effect_features']]
        
        all_part = list(set(list(train.participantCode) + list(test.participantCode)))
        train.loc[:,'clusterBy'] = [all_part.index(i) for i in train['participantCode']]
        test.loc[:,'clusterBy'] = [all_part.index(i) for i in test['participantCode']]

        clusters_train = train['clusterBy']
        clusters_test = test['clusterBy']
        
        #Intercept
        if self.feature_info['intercept']:
            Z_train['intercept'] = 1
            Z_test['intercept'] = 1 #add intercept term
        if self.feature_info['intercept']==False and len(list(Z_train)) == 0:
            Z_train['intercept'] = 0
            Z_test['intercept'] = 0 #add intercept term
            
        mrf = MERF(n_estimators=300, max_iterations=10)
        mrf.fit(X_train, Z_train, clusters_train, y_train)
        y_pred = mrf.predict(X_test, Z_test, clusters_test)
        self.model_summary = mrf
        self.randomFeatureList = list(Z_train)
        self.fixedFeatureList = list(X_train)
        self.participantOrder = all_part
        return y_pred
        
    def get_results(self, training_sets, testing_sets, validation_sets=None):
        """Fits a MERF Model to every training set, makes 
        predictions on its test split, and aggregates results"""
        result_df = []
        for i in range(len(training_sets)):
            train = training_sets[i].copy()
            test = testing_sets[i].copy()
            
            if validation_sets:
                validate = validation_sets[i].copy()
                all_training = pd.concat([train, validate]).reset_index(drop=True)
                ypred = self.fitMERF(all_training, test)
            else:
                ypred = self.fitMERF(train, test)
                
            test.loc[:,'prediction'] = ypred
            result_df.append(test.copy())
            
        if len(result_df) == 1:
            return result_df[0]
        final_result = pd.concat(result_df).reset_index(drop=True)
        self.results = final_result
        return final_result
            
    def RMSE(self):
        """Gets RMSE for results"""
        return math.sqrt(mean_squared_error(self.results[self.fullOutputName], self.results['prediction']))

    def MAE(self):
        """Gets MAE for results"""
        return np.mean(abs(self.results[self.fullOutputName]-self.results['prediction']))
    
    def getMERFModelCoefficients(self):
        """Get MERF Model coefficients"""
        model = self.model_summary
        featureOrder = self.fixedFeatureList
        
        print("Feature importances",list(zip(model.trained_rf.feature_importances_,featureOrder)))
        print("Dhat",model.D_hat_history[-1])
        print("sigma2hat",model.sigma2_hat_history[-1])
        
        coeff = model.b_hat_history[-1]
        coeff.columns = self.randomFeatureList
        coeff = coeff.reset_index(drop=False)
        coeff['participantName'] = coeff.apply(self.convertNumtoName,args=(self.participantOrder,),axis=1)
        return coeff
    
    def getMERFconvergencePlot(self):
        """Produce MERF Convergence Plot"""
        Z_feat_order = self.randomFeatureList
        self.plot_training_stats(self.model_summary,Z_feat_order)
        
    def convertNumtoName(self, row, participantList):
        return participantList[int(row['index'])]

    def plot_training_stats(self, model, attributes):
        """Plot training statistics"""
        n = len(attributes)+2
        f, axarr = plt.subplots(math.ceil(n/2.0),2, figsize=(20,10))

        # Plot trace and determinant of Sigma_b (covariance matrix)
        det_sigmaB_history = [np.linalg.det(x) for x in model.D_hat_history]
        trace_sigmaB_history = [np.trace(x) for x in model.D_hat_history]
        axarr[0,0].plot(det_sigmaB_history, label='Determinant of Covariance Matrix for Random Effects')
        axarr[0,0].plot(trace_sigmaB_history, label='Trace of Covariance Matrix for Random Effects')
        axarr[0,0].grid('on')
        axarr[0,0].legend()
        axarr[0,1].set_xlabel('Iteration')
        axarr[0,0].set_title('Metrics of Variance for Random Effect Coefficients')

        axarr[0,1].plot(model.sigma2_hat_history)
        axarr[0,1].grid('on')
        axarr[0,1].set_ylabel('Variance of Error')
        axarr[0,1].set_xlabel('Iteration')

        row = 1
        col = 0
        for feature in range(model.b_hat_history[0].shape[1]):
            for cluster in range(model.b_hat_history[0].shape[0]):
                a = [model.b_hat_history[i].iloc[cluster,feature] for i in range(len(model.b_hat_history))]
                axarr[row,col].plot(a)
            axarr[row,col].grid('on')
            axarr[row,col].set_ylabel(attributes[feature])
            axarr[row,col].set_xlabel('Iteration')
            col += 1
            if col > 1:
                row += 1
                col = 0

        if n % 2 != 0:
            axarr[math.ceil(n/2.0)-1,1].axis('off')
        plt.show()

