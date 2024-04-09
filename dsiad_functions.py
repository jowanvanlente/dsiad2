# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 07:04:02 2020

@author: anje.knottnerus
"""

from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec

# =============================================================================
# SML CHECKING FUNCTIONS 
# =============================================================================

class Check():
    def step_151(self, portion):
        if portion == 5:
            print("Correct!")
        else: 
            print('Not quite yet. \n Hint: Try dividing the tortillas equally among the people.')
        
    def step_152(self, price_total):
        if price_total == 5.16:
            print("Correct!")
        else: 
            print('Not quite yet. \n Hint: Try multiplying the price of the apple with the apples.')

    def step_153(self, celcius):
        if celcius == 25.0 or celcius == 25:
            print("Correct!")   
        else:
            print('Not quite yet. \n Hint: Look at the formula again.')


    def step_4(self, number_of_observations, mean_quality, minimum_alcohol, maximum_pH):
        #check number of variables
        if round(number_of_observations) == round(1599.000000): 
            print("number_of_observations is correct")
        else: 
            print("number_of_observations is not correct")
        
        #check mean_quality
        if  round(mean_quality) == round(5.636023): 
            print("mean_quality is correct")
        else: 
            print("mean_quality is not correct")
            
        #check minimum_alcohol
        if  round(minimum_alcohol) == round(8.400000): 
            print("minimum_alcohol is correct")
        else: 
            print("minimum_alcohol is not correct")
            
        #check maximum_pH
        if  round(maximum_pH) == round(4.010000): 
            print("maximum_pH is correct")
        else: 
            print("maximum_pH is not correct")
            
    def step_61(self, wine_df, wine_original_df):
        wine_columns = wine_original_df.drop(['ID', 'age', 'gender'], axis=1)
        
        if wine_columns.columns.equals(wine_df.columns):
            print("You correctly removed columns ID, age, and gender")
        else: 
            print("Not quite yet. Tip: remove the columns ID, age, and gender")

    def step_62(self, correlated_features):
        if correlated_features == "star_rating" or correlated_features == ["star_rating"]:
            print("Star_rating should indeed be removed, good job!")
        else: 
            print("This is not the correct feature, try again! Look at the correlation matrix to see which feature is highly correlated with another feature.")
        
    def step_63(self):
        print("The correct boolean statement is wine.var() > threshold")
        
    def step_81(self, y_test, pred, user_answer):
        true_answer = accuracy_score(y_test, pred)
        if round(user_answer) == round(true_answer):
            print("This is the correct accuracy score")
        else: 
            print("This is not the correct accuracy score, try again! Look at how many answers are correct, divide it by the total number of answers.")   

            
# =============================================================================
#  UML CHECKING FUNCTIONS 
# =============================================================================             
        
class Solution_UML():
    def __init__(self):
        print("Solutions loaded...")
        
    def step_4(self):
        print("X_normalized = ((X-X.min())/(X.max()-X.min()))")
        
    def step_61(self, k):
        if (k < 2) | (k > 6):
            print("This is not the elbow, try again!")
        else:
            print("Good job, try out different numbers for k too!")

# =============================================================================
#  UML CHECKING FUNCTIONS 
# =============================================================================             
 
class Solution_SMLreg():
    def __init__(self):
        print("Solutions loaded...")
        
    def step_4(self):
        print("sns.heatmap(data.corr(), annot = True)")
        
    def step_7(self):
        print("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)",
              "multi_reg = LinearRegression()",
              "multi_reg.fit(X_train, y_train)",
              "predictions = reg.predict(X_test)")
        
    def check9(self, sales):
        if sales == 211793.943 or sales == 211793.94 or sales == 211793.9 or sales == 211793. or sales == 211793:
            print("Correct, nice job!")
        else: 
            print("Not quite yet, try again!")
    
    def step_9(self):
        print("X_1 = 100000", 
              "X_2 = 50000",
              "X_3 = 20000",
              "sales = 94.283 + 1.0902*X_1 + 2.0189*X_2 + 0.086733*X_3")      

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
      
class plots():
    def __init__(self):
        print()
    
    def elbow_plot(self, X):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow plot')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Intertia') 
        
    def residual_plot(self, X, y): 
    #test train split
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size = 0.2, 
                                                            random_state = 42)
    
        # Fit the resgression line 
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        print("The model fitted is LinearRegression()")
        
        # Make predictions 
        predictions_train = reg.predict(X_train)
        predictions_test = reg.predict(X_test)
        
        #r_2 train, r_2 test
        r2_train = r2_score(y_train, predictions_train)
        r2_test = r2_score(y_test, predictions_test)
        
        #residuals 
        res_train = y_train - predictions_train
        res_test = y_test - predictions_test
        
        #grid plot
        gs = GridSpec(4,4)
        fig = plt.figure()
        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])
        
        #scatter plot residuals
        ax_joint.scatter(x=predictions_train, y=res_train, label=['Train R2:', round(r2_train, 3)])
        ax_joint.scatter(x=predictions_test, y=res_test, label=['Test R2:', round(r2_test, 3)])
        ax_joint.axhline(linewidth=3, color='black', alpha=0.3)
        ax_joint.set_xlabel('Predicted value')
        ax_joint.set_ylabel('Residuals')
        ax_joint.legend(frameon=1)
    
        #histogram residuals
        ax_marg_y.hist(res_train,orientation="horizontal")
        ax_marg_y.hist(res_test,orientation="horizontal")
        ax_marg_y.set_xlabel('Distribution')
        
    
            
