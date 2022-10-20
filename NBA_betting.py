# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 22:54:37 2022
@author: lucas
NExt steps: 
    1. making utilites module to reduce redundant code
    2. Altering how data is gather and stored to prevent the neccesity from 
        having to download the same data every time
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import model_selection
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model as lm
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from game import open_or_create_file
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection
from sklearn.metrics import accuracy_score
game_log_url= open_or_create_file()

#Convert all of the links to dataframes

def get_dict():
    strings = open_or_create_file()
    global years
    years = {str(i): 0 for i in range(2000,2023,1)}
    for url in strings:
        for year in list(years.keys()):
            if year in url:
                years[year]+=1
    

def get_tables():
    game_log_url= open_or_create_file()
    df_gl = []
    for i in range(len(game_log_url)):
        print(f"getting game log {i+1} out of {len(game_log_url)}")
        df = pd.read_html(requests.get(game_log_url[i]).content)
        df_gl.append(df)
    return df_gl
tables = get_tables()
def level_drop(df_list):
    for df in df_list:
        try:
            df.droplevel(0, axis=1)
        except:
            pass
    return df_list
sing_level = level_drop(tables)

def column_change(df_list):
    for df in df_list:
        
        df[0].columns = ([ 'Rk', 'G','Date_y', 'Unnamed: 3_level_1', 'OppT', 'W/L',
                      'Tm', 'OppP', 'FG', 'FGA','FG%', '3P', '3PA', '3P%', 'FT'
                      , 'FTA', 'FT%', 'ORB', 'TRB', 'AST','STL', 'BLK', 'TOV', 
                      'PF', 'Unnamed: 24_level_1', 'OppFG', 'OppFGA', 'OppFG%',
                      'Opp3P', 'Opp3PA', 'Opp3P%', 'OppFT', 'OppFTA', 'OppFT%', 
                      'OppORB', 'OppTRB', 'OppAST', 'OppSTL','OppBLK', 'OppTOV', 
                      'OppPF'])
    return [df_list[i][0] for i in range(len(df_list))]


def drop_rows(df_list):
    for gamelog in df_list:
        
        for game in range(len(gamelog)):
            if gamelog['OppFT'][game] == 'Opponent':
                gamelog.drop([game],inplace = True)
            elif gamelog['Date_y'][game] == 'Date':
                gamelog.drop([game],inplace = True)
        gamelog.reset_index(drop=True, inplace=True)
            
    return df_list

def convert_data(df_list):
    conv_dict = {'Tm':float, 'OppP':float,'FG':float, 'FGA':float, 'FG%':float, '3P':float, '3PA':float, '3P%':float, 'FT':float, 'FTA':float, 'FT%':float, 'ORB':float,'TRB':float, 'AST':float, 'STL':float, 'BLK':float, 'TOV':float, 'PF':float, 'OppFGA':float, 'OppFG':float,'OppFG%':float, 'Opp3P':float,  'Opp3PA':float, 'Opp3P%':float, 'OppFT':float, 'OppFTA':float, 'OppFT%':float, 'OppORB':float, 'OppTRB':float, 'OppAST':float, 'OppSTL':float, 'OppBLK':float, 'OppTOV':float,'OppPF':float}
    converted_df = [x.astype(conv_dict) for x in df_list]
    return converted_df

def calculate_stats(df_list):
    for gamelog in df_list:
        gamelog['Margin'] = gamelog['Tm'] - gamelog['OppP']
        gamelog['BWL'] = np.where(gamelog['Margin']>0, 1, 0)
        gamelog['% from 3P'] = (3 * gamelog['3P'])/(gamelog['Tm'])
        gamelog['% Opp 3'] = (3*gamelog['Opp3P'])/(gamelog['OppP'])
        gamelog['% from FT'] = gamelog['FT']/gamelog['Tm']
        gamelog['% Opp FT'] = gamelog['OppFT']/gamelog['OppP']
        gamelog['Offensive Rating'] = (gamelog['Tm']/(gamelog['FGA']-gamelog['ORB']+gamelog['TOV']+(.4*gamelog['FTA'])))*100
        gamelog['Defensive Rating'] = (gamelog['OppP']/(gamelog['OppFGA']-gamelog['OppORB']+gamelog['OppTOV']+(.4*gamelog['OppFTA'])))*100
        gamelog['One Game Lag'] = gamelog['BWL'].shift(1)
        gamelog['Assist Ratio'] = (gamelog['AST'] * 100) /(gamelog['FG'] + (gamelog['FTA'] * .44) + gamelog['AST'] + gamelog['TOV'])
        gamelog['Opp Assist Ratio'] = (gamelog['OppAST'] * 100)/(gamelog['OppFG'] + (gamelog['OppFTA'] * .44) + gamelog['AST'] + gamelog['TOV'])
        gamelog['Opp Assisted Points'] = gamelog['OppAST']/gamelog['OppFG']
        gamelog['5 Game Lag'] = gamelog['BWL'].shift(1) + gamelog['BWL'].shift(2) +gamelog['BWL'].shift(3) +gamelog['BWL'].shift(4) +gamelog['BWL'].shift(5)
        gamelog['10 Game Lag'] = gamelog['BWL'].shift(1) + gamelog['BWL'].shift(2) +gamelog['BWL'].shift(3) +gamelog['BWL'].shift(4) +gamelog['BWL'].shift(5) + gamelog['BWL'].shift(6) + gamelog['BWL'].shift(7) +gamelog['BWL'].shift(8) +gamelog['BWL'].shift(9) +gamelog['BWL'].shift(10)
        gamelog['ORB %'] = gamelog['ORB']/(gamelog['OppTRB']-gamelog['OppORB'] + gamelog['ORB'])
        gamelog['Opp ORB %'] = gamelog['OppORB'] / (gamelog['TRB'] - gamelog['ORB'] + gamelog['OppORB'])
        gamelog['DRB%'] = (gamelog['TRB'] - gamelog['ORB']) / (gamelog['OppORB'] + gamelog['TRB'] - gamelog['ORB'])
        gamelog['Opp DRB%'] = (gamelog['OppTRB'] - gamelog['OppORB']) / (gamelog['ORB'] + gamelog['OppTRB'] - gamelog['OppORB'])
        gamelog['A/T ratio'] = gamelog['AST']/gamelog['TOV']
        gamelog['Opp A/T ratio'] = gamelog['OppAST']/gamelog['OppTOV']
        gamelog['Effective FG %'] = gamelog['FG'] + (.5 * gamelog['3PA']) / gamelog['FGA']
        gamelog['True %'] = gamelog['Tm']/ (2 * (gamelog['FGA'] + (.44 * gamelog['FTA'])))
        gamelog['Opp Effective FG %'] = gamelog['OppFG'] + ((.5 * gamelog['Opp3PA'])/gamelog['OppFGA'])
        gamelog['Opp True %'] = gamelog['OppP'] / (2 * (gamelog['OppFGA'] + (.44 * gamelog['OppFTA'])))

    return df_list

def mean_list(gamelog_list):
    for gamelog in gamelog_list:
        gamelog['Mean OR'] = gamelog['Offensive Rating'].mean()
        gamelog['Mean DR'] = gamelog['Defensive Rating'].mean()
        gamelog['win %'] = gamelog['BWL'].mean()
        gamelog['Mean ORB %'] = gamelog['ORB'].mean()
        gamelog['Mean OppORB %'] = gamelog['OppORB'].mean()
        gamelog['Mean DRB%'] = gamelog['DRB%'].mean()
        gamelog['Mean Opp DRB%'] = gamelog['Opp DRB%'].mean()
        gamelog['Mean % from 3'] = gamelog['% from 3P'].mean()
        gamelog['Mean % from FT'] = gamelog['% from FT'].mean()
        gamelog['Mean % Opp 3'] = gamelog['% Opp 3'].mean()
        gamelog['Mean % Opp FT'] = gamelog['% Opp FT'].mean()
        gamelog['Mean Margin of Victory'] = gamelog['Margin'].mean()
        gamelog['Mean Assisted Ratio'] = gamelog['Assist Ratio'].mean()
        gamelog['Opp Mean Assist Ratio'] = gamelog['Opp Assist Ratio'].mean()
        gamelog['Mean A/T Ratio'] = gamelog['A/T ratio'].mean()
        gamelog['Mean Opp A/T Ratio'] = gamelog['Opp A/T ratio'].mean()
        gamelog['Mean Effective FG%'] = gamelog['Effective FG %'].mean()
        gamelog['Mean True %'] = gamelog['True %'].mean()
        gamelog['Mean Opp Effective FG%'] = gamelog['Opp Effective FG %'].mean()
        gamelog['Mean Opp True %'] = gamelog['Opp True %'].mean()
    
    return gamelog_list 

#Add a column of the means for each team

def drop_columns(gamelogs):
    for team in gamelogs:
        team.drop([ 'W/L', 'Tm', 'OppP','FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                'FT', 'FTA', 'FT%', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 
                'PF', 'Unnamed: 24_level_1', 'OppFG', 'OppFGA', 'OppFG%', 
                'Opp3P', 'Opp3PA', 'Opp3P%', 'OppFT', 'OppFTA','OppFT%', 
                'OppORB', 'OppTRB', 'OppAST', 'OppSTL', 'OppBLK', 'OppTOV',
                'OppPF', 'Margin',  '% Opp 3', '% from FT', '% Opp FT', 
                'Offensive Rating', 'Defensive Rating','Assist Ratio', 
                'Opp Assist Ratio', 'Opp Assisted Points','ORB %', 'Opp ORB %', 
                'DRB%', 'A/T ratio', 'Opp A/T ratio', 'Effective FG %', 
                'True %', 'Opp Effective FG %', 'Opp True %'], axis = 1, 
               inplace = True)
    return gamelogs

def create_season(gl_list):
    cut_num = 0
    df_by_season = []
    for year in list(years.values()):
        season = gl_list[cut_num:(cut_num+year)]
        
        df_by_season.append(season)
        cut_num+=year
    return df_by_season

def get_mean(gl_list):
    return [[team.describe() for team in season] for season in gl_list]

def create_total_season(season_list):
    new_season =  [pd.concat(season) for season in season_list]
    for season in new_season:
        season.reset_index(drop=True,inplace=True) 
    return new_season


#find the means

def clean_mean_list(mean_list):
    for season in mean_list:
        for team in season:
            
            team.drop(['One Game Lag','5 Game Lag', '10 Game Lag'], 
                      axis = 1 , inplace = True)
    mean = [[x.iloc[1:2,:] for x in x] for x in mean_list]
    meanlist = [pd.concat(x) for x in mean]
    for season in meanlist:
        season.reset_index(drop=True, inplace=True)
    
    return meanlist


def team_names(season_list):
    names = []
    for season in season_list:
        names.append(season['OppT'])
    for season in names:
        season.reset_index(drop=True, inplace=True)
        
    unique_names = [pd.DataFrame({'OppT':list(set(season.values.tolist()))}) for season 
                    in names]
    return_names = [season.sort_values(by="OppT") for season in unique_names]
    for season in return_names:
        season.reset_index(drop=True, inplace=True)
    return return_names

#Drop useless columns


def oppmeans(means, names, seasons):
    
    df = []
    for i in range(len(means)):
        OppMeans = pd.concat([means[i], names[i]], axis = 1)
        df_w_mean = seasons[i].merge(OppMeans, on = "OppT")
        df.append(df_w_mean)
    return df



def final_data_prep(season_list):
    for x in season_list:
        x.reset_index(drop = True, inplace = True)



    #Find the differential of the competing variables
    for x in season_list:
        x['Win Percentage Differntial'] = ((x['win %_x'] - x['win %_y']))
        x['Offensive Rating Differential'] = x['Mean OR_x'] - x['Mean DR_y']
        x['Defensive Rating Differential'] = x['Mean DR_x'] - x['Mean OR_y']
        x['ORB Dif'] = x['Mean ORB %_x'] - x['Mean DRB%_y']
        x['Opp ORB Dif'] = x['Mean ORB %_y'] - x['Mean DRB%_x']
        x['% from 3P']= (x['Mean % from 3_x'] - x['Mean % Opp 3_y'] ) 
        x['% from FT']= (x['Mean % from FT_x'] - x['Mean % Opp FT_y'])
        x['Margin of Victory']= (x['Mean Margin of Victory_x']-x['Mean Margin of Victory_y'])
        x['Assist Ratio D'] = x['Mean Assisted Ratio_x'] - x['Opp Mean Assist Ratio_y']
        x['D Assist Ratio'] = x['Mean Assisted Ratio_y'] - x['Opp Mean Assist Ratio_x']
        x[' D % from 3P']= (x['Mean % from 3_y'] - x['Mean % Opp 3_x'] )
        x['D % from FT']= (x['Mean % from FT_y'] - x['Mean % Opp FT_x'])
        x['A/T Ratio dif'] = x['Mean A/T Ratio_x'] - x['Mean Opp A/T Ratio_y']
        x['D A/T Ratio Dif'] = x['Mean A/T Ratio_y'] - x['Mean Opp A/T Ratio_x']


    #####  Drop the columns that arent needed
    for x in season_list:
        x.drop([ 'Opp DRB%_x', 'Mean OR_x', 'Mean DR_x',  'win %_x', 'Mean ORB %_x', 'Mean OppORB %_x', 'Mean DRB%_x','Mean Opp DRB%_x', 'Mean % from 3_x', 'Mean % from FT_x', 'Mean % Opp 3_x', 'Mean % Opp FT_x', 'Mean Margin of Victory_x', 'Mean Assisted Ratio_x', 'Opp Mean Assist Ratio_x', 'Mean A/T Ratio_x', 'Mean Opp A/T Ratio_x', 'Mean Effective FG%_x', 'Mean True %_x', 'Mean Opp Effective FG%_x', 'Mean Opp True %_x', 'Opp DRB%_y', 'Mean OR_y', 'Mean DR_y', 'win %_y', 'Mean ORB %_y', 'Mean OppORB %_y','Mean DRB%_y', 'Mean Opp DRB%_y', 'Mean % from 3_y', 'Mean % from FT_y', 'Mean % Opp 3_y', 'Mean % Opp FT_y', 'Mean Margin of Victory_y', 'Mean Assisted Ratio_y', 'Opp Mean Assist Ratio_y', 'Mean A/T Ratio_y',  'Mean Opp A/T Ratio_y', 'Mean Effective FG%_y', 'Mean True %_y', 'Mean Opp Effective FG%_y', 'Mean Opp True %_y'],axis = 1, inplace = True)

    for x in season_list:
        x.drop('BWL_y',axis = 1,inplace = True)

    #double check
    for x in season_list:
        x.drop(['One Game Lag', '5 Game Lag', '10 Game Lag'],axis = 1, inplace = True)

    #double check
    for x in season_list:
        x.drop(['% from 3P_x','% from 3P_y'],axis =1 , inplace = True)
    
    
    return_df = pd.concat(season_list)
    return_df.reset_index(drop=True, inplace=True)
    return_df.drop(['Rk', 'G', 'Date_y', 'Unnamed: 3_level_1', 'OppT'],axis =1,
                   inplace = True)
    return_df.fillna(0, inplace=True)
    return return_df

#Prep data for the pca


def scale_and_transform(full_df, split=True):
    y = full_df['BWL_x'].values
    no_y = full_df.drop(['BWL_x','Margin of Victory'],axis = 1)

    #Prepare the x variables

    scale = StandardScaler()
    scaled_data = scale.fit_transform( no_y)

    model = PCA( n_components = 9 )


    x = model.fit_transform(scaled_data)
    X = x.astype(np.float32)
    
    if split is True:
        X_tra, X_te, y_tr, y_te = model_selection.train_test_split( X, y.ravel(), 
                                                               test_size = 0.25,
                                                               random_state=7 )
        return X_tra, X_te, y_tr, y_te
    else:
        return X, y


def efficency_prep(binary,probability):
    global bins
    bins = np.linspace(0,1,101)
    wins = np.zeros(len(bins))
    total = np.zeros(len(bins))
    
    for result in range(len(binary)):
        for level in range(len(bins)-1):    
            if binary[result] == 1 and probability[result] >= bins[level] and \
            probability[result] < bins[level+1]:
                
                wins[level] +=1
                total[level] +=1
            elif binary[result] == 0 and probability[result] >= bins[level] and \
            probability[result] < bins[level+1]:
                
                total[level] +=1
    for i in range(len(bins)):
        if total[i] == 0:
            wins[i] = bins[i]
            total[i] = 1
    
    
    return wins, total

def mwa(probability, wins, total):
    diff_arr = [total[i]/len(probability) * abs(wins[i]/total[i] - bins[i]) 
                  for i in range(len(bins))]
    for i in range(len(diff_arr)):
        print(wins[i]/total[i])
        
    return np.sum(diff_arr)


def sig_dif(wins, totals):
    num_dif = 0
    diff_arr = [(wins[i]/totals[i]) - bins[i] for i in range(len(wins))]
     
    for i in range(len(diff_arr)):
        z = np.nan_to_num((diff_arr[i])/ (np.sqrt((bins[i] * 
                                                   (1 - bins[i]))/totals[i])))
       
        if stats.norm.cdf(z) > .99 or stats.norm.cdf(z) < .01:
            
            num_dif+=1
    return num_dif

def bake_off(X_train, y_train, X_test, y_test):
    global model_dict
    model_dict = {"Logistic Regression":lm.LogisticRegression(),
                  "Naive Bayes":GaussianNB(),
                  "Decision Tree":DecisionTreeClassifier( 
                    criterion = 'gini', min_samples_split = 30,max_depth = 15,
                    min_samples_leaf = 30,max_leaf_nodes=200), 
                    "K Nearest Neighbors": KNeighborsClassifier( 
                        n_neighbors = 15,  algorithm = 'kd_tree'),
                    "XGBoost":XGBClassifier(gamma = 18, max_depth = 3,
                                  min_child_weight = 0)}
    
    
    for model in list(model_dict.keys()):
        model_dict[model].fit(X_train, y_train)
        predictions = model_dict[model].predict(X_test)
        prob = model_dict[model].predict_proba(X_test)
        w,t = efficency_prep(y_test, prob.T[1])
        mean_score = mwa(prob, w,t)
        sig = sig_dif(w, t)
        print(f"{model} Performance Metrcs")
        print(f"\tAccuracy Score: {accuracy_score(y_test, predictions)}")
        print(f"\tMean Weighted Difference {mean_score}")
        print(f"\tSignificantly different levels {sig}")
        


def matchups(team1,team2):
    t1 = currentSeason[team1:team1]
    t2 = currentSeason[team2:team2]
    t1 = t1.reset_index()
    t2 = t2.reset_index()
    statline = pd.concat([t1,t2],axis = 1)
    statline.columns = (['Team1','BWL_1', '% from 3P_1', 'Opp DRB%_1', 'Mean OR_1', 'Mean DR_1', 'win %_1',
       'Mean ORB %_1', 'Mean OppORB %_1', 'Mean DRB%_1', 'Mean Opp DRB%_1',
       'Mean % from 3_1', 'Mean % from FT_1', 'Mean % Opp 3_1', 'Mean % Opp FT_1',
       'Mean Margin of Victory_1', 'Mean Assisted Ratio_1',
       'Opp Mean Assist Ratio_1', 'Mean A/T Ratio_1', 'Mean Opp A/T Ratio_1',
       'Mean Effective FG%_1', 'Mean True %_1', 'Mean Opp Effective FG%_1',
       'Mean Opp True %_1', 'Team2','BWL_2', '% from 3P_2', 'Opp DRB%_2', 'Mean OR_2', 'Mean DR_2', 'win %_2',
       'Mean ORB %_2', 'Mean OppORB %_2', 'Mean DRB%_2', 'Mean Opp DRB%_2',
       'Mean % from 3_2', 'Mean % from FT_2', 'Mean % Opp 3_2', 'Mean % Opp FT_2',
       'Mean Margin of Victory_2', 'Mean Assisted Ratio_2',
       'Opp Mean Assist Ratio_2', 'Mean A/T Ratio_2', 'Mean Opp A/T Ratio_2',
       'Mean Effective FG%_2', 'Mean True %_2', 'Mean Opp Effective FG%_2',
       'Mean Opp True %_2'])
    statline['Win Percentage Differntial'] = ((statline['win %_1'] - statline['win %_2']))
    statline['Offensive Rating Differential'] = statline['Mean OR_1'] - statline['Mean DR_2']
    statline['Defensive Rating Differential'] = statline['Mean DR_1'] - statline['Mean OR_2']
    statline['ORB Dif'] = statline['Mean ORB %_2'] - statline['Mean DRB%_2']
    statline['Opp ORB Dif'] = statline['Mean ORB %_2'] - statline['Mean DRB%_1']
    statline['% from 3P']= (statline['Mean % from 3_1'] - statline['Mean % Opp 3_2'] )
    statline['% from FT']= (statline['Mean % from FT_1'] - statline['Mean % Opp FT_2'])
    statline['Margin of Victory dif']= (statline['Mean Margin of Victory_1'] - statline['Mean Margin of Victory_2'])
    statline['Assist Ratio D'] = statline['Mean Assisted Ratio_1'] - statline['Opp Mean Assist Ratio_2']
    statline['D Assist Ratio'] = statline['Mean Assisted Ratio_2'] - statline['Opp Mean Assist Ratio_1']
    statline[' D % from 3P']= (statline['Mean % from 3_2'] - statline['Mean % Opp 3_1'] )
    statline['D % from FT']= (statline['Mean % from FT_2'] - statline['Mean % Opp FT_1'])
    statline['A/T Ratio dif'] = statline['Mean A/T Ratio_1'] - statline['Mean Opp A/T Ratio_2']
    statline['D A/T Ratio Dif'] = statline['Mean A/T Ratio_2'] - statline['Mean Opp A/T Ratio_1']
    statline.drop(['Team1' , 'BWL_1', '% from 3P_1', 'Opp DRB%_1',
       'Mean OR_1', 'Mean DR_1', 'win %_1', 'Mean ORB %_1', 'Mean OppORB %_1',
       'Mean DRB%_1', 'Mean Opp DRB%_1', 'Mean % from 3_1', 'Mean % from FT_1',
       'Mean % Opp 3_1', 'Mean % Opp FT_1', 'Mean Margin of Victory_1',
       'Mean Assisted Ratio_1', 'Opp Mean Assist Ratio_1', 'Mean A/T Ratio_1',
       'Mean Opp A/T Ratio_1', 'Mean Effective FG%_1', 'Mean True %_1',
       'Mean Opp Effective FG%_1', 'Mean Opp True %_1', 'Team2', 'BWL_2', '% from 3P_2', 'Opp DRB%_2', 'Mean OR_2', 'Mean DR_2',
       'win %_2', 'Mean ORB %_2', 'Mean OppORB %_2', 'Mean DRB%_2',
       'Mean Opp DRB%_2', 'Mean % from 3_2', 'Mean % from FT_2',
       'Mean % Opp 3_2', 'Mean % Opp FT_2', 'Mean Margin of Victory_2',
       'Mean Assisted Ratio_2', 'Opp Mean Assist Ratio_2', 'Mean A/T Ratio_2',
       'Mean Opp A/T Ratio_2', 'Mean Effective FG%_2', 'Mean True %_2',
       'Mean Opp Effective FG%_2', 'Mean Opp True %_2','Margin of Victory dif'],axis = 1,inplace = True)
    df = pd.concat([processed_season,statline],axis = 0)
    df.reset_index(drop = True,inplace = True)
    return df
def currentScaleAndPCAPredict(data):
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    model =  PCA(n_components = 9)
    print(scaled_data)
    PCAdata = model.fit_transform(scaled_data)
    cut = PCAdata[len(PCAdata) - 1:]
    print(cut)
    #model = XGBClassifier(gamma = 17, max_depth = 3,min_child_weight = 0)
    model = lm.LogisticRegression()
    model.fit(X_train,y_train)
    predictions = model.predict_proba(cut)[0]
    return predictions
def impliedMLDif(predictions,ML1,ML2,teams):
    #ML1 is the first team, ML2 is the second team
    if ML1 < 0:
        p1 = (-(ML1))/ (-(ML1) + 100)
    else:
        p1 = 100/(ML1 + 100)
    if ML2 < 0:
        p2 = (-(ML2))/ (-(ML2) + 100)
    else:
        p2 = 100/(ML2 + 100)
    difference1 = predictions[1] - p1
    difference2 = predictions[0] - p2
    if difference1 > difference2 and difference1> 0:
        return teams[0],difference1
    elif difference2 > difference1 and difference2 > 0:
        return teams[1],difference2
    else:
        print("do not bet on this game")
def prediction(teams,ML1,ML2):
    df = matchups(teams[0],teams[1])
    predictions = currentScaleAndPCAPredict(df)
    team = impliedMLDif(predictions,ML1,ML2,teams)
    return team
### STILL DEBUGGING
def user_chosen():
    print("Would you like to find positive expected value bets?")
    answer = input("Enter y or n")
    if answer == "y":
        while True:
            print("please enter teams, chose from given list below:")
            print(currentSeason.index.tolist())
            print("team 1:")
            team1 = str(input())
            print("team 2: ")
            team2 = str(input())
            print("Money Line 1: ")
            ml1 = int(input())
            print("Money Line 2: ")
            ml2 = (input())
            team, difference = prediction([team1, team2], ml1, ml2)
            print(f"Bet on this {team}. Their chance of winning is undervalue by {difference * 100}%")
            
            print("/nWould you like to bet find anymore more teams?")
            answer = input("Enter y or n")
            if answer == "y":
                pass
            elif answer == "n":
                break
    print("program finished")
    
def main():
    get_dict()
    new_col = column_change(sing_level)
    dropped = drop_rows(new_col)
    converted = convert_data(dropped)
    stats = calculate_stats(converted)
    mean_stats = mean_list(stats)
    cleaned = drop_columns(mean_stats)
    season = create_season(cleaned)
    total_season = create_total_season(season)
    means = get_mean(season)
    clean_means = clean_mean_list(means)
    names = team_names(total_season)
    global currentSeason
    currentSeason = pd.concat([clean_means[-1], names[-1]], axis = 1)
    currentSeason = currentSeason.set_index(keys = 'OppT')
    full_season = oppmeans(clean_means, names, total_season)
    global processed_season
    global X_train
    global y_train
    processed_season = final_data_prep(full_season)
    X, y = scale_and_transform(processed_season, split=False)
    X_train, X_test, y_train, y_test = scale_and_transform(processed_season)
    bake_off(X_train, y_train, X_test, y_test)
    
    
if __name__ == "__main__":
    main()
    ### STILL DEBUGGING
    user_chosen()



