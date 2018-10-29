import pandas as pd
import numpy as np
#import datetime as dt
#from sklearn import linear_model
import xgboost as xgb
from ggplot import *
from datetime import timedelta
from xgboost.sklearn import XGBRegressor,XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.linear_model import Lasso,LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import scikitplot as skplt


df1 = pd.read_csv('Batting_2014.csv',encoding='utf-8')
df1['Date'] = pd.to_datetime(df1['Date'],'coerce',format='%b %d')
df1['Date'] = [pd.to_datetime(i).replace(year=2014) for i in df1.Date]

df2 = pd.read_csv('Batting_2015.csv',encoding='utf-8')
df2['Date'] = pd.to_datetime(df2['Date'],'coerce',format='%b %d')
df2['Date'] = [pd.to_datetime(i).replace(year=2015) for i in df2.Date]

df3 = pd.read_csv('Batting_2016.csv',encoding='utf-8')
df3['Date'] = pd.to_datetime(df3['Date'],'coerce',format='%b %d')
df3['Date'] = [pd.to_datetime(i).replace(year=2016) for i in df3.Date]

df4 = pd.read_csv('Batting_2017.csv',encoding='utf-8')
df4['Date'] = pd.to_datetime(df4['Date'],'coerce',format='%b %d')
df4['Date'] = [pd.to_datetime(i).replace(year=2017) for i in df4.Date]

df5 = pd.read_csv('Batting_2018.csv',encoding='utf-8')
df5['Date'] = pd.to_datetime(df5['Date'],'coerce',format='%b %d')
df5['Date'] = [pd.to_datetime(i).replace(year=2018) for i in df5.Date]



df = pd.concat([df1,df2,df3,df4,df5],ignore_index=True)
df = df.drop_duplicates()
df = df[pd.notnull(df.Rk)]
df = df[df.Gcar != 'Gtm']
#df = df[df.Inngs.str.match('GS.*') == True]
df = df[df.Inngs.str.match('CG') == True]

##### Batting 
### Filter out pitchers from batting df
df = df[df.Pos != 'P']

#df['Date'] = pd.to_datetime(df['Date'],'coerce',format='%b %d')
#df['Date'] = [pd.to_datetime(i).replace(year=2017) for i in df.Date]
df['PA'] = pd.to_numeric(df['PA'],'coerce')
df['AB'] = pd.to_numeric(df['AB'],'coerce')
df['R'] = pd.to_numeric(df['R'],'coerce')
df['H'] = pd.to_numeric(df['H'],'coerce')
df['2B'] = pd.to_numeric(df['2B'],'coerce')
df['3B'] = pd.to_numeric(df['3B'],'coerce')
df['HR'] = pd.to_numeric(df['HR'],'coerce')
df['RBI'] = pd.to_numeric(df['RBI'],'coerce')
df['BB'] = pd.to_numeric(df['BB'],'coerce')
df['IBB'] = pd.to_numeric(df['IBB'],'coerce')
df['SO'] = pd.to_numeric(df['SO'],'coerce')
df['HBP'] = pd.to_numeric(df['HBP'],'coerce')
df['SB'] = pd.to_numeric(df['SB'],'coerce')
df['BA'] = pd.to_numeric(df['BA'],'coerce')
df['OBP'] = pd.to_numeric(df['OBP'],'coerce')
df['SLG'] = pd.to_numeric(df['SLG'],'coerce')
df['WPA'] = pd.to_numeric(df['WPA'],'coerce')

df = df.sort_values(['name','Date'],ascending=True)
df = df.fillna(0)

df['FDP'] = 3 * (df.H - df['2B'] - df['3B'] - df.HR) + 6*df['2B'] + 9*df['3B'] + 12*df.HR + 3.5*df.RBI + 3.2*df.R + 3*df.BB + 3*df.IBB + 6*df.SB + 3*df.HBP
def big_game(row):
    if row.FDP > 0:
        return 1
    else:
        return 0

df['BigGame'] = df.apply(big_game, axis=1)

def home_team(row):
    if row[5] == '@':
        return row.Opp
    else:
        return row.Tm
df['Home_Tm'] = df.apply(home_team, axis=1)

c = 'name'
stats = ['PA','AB','R','H','2B','3B','HR','RBI','BB','IBB','SO','HBP','SB','BA','OBP','OPS','SLG','WPA','FDP']

for d in stats:
    gp = df.groupby(c)[d]
    mean = gp.rolling(window=10).mean().shift(1).reset_index(level=['name'], drop=True)
    df['b_'+d+str(10)] = df.index.to_series().map(mean)
    df['b_'+d+str(10)] = pd.to_numeric(df['b_'+d+str(10)],'coerce')
    
for d in stats:
    gp = df.groupby(c)[d]
    mean = gp.rolling(window=20).mean().shift(1).reset_index(level=['name'], drop=True)
    df['b_'+d+str(20)] = df.index.to_series().map(mean)
    df['b_'+d+str(20)] = pd.to_numeric(df['b_'+d+str(20)],'coerce')

#mod_df = df
#model = xgb.XGBRegressor(max_depth=1)
#model.fit(mod_df[['PA10','AB10','R10','H10','2B10','3B10','HR10','RBI10','BB10'
#                  ,'IBB10','SO10','HBP10','SB10','BA10','OBP10','OPS10','SLG10','WPA10','FDP10',
#                  'PA20','AB20','R20','H20','2B20','3B20','HR20','RBI20','BB20'
#                  ,'IBB20','SO20','HBP20','SB20','BA20','OBP20','OPS20','SLG20','WPA20','FDP20']], mod_df['FDP'])
#xgb.plot_importance(model,max_num_features=20)
#mod_df['preds'] = model.predict(mod_df[['PA10','AB10','R10','H10','2B10','3B10','HR10','RBI10','BB10'
#      ,'IBB10','SO10','HBP10','SB10','BA10','OBP10','OPS10','SLG10','WPA10','FDP10',
#      'PA20','AB20','R20','H20','2B20','3B20','HR20','RBI20','BB20'
#      ,'IBB20','SO20','HBP20','SB20','BA20','OBP20','OPS20','SLG20','WPA20','FDP20']])

#df.groupby(pd.qcut(df['OPS20'],5,duplicates='drop'))['FDP'].mean()
    
###### Batter Splits #######
splits1 = pd.read_csv('Splits_2014.csv')
splits2 = pd.read_csv('Splits_2015.csv')
splits3 = pd.read_csv('Splits_2016.csv')
splits4 = pd.read_csv('Splits_2017.csv')
splits5 = pd.read_csv('Splits_2018.csv')

batter_splits = pd.concat([splits1,splits2,splits3,splits4,splits5],ignore_index=True)
batter_splits = batter_splits.groupby(['name','Split']).first().reset_index()

career = batter_splits.loc[batter_splits.Split == 'Career Totals']
career = career.groupby(['name']).first().reset_index()

#Career Stats have _y suffix
df_career = pd.merge(df,career,'inner',left_on='name',right_on='name')
    
########################### Pitching ###########################
def encode_date(x):
    try:
        return x.encode('ascii','ignore').replace("(1)","")
    except:
        return np.nan
#df_pitching.Date = [encode_date(i) for i in df_pitching.Date]
#df_pitching = df_pitching[df_pitching.Date != 'Opp']
#df_pitching = df_pitching[~pd.isnull(df_pitching.Date)]

df_pitching1 = pd.read_csv('Pitching_2014.csv',encoding='utf-8')
df_pitching1.Date = [encode_date(i) for i in df_pitching1.Date]
df_pitching1 = df_pitching1[df_pitching1.Date != 'Opp']
df_pitching1 = df_pitching1[~pd.isnull(df_pitching1.Date)]
df_pitching1['Date'] = pd.to_datetime(df_pitching1['Date'],'coerce',format='%b%d')
df_pitching1['Date'] = [pd.to_datetime(i).replace(year=2014) for i in df_pitching1.Date]

df_pitching2 = pd.read_csv('Pitching_2015.csv',encoding='utf-8')
df_pitching2.Date = [encode_date(i) for i in df_pitching2.Date]
df_pitching2 = df_pitching2[df_pitching2.Date != 'Opp']
df_pitching2 = df_pitching2[~pd.isnull(df_pitching2.Date)]
df_pitching2['Date'] = pd.to_datetime(df_pitching2['Date'],'coerce',format='%b%d')
df_pitching2['Date'] = [pd.to_datetime(i).replace(year=2015) for i in df_pitching2.Date]

df_pitching3 = pd.read_csv('Pitching_2016.csv',encoding='utf-8')
df_pitching3.Date = [encode_date(i) for i in df_pitching3.Date]
df_pitching3 = df_pitching3[df_pitching3.Date != 'Opp']
df_pitching3 = df_pitching3[~pd.isnull(df_pitching3.Date)]
df_pitching3['Date'] = pd.to_datetime(df_pitching3['Date'],'coerce',format='%b%d')
df_pitching3['Date'] = [pd.to_datetime(i).replace(year=2016) for i in df_pitching3.Date]

df_pitching4 = pd.read_csv('Pitching_2017.csv',encoding='utf-8')
df_pitching4.Date = [encode_date(i) for i in df_pitching4.Date]
df_pitching4 = df_pitching4[df_pitching4.Date != 'Opp']
df_pitching4 = df_pitching4[~pd.isnull(df_pitching4.Date)]
df_pitching4['Date'] = pd.to_datetime(df_pitching4['Date'],'coerce',format='%b%d')
df_pitching4['Date'] = [pd.to_datetime(i).replace(year=2017) for i in df_pitching4.Date]

df_pitching5 = pd.read_csv('Pitching_2018.csv',encoding='utf-8')
df_pitching5.Date = [encode_date(i) for i in df_pitching5.Date]
df_pitching5 = df_pitching5[df_pitching5.Date != 'Opp']
df_pitching5 = df_pitching5[~pd.isnull(df_pitching5.Date)]
df_pitching5['Date'] = pd.to_datetime(df_pitching5['Date'],'coerce',format='%b%d')
df_pitching5['Date'] = [pd.to_datetime(i).replace(year=2018) for i in df_pitching5.Date]


df_pitching = pd.concat([df_pitching1,df_pitching2,df_pitching3,df_pitching4,df_pitching5],ignore_index=True)
df_pitching = df_pitching.drop_duplicates()


### Filter for starting pitchers only
df_pitching = df_pitching[df_pitching.Inngs.str.match('GS.*') == True]
#df_pitching['Date'] = pd.to_datetime(df_pitching['Date'],'coerce',format='%b%d')
#df_pitching['Date'] = [pd.to_datetime(i).replace(year=2017) for i in df_pitching.Date]
def decision_parse(x):
    try:
        if x[0] == 'W':
            return 1
        elif x[0] == 'L':
            return 0
    except:
        return np.nan
df_pitching['D'] = [decision_parse(i) for i in df_pitching.Dec]
df_pitching['DR'] = pd.to_numeric(df_pitching['DR'],'coerce')
df_pitching['IP'] = pd.to_numeric(df_pitching['IP'],'coerce')
df_pitching['H'] = pd.to_numeric(df_pitching['H'],'coerce')
df_pitching['R'] = pd.to_numeric(df_pitching['R'],'coerce')
df_pitching['ER'] = pd.to_numeric(df_pitching['ER'],'coerce')
df_pitching['BB'] = pd.to_numeric(df_pitching['BB'],'coerce')
df_pitching['SO'] = pd.to_numeric(df_pitching['SO'],'coerce')
df_pitching['HR'] = pd.to_numeric(df_pitching['HR'],'coerce')
df_pitching['HBP'] = pd.to_numeric(df_pitching['HBP'],'coerce')
df_pitching['ERA'] = pd.to_numeric(df_pitching['ERA'],'coerce')
df_pitching['BF'] = pd.to_numeric(df_pitching['BF'],'coerce')
df_pitching['Pit'] = pd.to_numeric(df_pitching['Pit'],'coerce')
df_pitching['Str'] = pd.to_numeric(df_pitching['Str'],'coerce')
df_pitching['SB'] = pd.to_numeric(df_pitching['SB'],'coerce')
df_pitching['2B'] = pd.to_numeric(df_pitching['2B'],'coerce')
df_pitching['3B'] = pd.to_numeric(df_pitching['3B'],'coerce')
df_pitching['WPA'] = pd.to_numeric(df_pitching['WPA'],'coerce')
def func(row):
    if row['IP'] >=6 and row['ER'] <=3:
        return 1
    else:
        return 0
df_pitching['QS'] = df_pitching.apply(func, axis=1)

df_pitching = df_pitching.sort_values(['name','Date'],ascending=True)
df_pitching = df_pitching.fillna(0)
df_pitching['FDP'] = 6*df_pitching.D + 4*df_pitching.QS - 3*df_pitching.ER + 3*df_pitching.SO + 3*df_pitching.IP
df_pitching['FDP'] = pd.to_numeric(df_pitching['FDP'],'coerce')


#def pitching_big_game(row):
#    if row.FDP >=50:
#        return 1
#    else:
#        return 0
    
#df_pitching['BigGame'] = df_pitching.apply(pitching_big_game, axis=1)
df_pitching['Home_Tm'] = df_pitching.apply(home_team, axis=1)

c = 'name'
stats = ['D','DR','IP','H','ER','BB','SO','HR','HBP','ERA','BF','Pit','Str','SB','2B','3B','WPA','FDP']

for d in stats:
    gp = df_pitching.groupby(c)[d]
    mean = gp.rolling(window=7).mean().shift(1).reset_index(level=['name'], drop=True)
    df_pitching['p_'+d+str(10)] = df_pitching.index.to_series().map(mean)
    df_pitching['p_'+d+str(10)] = pd.to_numeric(df_pitching['p_'+d+str(10)],'coerce')
    
for d in stats:
    gp = df_pitching.groupby(c)[d]
    mean = gp.rolling(window=3).mean().shift(1).reset_index(level=['name'], drop=True)
    df_pitching['p_'+d+str(5)] = df_pitching.index.to_series().map(mean)
    df_pitching['p_'+d+str(5)] = pd.to_numeric(df_pitching['p_'+d+str(5)],'coerce')


#model_pitching = xgb.XGBRegressor(max_depth=1)
#model_pitching.fit(df_pitching[['D5','DR5','IP5','H5','ER5','BB5','SO5','HR5','HBP5','ERA5','BF5'
#                                ,'Pit5','Str5','SB5','2B5','3B5','WPA5','FDP5',
#                                'D10','DR10','IP10','H10','ER10','BB10','SO10','HR10','HBP10','ERA10','BF10'
#                                ,'Pit10','Str10','SB10','2B10','3B10','WPA10','FDP10']], df_pitching['FDP'])
#xgb.plot_importance(model_pitching,max_num_features=10)
#df_pitching['preds'] = model_pitching.predict(df_pitching[['D5','DR5','IP5','H5','ER5','BB5','SO5','HR5','HBP5','ERA5','BF5'
#                                ,'Pit5','Str5','SB5','2B5','3B5','WPA5','FDP5',
#                                'D10','DR10','IP10','H10','ER10','BB10','SO10','HR10','HBP10','ERA10','BF10'
#                                ,'Pit10','Str10','SB10','2B10','3B10','WPA10','FDP10']])

######## PITCHING SPLITS
psplits1 = pd.read_csv('Pitching_splits_2014.csv')
psplits2 = pd.read_csv('Pitching_splits_2015.csv')
psplits3 = pd.read_csv('Pitching_splits_2016.csv')
psplits4 = pd.read_csv('Pitching_splits_2017.csv')
psplits5 = pd.read_csv('Pitching_splits_2018.csv')

pitcher_splits = pd.concat([psplits1,psplits2,psplits3,psplits4,psplits5],ignore_index=True)
pitcher_splits = pitcher_splits.groupby(['name','Split']).first().reset_index()

pitcher_career = pitcher_splits.loc[pitcher_splits.Split == 'Career Totals']
pitcher_career = pitcher_career.groupby(['name']).first().reset_index()

pitcher_career['SO/PA'] = pitcher_career['SO'] / pitcher_career['PA']
pitcher_career['WHIP/PA'] = (pitcher_career['H'] + pitcher_career['BB']) / pitcher_career['PA']
pitcher_career['R/PA'] = pitcher_career['R'] / pitcher_career['PA']
pitcher_career['TB/PA'] = pitcher_career['TB'] / pitcher_career['PA']


#Career Stats have _y suffix
df_pitching_career = pd.merge(df_pitching,pitcher_career,'left',left_on='name',right_on='name')

#df_pitching_career['SO/PA'] = df_pitching_career['SO_y'] / df_pitching_career['PA']
#df_pitching_career['WHIP/PA'] = (df_pitching_career['H_y'] + df_pitching_career['BB_y']) / df_pitching_career['PA']



##### Add Park Factors
batting_park_factors = pd.read_csv('parkfactors.csv')
pitching_park_factors = pd.read_csv('pitching_park_factors.csv')
##### Add Pitcher Types
pitcher_types = pd.read_csv('pitcher_types.csv')
##### Add Batter Stats vs Pitcher Type

##### Add Opposing Pitcher Info to Batter DF
new_batting = pd.merge(df_career,df_pitching_career,'inner',left_on = ['Date','Tm'],right_on = ['Date','Opp'])
new_new_batting = pd.merge(new_batting,batting_park_factors,'left',left_on = ['Home_Tm_x','hand_x_x'],right_on=['Tm','Hand'])
new_new_batting = pd.merge(new_new_batting,pitcher_types,'left',left_on = ['name_y'],right_on = ['name'])
new_new_batting = pd.merge(new_new_batting,batter_splits,'left',left_on = ['name_x','Power'], right_on = ['name','Split'],suffixes=('_xx', '_POWER'))
new_new_batting = pd.merge(new_new_batting,batter_splits,'left',left_on = ['name_x','Ground'],right_on = ['name','Split'],suffixes=('_xxx', '_GROUND'))
new_new_batting['Pitcher_handedness'] = ['vs ' + str(x) + 'HP' for x in new_new_batting.hand_y_y]
new_new_batting = pd.merge(new_new_batting,batter_splits,'left',left_on = ['name_x','Pitcher_handedness'],right_on = ['name','Split'],suffixes=('_xxxx', '_HAND'))
new_new_batting = new_new_batting.loc[new_new_batting.name_x.notnull()]




mod_df = new_new_batting
mod_df['OPS_x'] = pd.to_numeric(mod_df['OPS_x'])
#model = xgb.XGBRegressor()
model = xgb.XGBRegressor(n_estimators=50,max_depth=3)
#mod_df = new_new_batting.dropna()
mod_df.Date = pd.to_datetime(mod_df.Date,'coerce')
mod_df = mod_df[pd.notnull(mod_df.Date)]
mod_df_train = mod_df.loc[pd.to_datetime(mod_df.Date) < pd.to_datetime('2018-1-1'),:]
#model = LogisticRegression()
model.fit(mod_df_train.loc[:,
                  (
#                  'b_PA10','b_AB10','b_R10','b_H10','b_2B10','b_3B10','b_HR10','b_RBI10','b_BB10'
#                  ,'b_IBB10','b_SO10','b_HBP10','b_BA10','b_OBP10','b_OPS10','b_SLG10','b_WPA10','b_FDP10'
#                  ,'b_PA20','b_AB20','b_R20','b_H20','b_2B20','b_3B20','b_HR20','b_RBI20','b_BB20'
#                  ,'b_IBB20','b_SO20','b_HBP20','b_BA20','b_OBP20','b_OPS20','b_SLG20','b_WPA20','b_FDP20'
#                  ,'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
#                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  ,'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
#                  'b_R20','bpf_HR'
#                  ,'p_ERA10','p_SO10'
#                  'OPS_x'
                  'SO/PA','WHIP/PA'
#                  ,'bpf_1B','bpf_2B','bpf_3B','bpf_HR'
                  ,'OPS_HAND','OPS_POWER','OPS_xx','OPS_xxxx'
                  )
                  ], mod_df_train.FDP_x)

# model = LinearRegression() 
# model.fit(mod_df_train[['b_PA20','b_OPS20','b_WPA20',
#                  'b_R20','b_H20','b_RBI20','b_SB20',
#                  'p_SO10','p_WPA10',
#                  'bpf_HR']]
#    , mod_df_train['BigGame_x']) 
xgb.plot_importance(model,max_num_features=10)
#mod_df['preds'] = model.predict(mod_df.loc[:,('b_PA20','b_OPS20','b_WPA20',
#                  'b_R20','b_H20','b_RBI20','b_SB20',
#                  'p_SO10','p_WPA10',
#                  'bpf_HR')])
#mod_df['preds'] = model.predict_proba(mod_df.loc[:,('b_PA20','b_OPS20','b_WPA20',
#                  'b_R20','b_H20','b_RBI20','b_SB20',
#                  'p_SO10','p_WPA10',
#                  'bpf_HR')])

mod_df['preds'] = model.predict(mod_df.loc[:,
                  (
#                  'b_PA10','b_AB10','b_R10','b_H10','b_2B10','b_3B10','b_HR10','b_RBI10','b_BB10'
#                  ,'b_IBB10','b_SO10','b_HBP10','b_BA10','b_OBP10','b_OPS10','b_SLG10','b_WPA10','b_FDP10'
#                  ,'b_PA20','b_AB20','b_R20','b_H20','b_2B20','b_3B20','b_HR20','b_RBI20','b_BB20'
#                  ,'b_IBB20','b_SO20','b_HBP20','b_BA20','b_OBP20','b_OPS20','b_SLG20','b_WPA20','b_FDP20'
#                  ,'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
#                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  ,'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
                  'SO/PA','WHIP/PA'
#                  ,'bpf_1B','bpf_2B','bpf_3B','bpf_HR'
                  ,'OPS_HAND','OPS_POWER','OPS_xx','OPS_xxxx')])
        
    
mod_df.groupby(pd.qcut(mod_df['p_ERA5'],5,duplicates='drop'))['FDP_x'].mean()    
    
    
param_test1 = {
 'n_estimators':[20,40,60,80]
# ,'max_depth':[3]
# ,'learning_rate':[.001]
# ,'colsample_bytree':[.5,.75,1]
# ,'min_child_weight':[1,1000]
#'alpha':[1,.1,.01,.001,.0001,.00001,.000001]
}



gsearch1 = GridSearchCV(estimator = XGBRegressor(), scoring = 'neg_mean_squared_error', param_grid = param_test1)
gsearch1.fit(mod_df_train[[
#                  'b_PA10','b_AB10','b_R10','b_H10','b_2B10','b_3B10','b_HR10','b_RBI10','b_BB10'
#                  ,'b_IBB10','b_SO10','b_HBP10','b_BA10','b_OBP10','b_OPS10','b_SLG10','b_WPA10','b_FDP10'
#                  ,'b_PA20','b_AB20','b_R20','b_H20','b_2B20','b_3B20','b_HR20','b_RBI20','b_BB20'
#                  ,'b_IBB20','b_SO20','b_HBP20','b_BA20','b_OBP20','b_OPS20','b_SLG20','b_WPA20','b_FDP20'
#                  ,'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
#                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  ,'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  ,'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
#                  ,'SO/PA','WHIP/PA'
#                  'bpf_1B','bpf_2B','bpf_3B','bpf_HR'
                  'OPS_HAND','OPS_xx'#,'OPS_POWER','OPS_xxxx'
        ]], mod_df_train.FDP_x)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#    
##### Add Opposing Team Batting Info to Pitcher DF
#team_batting = pd.read_csv('team_batting_2017.csv')
#team_batting = team_batting.rename(columns=lambda x: 't_' + x)
#new_pitching = pd.merge(df_pitching,team_batting,'left',left_on = ['Tm'],right_on = ['t_Tm'])
new_new_pitching = pd.merge(df_pitching_career,pitching_park_factors,left_on=['Home_Tm'],right_on=['Tm'])


#model_pitching = xgb.XGBRegressor()
model_pitching = xgb.XGBRegressor(n_estimators=50,max_depth=1)
#new_new_pitching = new_new_pitching.dropna()
new_new_pitching.Date = pd.to_datetime(new_new_pitching.Date,'coerce')
new_new_pitching = new_new_pitching[pd.notnull(new_new_pitching.Date)]
new_new_pitching_train = new_new_pitching.loc[pd.to_datetime(new_new_pitching.Date) < pd.to_datetime('2018-1-1'),:]
#model_pitching = LogisticRegression()
model_pitching.fit(new_new_pitching_train[[
#                  'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
#                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  ,'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  ,'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
#                  ,'t_R/G','t_G','t_PA','t_AB','t_R','t_H','t_2B','t_3B','t_HR','t_RBI','t_SB','t_CS','t_BB','t_SO','t_BA'
#                  ,'t_OBP','t_SLG','t_OPS','t_OPS+','t_TB','t_GDP','t_HBP','t_SH','t_SF','t_IBB','t_LOB',
#                  'ppf_RUNS','ppf_HR','ppf_H','ppf_2B','ppf_3B','ppf_BB'
                  'OPS','SO/PA','WHIP/PA','SO/W'
]], new_new_pitching_train.FDP)
xgb.plot_importance(model_pitching,max_num_features=10)
new_new_pitching['preds'] = model_pitching.predict(new_new_pitching[[
#                  'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
#                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
#                  ,'ppf_RUNS','ppf_HR','ppf_H','ppf_2B','ppf_3B','ppf_BB'
#                  'ppf_RUNS','ppf_HR','ppf_H','ppf_2B','ppf_3B','ppf_BB'
                  'OPS','SO/PA','WHIP/PA','SO/W'
]])
   
param_test1 = {
 'n_estimators':[20,50,100]
# ,'max_depth':[1,3,5,7]
# ,'learning_rate':[.001]
# ,'colsample_bytree':[.5,.75,1]
# ,'min_child_weight':[100,1000,5000]
#'alpha':[0,.1]
}
#
gsearch1 = GridSearchCV(estimator = XGBRegressor(), scoring = 'neg_mean_squared_error', param_grid = param_test1)
gsearch1.fit(new_new_pitching_train[[
                  'p_D5','p_DR5','p_IP5','p_H5','p_ER5','p_BB5','p_SO5','p_HR5','p_HBP5','p_ERA5','p_BF5'
                  ,'p_Pit5','p_Str5','p_SB5','p_2B5','p_3B5','p_WPA5','p_FDP5'
#                  ,'p_D10','p_DR10','p_IP10','p_H10','p_ER10','p_BB10','p_SO10','p_HR10','p_HBP10','p_ERA10','p_BF10'
#                  ,'p_Pit10','p_Str10','p_SB10','p_2B10','p_3B10','p_WPA10','p_FDP10'
#                  ,'t_R/G','t_G','t_PA','t_AB','t_R','t_H','t_2B','t_3B','t_HR','t_RBI','t_SB','t_CS','t_BB','t_SO','t_BA'
#                  ,'t_OBP','t_SLG','t_OPS','t_OPS+','t_TB','t_GDP','t_HBP','t_SH','t_SF','t_IBB','t_LOB'
#                  ,'ppf_RUNS','ppf_HR','ppf_H','ppf_2B','ppf_3B','ppf_BB'
                                    ,'OPS','SO/PA','WHIP/PA','SO/W'
]], new_new_pitching_train.FDP)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_  
     
    
##### Combine Batting and Pitching Predictions
batting_data = mod_df[['name_x','Date','preds','FDP_x','Tm_x']]
batting_data.columns = ['name','Date','preds','FDP','Tm']
pitching_data = new_new_pitching[['name','Date','preds','FDP','Tm_x']]
pitching_data.columns = ['name','Date','preds','FDP','Tm']
#
final = pd.concat([batting_data,pitching_data])
#
###### Read in Salaries
salaries = pd.read_csv('mlb_salaries.csv',encoding='utf-8')
salaries = salaries.drop_duplicates()
opt_df = final.merge(salaries,left_on='name',right_on='Player',how='inner')
#
###### Encode Positions for Optimization
opt_df['p_P'] = 0
opt_df.loc[opt_df.Position == 'P', 'p_P' ] = 1
opt_df['p_C'] = 0
opt_df.loc[opt_df.Position == 'C', 'p_C' ] = 1
opt_df['p_1B'] = 0
opt_df.loc[opt_df.Position == '1B', 'p_1B' ] = 1
opt_df['p_2B'] = 0
opt_df.loc[opt_df.Position == '2B', 'p_2B' ] = 1
opt_df['p_3B'] = 0
opt_df.loc[opt_df.Position == '3B', 'p_3B' ] = 1
opt_df['p_SS'] = 0
opt_df.loc[opt_df.Position == 'SS', 'p_SS' ] = 1
opt_df['p_OF'] = 0
opt_df.loc[opt_df.Position == 'OF', 'p_OF' ] = 1

##### Filter data for one day of games
#maxes = []
#averages = []
#reg = []
#for z in range(90):
#    reg = []
##    classif = []
#    for y in range(10):
#        try:
#            game_df = opt_df.loc[opt_df.Date == pd.to_datetime('2018-4-1') + timedelta(days=z)]
#            drop_indices = np.random.choice(game_df.index,int(len(game_df)*.3),replace=False) 
#            game_df = game_df.drop(drop_indices)
#            
#            
#            ##### Solve Multiple Constraint Knapsack Problem
#            import openopt as opt
#            N= len(game_df.index)
#            items = [{'P': game_df.p_P.iloc[i] ,
#                      'C': game_df.p_C.iloc[i] ,
#                      '1B': game_df['p_1B'].iloc[i] ,
#                      '2B': game_df['p_2B'].iloc[i] ,
#                      '3B': game_df['p_3B'].iloc[i] ,
#                      'SS': game_df.p_SS.iloc[i] ,
#                      'OF': game_df.p_OF.iloc[i] ,
#                      'salary': game_df.Salary.iloc[i], 
#                      'points': game_df.preds.iloc[i],
#                      'DKP': game_df['FDP'].iloc[i],
#                      } for i in range(N)]
#            
#            constraints = lambda values: (
#                                            values['salary'] <= 35000
#                                            ,values['P'] + values['C'] + values['1B'] + values['2B'] + values['3B']+ values['SS']+ values['OF'] == 9
#                                            ,values['P'] == 1
#                                            ,values['C'] >= 0
#                                            ,values['C'] <= 2
#                                            ,values['1B'] >= 0
#                                            ,values['1B'] <= 2
#                                            ,values['2B'] >= 1
#                                            ,values['2B'] <= 2
#                                            ,values['3B'] >= 1
#                                            ,values['3B'] <= 2
#                                            ,values['SS'] >= 1
#                                            ,values['SS'] <= 2
#                                            ,values['OF'] >= 3
#                                            ,values['OF'] <= 4                           
#                                            )
#              
#            objective = 'points'
#            knapsack = opt.KSP(objective, items, goal = 'max', constraints = constraints)
#            solution = knapsack.solve('glpk', iprint = 0)
#            print game_df.iloc[solution.xf,:][['Tm','Position','Player','Salary','preds','FDP']].sort_values('preds')
#            print 'Predicted Points: ' + str(game_df.iloc[solution.xf,:].preds.sum())
#            print 'Actual Points: ' + str(game_df.iloc[solution.xf,:]['FDP'].sum())
#            print 'Salary Used: ' + str(game_df.iloc[solution.xf,:].Salary.sum())
#            a = game_df.iloc[solution.xf,:]['FDP'].sum()
#            reg.append(a)
#        except:
#            continue
#    try:    
#        maxes.append(max(reg))
#        averages.append(np.mean(reg))
#    except:
#        continue
#
#
#print pd.Series(maxes).describe()
#print pd.Series(averages).describe()