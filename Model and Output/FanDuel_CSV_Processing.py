import pandas as pd
import numpy as np
import ggplot
#Preprocess
asdf = pd.read_csv('FanDuel-MLB-2018-07-05-26828-lineup-upload-template.csv',skiprows=6)
asdf = asdf.iloc[:,10:]
asdf = asdf.loc[asdf['Injury Indicator'].isnull()]
#Get Park Factor
asdf['Park'] = [x[-3:] for x in asdf.Game]
#Separate Batters and Pitchers
batters = asdf.loc[asdf.Position != 'P']
#Filter for Starters
#batters = batters.loc[batters.Played >= 80]
batters = batters.loc[batters['Batting Order'].notnull()]
batters = batters.loc[batters['Batting Order'] != 0]

pitchers = asdf.loc[asdf.Position == 'P']
#Get Starting Pitchers
sp = pitchers.loc[asdf['Probable Pitcher'] == 'Yes']
sp = sp[['Nickname','Team','Opponent','Salary','Position','FPPG']]
sp.columns = ['Opposing_Pitcher','Pitchers_Team','Pitchers_Opponent','Salary','Position','FPPG']

# Join opposing Pitcher to Batters
new_batters = pd.merge(batters,sp,left_on=['Team'],right_on=['Pitchers_Opponent'])

final_batters = pd.merge(new_batters,pitcher_career,'inner',left_on = ['Opposing_Pitcher'],right_on = ['name'])
final_batters = pd.merge(final_batters,pitcher_types,'left',left_on = ['Opposing_Pitcher'],right_on = ['name'])
final_batters = pd.merge(final_batters,batter_splits,'left',left_on = ['Nickname','Power'], right_on = ['name','Split'],suffixes=('_xx', '_POWER'))
final_batters = pd.merge(final_batters,batter_splits,'left',left_on = ['Nickname','Ground'],right_on = ['name','Split'],suffixes=('_xxx', '_GROUND'))

final_batters['Pitcher_handedness'] = ['vs ' + str(x) + 'HP' for x in final_batters.hand]
final_batters = pd.merge(final_batters,batter_splits,'left',left_on = ['Nickname','Pitcher_handedness'],right_on = ['name','Split'],suffixes=('_xxxx', '_HAND'))
final_batters = final_batters.loc[final_batters.name_x.notnull()]

final_batters['preds'] = model.predict(final_batters.loc[:,
                  (
                  'SO/PA','WHIP/PA'
                  ,'OPS_HAND','OPS_POWER','OPS_xx','OPS_xxxx')])

final_pitchers = sp
final_pitchers = pd.merge(final_pitchers,pitcher_career,left_on = 'Opposing_Pitcher',right_on='name')

final_pitchers['preds'] = model_pitching.predict(final_pitchers.loc[:,
                  ('OPS','SO/PA','WHIP/PA','SO/W')])

batting_data = final_batters[['Nickname','preds','Salary_x','Position_x','Team','Batting Order']]
batting_data.columns = ['name','preds','Salary','Position','Tm','Batting Order']
pitching_data = final_pitchers[['name','FPPG','Salary','Position','Pitchers_Team']]
pitching_data.columns = ['name','preds','Salary','Position','Tm']

final = pd.concat([batting_data,pitching_data])
opt_df = final

positions = ['P','C','1B','2B','3B','SS','OF']
for a in positions:
    opt_df[a] = 0
    opt_df.loc[opt_df.Position == a, a] = 1

teams = ['COL','TEX','ARI','DET','CHC','OAK','MIN','MIL','PHI','WSN'
,'BOS','BAL','CIN','NYY','CHW','ATL','CLE','LAD','LAA','PIT'
,'TOR','KCR','SEA','TBR','STL','NYM','SFG','MIA','SDP','HOU']
for a in teams:
    opt_df[a] = 0
    opt_df.loc[(opt_df.Tm == a)&(opt_df.P == 0), a] = 1

game_df = opt_df

####### manual removals for weather, personal hatred, etc.
#
#game_df = game_df.loc[game_df.name != 'Zack Greinke']
#game_df = game_df.loc[game_df.name != 'Mike Foltynewicz']
#
#game_df = game_df.loc[game_df.Tm != 'NYY']
#game_df = game_df.loc[game_df.Tm != 'NYM']

import openopt as opt

columns = positions + teams + ['Salary','preds']
items = []
for i in range(len(game_df.index)):
     dct = {}
     for j in columns:
          dct[j] = game_df[j].iloc[i]
     items.append(dct)
     
constraints = lambda values: (
                                values['Salary'] <= 35000
                                ,values['P'] + values['C'] + values['1B'] + values['2B'] + values['3B']+ values['SS']+ values['OF'] == 9
                                ,values['P'] == 1
                                ,values['C'] + values['1B'] >= 1
                                ,values['C'] + values['1B'] >= 2
                                ,values['2B'] >= 1
                                ,values['2B'] <= 2
                                ,values['3B'] >= 1
                                ,values['3B'] <= 2
                                ,values['SS'] >= 1
                                ,values['SS'] <= 2
                                ,values['OF'] >= 3
                                ,values['OF'] <= 4 
#                                ,values['ARI'] == 4
#                                ,values['CLE'] == 4
#                                ,values['ATL'] == 4
                                )
  
objective = 'preds'
knapsack = opt.KSP(objective, items, goal = 'max', constraints = constraints)
solution = knapsack.solve('glpk', iprint = 0)
print game_df.iloc[solution.xf,:][['name','Tm','Batting Order','Position','Salary','preds']].sort_values(['preds'], ascending=False)
print 'Predicted Points: ' + str(game_df.iloc[solution.xf,:].preds.sum())
print 'Salary Used: ' + str(game_df.iloc[solution.xf,:].Salary.sum())

################ WRITE TO CSV ##############
## One loop for each set of players
#final_df = pd.DataFrame()
##for i in [0,1]:
#k = game_df.iloc[solution.xf,:][['name','Tm','Position','Salary','preds']].sort_values('preds')
#ids = asdf[['Nickname','Id']]
#l = pd.merge(k,ids,left_on = 'name',right_on = 'Nickname')
#positions = ['P','C/1B','2B','3B','SS','OF','OF','OF','UTIL']
#row = [0,0,0,0,0,0,0,0,0]
#for a in range(len(l.name)):
#    for b in range(len(positions)): 
#        if row[b] == 0 and l.Position[a] in positions[b]:
#            row[b] = l.Id[a]
#            break
##utility is the Id that has not been assigned yet
#util = [y for y in l.Id if y not in row]
#row[8] = util[0]      
#z = pd.DataFrame(row)
#z = z.T
#z.columns = positions
#z.to_csv('EZ.csv',index=False)
#    final_df = pd.concat([final_df,z])
#final_df.to_csv('EZ.csv',index=False)