import pandas as pd

splits1 = pd.read_csv('Pitching_splits_2014.csv')
splits2 = pd.read_csv('Pitching_splits_2015.csv')
splits3 = pd.read_csv('Pitching_splits_2016.csv')
splits4 = pd.read_csv('Pitching_splits_2017.csv')
splits5 = pd.read_csv('Pitching_splits_2018.csv')


pitcher_splits = pd.concat([splits1,splits2,splits3,splits4,splits5],ignore_index=True)
pitcher_splits = pitcher_splits.drop_duplicates()
pitcher_splits = pitcher_splits.groupby(['name','Split']).first().reset_index()

df = pitcher_splits
#df = pd.read_csv('mlb2017_pitching_splits.csv')

df_power = df.loc[df['Split'] == 'Career Totals']
df_power['SOBBPA'] = (df_power.SO + df_power.BB) / df_power.AB
df_power['PowFin'] = pd.qcut(df_power.SOBBPA,3)
df_power['PowerFinesse'] = df_power.PowFin.cat.rename_categories(['vs. Finesse','vs. avg.P/F','vs. Power'])
df1 = df_power.loc[:,('name','PowerFinesse')]

df_ground = df.loc[df['Split'].isin(['Ground Balls','Fly Balls'])]
df_ground['Outs'] = (df_ground.AB - df_ground.H)
df_ground2 = df_ground.pivot_table(index='name',columns='Split',values=['Outs'])
df_ground2['Ratio'] = df_ground2.iloc[:,0] / df_ground2.iloc[:,1]
df_ground2['GBFB'] = pd.qcut(df_ground2.Ratio,3)
df_ground2['GroundFly'] = df_ground2.GBFB.cat.rename_categories(['vs. GrndBall','vs. avg.F/G','vs. Fly Ball'])
df2 = df_ground2.iloc[:,4]
df2 = df2.reset_index()

types = pd.merge(df1,df2,'inner','name')
types.columns = ['name','Power','Ground']
types.to_csv('pitcher_types.csv',encoding='utf-8',index=False)