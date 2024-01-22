import datetime
import pandas as pd

folder_path = 'datasets_cleared/'


appearances_csv = pd.read_csv(folder_path + 'appearances.csv')
player_valuations_csv = pd.read_csv(folder_path + 'player_valuations.csv')
players_csv = pd.read_csv(folder_path + 'players.csv')
def getPlayerValuation(date, player_id):
    try:
        val = player_valuations_csv.query('player_id == @player_id')
        reduced_vals = []
        for i in range(len(val)):
            valDate = datetime.datetime.strptime(val.iloc[i]['date'], '%Y-%m-%d')
            if(int((date - valDate).days) >= 0):
                reduced_vals.append(val.iloc[i])
        max_valuation = 0
        most_recent_val = 0
        days_since_most_recent = int((date - datetime.datetime(1,1,1)).days)
        for i in range(len(reduced_vals)):
            valDate = datetime.datetime.strptime(reduced_vals[i]['date'], '%Y-%m-%d')
            days_since = int((date - valDate).days)
            value = int(reduced_vals[i]['market_value_in_eur'])
            if(days_since < days_since_most_recent):
                days_since_most_recent = days_since
                most_recent_val = value
            if(value > max_valuation):
                max_valuation = value
        return [most_recent_val, max_valuation]
    except:
        print(f'valuation {player_id}')
        return [0,0]
    
        
def getAgeOfPlayer(matchDate: datetime, player_id):
    try:
        birthDateRaw = players_csv.query('player_id == @player_id')['date_of_birth'].iloc[0]
        birthDate = datetime.datetime.strptime(birthDateRaw, '%Y-%m-%d')
        return int((matchDate - birthDate).days)//365
    except:
        print(f'age {player_id}')
        return 18
columns=['age', 'max_valuation', 'recent_valuation']
new_cols = pd.DataFrame(columns=columns)
for i in range(len(appearances_csv)):
    print(i)
    new_row = {}
    date = datetime.datetime.strptime(appearances_csv.iloc[i]['date'], '%Y-%m-%d')
    id =appearances_csv.iloc[i]['player_id']
    new_row['age'] = [getAgeOfPlayer(date, id)]
    val = getPlayerValuation(date, id)
    new_row['recent_valuation'] = [val[0]]
    new_row['max_valuation'] = [val[1]]
    new_cols = pd.concat([new_cols, pd.DataFrame(new_row, columns=columns)], ignore_index=True)
new_appearances = pd.concat([appearances_csv, new_cols], axis=1)
new_appearances.to_csv(folder_path+'new_appearances.csv', index=False)