import pandas as pd
import datetime

wanted_competitions_ids = ['EL', 'ES1', 'FR1', 'IT1', 'GB1', 'L1', 'PO1', 'CL']
indexes_to_remove_competitions = []

competitions_csv = pd.read_csv('datasets_full/competitions.csv')
for i in range(len(competitions_csv)):
    if competitions_csv.iloc[i]["competition_id"] not in wanted_competitions_ids:
        indexes_to_remove_competitions.append(i)
competitions_csv.drop(indexes_to_remove_competitions, inplace=True)
competitions_csv.to_csv('datasets_cleared/competitions.csv',index=False)

games_csv = pd.read_csv('datasets_full/games.csv')
indexes_to_remove_games = []
minimal_date = datetime.datetime(2016, 7, 15)


for i in range(len(games_csv)):
    date_of_match = datetime.datetime.strptime(games_csv.iloc[i]['date'], '%Y-%m-%d')
    if games_csv.iloc[i]['competition_id'] not in wanted_competitions_ids or date_of_match < minimal_date:
        indexes_to_remove_games.append(i)
    elif games_csv.iloc[i]['game_id'] in [3462393, 2848788]:
        indexes_to_remove_games.append(i)

games_csv.drop(indexes_to_remove_games, inplace=True)
games_csv.to_csv('datasets_cleared/games.csv', index=False)

wanted_games_ids = games_csv['game_id'].values

club_games_csv = pd.read_csv('datasets_full/club_games.csv')
club_games_csv.query("game_id in @wanted_games_ids", inplace=True)
club_games_csv.to_csv("datasets_cleared/club_games.csv", index=False)

clubs_csv = pd.read_csv('datasets_full/clubs.csv')
clubs_csv.query("domestic_competition_id in @wanted_competitions_ids", inplace=True)
clubs_csv.to_csv("datasets_cleared/clubs.csv", index=False)

players_csv = pd.read_csv('datasets_full/players.csv')
players_csv.query("last_season > 2015", inplace=True)
players_csv.to_csv("datasets_cleared/players.csv", index=False)

wanted_players_ids = players_csv['player_id'].values

player_valuations_csv = pd.read_csv('datasets_full/player_valuations.csv')
player_valuations_csv.query("player_id in @wanted_players_ids", inplace=True)
player_valuations_csv.to_csv("datasets_cleared/player_valuations.csv", index=False)

appearances_csv = pd.read_csv('datasets_full/appearances.csv')
appearances_csv.query("game_id in @wanted_games_ids", inplace=True)
appearances_csv.to_csv("datasets_cleared/appearances.csv", index=False)

game_events_csv = pd.read_csv('datasets_full/game_events.csv')
game_events_csv.query("game_id in @wanted_games_ids", inplace=True)
game_events_csv.to_csv("datasets_cleared/game_events.csv", index=False)

game_lineups_csv = pd.read_csv('datasets_full/game_lineups.csv')
game_lineups_csv.query("game_id in @wanted_games_ids", inplace=True)
game_lineups_csv.to_csv("datasets_cleared/game_lineups.csv", index=False)