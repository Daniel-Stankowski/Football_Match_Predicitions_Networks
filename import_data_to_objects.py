import pandas as pd
from objects.game import Game
from objects.player import Player
from objects.team import Team
from objects.team_stats import TeamStats
import datetime
import time
from typing import List




def getObjects(date = None, to=1) -> List[Game]:
    folder_path = 'datasets_cleared/'

    games_csv = pd.read_csv(folder_path + 'games.csv')
    clubs_csv = pd.read_csv(folder_path + 'clubs.csv')
    appearances_csv = pd.read_csv(folder_path + 'new_appearances.csv')
    additional_tademarkt_stats_csv = pd.read_csv(folder_path + 'additional_trademarkt_stats_no_na.csv')
    nonDomesticLeagues = ['EL','CL']
    games = []
    start = time.time()
    if date != None:
        filter_csv = pd.DataFrame(columns=['date'])
        filter_csv['date'] = pd.to_datetime(games_csv['date'])
        filter = (filter_csv['date']  < date) if to == 1 else (filter_csv['date']  >= date)
        games_csv = games_csv[filter]
    print(f'starting import of {len(games_csv)}')
    for i in range(len(games_csv)): #len(games_csv)
        
        date = datetime.datetime.strptime(games_csv.iloc[i]['date'], '%Y-%m-%d')
        game_row = games_csv.iloc[i]
        game_id = game_row['game_id']
        home_club_id = game_row['home_club_id']
        away_club_id = game_row['away_club_id']
        home_club = clubs_csv.query('club_id == @home_club_id')
        away_club = clubs_csv.query('club_id == @away_club_id')
        isDomestic = game_row['competition_id'] not in nonDomesticLeagues
        home_club_lineup = appearances_csv.query('game_id == @game_id and player_club_id == @home_club_id').apply(lambda x: Player(x['player_id'], x['age'],
                             x['player_name'], x['yellow_cards'], x['red_cards'], x['goals'], x['assists'], x['minutes_played'], x['recent_valuation'], x['max_valuation'], 1), axis=1)
        away_club_lineup = appearances_csv.query('game_id == @game_id and player_club_id == @away_club_id').apply(lambda x: Player(x['player_id'], x['age'],
                             x['player_name'], x['yellow_cards'], x['red_cards'], x['goals'], x['assists'], x['minutes_played'], x['recent_valuation'], x['max_valuation'], 0), axis=1)
        game_stats = additional_tademarkt_stats_csv.query('game_id == @game_id')
        home_club_stats = TeamStats(game_row['home_club_goals'], game_stats['home_club_total_shots'], game_stats['home_club_shots_off_target'], game_stats['home_club_shots_saved'],
                                     game_stats['home_club_corners'], game_stats['home_club_free_kicks'], game_stats['home_club_fouls'], game_stats['home_club_offsides'], 1)
        away_club_stats = TeamStats(game_row['away_club_goals'], game_stats['away_club_total_shots'], game_stats['away_club_shots_off_target'], game_stats['away_club_shots_saved'],
                                     game_stats['away_club_corners'], game_stats['away_club_free_kicks'], game_stats['away_club_fouls'], game_stats['away_club_offsides'], 0)
        home_club_team = Team(home_club_id, home_club['name'], list(home_club_lineup), home_club_stats)
        away_club_team = Team(away_club_id, away_club['name'], list(away_club_lineup), away_club_stats)
        game = Game(game_id, date, home_club_team, away_club_team, isDomestic)
        games.append(game)
    end = time.time()
    print(f'Created {len(games)} objects in {end-start}')  
    return games

