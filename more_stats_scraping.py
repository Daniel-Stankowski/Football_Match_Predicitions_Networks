import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

games_csv = pd.read_csv('datasets_cleared/games.csv')
columns = ['game_id', 'home_club_id', 'away_club_id', 'home_club_total_shots',
            'away_club_total_shots', 'home_club_shots_off_target', 'away_club_shots_off_target',
              'home_club_shots_saved', 'away_club_shots_saved', 'home_club_corners', 'away_club_corners',
                'home_club_free_kicks', 'away_club_free_kicks', 'home_club_fouls', 'away_club_fouls', 'home_club_offsides', 'away_club_offsides']
knumber = 1758
per_dataset = 10
starting_set = 1756
datasets = [0] * knumber
for i in range(knumber):
    datasets[i] = pd.DataFrame(columns=columns)

for i in range(starting_set, knumber):
    for k in range(per_dataset):
        index = i * per_dataset + k
        if index>=len(games_csv):
            break
        url = games_csv.iloc[index]['url']
        url_segments = url.split(sep='/')

        url = f"https://{'/'.join(url_segments[2:4])}/statistik/{'/'.join(url_segments[5:])}"

        new_row = {}
        new_row['game_id'] = [games_csv.iloc[index]['game_id']]
        new_row['home_club_id'] = [games_csv.iloc[index]['home_club_id']]
        new_row['away_club_id'] = [games_csv.iloc[index]['away_club_id']]


        pageTree = requests.get(url, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
        pageTree.close

        elements = pageSoup.find_all("div", {"class": "sb-statistik-zahl"})
        for j in range(len(elements)):
            new_row[columns[3+j]] = elements[j].text

        datasets[i] = pd.concat([pd.DataFrame(new_row, columns=columns),datasets[i]], ignore_index=True)
        print(f"{index} {len(elements)}")
    print(f'Saving scraping/{i}.csv')
    datasets[i].to_csv(f'scraping/{i}.csv', index=False)

bigdf = pd.DataFrame(columns=columns)
for i in range(knumber):
    localdf = pd.read_csv(f'scraping/{i}.csv')
    bigdf = pd.concat([bigdf, localdf], ignore_index=True)
bigdf.to_csv('datasets_cleared/additional_trademarkt_stats.csv', index=False)
