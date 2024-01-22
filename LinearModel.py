import torch
import torch.nn as nn
import torch.nn.functional as F
from import_data_to_objects import getObjects
import torch.optim as optim
import datetime

class LinearFootballPredictionModel(nn.Module):
    def __init__(self, team_output=32, player_output = 32, embedding_output = 64,
                  player_embedding_output = 64, combined = 64, combined_player = 64, combine_team_players_output = 256):
        super(LinearFootballPredictionModel, self).__init__()
        self.team_output = team_output
        self.embedding_output = embedding_output
        self.player_embedding_output = player_embedding_output
        self.player_output = player_output
        self.combined = combined
        self.combined_player = combined_player
        self.combine_team_players_output = combine_team_players_output
        self.teams = 2
        self.team_input = 9
        self.players_input = 9
        self.players = 20

        self.team_embedding = nn.Embedding(68700, self.embedding_output)
        self.player_embedding = nn.Embedding(1111780, self.player_embedding_output)
        self.team_embedding.cuda()
        self.player_embedding.cuda()
        self.player_stats = nn.Linear(self.players_input, self.player_output)
        self.team_stats_layer = nn.Linear(self.team_input, self.team_output)
    
        self.team_combined = nn.Linear(self.embedding_output + self.team_output, self.combined)
        self.player_combined = nn.Linear(self.player_embedding_output + self.player_output, self.combined_player)

        self.combine_team_players = nn.Linear(self.embedding_output + self.players * self.combined_player, self.combine_team_players_output)

        self.combine_team_players_embedding_only = nn.Linear(self.embedding_output + self.players * self.player_embedding_output, self.combine_team_players_output)

        self.output_layer = nn.Linear(self.teams * self.combine_team_players_output, 3)




    def forward(self, input_data=None, prediction_data = None):
        # Unpack the input data
        if prediction_data is None:
            input_data = input_data.cuda()
            home_tensor, away_tensor = input_data
            home_id, away_id = home_tensor[0, 0], away_tensor[0, 0]
            home_stats, away_stats = home_tensor[:, 1:], away_tensor[:, 1:]
            home_player_ids, away_player_ids = home_tensor[1:, 0], away_tensor[1:, 0]

            home_embedding = self.team_embedding(home_id.long().cuda())
            away_embedding = self.team_embedding(away_id.long().cuda())

            home_team_stats = self.team_stats_layer(home_stats[0])
            away_team_stats = self.team_stats_layer(away_stats[0])

            home_combined = F.relu(self.team_combined(torch.cat([home_embedding, home_team_stats])))
            away_combined = F.relu(self.team_combined(torch.cat([away_embedding, away_team_stats])))

            home_players = F.relu(self.player_combined(torch.cat([self.player_embedding(home_player_ids.long().cuda()),self.player_stats(home_stats[1:])], dim=1)))
            away_players = F.relu(self.player_combined(torch.cat([self.player_embedding(away_player_ids.long().cuda()),self.player_stats(away_stats[1:])], dim=1)))

            team_players_home = F.relu(self.combine_team_players(torch.cat([home_combined.view(1, -1), home_players], dim=0).view(1, -1)))
            team_players_away = F.relu(self.combine_team_players(torch.cat([away_combined.view(1, -1), away_players], dim=0).view(1, -1)))

            
            output = F.relu(self.output_layer(torch.cat([team_players_home, team_players_away]).view(1, -1)))
            output = nn.functional.softmax(output, dim=1)
            return output.squeeze()
        else:
            prediction_data = prediction_data.cuda()
            home_team_id, away_team_id = prediction_data[:2]
            home_players_ids, away_players_ids = prediction_data[2:22], prediction_data[22:]

            home_embedding = self.team_embedding(home_team_id.long().cuda())
            away_embedding = self.team_embedding(away_team_id.long().cuda())

            home_players_embeddings = self.player_embedding(home_players_ids.long().cuda())
            away_players_embeddings = self.player_embedding(away_players_ids.long().cuda())
            
            team_players_home = F.relu(self.combine_team_players_embedding_only(torch.cat([home_embedding.view(1, -1), home_players_embeddings], dim=0).view(1, -1)))
            team_players_away = F.relu(self.combine_team_players_embedding_only(torch.cat([away_embedding.view(1, -1), away_players_embeddings], dim=0).view(1, -1)))

            output = F.relu(self.output_layer(torch.cat([team_players_home, team_players_away]).view(1, -1)))
            output = nn.functional.softmax(output, dim=1)
            return output.squeeze()





# games= getObjects(date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'), to=0)

# model = LinearFootballPredictionModel()
# model.cuda()
# input_data = games[0].to_single_tensor()
# # home_tensor, away_tensor = input_data
# # home_id, away_id = home_tensor[0, 0], away_tensor[0, 0]
# # home_stats, away_stats = home_tensor[:, 1:], away_tensor[:, 1:]
# # home_player_ids, away_player_ids = home_tensor[1:, 0], away_tensor[1:, 0]
# # print(list([p for p in home_player_ids]))
# # print(input_data)
# # print(input_data[0,0,0])
# data = torch.cat([input_data[:,0,0], input_data[0,1:,0], input_data[1,1:,0]])
# print(data)
# output = model(prediction_data = data)
# print(output)



