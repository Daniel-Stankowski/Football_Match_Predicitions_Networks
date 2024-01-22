import torch
import torch.nn as nn
import torch.nn.functional as F
from import_data_to_objects import getObjects
import torch.optim as optim
import datetime

class LSTMFootballPredictionModel(nn.Module):
    def __init__(self, game_output=32, team_output=32, embedding_output = 9, combined = 32, hidden_size = 256):
        super(LSTMFootballPredictionModel, self).__init__()
        self.game_output = game_output
        self.team_output = team_output
        self.hidden_size = hidden_size
        self.embedding_output = embedding_output
        self.combined = combined
        self.teams = 2
        self.team_input = 9
        self.players = 20

        self.team_embedding = nn.Embedding(68700, self.embedding_output)
        self.player_embedding = nn.Embedding(1111800, self.embedding_output)
        self.team_embedding.cuda()
        self.player_embedding.cuda()

        self.team_stats_layer = nn.LSTM(self.team_input, self.hidden_size, batch_first=True)

        self.combined_layer = nn.Linear(self.teams * self.hidden_size, 3)




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

            home_player_embeddings = self.player_embedding(home_player_ids.long().cuda())
            away_player_embeddings = self.player_embedding(away_player_ids.long().cuda())
            
            home_list = []
            away_list = []
            for i in range(20):
                home_list.extend([home_stats[i+1], home_player_embeddings[i, :]])
                away_list.extend([away_stats[i+1], away_player_embeddings[i, :]])
            home_list.extend([home_stats[0],home_embedding])
            away_list.extend([away_stats[0],away_embedding])
            home, _ = self.team_stats_layer(torch.stack(tuple(home_list)))
            away, _ = self.team_stats_layer(torch.stack(tuple(away_list)))
            output = self.combined_layer(torch.cat([home[-1, :], away[-1, :]]))
            output = nn.functional.softmax(output, dim=0)
            return output
        else:
            prediction_data = prediction_data.cuda()
            home_team_id, away_team_id = prediction_data[:2]
            home_players_ids, away_players_ids = prediction_data[2:22], prediction_data[22:]

            home_embedding = self.team_embedding(home_team_id.long().cuda())
            away_embedding = self.team_embedding(away_team_id.long().cuda())

            home_player_embeddings = self.player_embedding(home_players_ids.long().cuda())
            away_player_embeddings = self.player_embedding(away_players_ids.long().cuda())
            home_stats = torch.cat([home_embedding.unsqueeze(0), home_player_embeddings])
            away_stats = torch.cat([away_embedding.unsqueeze(0), away_player_embeddings])
            home, _ = self.team_stats_layer(home_stats)
            away, _ = self.team_stats_layer(away_stats)
            output = self.combined_layer(torch.cat([home[-1, :], away[-1, :]]))
            output = nn.functional.softmax(output, dim=0)
            return output





# games= getObjects(date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'), to=0)

# model = RNNFootballPredictionModel()
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
# output = model(input_data = input_data)
# print(output)



