from flask import Flask, jsonify, request
from LinearModel import LinearFootballPredictionModel
from RNNModel import RNNFootballPredictionModel
from LSTMModel import LSTMFootballPredictionModel
import torch
import pandas as pd
app = Flask(__name__)


linearModel = LinearFootballPredictionModel()
RNNModel = RNNFootballPredictionModel()
LSTMModel = LSTMFootballPredictionModel()

linearModel.load_state_dict(torch.load('models/v1/Linear_model.pth'), strict=False)
RNNModel.load_state_dict(torch.load('models/v1/RNN_model.pth'), strict=False)
LSTMModel.load_state_dict(torch.load('models/v1/LSTM_model.pth'), strict=False)
linearModel.to('cuda')
RNNModel.to('cuda')
LSTMModel.to('cuda')
linearModel.eval()
RNNModel.eval()
LSTMModel.eval()
models = {'Linear': linearModel, 'RNN': RNNModel, 'LSTM': LSTMModel}
players_csv = pd.read_csv('datasets_cleared/players.csv', index_col=None)
players_csv = players_csv.reset_index(drop=True)

@app.route('/api/predict/<path_variable>', methods=['GET'])
def predict_with_path_and_body(path_variable):
    # Extract data from the request body
    request_data = request.get_json()
    homeId = request_data['homeId']
    awayId = request_data['awayId']
    tensor = get_input_tensor(homeId, awayId).cuda()
    with torch.no_grad():
        output = models[path_variable](prediction_data = tensor)
    return {'kurs': calculate_kurs(output.tolist()), 'szanse': output.tolist() }

def get_input_tensor(home_id, away_id):
    homePlayers = get_players_of_team(home_id)
    awayPlayers = get_players_of_team(away_id)
    tensor = torch.tensor([
        home_id,
        away_id,
        *homePlayers,
        *awayPlayers
    ])
    return tensor

def get_players_of_team(teamId):
    players = players_csv.query('current_club_id == @teamId').sort_values(by='market_value_in_eur').iloc[:20,0]
    return [p for p in players]

def calculate_kurs(odds):
    return [1/o for o in odds]

if __name__ == '__main__':
    app.run(debug=True)