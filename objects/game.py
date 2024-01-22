from objects.team import Team
from datetime import datetime
import torch
class Game:
    def __init__(self, gameId, date: datetime, homeTeam: Team, awayTeam: Team, isDomestic: bool) -> None:
        self.gameId = gameId
        self.date = date
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.isDomestic = isDomestic

    def to_tensor(self):
        home_team_tensor = self.homeTeam.to_tensor()
        away_team_tensor = self.awayTeam.to_tensor()

        year, month, day = self.date.year, self.date.month, self.date.day
        overall_game_stats_tensor = torch.tensor([          
            int(self.isDomestic) 
        ], dtype=torch.float32)
        return [overall_game_stats_tensor, home_team_tensor, away_team_tensor]
    def to_single_tensor(self):
        home_team_tensor = self.homeTeam.to_tensor_stack_players()
        away_team_tensor = self.awayTeam.to_tensor_stack_players()
        return torch.stack((home_team_tensor, away_team_tensor))
    
    def to_single_tensor_no_padding(self):
        home_team_tensor = self.homeTeam.to_tensor_stack_players_no_padding()
        away_team_tensor = self.awayTeam.to_tensor_stack_players_no_padding()
        return torch.stack((home_team_tensor, away_team_tensor))

    def getOutput(self):
        homeGoals = self.homeTeam.teamStats.goals
        awayGoals = self.awayTeam.teamStats.goals
        if homeGoals > awayGoals:
            return torch.tensor([1., 0., 0.])
        elif awayGoals > homeGoals:
            return torch.tensor([0., 0., 1.])
        else:
            return torch.tensor([0., 1., 0.])