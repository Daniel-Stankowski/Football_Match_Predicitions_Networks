import torch
from objects.player import Player
from objects.team_stats import TeamStats
from typing import List
class Team:
    def __init__(self, teamId, teamName, lineUp: List[Player], teamStats: TeamStats) -> None:
        self.teamId = teamId
        self.teamName = teamName
        self.lineUp = lineUp
        self.teamStats = teamStats
    
    def to_tensor(self, max_players=20):
        padded_lineup = self.lineUp + [Player(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.teamStats.isHome)] * (max_players - len(self.lineUp))

        team_stats_tensor = self.teamStats.to_tensor()

        players = [player.to_tensor() for player in padded_lineup]

        team_id_tensor = torch.tensor([self.teamId], dtype=torch.int64)
        

        all_tensors = [list(players), torch.cat((team_id_tensor, team_stats_tensor))]

        return all_tensors
    
    def to_tensor_stack_players(self, max_players=20):
        padded_lineup = self.lineUp + [Player(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.teamStats.isHome)] * (max_players - len(self.lineUp))

        team_stats_tensor = self.teamStats.to_tensor()

        players = list(player.to_single_tensor() for player in padded_lineup)
        team_id_tensor = torch.tensor([self.teamId], dtype=torch.int64)
        team_tensor = [torch.cat([team_id_tensor, team_stats_tensor])]
        team_tensor.extend(players)
        all_tensors = torch.stack(tuple(team_tensor))

        return all_tensors
    
    def to_tensor_stack_players_no_padding(self):
        team_stats_tensor = self.teamStats.to_tensor()

        players = list(player.to_single_tensor() for player in self.lineUp)
        team_id_tensor = torch.tensor([self.teamId], dtype=torch.int64)
        team_tensor = [torch.cat([team_id_tensor, team_stats_tensor])]
        team_tensor.extend(players)
        all_tensors = torch.stack(tuple(team_tensor))

        return all_tensors
    