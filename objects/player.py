import torch
class Player:
    def __init__(self, player_id, age, name, yellow_cards, red_cards, goals, assists, minutes_played, most_recent_valuation, max_valuation, isHome) -> None:
        self.player_id = player_id
        self.name = name
        self.age = age
        self.yellow_cards = yellow_cards
        self.red_cards = red_cards
        self.goals = goals
        self.assists = assists
        self.minutes_played = minutes_played
        self.most_recent_valuation = (most_recent_valuation / 1000000)
        self.max_valuation = (max_valuation / 1000000)
        self.isHome = isHome

    @staticmethod
    def normalize_player_data(player_list):
        tensors = [player.to_tensor() for player in player_list]
        player_tensors = torch.stack([tensor[0] for tensor in tensors], dim=0)
        ids = [tensor[1] for tensor in tensors]

        min_val = torch.min(player_tensors)
        max_val = torch.max(player_tensors)

        normalized_player_tensors = (player_tensors - min_val) / (max_val - min_val)

        individual_player_tensors = torch.unbind(normalized_player_tensors, dim=0)
        individual_player_tensors = [[ids[i], tensor] for i,tensor in enumerate(normalized_player_tensors)]
        for i in individual_player_tensors:
            if i[0][0] == 1:
                i[1].fill_(0)
                i[1][-1] = player_list[0].isHome
        tensors_to_return = [torch.cat((tensors[0], tensors[1])) for tensors in individual_player_tensors]
        return tensors_to_return

    def to_tensor(self):
        player_data_tensor = torch.tensor([
            self.age,
            self.yellow_cards,
            self.red_cards,
            self.goals,
            self.assists,
            self.minutes_played,
            self.most_recent_valuation,
            self.max_valuation,
            self.isHome
        ], dtype=torch.float32)

        player_id_tensor = torch.tensor([self.player_id], dtype=torch.int64)

        return [player_data_tensor, player_id_tensor]
    
    def to_single_tensor(self):
        player_data_tensor = torch.tensor([
            self.age,
            self.yellow_cards,
            self.red_cards,
            self.goals,
            self.assists,
            self.minutes_played,
            self.most_recent_valuation,
            self.max_valuation,
            self.isHome
        ], dtype=torch.float32)

        player_id_tensor = torch.tensor([self.player_id], dtype=torch.int64)
         
        return torch.cat([player_id_tensor, player_data_tensor])