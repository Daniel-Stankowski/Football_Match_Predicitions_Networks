import torch
class TeamStats:
    def __init__(self, goals, totalShots, shotsOffTarget, shotsSaved, corners, freeKicks, fouls, offsides, isHome) -> None:
        self.goals = goals
        self.totalShots = totalShots.iloc[0]
        self.shotsOffTarget = shotsOffTarget.iloc[0]
        self.shotsSaved = shotsSaved.iloc[0]
        self.corners = corners.iloc[0]
        self.freeKicks = freeKicks.iloc[0]
        self.fouls = fouls.iloc[0]
        self.offsides = offsides.iloc[0]
        self.isHome = isHome
    
    def to_tensor(self):
       team_stats_tensor = torch.tensor([
           self.goals,
           self.totalShots,
           self.shotsOffTarget,
           self.shotsSaved,
           self.corners,
           self.freeKicks,
           self.fouls,
           self.offsides,
           self.isHome
       ], dtype=torch.float32)
       return team_stats_tensor