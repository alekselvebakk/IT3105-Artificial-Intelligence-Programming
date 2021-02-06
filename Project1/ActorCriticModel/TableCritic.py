class TableCritic:
    def __init__(self):
        self.table = dict()
        self.elig = dict()

    
    
    def episode_reset(self):
        for state in self.elig:
            self.elig[state] = 0