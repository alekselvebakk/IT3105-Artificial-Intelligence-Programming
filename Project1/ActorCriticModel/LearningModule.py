from Actor import Actor
from NetCritic import NetCritic
from TableCritic import TableCritic




class LearningModule:
    def __init__(   self, 
                    initial_state, 
                    neural_net_critic = False, 
                    epsilon = 0.1, 
                    alpha_actor = 0.1,
                    alpha_critic = 0.1,
                    gamma = 0.1):
        
        #Initialize Actor Critic Objects
        self.actor = Actor(epsilon, alpha_actor, gamma)
        if neural_net_critic:
            self.critic = NetCritic()
        else:
            self.critic = TableCritic()
        

        


