from Project1.ActorCriticModel.Actor import Actor
from Project1.ActorCriticModel.NetCritic import NetCritic
from Project1.ActorCriticModel.TableCritic import TableCritic




class LearningModule:
    def __init__(   self,
                    neural_net_critic = True,
                    epsilon = 0.1, 
                    alpha_actor = 0.1,
                    alpha_critic = 0.1,
                    gamma = 0.1,
                    elig_decay = 0.1,
                    layers = [20, 30, 50, 1]):
        #Initialize Actor Object
        self.actor = Actor( alpha_actor, 
                            gamma, 
                            elig_decay)
        #Initialize Critic Object
        if neural_net_critic:
            self.critic = NetCritic(    alpha_critic,
                                        layers,
                                        )
        else:
            self.critic = TableCritic(  alpha_critic,
                                        gamma,
                                        elig_decay)
        #Set constants
        self.epsilon = epsilon
        self.current_episode = [[]]

    def initialize_episode( self, 
                            initial_state, 
                            initial_actions):
        self.state = initial_state
        self.actions = initial_actions
        self.episode_finished = False
        self.actor.episode_reset()
        self.critic.episode_reset()
        self.action = self.actor.get_action(self.state, 
                                            self.actions,
                                            self.epsilon)
        

        self.current_episode = [[self.state, self.action]]
        return self.action
    
    def episode_step(   self, 
                        next_state, 
                        next_possible_actions,
                        reward,
                        next_state_is_final = False):




        #Check if next step is final step
        if next_state_is_final == False:
            next_action = self.actor.get_action(next_state, 
                                                next_possible_actions, 
                                                self.epsilon)
        else:
            next_action = []
        
        #Set eligibilities for current states to 1
        #CRITIC                                
        self.critic.set_unit_eligibility(self.state)
        #ACTOR
        self.actor.set_unit_eligibility(self.state, self.action)

        #Calculate delta
        delta = self.critic.get_delta(  self.state,
                                        next_state,
                                        reward)

        #Update for all states and actions in current epsisode TODO: Check with Aleks that this change is OK
        for SAP in self.current_episode:
            state = SAP[0]
            action = SAP[1]
            #CRITIC update
            self.critic.update_value_function(state, delta)
            #ACTOR update
            self.actor.update_policy_table( state, 
                                            action,
                                            delta)

        if next_state_is_final == False:
            self.state = next_state
            self.action = next_action
            self.current_episode.append([next_state, next_action])
        return next_action