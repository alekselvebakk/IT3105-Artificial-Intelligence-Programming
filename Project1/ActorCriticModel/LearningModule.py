from ActorCriticModel.Actor import Actor
from ActorCriticModel.NetCritic import NetCritic
from ActorCriticModel.TableCritic import TableCritic




class LearningModule:
    def __init__(   self,
                    NNCriticBool = True,
                    epsilon = 0.1, 
                    alpha_actor = 0.1,
                    alpha_critic = 0.1,
                    gamma_actor = 0.1,
                    gamma_critic = 0.1,
                    elig_decay_actor = 0.1,
                    elig_decay_critic = 0.1,
                    epsilon_decay = 0.9,
                    hidden_layers = [20, 30, 50],
                    input_size = 16):
        #Set constants and flags
        self.epsilon = epsilon
        self.NNCriticBool = NNCriticBool
        self.current_episode = [[]]
        self.epsilon_decay = epsilon_decay
        hidden_layers = [int(x) for x in hidden_layers.split(",")]
        
        
        
        
        #Initialize Actor Object
        self.actor = Actor( alpha_actor, 
                            gamma_actor,
                            elig_decay_actor)
        #Initialize Critic Object
        if self.NNCriticBool:
            self.critic = NetCritic(    alpha_critic,
                                        gamma_critic,
                                        elig_decay_critic,
                                        hidden_layers,
                                        input_size)
        else:
            self.critic = TableCritic(  alpha_critic,
                                        gamma_critic,
                                        elig_decay_critic)
        

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
    def decay_epsilon(self, zero = False):
        if zero:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon*self.epsilon_decay
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
        if self.NNCriticBool == False:                                
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
