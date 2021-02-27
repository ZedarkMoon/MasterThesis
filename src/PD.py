from __future__ import annotations
import numpy as np
from collections import Counter
import itertools as it
import typing
import random

#All RL packafe
import torch as T
import torch.nn as nn
import torch.nn.functional as F #for the activation function
import torch.optim as optim

from numpy.core.arrayprint import _leading_trailing
from numpy.core.numeric import NaN

Strategy = {'C':0 , 'D':1}
PD = [[3, 0], [5, 1]]

class DeepQNetwork(nn.Module): #Give access to a lot of function

    def __init__(self, lr, input_dim, fc1_dim, fc2_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.output_dim)

        self.optimize = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)    

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)

        return action    

class Agent:

    def takeAction(self) -> None:
        raise NotImplementedError 

    def getAction(self) -> str:
        raise NotImplementedError  

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError 

class DeepRLNetwork(Agent):

    def __init__(self, gamma, epsilon, lr, input_dim, batch_size, n_actions, max_mem_size=100000,
        eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.actions_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, input_dim=input_dim, fc1_dim=256, fc2_dim=256, output_dim=n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)

class RLAgent(Agent):

    def __init__(self, learning_rate, gamma, epsilon) -> None:
        temp = list("".join(elem) for elem in it.product(Strategy.keys(), repeat=2))
        self.state = None
        self.action = None
        self.q_table = {row: {col:0 for col in Strategy} for row in temp}
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.avg_reward =  np.array([])
        self.cooperation = [0, 0]

    def __str__(self) -> str:
        return f"Current State: {self.state}\n" \
        f"Q-Tables: {self.q_table}\n"\
        f"Level of Cooperation: {self.cooperation[0]/sum(self.cooperation) *100:.2f}%\n"\
        f"Average reward at each round: {np.mean(self.avg_reward)}"
        

    def takeAction(self) -> None:
        if random.uniform(0, 1) < self.epsilon or self.action == None: # Exploration
            self.action = random.choice(list(Strategy))
        else: # Exploitation
            temp = self.q_table[self.state] #+ Counter(self.q_table_2[self.state])
            self.action = max(temp, key=temp.get)

        self.cooperation[Strategy[self.action]] += 1

    def getAction(self) -> str:
        return self.action

    def getQTable(self) -> dict:
        return self.q_table

    def update(self, opponent_action, reward) -> None:
        self.__updateQtable(opponent_action, reward)

    def __updateQtable(self, opponent_action, reward) -> None:
        print(self.learning_rate, self.gamma, reward)
        new_state = "".join((self.action, opponent_action))
        self.avg_reward = np.append(self.avg_reward, reward)
        if self.state == None:
            self.state = new_state

        self.q_table[self.state][self.action] = (1-self.learning_rate)*self.q_table[self.state][self.action] + \
            self.learning_rate*(reward + (self.gamma * max(self.q_table[new_state].values())))

        #self.q_table[self.state][self.action] *= 0.95

        self.state = new_state

    

class CooperativeAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 0 #->cooperate

    def getAction(self) -> str:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class DefectiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 1 #->Defect

    def getAction(self) -> str:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class TitForTatAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 0

    def getAction(self) -> str:
            return self.action

    def update(self, opponent_action, *args, **kwargs) -> None:
        self.action = opponent_action
    


if __name__ == "__main__":

    rl_agent = RLAgent(0.1, 0.8, 0.1)
    adversary  = DefectiveAgent()

    nbr_episode = 100000

    for i in range(nbr_episode):
        rl_agent.takeAction()

        adversary_action = adversary.getAction()

        reward = PD[Strategy[rl_agent.getAction()]][Strategy[adversary_action]]

        rl_agent.update(adversary_action, reward)
        adversary.update(rl_agent.getAction())


    print(rl_agent)


