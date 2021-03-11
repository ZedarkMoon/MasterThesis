from __future__ import annotations
import numpy as np
from collections import Counter
import itertools as it
import typing
import random
from tqdm import tqdm

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

class DeepRLNetwork():

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
        
        #*input_dim -> actions of all the agent
        self.state_memory = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)


    def storeTransition(self, state, action, reward, new_state):
        index = self.mem_cntr%self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def chooseAction(self, observation):
        if np.random.rand() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item() #Because return a tensor
        else:
            action = np.random.choice(self.actions_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return #not enough data to learn

        self.Q_eval.optimize.zero_grad() #Particular in pytorch
        max_mem = min(self.mem_size, self.mem_cntr) #until self.mem_size < self.mem_cntr
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch= T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimize.step()

class Agent:

    def takeAction(self) -> None:
        raise NotImplementedError 

    def getAction(self) -> str:
        raise NotImplementedError  

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError 

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

    rl_agent =DeepRLNetwork(gamma=0.8, epsilon=0.1, lr=0.01, input_dim=[2], batch_size=64, n_actions=2)
    adversary  = DefectiveAgent()

    nbr_episode = 100000

    state = np.array([0,0])
    new_state = np.zeros(2)

    cooperation_level = 0

    print(rl_agent.Q_eval.forward(T.tensor([0, 0], dtype=T.float)))

    for i in tqdm(range(nbr_episode)):

        rl_action = rl_agent.chooseAction(state)
        adversary_action = adversary.getAction()

        new_state[0], new_state[1] = rl_action, adversary_action
        reward = PD[rl_action][adversary_action]

        rl_agent.storeTransition(state, rl_action, reward, new_state)
        rl_agent.learn()

        state = new_state
        adversary.update(rl_action)
        cooperation_level += rl_action

    print(1-(cooperation_level/nbr_episode))
    print(rl_agent.Q_eval.forward([0, 0]))
    print(rl_agent.Q_eval.forward([0, 1]))
    print(rl_agent.Q_eval.forward([1, 0]))
    print(rl_agent.Q_eval.forward([1, 1]))


    print(rl_agent)


