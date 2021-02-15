from __future__ import annotations
import numpy as np
from collections import Counter
import itertools as it
import typing
import random

from numpy.core.arrayprint import _leading_trailing
from numpy.core.numeric import NaN

Strategy = {'C':0 , 'D':1}
PD = [[3, 0], [5, 1]]

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
        self.action = 'C'

    def getAction(self) -> str:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class DefectiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 'D'

    def getAction(self) -> str:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class TitForTatAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 'C'

    def getAction(self) -> str:
            return self.action

    def update(self, opponent_action, *args, **kwargs) -> None:
        self.action = opponent_action
    


if __name__ == "__main__":

    rl_agent = RLAgent(1, 0.8, 0.1)
    adversary  = DefectiveAgent()

    nbr_episode = 100000

    for i in range(nbr_episode):
        rl_agent.takeAction()

        adversary_action = adversary.getAction()

        reward = PD[Strategy[rl_agent.getAction()]][Strategy[adversary_action]]

        rl_agent.update(adversary_action, reward)
        adversary.update(rl_agent.getAction())


    print(rl_agent)


