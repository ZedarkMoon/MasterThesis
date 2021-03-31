from __future__ import annotations
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from tqdm import tqdm
from DQN import DeepQNetwork, DeepRLNetwork
import torch as T

Strategy = {'C': 0, 'D': 1}
PD = [[3, 0], [5, 1]]
MODEL_NAME = "DQN_PD_TIT_FOR_TAT"

class Agent:

    def takeAction(self) -> None:
        raise NotImplementedError 

    def getAction(self) -> str:
        raise NotImplementedError  

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError 

class RLAgent(Agent):

    def __init__(self, learning_rate, gamma, epsilon, nbr_action, nbr_state) -> None:
        self.state = None
        self.action = None
        self.q_table = np.zeros((nbr_state, nbr_action))
        self.nbr_actions = nbr_action
        self.nbr_state = nbr_state
        
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
        if np.random.uniform() < self.epsilon or self.action == None: # Exploration
            self.action = np.random.randint(self.nbr_actions)
        else: # Exploitation
            temp = self.q_table[self.state] #+ Counter(self.q_table_2[self.state])
            self.action = np.argmax(temp)

        self.cooperation[self.action] += 1

    def getAction(self) -> int:
        return self.action

    def getQTable(self) -> dict:
        return self.q_table

    def getCoopearationLevel(self) ->float:
        return self.cooperation[0]/sum(self.cooperation)

    def update(self, opponent_action, reward) -> None:
        self.__updateQtable(opponent_action, reward)

    def __updateQtable(self, opponent_action, reward) -> None:
        new_state = int(f'0b{self.action}{opponent_action}', 2)
        self.avg_reward = np.append(self.avg_reward, reward)
        if self.state == None:
            self.state = new_state

        self.q_table[self.state][self.action] = (1-self.learning_rate)*self.q_table[self.state][self.action] + \
            self.learning_rate*(reward + (self.gamma * max(self.q_table[new_state]) ) )

        #self.q_table[self.state][self.action] *= 0.95

        self.state = new_state

    

class CooperativeAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 0 #->cooperate

    def getAction(self) -> int:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class DefectiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 1 #->Defect

    def getAction(self) -> int:
        return self.action

    def update(self, *args, **kwargs) -> None:
        pass

class TitForTatAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.action = 0

    def getAction(self) -> int:
            return self.action

    def update(self, opponent_action, *args, **kwargs) -> None:
        self.action = opponent_action

def saveTrainedModel(model, name):
    T.save(model.state_dict(), f"Models/{name}.pt")

def loadTrainedModel(name):
    model = DeepQNetwork(fc1_dim=256, fc2_dim=256, lr=0.01, input_dim=[2], output_dim=2)
    model.load_state_dict(T.load(f"Models/{name}.pt"))
    model.eval()
    return model

def trainDRL():
    rl_agent = DeepRLNetwork(gamma=0.8, epsilon=0.1, lr=0.1, input_dim=[4], batch_size=64, n_actions=2)
    adversary  = TitForTatAgent()
    nbr_episode = 100000
    state = np.random.randint(2, size=4)
    new_state = np.zeros(4)
    cooperation_level = 0
    loss_over_time=[]
    for _ in tqdm(range(nbr_episode)):
        rl_action = rl_agent.chooseAction(state)
        adversary_action = adversary.getAction()
        new_state[-2], new_state[-1] = rl_action, adversary_action
        reward = PD[rl_action][adversary_action]
        rl_agent.storeTransition(state, rl_action, reward, new_state)
        rl_agent.learn()
        state = new_state
        adversary.update(rl_action)
        cooperation_level += rl_action
        loss_over_time.append(sum(loss_over_time)+PD[1-rl_action][adversary_action])
        
    # saveTrainedModel(rl_agent.Q_eval, MODEL_NAME)
    print(1-(cooperation_level/nbr_episode))
    

    plt.plot(loss_over_time, np.arange(len(loss_over_time)))
    plt.show()

def train():
    rl_agent = RLAgent(0.1, 0.8, 0.1, 2, 4)
    adversary  = CooperativeAgent()
    nbr_episode = 10000
    cooperation_level = [0]
    loss_over_time=[0]

    for _ in tqdm(range(nbr_episode)):
        rl_agent.takeAction()
        rl_action = rl_agent.getAction()
        adversary_action = adversary.getAction()

        reward = PD[rl_action][adversary_action]

        adversary.update(rl_action)
        rl_agent.update(adversary_action, reward)

        cooperation_level.append(rl_agent.getCoopearationLevel())
        loss_over_time.append(loss_over_time[-1]+PD[1-rl_action][adversary_action] - reward)

    # plt.plot(np.arange(len(loss_over_time)), loss_over_time)
    # plt.save("tit_for_tat.png")


    f, (ax1, ax2)= plt.subplots(2)
    ax1.plot(np.arange(len(loss_over_time)),loss_over_time, 'b-', label="Loss Over Time")
    ax1.legend(loc="lower right")
    ax2.plot(np.arange(len(cooperation_level)), cooperation_level, 'r-', label="Cooperation Level")
    ax2.legend(loc="lower right")
    plt.savefig("cooperative.png")



def test():
    rl_agent = DeepRLNetwork(gamma=0.8, epsilon=0.1, lr=0.1, input_dim=[2], batch_size=64, n_actions=2)
    rl_agent.Q_eval = loadTrainedModel(MODEL_NAME)

    print("0 - 0:", rl_agent.testState([0,0]))
    print("0 - 1:", rl_agent.testState([0,1]))
    print("1 - 0:", rl_agent.testState([1,0]))
    print("1 - 1:", rl_agent.testState([1,1]))

if __name__ == "__main__":
    train()
    # test()
    
    




