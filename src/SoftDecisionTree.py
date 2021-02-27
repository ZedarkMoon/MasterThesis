import typing
import torch
import numpy as np
from scipy.stats import truncnorm


class TreeParameters:

    def __init__(self, max_depth, max_leafs, nbr_features, nbr_output,
    regularization_penality=10., decay_penality = 0.9) -> None:
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.nbr_features = nbr_features
        self.nbr_output = nbr_output

        self.epsilon = 1e-8  #For calculation (prevent log(0))

        self.decay_penality = decay_penality
        self.regularisation_penality = regularization_penality

    def getMaxDepth(self):
        return self.max_depth

    def getMaxLeafs(self):
        return self.max_leafs

    def getDecayPenality(self):
        return self.decay_penality

    def getRegularisationPenality(self):
        return self.regularisation_penality

class Node:

    def __init__(self, depth, path_probability, tree: TreeParameters) -> None:
        self.left_child = None
        self.right_child = None
        self.pathprob = path_probability

        self.depth = depth
        self.epsilon = 1e-8 #this is a correction to avoid log(0)

        if self.depth == tree.getMaxDepth():
            self.is_leaf = True

        if self.is_leaf:
            self.W = torch.randn((tree.getNumberFeatures(), tree.getNumberoutput()), requires_grad=True)
            #truncnorm -> normal but values over 2 times the standard deviation got truncate
            self.B = torch.randn((truncnorm(-2, 2).rvs(tree.getNumberoutput()), ), requires_grad=True)
        else:
            self.W = torch.randn((tree.getNumberFeatures(), 1), requires_grad=True)
            #truncnorm -> normal but values over 2 times the standard deviation got truncate
            self.B = torch.randn(truncnorm(-2, 2).rvs(1), requires_grad=True)

    def forward(self, x):
        if self.is_leaf:
            softmax = torch.nn.Softmax()
            return softmax(np.multiply(x, self.W) + self.B)
        else:
            hard_sigmoid = torch.nn.Hardsigmoid()
            return hard_sigmoid(np.multiply(x, self.W) + self.B)

    def build(self, x, tree):
        self.prob = self.forward(x)
        if not self.is_leaf:
            self.right_child = Node(self.depth+1, self.pathprob*self.prob, tree)
            self.left_child = Node(self.depth+1, self.pathprob*self.prob, tree)

    def regularise(self, tree):
        if self.is_leaf:
            return 0.0
        else:
            alpha = torch.mean(self.pathprob*self.prob)/(self.epsilon* torch.mean(self.pathprob))
            return (-0.5 * torch.log(alpha + self.epsilon) - 0.5 * torch.log(
                1. - alpha + self.epsilon)) * (tree.getDecayPenality()** self.depth)

    def getLoss(self, y, tree):
        if self.is_leaf:
            -torch.mean(torch.log(self.epsilon + torch.sum(y *self.prob, 1)) * self.pathprob)
        else:
            return tree.getRegularisationPenality() * self.regularise(tree)


class SoftDecisiontree(torch.nn.Module):
    pass
