from hashlib import sha1
import numpy as np
from torch.nn import Module, Sequential, Conv2d, ReflectionPad2d, Conv1d, PReLU, BatchNorm2d, BatchNorm1d, Linear
import torch.nn.functional as F
import torch

import Game
from helpers import Position
from ValueFunctions.ActionValueFunction import ActionValueFunction
import helpers as h

class CNN(ActionValueFunction, Module):

    def __init__(self, gamma, lambd, alpha):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.e = {}
        self.conv1 = Conv2d(3, 64, kernel_size=5, padding=2).to('cuda')
        # self.act1 = PReLU()
        self.act1 = torch.tanh
        # self.norm1 = BatchNorm2d(24).to('cuda')
        self.conv2 = Conv2d(64, 128, kernel_size=5, padding=2).to('cuda')
        # self.act2 = PReLU()
        self.act2 = torch.tanh
        # self.norm2 = BatchNorm2d(48).to('cuda')
        self.conv3 = Conv2d(128, 128, kernel_size=5, padding=2).to('cuda')
        # self.act3 = PReLU()
        self.act3 = torch.tanh
        # self.norm3 = BatchNorm2d(64).to('cuda')
        # self.conv_final = Conv2d(64, 1, kernel_size=3, padding=1)
        self.dense_final = Linear(128 * 3 * 3, 9).to('cuda')
        # self.act_final = PReLU()
        self.act_final = torch.tanh
        # self.norm_final = BatchNorm1d(1).to('cuda')

        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                print(name)
                self.e[name] = {}
        pass

    def forward(self, x):


        x = torch.from_numpy(x).unsqueeze(0).float().to('cuda')
        # x = self.pad1(x)
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.act1(x) * 2

        # x = self.pad2(x)
        x = self.conv2(x)
        # x = self.norm2(x)
        x = self.act2(x) * 2

        # x = self.pad3(x)
        x = self.conv3(x)
        # x = self.norm3(x)
        x = self.act3(x) * 2

        x = x.view(x.size(0), 1, -1)
        x = self.dense_final(x)
        # x = self.norm_final(x)
        x = self.act_final(x) * 2
        x = torch.reshape(x, (1, 1, 3, 3))

        # initial_gradient = torch.ones([1, 1, 3, 3])
        # x.retain_grad()
        # # x.zero_()
        # x.backward(initial_gradient, retain_graph=True)

        return x

    def get_action_value(self, X_s, O_s, p: Position):
        pass

    def get_action_values(self, X_s, O_s):
        ## Fake the list of empty spaces
        empty_spaces = np.logical_not(np.logical_or(X_s, O_s)).astype(int)
        x = np.array([X_s, O_s, empty_spaces])
        res = self.forward(x)
        return res



    def set_action_value(self, X_s, O_s, p: Position, value):
        pass

    def set_action_values(self, X_s, O_s, values):
        pass

    def update_action_values(self, guess: torch.Tensor, error, state_hash):
        # error = torch.from_numpy(error).float()
        # initial_gradient = torch.randn([1, 1, 3, 3])
        # guess.retain_grad()
        # guess.backward(initial_gradient) ##  We now have the grad in each layer
        # torch.nn.utils.clip_grad_value_(self.conv1.weight, 0.1)
        # torch.nn.utils.clip_grad_value_(self.conv2.weight, 0.1)
        # torch.nn.utils.clip_grad_value_(self.conv3.weight, 0.1)
        # torch.nn.utils.clip_grad_value_(self.conv_final.weight, 0.1)



        initial_gradient = torch.ones([1, 1, 3, 3]).to('cuda')
        guess.retain_grad()
        guess.backward(initial_gradient, retain_graph=True)


        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if state_hash not in self.e[name]:
                self.e[name][state_hash] = 0
            for state in self.e[name].keys():
                self.e[name][state] = self.e[name][state] * self.lambd * self.gamma
            self.e[name][state_hash] += parameter.grad
            for state in self.e[name].keys():
                parameter.data -= self.alpha * error * self.e[name][state]
            parameter.grad.zero_()


        # with torch.no_grad():
        # self.e1 = self.e1 * self.lambd * self.gamma + self.conv1.weight.grad
        # self.e2 = self.e2 * self.lambd * self.gamma + self.conv2.weight.grad
        # self.e3 = self.e3 * self.lambd * self.gamma + self.conv3.weight.grad
        # self.e_final = self.e_final * self.lambd * self.gamma + self.conv_final.weight.grad
        # self.conv1.weight.data -= self.alpha * error * self.e1
        # self.conv2.weight.data -= self.conv2.weight - self.alpha * error * self.e2
        # self.conv3.weight.data -= self.conv3.weight - self.alpha * error * self.e3
        # self.conv_final.weight.data -= self.conv_final.weight - self.alpha * error * self.e_final
        # self.conv1.zero_grad()
        # self.conv2.zero_grad()
        # self.conv3.zero_grad()
        # self.conv_final.zero_grad()

    def reset(self):
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                self.e[name] = {}

    def save(self, path):
        np.savez('objects/' + path + '.npz', table=self.lookup_table)

    def load(self, path):
        res = np.load('objects/' + path + '.npz', allow_pickle=True)
        self.lookup_table = res['table'].item()




