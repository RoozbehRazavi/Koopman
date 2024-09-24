import torch
import numpy as np 
import random 
from collections import deque
import gym

import imageio
import os

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import shutil
import torchvision
from termcolor import colored

import time
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset, DataLoader

import pickle
import ast
import torch
import torch.nn as nn
        
def loss_function1(self, A: nn.Parameter, B: nn.Parameter, states, actions):
        """
        Arguments:
            A: Dynamics matrix of size (d x d)
            B: Control matrix of size (d x m)
            states: Batch of initial states of size (N x T x d)
            actions: Actions taken at each time step of size (N x T x m)
            positive_samples: Positive samples (ground truth future states) of size (N x T x d)
            negative_samples: Negative samples (unrelated states) of size (N x T x K x d)
        """
        A.requires_grad = False
        B.requires_grad = False
        
        N, T, m = actions.shape  # Batch size, time steps, action dimension
        d = states.shape[2]  # State dimension
        
        # Step 1: Compute the anchor (future) state in the embedding space
        z_t = self(states)
        
        z_future = torch.zeros((N, T, T, self.embedding_dim))  # Initialize future state embeddings (N x hidden_dim)

        # TODO we should have masking here too!

        for i in range(N):
            for n in range(T):
                for m in range(n+1, T):
                    action = actions[i, n, :].unsqueeze(0)
                    if m == n+1:
                        z_t_ = z_t[i, n, :].unsqueeze(-1)
                        temp = A @ z_t_ + B @ action.T
                        z_future[i, n, m] = temp.squeeze(-1)
                    else:
                        temp = A @ z_future[i, n, m-1, :].unsqueeze(-1) + B @ action.T
                        z_future[i, n, m] = temp.squeeze(-1)

        # Pass positive samples through the RNN and get embeddings
        with torch.no_grad():
            z_positive = self(states)  # (1 x N x hidden_dim)
        loss = 0
        counter = 0
        nan_counter = 0
        inf_counter = 0
        for i in range(N):
            for n in range(T):
                for m in range(n, T):
                    if m > n:
                        positive_similarity = self.similarity(z_future[i, n, m], z_positive[i, m])
                        negative_similarity = 0
                        for k in range(10):
                            if k != m:
                                negative_similarity += self.similarity(z_future[i, n, m], z_positive[i, k])
                        loss1 = positive_similarity / (negative_similarity + positive_similarity)
                        if torch.isnan(loss1):
                            nan_counter += 1
                        elif torch.isinf(loss1):
                            inf_counter += 1
                        else:
                            loss += loss1
                            counter += 1
                    #print(f'Computing loss for trajectory {i}, time step {n}, future time step {m}')
        
        # TODO what is the purpose of this one? Log-softmax over the positive and negative similarities
        # logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)  # (N x (1 + K))
        # labels = torch.zeros(N, dtype=torch.long).to(logits.device)  # Positive class index is 0
        # loss = nn.CrossEntropyLoss()(logits, labels)
        
        A.requires_grad = True
        B.requires_grad = True
        print('Nan: ', nan_counter, ' Inf: ', inf_counter)
        return loss/counter