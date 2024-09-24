import torch
import torch.nn as nn
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import dmc2gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym.wrappers import NormalizeObservation
from collections import deque
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import os 
import shutil

def setup_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

setup_directory('./logs13_mapping_epoch_2_b_128')
writer = SummaryWriter(log_dir="./logs13_mapping_epoch_2_b_128")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



class KoopmanMapping(nn.Module):
    def __init__(self, obs_dim, hidden_dim, embedding_dim) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, obs_dim)
        )
        # Define an RNN for processing sequences (trajectories)
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        
        # MLP to get anchor embedding from RNN hidden state
        self.anchor_emebedding_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # MLP to get positive/negative embedding from RNN hidden state
        # Only one 256
        self.pos_neg_embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        # Copy weights from anchor_embedding_head to pos_neg_embedding_head using state_dict
        #self.pos_neg_embedding_head.load_state_dict(self.anchor_emebedding_head.state_dict())

    def forward(self, trajectories, anchor=True, h_0=None):
        """
        Arguments:
            trajectories: Batch of trajectories (N x T x d)
        Returns:
            Embedding of the anchor (last hidden state of RNN)
        """
        #hidden_state = hidden_state.squeeze(0)  # Remove extra dimension, result is (N, hidden_dim)
        B = trajectories.shape[0]  # Batch size
        T = trajectories.shape[1]  # Time steps
        trajectories = trajectories.reshape((-1, trajectories.shape[-3], trajectories.shape[-2], trajectories.shape[-1]))
        trajectories = self.image_encoder(trajectories).reshape((B, T, -1))
        if h_0 is None:
            # Pass the trajectories through the RNN
            hidden_state, h_n = self.rnn(trajectories)  # hidden_state: (1, N, hidden_dim)
        else:
            hidden_state, h_n = self.rnn(trajectories, h_0)
        # Pass hidden state through anchor embedding head
        if anchor:
            z_anchor = self.anchor_emebedding_head(hidden_state)  # (N x embedding_dim)
        else:
            
            z_anchor = self.pos_neg_embedding_head(hidden_state)
        return z_anchor, h_n

    # def similarity(self, z1, z2, temperature=0.1):
    #     # Normalize z1 and z2 to unit vectors
    #     z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    #     z2_norm = z2 / z2.norm(dim=1, keepdim=True)
        
    #     # Compute the dot product of the normalized vectors
    #     dot_product = torch.einsum('bi,bi -> b', z1_norm, z2_norm)  # Shape: (batch_size,)
        
    #     # Apply the temperature scaling and take the exponential
    #     result = torch.exp(dot_product / temperature)
        
    #     # Handle potential NaN or infinite values
    #     if torch.isnan(result).any() or torch.isinf(result).any():
    #         pass  # You can add any handling logic here if necessary
        
    #     return result
    @staticmethod
    def log_sum_exp_with_temperature(dot_products, temperature=0.1):
        # Scale dot products by the temperature
        scaled_dot_products = dot_products / temperature
        
        # Apply the Log-Sum-Exp trick for numerical stability
        max_scaled = torch.max(scaled_dot_products, dim=-1, keepdim=True)[0]  # Find the max for each row
        log_sum_exp = max_scaled + torch.log(torch.sum(torch.exp(scaled_dot_products - max_scaled), dim=-1, keepdim=True))
        
        return log_sum_exp
    

    def similarity(self, z1, z2, temperature = 0.1):
        dot_product = torch.einsum('bi,bi -> b', z1, z2)  # Shape: (batch_size,)
        #dot_product =  torch.clamp(dot_product, max=2.5, min=2.5) #self.log_sum_exp_with_temperature(dot_product, temperature)
        result = torch.exp(dot_product / temperature)
        # mask = torch.isfinite(result).float()
        # result = result * mask
        if torch.isnan(result).any() or torch.isinf(result).any():
            pass 
            #breakpoint()
            print('Nan or Inf')
        return result

    def loss_function1(self, A: nn.Parameter, B: nn.Parameter, states, actions, LEN_PRED, nagative_count=10):
        """
        Arguments:
            A: Dynamics matrix of size (d x d)
            B: Control matrix of size (d x m)
            states: Batch of initial states of size (N x T x d)
            actions: Actions taken at each time step of size (N x T x m)
            positive_samples: Positive samples (ground truth future states) of size (N x T x d)
            negative_samples: Negative samples (unrelated states) of size (N x T x K x d)
        """

        
        N, T, m = actions.shape  # Batch size, time steps, action dimension
        d = states.shape[2]  # State dimension
        
        # Step 1: Compute the anchor (future) state in the embedding space
        z_t, h_n = self(states)
        
        # Increase this as traning goes on
        #LEN_PRED = 2

        z_future = torch.zeros((N, T, LEN_PRED, self.embedding_dim))  # Initialize future state embeddings (N x hidden_dim)

        # TODO we should have masking here too!
        # We should do transpose here 
        for n in range(T):
            for m in range(0, LEN_PRED):
                action = actions[:, n, :]#.unsqueeze(1)
                if m == 0:
                    z_t_ = z_t[:, n, :]#.unsqueeze(-1)
                    temp = torch.matmul(z_t_, A.transpose(0, 1)).unsqueeze(-1) #A.unsqueeze(0) @ z_t_ 
                    temp1 = B.unsqueeze(0) * action.unsqueeze(1)
                    temp = temp + temp1
                    z_future[:, n, m] = temp.squeeze(-1)
                else:
                    torch.matmul(z_future[:, n, m-1, :], A.transpose(0, 1)).unsqueeze(-1)
                    temp = torch.matmul(z_future[:, n, m-1, :], A.transpose(0, 1)).unsqueeze(-1) + B.unsqueeze(0) * action.unsqueeze(1)
                    z_future[:, n, m] = temp.squeeze(-1)

        # Pass positive samples through the RNN and get embeddings
        z_positive, h_n = self(states, anchor=False)  # (1 x N x hidden_dim)
        loss = torch.tensor(0.0)
        counter = 0
        nan_counter = 0
        inf_counter = 0
        for n in range(T):
            for m in range(0, LEN_PRED):
                if n + m + 1 >= T:
                    break

                positive_similarity = self.similarity(z_future[:, n, m, :], z_positive[:, n + m + 1, :])
                negative_similarity = 0
                
                negative_indices = np.random.choice([i for i in range(T) if i != n + m + 1], nagative_count, replace=False)
                
                for k in negative_indices:
                    negative_similarity += self.similarity(z_future[:, n, m], z_positive[:, k])
                
                loss1 = -torch.log(positive_similarity / (negative_similarity + positive_similarity))
                loss1 = loss1.mean()
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
        
        print('Nan: ', nan_counter, ' Inf: ', inf_counter)
        if loss == 0:
            return loss, False
        return loss, True

    # def update_momentum_encoder(self, momentum=0.999):
    #     # Update the momentum encoder (positive/negative embedding head) using EMA of anchor embedding head
    #     for param_q, param_k in zip(self.anchor_emebedding_head.parameters(), self.pos_neg_embedding_head.parameters()):
    #         param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data



class KoopamnOperator(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty((state_dim, state_dim)))
        self.B = nn.Parameter(torch.empty((state_dim, action_dim)))
        
        self.A.requires_grad = True
        self.B.requires_grad = True
        
        # try to avoid degenerated case, can it be fixed with initialization?
        torch.nn.init.normal_(self.A, mean=0, std=1)
        torch.nn.init.normal_(self.B, mean=0, std=1)

    def forward(self, states, actions):
        result = torch.matmul(states, self.A.transpose(0, 1))+torch.matmul(actions, self.B.transpose(0, 1))
        return result
    

    def loss_function(self, states, actions, next_states):
        return nn.MSELoss()(self(states, actions), next_states)#.mean()
        # result = self(states, actions)
        # result = result - next_states
        # result =  (result ** 2).mean()
        # return result


class CostLearning(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        # parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(state_dim))  # to use: Q = diag(_q_diag_log.exp())
        self._r_diag_log = nn.Parameter(torch.zeros(action_dim)) # gain of control penalty, in theory need to be parameterized...

        torch.nn.init.normal_(self._q_diag_log, mean=0, std=.1)
        torch.nn.init.normal_(self._r_diag_log, mean=0, std=.1)
    
    def forward(self, states, actions):
        B, T, d = states.shape
        Q = torch.diag(self._q_diag_log.exp())
        R = torch.diag(self._r_diag_log.exp())
        states = states.reshape(B*T, d)
        actions = actions.reshape(B*T, -1)
        state_cost = (states @ Q @ states.T).diag()
        action_cost = (actions @ R @ actions.T).diag()
        return state_cost + action_cost

    def loss_function(self, states, actions, rewards):
        B, T = rewards.shape
        rewards = rewards.reshape(B*T, -1)
        return ((self(states, actions) - rewards) ** 2).mean()


class TrajectoryBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)

    def store_trajectory(self, states, actions, rewards, next_states):
        trajectory = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states
        }
        self.buffer.append(trajectory)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.buffer = pickle.load(f)


class FixLenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.step_count = 0
        assert self.env._max_episode_steps
    
    def reset(self, **kwargs):
        result =  super().reset(**kwargs)
        self.step_count = 0
    
    def step(self, action):
        state, reward, truncated, terminated, info =  super().step(action)
        done = terminated or truncated
        
        if self.step_count >= self.env._max_episode_steps:
            terminated = done
        else:
            self.step_count += 1
            done = False
        return state, reward, done, done, info

class KoopmanLQR(nn.Module):
    def __init__(self, k, T, g_dim, u_dim, g_goal=None, u_affine=None):
        """
        k:          rank of approximated koopman operator
        T:          length of horizon
        g_dim:      dimension of latent state
        u_dim:      dimension of control input
        g_goal:     None by default. If not, override the x_goal so it is not necessarily corresponding to a concrete goal state
                    might be useful for non regularization tasks.  
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        """
        super().__init__()
        self._k = k
        self._T = T
        self._g_dim = g_dim
        self._u_dim = u_dim
        self._g_goal = g_goal
        self._u_affine = u_affine
        
        # prepare linear system params
        self._g_affine = nn.Parameter(torch.empty((k, k)))
        
        if self._u_affine is None:
            self._u_affine = nn.Parameter(torch.empty((k, u_dim)))
        else:
            self._u_affine = nn.Parameter(self._u_affine)
        
        # try to avoid degenerated case, can it be fixed with initialization?
        torch.nn.init.normal_(self._g_affine, mean=0, std=1)
        torch.nn.init.normal_(self._u_affine, mean=0, std=1)

        # parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(self._k))  # to use: Q = diag(_q_diag_log.exp())
        self._r_diag_log = nn.Parameter(torch.zeros(self._u_dim)) # gain of control penalty, in theory need to be parameterized...
        self._r_diag_log.requires_grad = False

        # zero tensor constant for k and v in the case of fixed origin
        # these will be automatically moved to gpu so no need to create and check in the forward process
        self.register_buffer('_zero_tensor_constant_k', torch.zeros((1, self._u_dim)))
        self.register_buffer('_zero_tensor_constant_v', torch.zeros((1, self._k)))

        # we may need to create a few cache for K, k, V and v because they are not dependent on x
        # unless we make g_goal depend on it. This allows to avoid repeatively calculate riccati recursion in eval mode
        self._riccati_solution_cache = None
        return

    def forward(self, g0):
        '''
        perform mpc with current parameters given the initial x0
        '''
        breakpoint()
        K, k, V, v = self._retrieve_riccati_solution()
        u = -self._batch_mv(K[0], g0) + k[0]  # apply the first control as mpc
        return u
    
    @staticmethod
    def _batch_mv(bmat, bvec):
        """
        Performs a batched matrix-vector product, with compatible but different batch shapes.

        This function takes as input `bmat`, containing :math:`n \times n` matrices, and
        `bvec`, containing length :math:`n` vectors.

        Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
        to a batch shape. They are not necessarily assumed to have the same batch shape,
        just ones which can be broadcasted.
        """
        return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)
    
    def _retrieve_riccati_solution(self):
        if self.training or self._riccati_solution_cache is None:
            Q = torch.diag(self._q_diag_log.exp()).unsqueeze(0)
            R = torch.diag(self._r_diag_log.exp()).unsqueeze(0)

            # use g_goal
            if self._g_goal is not None:
                goals = torch.repeat_interleave(self._g_goal.unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
            else:
                goals = None

            # solve the lqr problem via a differentiable process.
            K, k, V, v = self._solve_lqr(self._g_affine.unsqueeze(0), self._u_affine.unsqueeze(0), Q, R, goals)
            self._riccati_solution_cache = (
                [tmp.detach().clone() for tmp in K], 
                [tmp.detach().clone() for tmp in k], 
                [tmp.detach().clone() for tmp in V], 
                [tmp.detach().clone() for tmp in v])
                 
        else:
            K, k, V, v = self._riccati_solution_cache
        return K, k, V, v
    

    def _solve_lqr(self, A, B, Q, R, goals):
        # a differentiable process of solving LQR, 
        # time-invariant A, B, Q, R (with leading batch dimensions), but goals can be a batch of trajectories (batch_size, T+1, k)
        #       min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        # s.t.  x_{t+1} = A x_t + B u_t
        # return feedback gain and feedforward terms such that u = -K x + k

        T = self._T
        K = [None] * T
        k = [None] * T
        V = [None] * (T+1)
        v = [None] * (T+1)

        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)

        V[-1] = Q  # initialization for backpropagation
        breakpoint()
        if goals is not None:
            v[-1] = self._batch_mv(Q, goals[:, -1, :])
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans)
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A)
                k[i] = self._batch_mv(V_uu_inv_B_trans, v[i+1])

                # riccati difference equation, A-BK
                A_BK = A - torch.matmul(B, K[i])
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
                v[i] = self._batch_mv(A_BK.transpose(-2, -1), v[i+1]) + self._batch_mv(Q, goals[:, i, :])
        else:
            # None goals means a fixed regulation point at origin. ignore k and v for efficiency
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of A X = B to avoid direct inverse, note it also returns LU
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans)
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A)
                
                A_BK = A - torch.matmul(B, K[i]) #riccati difference equation: A-BK
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
            k[:] = self._zero_tensor_constant_k
            v[:] = self._zero_tensor_constant_v       

        # we might need to cat or 
        #  to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k, V, v

    def _predict_koopman(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        return torch.matmul(G, self._g_affine.transpose(0, 1))+torch.matmul(U, self._u_affine.transpose(0, 1))

def eval_lqr(env, koopman_mapping, koopman_operator, cost_learning, num_episodes=10):


def main():
    EPOCH = 1000
    EPISODE_COUNT_TRANING = 1000
    KOOPMAN_MAPPING_FRE = 32
    KOOPMAN_MAPPING_EPOCH = 2
    KOOPMAN_MAPPING_BATCH_SIZE = 128

    RNN_HIDDEN_DIM = 256
    OBS_EMBEDDING_DIM = 64
    KOOPMAN_DIM = 64
    BATCH_SIZE = 64
    LEN_PRED = 2

    EVAL_INTERVAL = 10 


    env = dmc2gym.make(
            domain_name="cartpole",
            task_name='swingup',
            seed=1,
            visualize_reward=False,
            from_pixels='pixel',
            height=64,
            width=64,
            frame_skip=1)
    env = NormalizeObservation(env)
    env = FixLenWrapper(env)

    buffer = TrajectoryBuffer(buffer_size=EPISODE_COUNT_TRANING)

    for i in range(EPISODE_COUNT_TRANING):
        states, actions, rewards, next_states = [], [], [], []
        state = env.reset()
        done = False
        while done is False:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = next_state
    
    koopman_mapping = KoopmanMapping(64, 256, 64)
    koopman_operator = KoopamnOperator(64, 1)
    cost_learning = CostLearning(64, 1)

    koopman_mapping_optimizer = torch.optim.Adam(koopman_mapping.parameters(), lr=1e-3)
    koopman_operator_optimizer = torch.optim.Adam(koopman_operator.parameters(), lr=1e-3)  
    cost_learning_optimizer = torch.optim.Adam(cost_learning.parameters(), lr=1e-3)

    for i in range(EPOCH):
        if i % KOOPMAN_MAPPING_FRE == 0:
            for j in range(KOOPMAN_MAPPING_EPOCH):
                states, actions, rewards, next_states = buffer.sample_batch(KOOPMAN_MAPPING_BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.float32).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                koopman_mapping_loss, flag = koopman_mapping.loss_function1(koopman_operator.A.clone().detach(),
                                                            koopman_operator.B.clone().detach(),
                                                            states, actions, LEN_PRED)
                if flag:
                    koopman_mapping_optimizer.zero_grad()
                    koopman_mapping_loss.backward()
                    koopman_mapping_optimizer.step()

        states, actions, rewards, next_states = buffer.sample_batch(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        with torch.no_grad():
            states = koopman_mapping(states)
            next_states = koopman_mapping(next_states)
            
        koopman_operator_loss = koopman_operator.loss_function(states, actions, next_states)
        cost_learning_loss = cost_learning.loss_function(states, actions, rewards)

        koopman_operator_optimizer.zero_grad()
        cost_learning_optimizer.zero_grad()

        koopman_operator_loss.backward()
        cost_learning_loss.backward()

        koopman_operator_optimizer.step()
        cost_learning_optimizer.step()

    if i 
    writer.add_scalar('Loss/koopman_mapping', koopman_mapping_loss, i)
    writer.add_scalar('Loss/koopman_operator', koopman_operator_loss, i)
    writer.add_scalar('Loss/cost_learning', cost_learning_loss, i)



    
