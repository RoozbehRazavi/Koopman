import torch
import torch.nn as nn
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import dmc2gym
from gym import spaces
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
import time
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

TAU = 0.02

# TODO fix this part!
def setup_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def tune_action(ppo, action_space, action):
    # Rescale and perform action
    clipped_actions = action

    if isinstance(action_space, spaces.Box):
        if ppo.policy.squash_output:
            # Unscale the actions to match env bounds
            # if they were previously squashed (scaled in [-1, 1])
            clipped_actions = ppo.policy.unscale_action(clipped_actions)
        else:
            # Otherwise, clip the actions to avoid out of bound error
            # as we are sampling from an unbounded Gaussian distribution
            clipped_actions = np.clip(action, action_space.low, action_space.high)
    return clipped_actions

class KoopmanMapping(nn.Module):
    def __init__(self, obs_dim, hidden_dim, embedding_dim, image_size=64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, obs_dim)
        )
        # Define an RNN for processing sequences (trajectories)
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)

        # Define decoder 
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, image_size*image_size*3))
        
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
        # TODO should all rnn and encoder train with pos. and neg. samples?
        #hidden_state = hidden_state.squeeze(0)  # Remove extra dimension, result is (N, hidden_dim)
        B = trajectories.shape[0]  # Batch size
        T = trajectories.shape[1]  # Time steps
        trajectories = trajectories.reshape((-1, trajectories.shape[-3], trajectories.shape[-2], trajectories.shape[-1]))
        trajectories = self.image_encoder(trajectories).reshape((B, T, -1))
        if h_0 is None:
            h_0 = torch.zeros(1, B, self.hidden_dim).to(trajectories.device)  # Initial hidden state
            hidden_state, h_n = self.rnn(trajectories, h_0)  # hidden_state: (1, N, hidden_dim)
        else:
            hidden_state, h_n = self.rnn(trajectories, h_0)
        
        # Pass hidden state through anchor embedding head
        if anchor:
            z_anchor = self.anchor_emebedding_head(hidden_state)  # (N x embedding_dim)
        else:
            z_anchor = self.pos_neg_embedding_head(hidden_state)
        return z_anchor, h_n, hidden_state

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
            print('Nan or Inf')
        return result

    def loss_function1(self, A: nn.Parameter, B: nn.Parameter, states, actions, LEN_PRED, nagative_count=20):
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
        z_t, h_n, hidden_state = self(states)
        
        if False:
            decoded_image = self.decoder(hidden_state)

            decoder_loss = nn.MSELoss()(decoded_image.reshape(N*T, -1), states.reshape((N*T, -1))).mean()
        else:
            decoder_loss = torch.tensor(0.0, requires_grad=True)
        # Increase this as traning goes on
        z_future = torch.zeros((N, T, LEN_PRED, self.embedding_dim))  # Initialize future state embeddings (N x hidden_dim)

        # TODO we should have masking here too!
        # We should do transpose here 
        for n in range(T):
            for m in range(0, LEN_PRED):
                action = actions[:, n, :]#.unsqueeze(1)
                if m == 0:
                    z_future[:, n, m] = z_t[:, n, :]
                    z_t_ = z_t[:, n, :]#.unsqueeze(-1)
                    state_effect = torch.matmul(z_t_, A.transpose(0, 1)).unsqueeze(-1) #A.unsqueeze(0) @ z_t_ 
                    action_effect = B.unsqueeze(0) * action.unsqueeze(1)
                    effect = state_effect + action_effect
                    z_future[:, n, m] += TAU * effect.squeeze(-1)
                else:
                    z_future[:, n, m] = z_future[:, n, m-1]
                    torch.matmul(z_future[:, n, m-1, :], A.transpose(0, 1)).unsqueeze(-1)
                    effect = torch.matmul(z_future[:, n, m-1, :], A.transpose(0, 1)).unsqueeze(-1) + B.unsqueeze(0) * action.unsqueeze(1)
                    z_future[:, n, m] += TAU * effect.squeeze(-1)

        # Pass positive samples through the RNN and get embeddings
        z_positive, h_n, _ = self(states, anchor=False)  # (1 x N x hidden_dim)
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
        return loss/counter, decoder_loss, True

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
        # TODO changed from 1 to 0.1
        torch.nn.init.normal_(self.A, mean=0, std=.1)
        torch.nn.init.normal_(self.B, mean=0, std=.1)

    def forward(self, states, actions):
        result = states + TAU * torch.matmul(states, self.A.transpose(0, 1))+torch.matmul(actions, self.B.transpose(0, 1))
        return result
    
    def regularize_largest_eigenvalue(self):
        # Compute the eigenvalues of matrix A
        eigenvalues = torch.linalg.eigvals(self.A)

        # Take the absolute values (to handle complex eigenvalues) and find the largest
        largest_eigenvalue = torch.max(eigenvalues.abs())

        # Return the largest eigenvalue as the regularization term
        return largest_eigenvalue

    # TODO if reg should have lower weight?
    def loss_function(self, states, actions, next_states):
        rec_loss = nn.MSELoss()(self(states, actions), next_states).mean()
        reg_loss = self.regularize_largest_eigenvalue()
        return rec_loss, reg_loss



class CostLearning(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        # parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(state_dim))  # to use: Q = diag(_q_diag_log.exp())
        self._r_diag_log = nn.Parameter(torch.zeros(action_dim)) # gain of control penalty, in theory need to be parameterized...

        # torch.nn.init.normal_(self._q_diag_log, mean=0, std=1)
        # torch.nn.init.normal_(self._r_diag_log, mean=0, std=1)
    
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
        return ((self(states, actions) + rewards) ** 2).mean()


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

class QuadraticRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(QuadraticRewardWrapper, self).__init__(env)
    
    def step(self, action):
        # Take a step in the environment
        state, reward, done, info = self.env.step(action)
        
        # Unpack the state vector for CartPole
        # state = [cart position, cart velocity, pole angle, pole angular velocity]
        tmp = self.env.env.env.env.env.current_state
        cart_pos, cart_vel, pole_angle, pole_angular_vel = tmp[0], tmp[1], tmp[2], tmp[3]
        
        # Define the quadratic reward with respect to state and action
        # You can customize the coefficients for each term here
        quadratic_reward = -(
            1.0 * (cart_pos**2) +  # quadratic penalty on cart position
            #0.5 * (cart_vel**2) +  # quadratic penalty on cart velocity
            2.0 * (pole_angle**2) #+  # quadratic penalty on pole angle
            #0.5 * (pole_angular_vel**2) +  # quadratic penalty on pole angular velocity
            #0.1 * (action**2)  # quadratic penalty on action
        )
        
        # Return the new state, modified quadratic reward, done flag, and info
        return state, quadratic_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

class FixLenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.step_count = 0
        self.env._max_episode_steps = max_steps
    
    def reset(self, **kwargs):
        result =  super().reset(**kwargs)
        self.step_count = 0
        return result
    
    def step(self, action):
        state, reward, done, info =  super().step(action)
        
        if self.step_count == self.env._max_episode_steps - 1:
            done = True
        else:
            self.step_count += 1
            done = False
        return state, -1/reward , done, info

class KoopmanLQR(nn.Module):
    def __init__(self, A, B, q_diag_log, r_diag_log, koopman_dim, horizen, action_dim, g_goal=None):
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
        self._k = koopman_dim
        self._T = horizen
        self._u_dim = action_dim
        self._g_goal = g_goal
        
        # prepare linear system params
        self._g_affine = A

        self._u_affine = B

        # parameters of quadratic functions
        self._q_diag_log = q_diag_log
        self._r_diag_log = r_diag_log 

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
        K, k, V, v = self._retrieve_riccati_solution()
        u = -self._batch_mv(K[0], g0) + k[0]  # apply the first control as mpc
        return torch.clip(u, -1, 1)
    
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

def log_weights(koopman_mapping:KoopmanMapping, koopman_operator:KoopamnOperator, cost_learning: CostLearning, epoch: int, writer: SummaryWriter):
    
    for name, param in koopman_mapping.named_parameters():
        writer.add_scalar(f'Weights/{name}', param.norm(), epoch)
        if param.grad is not None:
            writer.add_scalar(f'Gradients/{name}', param.grad.norm(), epoch)
    
    for name, param in koopman_operator.named_parameters():
        writer.add_scalar(f'Weights/{name}', param.norm(), epoch)
        if param.grad is not None:
            writer.add_scalar(f'Gradients/{name}', param.grad.norm(), epoch)

    for name, param in cost_learning.named_parameters():
        writer.add_scalar(f'Weights/{name}', param.norm(), epoch)
        if param.grad is not None:
            writer.add_scalar(f'Gradients/{name}', param.grad.norm(), epoch)

def eval_lqr(epoch, env, koopman_mapping, koopman_operator, cost_learning, koopman_dim, writer, device, horizen=10, num_episodes=10, gamma=0.99):

    controller_transpose = KoopmanLQR(A=koopman_operator.A.transpose(0, 1),
                            B=koopman_operator.B,
                            q_diag_log=cost_learning._q_diag_log,
                            r_diag_log=cost_learning._r_diag_log,
                            koopman_dim=koopman_dim, horizen=horizen, action_dim=1)
    values2 = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        hidden_state = None
        ret = 0
        while done is False:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                state, hidden_state, _ = koopman_mapping(state, h_0=hidden_state)
                action = controller_transpose(state[0]).detach().cpu().numpy()[0]
            state, reward, done, _ = env.step(action)
            ret = ret * gamma + reward
        values2.append(ret)
    writer.add_scalar('Evaluation/mean_return_transpose', np.mean(values2), epoch)
    writer.add_scalar('Evaluation/std_return_transpose', np.std(values2), epoch)

    return np.mean(values2) 


def main():
    EPOCH = 1000
    EPISODE_COUNT_TRANING = 1000
    KOOPMAN_MAPPING_FRE = 32
    KOOPMAN_MAPPING_EPOCH = 2
    KOOPMAN_MAPPING_BATCH_SIZE = 64
    KOOPMAN_OPT_REG = 0.001

    RNN_HIDDEN_DIM = 256
    OBS_EMBEDDING_DIM = 64
    KOOPMAN_DIM = 64
    BATCH_SIZE = 32
    LEN_PRED = 1
    MAX_STEPS = 200
    IMAGE_SIZE = 64

    # TODO 
    EVAL_INTERVAL = 5
    SAVE_INTERVAL = 100
    LOAD=False
    EXPERT_POLICY_FRAME  = 1000000
    PORTION_EXPERT = 0.0

    ENV_NAME = 'cartpole'
    TASK_NAME = 'swingup'

    DATA_TYPE = 'MIX'
    if PORTION_EXPERT == 0:
        DATA_TYPE = 'RANDOM'
    elif PORTION_EXPERT == 1:
        DATA_TYPE = 'EXPERT'
    else:
        DATA_TYPE = 'MIX'

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{DATA_TYPE}_len{LEN_PRED}_{start_time}"

    save_dir = f'saved/{TASK_NAME}_{ENV_NAME}'
    logdir = F'log/task_{TASK_NAME}_env_{ENV_NAME}'

    setup_directory(save_dir)
    setup_directory(logdir)
    
    writer = SummaryWriter(log_dir=logdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = dmc2gym.make(
            domain_name=ENV_NAME,
            task_name=TASK_NAME,
            seed=1,
            visualize_reward=False,
            from_pixels='pixel',
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            frame_skip=1)
    
    env = NormalizeObservation(env)
    
    env = FixLenWrapper(env, max_steps=MAX_STEPS)

    #env = QuadraticRewardWrapper(env)

    expert_policy = PPO('CnnPolicy', env=env, verbose=1, device=device)

    if os.path.exists(f'./{save_dir}/ppo_expert_policy_{EXPERT_POLICY_FRAME}.zip'):
        print('Load Expert Policy')
        expert_policy.load(f'./{save_dir}/ppo_expert_policy_{EXPERT_POLICY_FRAME}')
    else:
        print('Train Expert Policy')
        expert_policy.learn(total_timesteps=EXPERT_POLICY_FRAME)
        expert_policy.save(f'./{save_dir}/ppo_expert_policy_{EXPERT_POLICY_FRAME}')
    
    print('Expert Policy is Ready!')

    buffer = TrajectoryBuffer(buffer_size=EPISODE_COUNT_TRANING)

    if os.path.exists(f'./{save_dir}/buffer_{EPISODE_COUNT_TRANING}_{DATA_TYPE}_{PORTION_EXPERT}.pkl'):
        print('Loading buffer')
        buffer.load(f'./{save_dir}/buffer_{EPISODE_COUNT_TRANING}_{DATA_TYPE}_{PORTION_EXPERT}.pkl')
    else:
        print('Generating buffer')
        for i in range(EPISODE_COUNT_TRANING):
            states, actions, rewards, next_states = [], [], [], []
            state = env.reset()
            done = False
            policy = 'random'
            if np.random.uniform(0, 1) < PORTION_EXPERT:
                policy = 'expert'
            while done is False:
                if policy == 'expert':
                    state_ = torch.tensor(state).unsqueeze(0).float().to(device)
                    action, _, _ = expert_policy.policy(torch.tensor(state_))
                    action = action.detach().cpu().numpy()[0]
                    action = tune_action(expert_policy, env.action_space, action)
                else:
                    action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = next_state
            buffer.store_trajectory(states, actions, rewards, next_states)
    
    if not os.path.exists(f'./{save_dir}/buffer_{EPISODE_COUNT_TRANING}_{DATA_TYPE}_{PORTION_EXPERT}.pkl'):
        print('Saving buffer')
        buffer.save(f'./{save_dir}/buffer_{EPISODE_COUNT_TRANING}_{DATA_TYPE}_{PORTION_EXPERT}.pkl')
    
    print('Buffer is Ready!')
    
    koopman_mapping = KoopmanMapping(obs_dim=OBS_EMBEDDING_DIM, hidden_dim=RNN_HIDDEN_DIM, embedding_dim=KOOPMAN_DIM).to(device)
    koopman_operator = KoopamnOperator(state_dim=KOOPMAN_DIM, action_dim=1).to(device)
    cost_learning = CostLearning(state_dim=KOOPMAN_DIM, action_dim=1).to(device)

    if LOAD:
        koopman_mapping.load_state_dict(torch.load(f'{save_dir}_{run_id}/koopman_mapping_100'))
        koopman_operator.load_state_dict(torch.load(f'{save_dir}_{run_id}/koopman_operator_100'))
        cost_learning.load_state_dict(torch.load(f'{save_dir}_{run_id}/cost_learning_100'))

    koopman_mapping_optimizer = torch.optim.Adam(koopman_mapping.parameters(), lr=1e-3)
    koopman_operator_optimizer = torch.optim.Adam(koopman_operator.parameters(), lr=1e-3)  
    cost_learning_optimizer = torch.optim.Adam(cost_learning.parameters(), lr=1e-2)

    for i in range(EPOCH):
        if i % KOOPMAN_MAPPING_FRE == 0:
            for j in range(KOOPMAN_MAPPING_EPOCH):
                batch = buffer.sample_batch(KOOPMAN_MAPPING_BATCH_SIZE)
                states = [torch.tensor(x['states'], dtype=torch.float32) for x in batch]
                states = torch.stack(states, dim=0).to(device)
                actions = [torch.tensor(x['actions'], dtype=torch.float32) for x in batch]
                actions = torch.stack(actions, dim=0).to(device)
                rewards = [torch.tensor(x['rewards'], dtype=torch.float32) for x in batch]
                rewards = torch.stack(rewards, dim=0).to(device)
                next_states = [torch.tensor(x['next_states'], dtype=torch.float32) for x in batch]
                next_states = torch.stack(next_states, dim=0).to(device)
                # states = torch.tensor([x['states'] for x in batch], dtype=torch.float32).to(device)
                # actions = torch.tensor([x['actions'] for x in batch], dtype=torch.float32).to(device)
                # rewards = torch.tensor([x['rewards'] for x in batch], dtype=torch.float32).to(device)
                # next_states = torch.tensor([x['next_states'] for x in batch], dtype=torch.float32).to(device)
                # TODO !
                koopman_mapping_loss, decoder_loss, flag = koopman_mapping.loss_function1(koopman_operator.A.clone().detach(),
                                                            koopman_operator.B.clone().detach(),
                                                            states, actions, LEN_PRED)
                if flag:
                    koopman_mapping_optimizer.zero_grad()
                    koopman_mapping_loss.backward()
                    koopman_mapping_optimizer.step()

        batch = buffer.sample_batch(BATCH_SIZE)
        states = [torch.tensor(x['states'], dtype=torch.float32) for x in batch]
        states = torch.stack(states, dim=0).to(device)
        actions = [torch.tensor(x['actions'], dtype=torch.float32) for x in batch]
        actions = torch.stack(actions, dim=0).to(device)
        rewards = [torch.tensor(x['rewards'], dtype=torch.float32) for x in batch]
        rewards = torch.stack(rewards, dim=0).to(device)
        next_states = [torch.tensor(x['next_states'], dtype=torch.float32) for x in batch]
        next_states = torch.stack(next_states, dim=0).to(device)

        with torch.no_grad():
            states, _, _ = koopman_mapping(states)
            next_states, _, _ = koopman_mapping(next_states)
        
        koopman_operator_loss, reg_loss = koopman_operator.loss_function(states, actions, next_states)
        cost_learning_loss = cost_learning.loss_function(states, actions, rewards)

        koopman_operator_optimizer.zero_grad()
        cost_learning_optimizer.zero_grad()

        (koopman_operator_loss + KOOPMAN_OPT_REG * reg_loss).backward()
        cost_learning_loss.backward()

        koopman_operator_optimizer.step()
        cost_learning_optimizer.step()

        if i % EVAL_INTERVAL == 0:
            values2 = eval_lqr(i, env, koopman_mapping, koopman_operator, cost_learning, KOOPMAN_DIM, writer, device)
            log_weights(koopman_mapping, koopman_operator, cost_learning, i, writer)
            print(f'Epoch: {i}, Mapping Loss: {koopman_mapping_loss.item()}, Operator Loss: {koopman_operator_loss.item()}, Reg Loss: {reg_loss.item()}, Cost Loss: {cost_learning_loss.item()}, Controller_T: {values2}')

        if i % SAVE_INTERVAL == 0:
            torch.save(koopman_mapping.state_dict(), f'{save_dir}_{run_id}/koopman_mapping_{i}.pt')
            torch.save(koopman_operator.state_dict(), f'{save_dir}_{run_id}/koopman_operator_{i}.pt')
            torch.save(cost_learning.state_dict(), f'{save_dir}_{run_id}/cost_learning_{i}.pt')

        writer.add_scalar('Loss/koopman_mapping', koopman_mapping_loss, i)
        writer.add_scalar('Loss/koopman_operator', koopman_operator_loss, i)
        writer.add_scalar('Loss/reg_loss', reg_loss, i)
        writer.add_scalar('Loss/cost_learning', cost_learning_loss, i)


if __name__ == '__main__':
    main()