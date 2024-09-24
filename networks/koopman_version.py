import torch 
import torch.nn as nn
from networks.encoder import *
import pickle
from networks.actor_critic import *


class KoopmanActor(nn.Module):
    """Koopman LQR actor."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
        config
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.action_shape = action_shape
        self.config = config
        # XL: set up goal reference for different situations
        goal_meta = self.config['koopman']['koopman_goal_image_path']
        if isinstance(goal_meta, str) and goal_meta.endswith(".pkl"):
            with open(self.config['koopman']['koopman_goal_image_path'], "rb") as f:
                self.goal_obs = torch.from_numpy(pickle.load(f)).unsqueeze(0).to(torch.device(self.config['device']))
        elif isinstance(goal_meta, list):
            self.goal_obs = torch.from_numpy(np.array(self.config['koopman']['koopman_goal_image_path'], dtype=np.float32)).unsqueeze(0).to(torch.device(self.config['device']))
        else:
            self.goal_obs = None

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_init = torch.nn.Parameter(torch.Tensor([1.0]).log()) # XL: initialize a log_std

        # XL: Koopman control module as trunk
        self.trunk = KoopmanLQR(k=encoder_feature_dim, 
                                T=5,
                                g_dim=encoder_feature_dim,
                                u_dim=action_shape[0],
                                g_goal=None,
                                u_affine=None)

        self.outputs = dict()
        self.apply(weight_init)

        

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        # XL: encode the goal images to be used in self.trunk
        if self.goal_obs is None:
            self.trunk._g_goal = torch.zeros((1, obs.shape[1])).squeeze(0).to(torch.device(self.config['device']))
        else:
            goal_obs = self.encoder(self.goal_obs, detach=detach_encoder)
            self.trunk._g_goal = goal_obs.squeeze(0)

        # XL: do not chunk to 2 parts as LQR directly gives mu; use constant log_std
        broadcast_shape = list(obs.shape[:-1]) + [self.action_shape[0]]
        mu, log_std = self.trunk(obs).chunk(1, dim=-1)[0], \
                    self.log_std_init + torch.zeros(*broadcast_shape).to(torch.device(self.config['device']))

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        # L.log_param('train_actor/fc1', self.trunk[0], step)
        # L.log_param('train_actor/fc2', self.trunk[2], step)
        # L.log_param('train_actor/fc3', self.trunk[4], step)

    

class KoopmanCritic(Critic):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters):
        super(KoopmanCritic, self).__init__(obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters)



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
    