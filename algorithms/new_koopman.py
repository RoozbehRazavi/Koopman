from networks.encoder import *
from algorithms.curl_sac import CurlSacKoopmanAgent
from utils.utils import ReplayBuffer

# TODO write the program from scratch 
# First train a controller on pixel cartpole, an store trajectory based  
# Then train encoder with new contrastive loss
# Train A, B with reconstruction loss
# Train Q and R to estimate reward in the embedding space (z, u)
# Use LQR to solve the problem  

class NewReplayBuffer(ReplayBuffer):
    # TODO we should make the replay buffer recurrent
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, from_pixel=True):
        super(NewReplayBuffer, self).__init__(obs_shape, action_shape, capacity, batch_size, device,image_size,transform, from_pixel)

    def add(self, obs, action, reward, next_obs, done, cpc_kwargs):
        super().add(obs, action, reward, next_obs, done)
        for k, v in cpc_kwargs.items():
            setattr(self, k, np.copy(v))
    
    def sample_cpc(self):
        obses, actions, rewards, next_obses, not_dones, cpc_kwargs = super().sample()

class NewCURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(NewCURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    

class CPCKoopmanSAC(CurlSacKoopmanAgent):

    def __init__(self, *args, **kwargs):
        super(CPCKoopmanSAC, self).__init__(*args, **kwargs)
        
        assert self.koopman_update_freq > 0
        assert self.koopman_fit_coeff > 0

        self.critic = None 
        self.critic_target = None 
        self.log_alpha = None
        self.target_entropy = None

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.log_alpha_optimizer = None

        assert self.koopman_optimizers is not None

        self.CURL = NewCURL()

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = None

        self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=kwargs['encoder_lr'])
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()

    
    def update_actor_and_alpha(self,  obs, next_obs, action, L, step):
        return 
    
    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        # TODO update CPC loss  
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)
        pass

    def update_kpm(self, obs, next_obs, action, L, step, use_ls=False):
        # TODO whether we should have reconstruction loss or not? 
        return super().update_kpm(obs, next_obs, action, L, step, use_ls)

    def update(self, replay_buffer, L, step):
        # TODO replay buffer 
        if self.encoder_type in ['pixel','fc']:
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        #self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            # TODO does actor needs a separate encoder?
            self.update_actor_and_alpha(obs, next_obs, action, L, step)  # XL: fit the form of new update_actor_and_alpha()

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.cpc_update_freq == 0 and self.encoder_type in ['pixel','fc']:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)
        
        
        if step % self.koopman_update_freq == 0 and self.encoder_type in ['pixel','fc'] and self.koopman_fit_coeff > 0:
            self.update_kpm(obs, next_obs, action, L, step)   