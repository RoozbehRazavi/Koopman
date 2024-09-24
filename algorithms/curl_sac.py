import torch
import torch.nn as nn
from networks.actor_critic import *
from algorithms.curl import *
import numpy as np
from utils.utils import *
from networks.encoder import *
from networks.koopman_version import *
from networks.decoder import *

class CurlSacAgent(object):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        config,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128
    ):
        self.config=config
        self.device = device
        self.hidden_dim = hidden_dim
        self.discount = discount
        self.init_temperature=init_temperature
        self.alpha_lr=alpha_lr
        self.alpha_beta=alpha_beta
        self.actor_log_std_min=actor_log_std_min
        self.actor_log_std_max=actor_log_std_max
        self.actor_update_freq = actor_update_freq
        self.critic_lr=critic_lr
        self.critic_beta=critic_beta
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.encoder_type = encoder_type
        self.encoder_featured_dim = encoder_feature_dim
        self.encoder_lr=encoder_lr
        self.encoder_tau = encoder_tau
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.detach_encoder = detach_encoder
        self.curl_latent_dim = curl_latent_dim
        
        
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type in ['pixel','fc']:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type in ['pixel','fc']:
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)  # TODO: debug this actor, the policy_action has 1 more dimension
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        
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

    def update(self, replay_buffer, L, step):
        if self.encoder_type in ['pixel','fc']:
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

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

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 

# TODO THIS IS THE PLACE     
class CurlSacKoopmanAgent(CurlSacAgent):
    def __init__(self, obs_shape, 
                 action_shape, 
                 device,
                 config,
                 hidden_dim=256,
                 discount=0.99,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 encoder_type='pixel',
                 encoder_feature_dim=50,
                 encoder_lr=1e-3,
                 encoder_tau=0.005,
                 num_layers=4,
                 num_filters=32,
                 cpc_update_freq=1,
                 log_interval=100,
                 detach_encoder=False,
                 curl_latent_dim=128):
        super(CurlSacKoopmanAgent, self).__init__(obs_shape, action_shape, device, config,
                                                  hidden_dim=hidden_dim,
                                                  discount=discount,
                                                  init_temperature=init_temperature,
                                                  alpha_lr=alpha_lr,
                                                  alpha_beta=alpha_beta,
                                                  actor_lr=actor_lr,
                                                  actor_beta=actor_beta,
                                                  actor_log_std_min=actor_log_std_min,
                                                  actor_log_std_max=actor_log_std_max,
                                                  actor_update_freq=actor_update_freq,
                                                  critic_lr=critic_lr,
                                                  critic_beta=critic_beta,
                                                  critic_tau=critic_tau,
                                                  critic_target_update_freq=critic_target_update_freq,
                                                  encoder_type=encoder_type,
                                                  encoder_feature_dim=encoder_feature_dim,
                                                  encoder_lr=encoder_lr,
                                                  encoder_tau=encoder_tau,
                                                  num_layers=num_layers,
                                                  num_filters=num_filters,
                                                  cpc_update_freq=cpc_update_freq,
                                                  log_interval=log_interval,
                                                  detach_encoder=detach_encoder,
                                                  curl_latent_dim=curl_latent_dim)

        print("encoder type:{}".format(encoder_type))

        self.koopman_update_freq = config['koopman']['koopman_update_freq']
        self.koopman_fit_coeff   = config['koopman']['koopman_fit_coeff']

        self.actor = KoopmanActor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, config
        ).to(device)

        self.critic = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # XL: additional optimizers
        self.koopman_optimizers = torch.optim.Adam(
            self.actor.trunk.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        if self.encoder_type in ['pixel','fc']:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()
    
    # TODO: check loss terms in koopmanlqr_sac_garage.py
    def update_actor_and_alpha(self, obs, next_obs, action, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # XL: [not useful] maybe add more loss terms for embedlqr
        # koopman_fit_loss = self.koopman_fit_loss(obs, next_obs, action, self.config['koopman']['least_square_fit_coeff'])
        # actor_loss += self.config['koopman']['koopman_fit_coeff'] * koopman_fit_loss
        # print("actor loss: {} || koopman fit loss: {}".format(actor_loss, koopman_fit_loss))

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step() 


    def update_kpm(self, obs, next_obs, action, L, step, use_ls=False):
        # XL: we only use fit_loss for now, 
        # we do not use recon_loss as (1) curl is kind of recon, (2) we don't have a decoder spec.
        # we do not use reg_loss as it is not very important.
        g = self.actor.encoder(obs)
        g_next = self.actor.encoder(next_obs)
        g_pred = self.actor.trunk._predict_koopman(g, action)
        loss_fn = nn.MSELoss()
        fit_loss = loss_fn(g_pred, g_next)   
        
        if step % self.log_interval == 0: 
            L.log('train/kpm_fitting_loss', fit_loss, step)
        
        # XL: [not useful] update critic's encoder (should we update actor's encoder? maybe not because actor's encoder is tied to critic's)
        # self.encoder_optimizers.zero_grad()
        # fit_loss.backward()
        # self.encoder_optimizer.zero_grad()

        # update self.actor.trunk's parameters (A, B, Q, R)
        self.koopman_optimizers.zero_grad()
        fit_loss.backward()
        self.koopman_optimizers.step()


    def update(self, replay_buffer, L, step):
        if self.encoder_type in ['pixel','fc']:
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
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

