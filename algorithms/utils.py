import time
from utils.utils import *
import os 
import pickle
from algorithms.ae_sac import *
from algorithms.curl_sac import *

def evaluate(env, agent, video, num_episodes, L, step, config, save_transitions=False, writer=None):
    all_ep_rewards = []
    all_ep_costs   = []

    all_fit_losses = []
    gamma = .99
    seed = range(num_episodes)
    all_transitions = [{"obs":[], "act":[], "rew":[], "done":[]}]*num_episodes
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            env.seed(seed[i])
            set_seed_everywhere(seed[i])
            obs = env.reset()
            # video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_cost = 0  # XL: record the cost
            episode_fit_loss = 0 # XL: record the fit loss
            episode_lat_trans = []
            episode_timestep = 0
            if save_transitions:
                all_transitions[i]["obs"].append(obs)
                all_transitions[i]["done"].append(done)
            while not done:
                # center crop image
                if config['env']['encoder_type'] == 'pixel':
                    obs = center_crop_image(obs,config['env']['image_size'])
                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)

                # XL: latent state for curr obs
                with eval_mode(agent):
                    with torch.no_grad():
                        th_obs = torch.FloatTensor(obs).to(config['device'])
                        th_obs = th_obs.unsqueeze(0)
                        th_act = torch.FloatTensor(action).to(config['device'])
                        g = agent.actor.encoder(th_obs)
                        g_pred = agent.actor.trunk._predict_koopman(g, th_act)

                obs, reward, done, _ = env.step(action)
                #video.record(env)
                episode_reward = episode_reward * gamma + reward

                cost = -reward  # XL: record the cost
                episode_cost += cost
                episode_timestep += 1


                if save_transitions:
                    all_transitions[i]["obs"].append(obs)
                    all_transitions[i]["act"].append(action)
                    all_transitions[i]["rew"].append(reward)
                    all_transitions[i]["done"].append(done)
                # XL: latent state for next obs
                with eval_mode(agent):
                    with torch.no_grad():
                        # center crop image
                        if config['env']['encoder_type'] == 'pixel':
                            obs = center_crop_image(obs,config['env']['image_size'])
                        th_obs = torch.FloatTensor(obs).to(config['device'])
                        th_obs = th_obs.unsqueeze(0)
                        g_next = agent.actor.encoder(th_obs)
                        loss_fn = nn.MSELoss()
                        fit_loss = loss_fn(g_pred, g_next)

                episode_fit_loss += (fit_loss.detach().cpu().numpy())
                episode_lat_trans.append({'g':g.detach().cpu().numpy(), 
                                          'g_next':g_next.detach().cpu().numpy(), 
                                          'g_pred':g_pred.detach().cpu().numpy()})

            #video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
            
            # XL: record the cost
            L.log('eval/' + prefix + 'episode_cost', episode_cost, step)
            all_ep_costs.append(episode_cost) 

            # XL: record mean fitting loss
            L.log('eval/' + prefix + 'episode_fit_loss', episode_fit_loss, step)
            all_fit_losses.append(episode_fit_loss/episode_timestep)

            # XL: record latent transition info
            # path = '/'.join(video.dir_name.split('/')[0:-1] + ['lat'] + ['{}.pkl'.format(step)])
            # folder_path = os.path.dirname(path)
            # os.makedirs(folder_path, exist_ok=True)
            # with open(path, 'wb') as f:
            #     pickle.dump(episode_lat_trans, f)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        # XL: record the cost
        mean_ep_cost = np.mean(all_ep_costs)
        writer.add_scalar('Evaluation/mean_return_transpose', mean_ep_reward, step)
        best_ep_cost = np.min(all_ep_costs)
        L.log('eval/' + prefix + 'mean_episode_cost', mean_ep_cost, step)
        L.log('eval/' + prefix + 'best_episode_cost', best_ep_cost, step)

        # XL: record the fit loss
        mean_ep_fit_loss = np.mean(all_fit_losses)
        best_ep_fit_loss = np.min(all_fit_losses)
        L.log('eval/' + prefix + 'mean_episode_fit_loss', mean_ep_fit_loss, step)
        L.log('eval/' + prefix + 'best_episode_fit_loss', best_ep_fit_loss, step)

        # if save_transitions:
        #     eval_path = '/'.join(video.dir_name.split('/')[0:-1] + ['eval_transitions'] + ['{}.pkl'.format(step)])
        #     eval_folder_path = os.path.dirname(eval_path)
        #     os.makedirs(eval_folder_path, exist_ok=True)
        #     with open(eval_path, 'wb') as f:
        #         pickle.dump(all_transitions, f)
                
    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    set_seed_everywhere(30)


def make_agent(obs_shape, action_shape, config, device):
    if config['agent']['name'] == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            config=config,
            hidden_dim=config['agent']['hidden_dim'],
            discount=config['agent']['discount'],
            init_temperature=config['agent']['init_temperature'],
            alpha_lr=config['agent']['alpha_lr'],
            alpha_beta=config['agent']['alpha_beta'],
            actor_lr=config['agent']['actor_lr'],
            actor_beta=config['agent']['actor_beta'],
            actor_log_std_min=config['agent']['actor_log_std_min'],
            actor_log_std_max=config['agent']['actor_log_std_max'],
            actor_update_freq=config['agent']['actor_update_freq'],
            critic_lr=config['agent']['critic_lr'],
            critic_beta=config['agent']['critic_beta'],
            critic_tau=config['agent']['critic_tau'],
            critic_target_update_freq=config['agent']['critic_target_update_freq'],
            encoder_type=config['env']['encoder_type'],
            encoder_feature_dim=config['agent']['encoder_feature_dim'],
            encoder_lr=config['agent']['encoder_lr'],
            encoder_tau=config['agent']['encoder_tau'],
            num_layers=config['agent']['num_layers'],
            num_filters=config['agent']['num_filters'],
            log_interval=config['log_interval'],
            detach_encoder=config['agent']['detach_encoder'],
            curl_latent_dim=config['agent']['curl_latent_dim']

        )
    elif config['agent']['name'] == 'curl_sac_koopmanlqr':
        return CurlSacKoopmanAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            config=config,
            hidden_dim=config['agent']['hidden_dim'],
            discount=config['agent']['discount'],
            init_temperature=config['agent']['init_temperature'],
            alpha_lr=config['agent']['alpha_lr'],
            alpha_beta=config['agent']['alpha_beta'],
            actor_lr=config['agent']['actor_lr'],
            actor_beta=config['agent']['actor_beta'],
            actor_log_std_min=config['agent']['actor_log_std_min'],
            actor_log_std_max=config['agent']['actor_log_std_max'],
            actor_update_freq=config['agent']['actor_update_freq'],
            critic_lr=config['agent']['critic_lr'],
            critic_beta=config['agent']['critic_beta'],
            critic_tau=config['agent']['critic_tau'],
            critic_target_update_freq=config['agent']['critic_target_update_freq'],
            encoder_type=config['env']['encoder_type'],
            encoder_feature_dim=config['agent']['encoder_feature_dim'],
            encoder_lr=config['agent']['encoder_lr'],
            encoder_tau=config['agent']['encoder_tau'],
            num_layers=config['agent']['num_layers'],
            num_filters=config['agent']['num_filters'],
            log_interval=config['log_interval'],
            detach_encoder=config['agent']['detach_encoder'],
            curl_latent_dim=config['agent']['curl_latent_dim']
        )
    else:
        assert 'agent is not supported: %s' % config['agent']['name']

