from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch 
import torch.nn as nn
from utils.utils import *
import time

class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, from_pixel=True):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        # XL: use contrastive loss for vectorized states, not pixels
        self.from_pixel = from_pixel

    def add(self, obs, action, reward, next_obs, done):
       
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # XL: only used by SAC_AE module
    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = center_crop(obses, self.image_size) if self.from_pixel else obses
        next_obses = center_crop(next_obses, self.image_size) if self.from_pixel else next_obses

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()
    
        # XL: handle pixel and vector states
        obses = random_crop(obses, self.image_size) if self.from_pixel else random_noise(obses)  
        next_obses = random_crop(next_obses, self.image_size) if self.from_pixel else random_noise(next_obses)
        pos = random_crop(pos, self.image_size) if self.from_pixel else random_noise(pos)
    
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 
    
class ReplayBufferT3(ReplayBuffer):
    def __init__(self,obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, from_pixel=True):
        super(ReplayBufferT3,self).__init__(obs_shape, action_shape, capacity, batch_size, device,image_size=image_size,transform=transform, from_pixel=from_pixel)

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.nexxt_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.nexxxt_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        
class ReplayBufferT5(ReplayBuffer):
    def __init__(self):  
        pass

