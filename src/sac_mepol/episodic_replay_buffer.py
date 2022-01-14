import random
import numpy as np


class EpisodicReplayBuffer(object):
    """
    Wrapper around a replay buffer in order to use Episodic replay buffer.
    :param replay_buffer: (ReplayBuffer)
    """

    def __init__(self, capacity, seed):
        super(EpisodicReplayBuffer, self).__init__()
        # Buffer for storing transitions of the current episode
        random.seed(seed)

        self.episode_transitions = []
        self._buffer = []
        self._episode_lengths = []
        self.capacity = capacity
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        add a new transition to the buffer
        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        :param info: (dict) extra values used to compute reward
        """
        # Update current episode buffer
        self.episode_transitions.append([state, action, reward, next_state, done])
    
    def end_episode(self):
        self._store_episode()
        # Reset episode buffer
        self.episode_transitions = []

    def sample(self, batch_size, num_steps=1):
        # sample episodes
        episode_idx = np.random.choice(
            np.arange(len(self._buffer)),
            batch_size,
            p=self._episode_lengths / np.sum(self._episode_lengths)
        )

        frame_idx = np.random.randint(0, self._episode_lengths[0] - num_steps, batch_size)
        
        batch = []
        for i, d in enumerate(episode_idx):
            start = frame_idx[i]
            x = [self._buffer[d][j] for j in range(start, start + num_steps)] 
            x = map(np.stack, zip(*x))
            batch.append(x)
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
  
    def __len__(self):
        return len(self._buffer)

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        if len(self._buffer) < self.capacity:
            self._buffer.append(None)
            self._episode_lengths.append(None)
        self._buffer[self.position] = self.episode_transitions
        self._episode_lengths[self.position] = len(self.episode_transitions)
        self.position = (self.position + 1) % self.capacity
 
    def get_episode_states(self):
        _, _, _, next_states, _ = zip(*self.episode_transitions)
        return np.stack(next_states)

    def set_episodic_reward(self, reward):
        for transition_idx, transition in enumerate(self.episode_transitions):
            self.episode_transitions[transition_idx][2] = reward
