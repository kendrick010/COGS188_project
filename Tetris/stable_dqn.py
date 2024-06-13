import gym_simpletetris
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import pygame

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        done = self.locals['dones']
        reward = self.locals['rewards']

        if done:
            episode_reward = np.sum(reward)
            self.episode_rewards.append(episode_reward)
            if self.verbose > 0:
                print(f"Episode Reward: {episode_reward}")

        return True

    def _on_training_end(self) -> None:
        np.save("episode_rewards.npy", self.episode_rewards)  # Save the rewards to a file

env = gym.make('SimpleTetris-v0',
               height=20,                       # Height of Tetris grid
               width=10,                        # Width of Tetris grid
               obs_type='ram',                  # ram | grayscale | rgb
               extend_dims=False,               # Extend ram or grayscale dimensions
               render_mode='rgb_array',         # Unused parameter
               reward_step=True,               # See reward table
               penalise_height=False,           # See reward table
               penalise_height_increase=True,  # See reward table
               advanced_clears=True,           # See reward table
               high_scoring=False,              # See reward table
               penalise_holes=False,            # See reward table
               penalise_holes_increase=True,   # See reward table
               lock_delay=0,                    # Lock delay as number of steps
               step_reset=False                 # Reset lock delay on step downwards
               )

model = DQN.load("tetris_100000.zip", env=env)

pygame.init()
window_size = 512
combined_window_size = (window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    # Rendering
    window.fill((0, 0, 0))
    vec_env.render(surface=window, offset=(0, 0))

    # Update the display
    pygame.display.update()
    clock.tick(30)