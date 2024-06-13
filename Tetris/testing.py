import gym
import gym_simpletetris
import pygame
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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
               high_scoring=True,              # See reward table
               penalise_holes=False,            # See reward table
               penalise_holes_increase=True,   # See reward table
               lock_delay=0,                    # Lock delay as number of steps
               step_reset=False                 # Reset lock delay on step downwards
               )


import numpy as np

state_array = np.zeros((10, 20))

def state_to_tuple(state_array):
    return tuple(map(tuple, state_array))

# Initialize Q-table as a dictionary
# q_table = {}
# with open('q_table_10k.pkl', 'rb') as f:
#     q_table = pickle.load(f)

pygame.init()
window_size = 512
combined_window_size = (window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()

# Define the state and actions
state = state_to_tuple(state_array)
actions = 7

q_table = np.zeros((10, 20) + (7,))
q_table = {}
state = tuple(map(tuple, state_array))  # Converting 10x20 array to a hashable state representation
q_table[state] = [0] * 7  # Initializing Q-values for the 7 actions

alpha = 0.2
epsilon = 1.0
gamma = 0.99

num_episodes = 5010

for i in tqdm(range(num_episodes)):
    if i == 5001:
        pygame.init()
        window_size = 512
        combined_window_size = (window_size, window_size)
        window = pygame.display.set_mode(combined_window_size)
        clock = pygame.time.Clock()
    state = env.reset()
    done = False
    curr_rewards = 0
    while not done:
        state = state_to_tuple(state)
        if state not in q_table:
                q_table[state] = [0] * actions
                action = np.random.randint(0, env.action_space.n)
        else:
            if np.random.rand() < epsilon: # * (1 - episode/num_episodes): # random pick with decay
                action = np.random.randint(0, env.action_space.n)
            else: # greedy pick
                action = np.argmax(q_table[state])
        epsilon *= 0.995
        next_state, reward, done, _ = env.step(action)
        curr_rewards += reward

        temp = next_state
        next_state = state_to_tuple
        if next_state not in q_table:
            q_table[next_state] = [0] * actions
        next_max = np.max(q_table[next_state])
        update = alpha * (reward + gamma * next_max - q_table[state][action])
        q_table[state][action] = q_table[state][action] + update

        state = temp

        if i > 5000:
            # Rendering
            window.fill((0, 0, 0))
            env.render(surface=window, offset=(0, 0))

            # Update the display
            pygame.display.update()
            clock.tick(30)

with open('q_table_5k.pkl', 'wb') as f:
    pickle.dump(q_table, f)

# plt.plot(range(len(total_rewards)), total_rewards)
env.close()
# pygame.quit()