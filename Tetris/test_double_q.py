import gym
import gym_simpletetris
import pygame
import numpy as np

# Load the trained Q-tables
Q1_agent = np.load("double_q/Q1_agent.npy")
Q2_agent = np.load("double_q/Q2_agent.npy")
Q1_adversary = np.load("double_q/Q1_adversary.npy")
Q2_adversary = np.load("double_q/Q2_adversary.npy")

# Environment setup
env_agent = gym.make('SimpleTetris-v0',
                     obs_type='grayscale', reward_step=True, penalise_height_increase=True,
                     advanced_clears=True, penalise_holes_increase=True)

env_adversary = gym.make('SimpleTetris-v0',
                         obs_type='grayscale', reward_step=True, penalise_height_increase=True,
                         advanced_clears=True, penalise_holes_increase=True)

# Window setup
pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()

num_episodes = 10

def flatten_state(state):
    return state.flatten().astype(int)

def state_to_index(state, state_size):
    return hash(state.tobytes()) % state_size

def select_action(state, Q1, Q2, state_size):
    state_index = state_to_index(state, state_size)
    q_values = Q1[state_index, :] + Q2[state_index, :]
    return np.argmax(q_values)

if __name__ == "__main__":
    state_shape = env_agent.observation_space.shape
    state_size = np.prod(state_shape)
    action_size = env_agent.action_space.n

    for ep in range(num_episodes):
        state_agent = flatten_state(env_agent.reset())
        state_adversary = flatten_state(env_adversary.reset())
        done_agent = False
        done_adversary = False

        while not done_agent or not done_adversary:
            if not done_agent:
                action_agent = select_action(state_agent, Q1_agent, Q2_agent, state_size)
                next_state_agent, reward_agent, done_agent, _ = env_agent.step(action_agent)
                next_state_agent = flatten_state(next_state_agent)
                state_agent = next_state_agent

            if not done_adversary:
                action_adversary = select_action(state_adversary, Q1_adversary, Q2_adversary, state_size)
                next_state_adversary, reward_adversary, done_adversary, _ = env_adversary.step(action_adversary)
                next_state_adversary = flatten_state(next_state_adversary)
                state_adversary = next_state_adversary

            # Rendering
            window.fill((0, 0, 0))
            env_agent.render(surface=window, offset=(0, 0))
            env_adversary.render(surface=window, offset=(window_size, 0))
            pygame.display.update()
            clock.tick(30)

        print(f"Episode {ep + 1} finished.")

    env_agent.close()
    env_adversary.close()
    pygame.quit()