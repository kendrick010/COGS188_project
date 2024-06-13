import gym
import gym_simpletetris
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# Parameters
num_episodes = 4000
max_steps = 500
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.4

# Environment setup
env_agent = gym.make('SimpleTetris-v0', obs_type='grayscale', reward_step=True, penalise_height_increase=True, advanced_clears=True, penalise_holes_increase=True)
env_adversary = gym.make('SimpleTetris-v0', obs_type='grayscale', reward_step=True, penalise_height_increase=True, advanced_clears=True, penalise_holes_increase=True)

# Check environment observation and action spaces
state_shape = env_agent.observation_space.shape
state_size = np.prod(state_shape)
action_size = env_agent.action_space.n

# Initialize Q-tables
Q1_agent = np.zeros((state_size, action_size))
Q2_agent = np.zeros((state_size, action_size))
Q1_adversary = np.zeros((state_size, action_size))
Q2_adversary = np.zeros((state_size, action_size))

# Functions to flatten state, select action, and update Q-tables
def flatten_state(state):
    return state.flatten()

def state_to_index(state):
    return hash(state.tobytes()) % state_size

def select_action(state, Q1, Q2, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    state_index = state_to_index(state)
    q_values = Q1[state_index, :] + Q2[state_index, :]
    return np.argmax(q_values)

def update_q_tables(state, action, reward, next_state, done, Q1, Q2, alpha, gamma):
    state_index = state_to_index(state)
    next_state_index = state_to_index(next_state)
    if np.random.rand() < 0.5:
        best_next_action = np.argmax(Q1[next_state_index, :])
        td_target = reward + (1 - done) * gamma * Q2[next_state_index, best_next_action]
        td_error = td_target - Q1[state_index, action]
        Q1[state_index, action] += alpha * td_error
    else:
        best_next_action = np.argmax(Q2[next_state_index, :])
        td_target = reward + (1 - done) * gamma * Q1[next_state_index, best_next_action]
        td_error = td_target - Q2[state_index, action]
        Q2[state_index, action] += alpha * td_error

# Track rewards and times
rewards_agent = []
rewards_adversary = []
times = []

# Training loop
for ep in range(num_episodes):
    start_time = time.time()

    state_agent = env_agent.reset()
    state_adversary = env_adversary.reset()
    state_agent = flatten_state(state_agent)
    state_adversary = flatten_state(state_adversary)
    done_agent = False
    done_adversary = False

    total_reward_agent = 0
    total_reward_adversary = 0

    for t in range(max_steps):
        if not done_agent:
            action_agent = select_action(state_agent, Q1_agent, Q2_agent, epsilon)
            next_state_agent, reward_agent, done_agent, _ = env_agent.step(action_agent)
            next_state_agent = flatten_state(next_state_agent)
            total_reward_agent += reward_agent
            update_q_tables(state_agent, action_agent, reward_agent, next_state_agent, done_agent, Q1_agent, Q2_agent, alpha, gamma)
            state_agent = next_state_agent

        if not done_adversary:
            action_adversary = select_action(state_adversary, Q1_adversary, Q2_adversary, epsilon)
            next_state_adversary, reward_adversary, done_adversary, _ = env_adversary.step(action_adversary)
            next_state_adversary = flatten_state(next_state_adversary)
            total_reward_adversary += reward_adversary
            update_q_tables(state_adversary, action_adversary, reward_adversary, next_state_adversary, done_adversary, Q1_adversary, Q2_adversary, alpha, gamma)
            state_adversary = next_state_adversary

        if done_agent and done_adversary:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_agent.append(total_reward_agent)
    rewards_adversary.append(total_reward_adversary)
    times.append(time.time() - start_time)
    print(f"Episode {ep + 1}/{num_episodes} finished with epsilon {epsilon:.4f} agent: {total_reward_agent} adversary: {total_reward_adversary}")

# Save the Q-tables to the 'double_q' folder
os.makedirs("double_q", exist_ok=True)
np.save(os.path.join("double_q", "Q1_agent.npy"), Q1_agent)
np.save(os.path.join("double_q", "Q2_agent.npy"), Q2_agent)
np.save(os.path.join("double_q", "Q1_adversary.npy"), Q1_adversary)
np.save(os.path.join("double_q", "Q2_adversary.npy"), Q2_adversary)

# Plot the rewards and times
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rewards_agent, label='Agent')
plt.plot(rewards_adversary, label='Adversary')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(times, label='Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Time (s)')
plt.title('Time per Episode')
plt.legend()

plt.tight_layout()
plt.show()