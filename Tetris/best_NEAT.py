import gym
import gym_simpletetris
import pygame
import neat
import pickle
import os
import numpy as np

# Config and checkpoint paths
relative_path = os.path.dirname(__file__)
config_path = os.path.join(relative_path, "assets/config.txt")
best_model_path = os.path.join(relative_path, "models/best_model.pickle")

# Window setup, adjust for two environments side by side
rendering = True

pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

enumerated_shapes = {'T': [1, 0, 0, 0, 0, 0, 0], 'J': [0, 1, 0, 0, 0, 0, 0], 
                     'L': [0, 0, 1, 0, 0, 0, 0], 'Z': [0, 0, 0, 1, 0, 0, 0], 
                     'S': [0, 0, 0, 0, 1, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0], 
                     'O': [0, 0, 0, 0, 0, 0, 1]}

def penalize(info, info_cache, env_agent, env_adversary, sending_agent="agent"):
    # Initialize state_cache if empty
    if not info_cache[sending_agent]: 
        info_cache[sending_agent] = info
        return

    delta = info["lines_cleared"] - info_cache[sending_agent]["lines_cleared"]
    if sending_agent == "agent":
        env_adversary.add_lines(delta)
    else:
        env_agent.add_lines(delta)

    info_cache[sending_agent] = info

def input_package(info):
    # Default game board 10x20
    package = info["current_board"].flatten().tolist()
    
    # Append the current piece, next piece, lines cleared, and holes
    package.extend(enumerated_shapes[info["current_piece"]])
    package.extend(enumerated_shapes[info["next_piece"]])
    package.append(info["lines_cleared"])
    package.append(info["holes"])
    
    # Convert back to a numpy array
    return np.array(package)

def simulate_tetris_game(genome1, genome2, config):
    # Initialize the environments
    env_agent = gym.make('SimpleTetris-v0',
                        obs_type='grayscale',              # ram | grayscale | rgb
                        reward_step=True,                  # See reward table
                        penalise_height_increase=True,     # See reward table
                        advanced_clears=True,              # See reward table
                        penalise_holes_increase=True)      # See reward table

    env_adversary = gym.make('SimpleTetris-v0',
                        obs_type='grayscale',              # ram | grayscale | rgb
                        reward_step=True,                  # See reward table
                        penalise_height_increase=True,     # See reward table
                        advanced_clears=True,              # See reward table
                        penalise_holes_increase=True)      # See reward table
    
    # Initialize network
    net_agent = neat.nn.FeedForwardNetwork.create(genome1, config)
    net_adversary = neat.nn.FeedForwardNetwork.create(genome2, config)

    # Reset environments
    state_agent = env_agent.reset()
    state_adversary = env_adversary.reset()
    window.fill((0, 0, 0))

    done_agent = False
    done_adversary = False

    # Info cache
    info_cache = {
        "agent": {},
        "adversary": {}
    }

    while True:
        # First move random
        if not info_cache["agent"]:
            action_agent = env_agent.action_space.sample()
            action_adversary = env_adversary.action_space.sample()
        else:
            action_agent = np.argmax(net_agent.activate(input_package(info_agent)))
            action_adversary = np.argmax(net_adversary.activate(input_package(info_adversary)))

        _, reward_agent, done_agent, info_agent = env_agent.step(action_agent)
        _, reward_adversary, done_adversary, info_adversary = env_adversary.step(action_adversary)

        # Penalize opposings for line clears
        penalize(info_agent, info_cache, env_agent, env_adversary, "agent")
        penalize(info_adversary, info_cache, env_agent, env_adversary, "adversary")

    env_agent.close()
    env_adversary.close()

if __name__ == "__main__":
    # Load the best model
    with open(best_model_path, "rb") as f:
        best_genome_agent = pickle.load(f)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    simulate_tetris_game(best_genome_agent, best_genome_agent, config)

    pygame.quit()
