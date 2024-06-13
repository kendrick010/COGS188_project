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
best_model_path = os.path.join(relative_path, "models/best_neat_model.pickle")

# Window setup, adjust for two environments side by side
rendering = True

pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

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

def simulate_tetris_game(genome1, genome2, config):
    # Initialize the environments
    env_agent = gym.make('SimpleTetris-v0',
                        obs_type='ram',              # ram | grayscale | rgb
                        reward_step=True,                  # See reward table
                        penalise_height_increase=True,     # See reward table
                        advanced_clears=True,              # See reward table
                        penalise_holes_increase=True)      # See reward table

    env_adversary = gym.make('SimpleTetris-v0',
                        obs_type='ram',              # ram | grayscale | rgb
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
        input_package_agent = np.argmax(net_agent.activate(state_agent.flatten()))
        input_package_adversary = np.argmax(net_adversary.activate(state_adversary.flatten()))

        state_agent, reward_agent, done_agent, info_agent = env_agent.step(input_package_agent)
        state_adversary, reward_adversary, done_adversary, info_adversary = env_adversary.step(input_package_adversary)

        # Penalize opposings for line clears
        penalize(info_agent, info_cache, env_agent, env_adversary, "agent")
        penalize(info_adversary, info_cache, env_agent, env_adversary, "adversary")

        if rendering == True:
            window.fill((0, 0, 0))
            env_agent.render(surface=window, offset=(0, 0))
            env_adversary.render(surface=window, offset=(window_size, 0))
            pygame.display.flip()  # Update the full display Surface to the screen
            clock.tick(30)  # Control the game's frame rate

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
