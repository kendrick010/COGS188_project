import gym
import gym_simpletetris
import pygame
import neat
import pickle
import os
import numpy as np
from CSVReporter import CSVReporter

# Window setup, adjust for two environments side by side
rendering = True
check_point = False

# Config and checkpoint paths
relative_path = os.path.dirname(__file__)
config_path = os.path.join(relative_path, "assets/config.txt")
checkpoints_dir = os.path.join(relative_path, "models")
best_model_path = os.path.join(relative_path, "models/best_neat_model.pickle")
csv_log_path = os.path.join(relative_path, "models/training_log.csv")
latest_checkpoint_file = 0

pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Out of how many games per versus
num_episodes = 3

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

def render_text(window, text, position):
    text_surface = font.render(text, True, (255, 255, 255))
    window.blit(text_surface, position)

def simulate_tetris_game(genome1, genome2, config, generation, population_pair):
    # Initialize the environments
    env_agent = gym.make('SimpleTetris-v0',
                        obs_type='ram',                     # ram | grayscale | rgb
                        reward_step=True,                  # See reward table
                        penalise_height=True,              # See reward table
                        advanced_clears=True,              # See reward table
                        penalise_holes=True)               # See reward table

    env_adversary = gym.make('SimpleTetris-v0',
                        obs_type='ram',                     # ram | grayscale | rgb
                        reward_step=True,                  # See reward table
                        penalise_height=True,              # See reward table
                        advanced_clears=True,              # See reward table
                        penalise_holes=True)               # See reward table
    
    # Initialize network
    net_agent = neat.nn.FeedForwardNetwork.create(genome1, config)
    net_adversary = neat.nn.FeedForwardNetwork.create(genome2, config)

    for episode in range(num_episodes):
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

        while not done_agent or not done_adversary:
            input_package_agent = np.argmax(net_agent.activate(state_agent.flatten()))
            input_package_adversary = np.argmax(net_adversary.activate(state_adversary.flatten()))

            state_agent, reward_agent, done_agent, info_agent = env_agent.step(input_package_agent)
            state_adversary, reward_adversary, done_adversary, info_adversary = env_adversary.step(input_package_adversary)

            # Penalize opposings for line clears
            penalize(info_agent, info_cache, env_agent, env_adversary, "agent")
            penalize(info_adversary, info_cache, env_agent, env_adversary, "adversary")

            # Rendering
            if rendering:
                window.fill((0, 0, 0))
                env_agent.render(surface=window, offset=(0, 0))
                env_adversary.render(surface=window, offset=(window_size, 0))
                render_text(window, f"Generation: {generation + latest_checkpoint_file}", (10, 10))
                render_text(window, f"Population Pair: {population_pair}", (10, 30))
                render_text(window, f"Episode: {episode}/{num_episodes}", (10, 50))
                pygame.display.flip()  # Update the full display Surface to the screen
                clock.tick(30)  # Control the game's frame rate

            genome1.fitness += reward_agent
            genome2.fitness += reward_adversary

    env_agent.close()
    env_adversary.close()

def eval_genomes(genomes, config, generation):
    # Ensure population size is even
    for idx in range(0, len(genomes), 2):
        _, genome1 = genomes[idx]
        _, genome2 = genomes[idx+1]
        
        genome1.fitness = 0
        genome2.fitness = 0
        
        simulate_tetris_game(genome1, genome2, config, generation, idx // 2 + 1)

def run_neat(config, generations=50, run_last_checkpoint=False):
    global latest_checkpoint_file

    # Create a NEAT population
    p = neat.Population(config)
    
    # If requested, restore from the last checkpoint
    if run_last_checkpoint:
        checkpoint_files = [int(f.replace('neat-checkpoint-', '')) for f in os.listdir(checkpoints_dir) if f.startswith("neat-checkpoint-")]
        if checkpoint_files:
            latest_checkpoint_file = max(checkpoint_files)
            p = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoints_dir, f"neat-checkpoint-{latest_checkpoint_file}"))
        # else: start from scratch
    
    # Add reporters to the population
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))  # Output statistics to console
    p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(checkpoints_dir, "neat-checkpoint-")))

    csv_reporter = CSVReporter(csv_log_path)
    p.add_reporter(csv_reporter)

    for generation in range(generations):
        p.run(lambda genomes, config: eval_genomes(genomes, config, generation), 1)

    winner = p.best_genome
    with open(best_model_path, "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config, generations=50, run_last_checkpoint=check_point)

    pygame.quit()
