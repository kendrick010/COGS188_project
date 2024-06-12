import gym
import neat
import numpy as np
import gym_simpletetris
import pygame
import os

relative_path = os.path.dirname(__file__)
config_path = os.path.join(relative_path, "assets/config.txt")
checkpoints_dir = os.path.join(relative_path, "models")
best_model_path = os.path.join(relative_path, "models/best_model.pickle")

pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Create Tetris environment
env = gym.make('SimpleTetris-v0',
                obs_type='grayscale',              # ram | grayscale | rgb
                reward_step=True,                  # See reward table
                penalise_height=True,              # See reward table
                advanced_clears=True,              # See reward table
                penalise_holes=True)               # See reward table

# Define the fitness function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()
        fitness = 0
        done = False

        while not done:
            action = net.activate(observation.flatten())
            observation, reward, done, info = env.step(np.argmax(action))
            fitness += reward

            window.fill((0, 0, 0))
            env.render(surface=window, offset=(0, 0))
            pygame.display.flip()  # Update the full display Surface to the screen
            clock.tick(30)  # Control the game's frame rate


        genome.fitness = fitness

# Load NEAT configuration
config = neat.config.Config(
    neat.DefaultGenome, 
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet, 
    neat.DefaultStagnation,
    config_path
)

# Create the population
population = neat.Population(config)

# Add reporters to visualize progress
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.StdOutReporter(True))  # Output statistics to console
population.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(checkpoints_dir, "neat-checkpoint-")))
population.add_reporter(stats)

# Run the NEAT algorithm
winner = population.run(eval_genomes, 50)

# Save the winning network
import pickle

with open(best_model_path, "wb") as f:
    pickle.dump(winner, f)
