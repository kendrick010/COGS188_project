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
               obs_type='ram',                    # ram | grayscale | rgb
               reward_step=True,                  # See reward table
               penalise_height_increase=True,     # See reward table
               advanced_clears=True,              # See reward table
               penalise_holes_increase=True)      # See reward table

# Define the function to extract features from the board
def get_board_features(board):
    heights = [0] * len(board[0])
    holes = 0
    for x in range(len(board[0])):
        column_height = 0
        column_holes = 0
        for y in range(len(board)):
            if board[y][x] > 0:
                column_height = len(board) - y
                break
        for y in range(len(board) - column_height, len(board)):
            if board[y][x] == 0:
                column_holes += 1
        heights[x] = column_height
        holes += column_holes
    aggregate_height = sum(heights)
    bumpiness = sum([abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1)])
    
    return heights + [holes, aggregate_height, bumpiness]

# Define the function to evaluate a move
def evaluate_move(board, net):
    features = get_board_features(board)
    output = net.activate(features)
    return np.argmax(output)

# Simulate all possible moves and choose the best one
def choose_best_action(board, env, net):
    best_action = None
    best_score = float('-inf')
    original_board = board.copy()
    
    for action in range(env.action_space.n):
        _, _, _, info = env.step(action)
        new_board = info['board']
        score = evaluate_move(new_board, net)
        if score > best_score:
            best_score = score
            best_action = action
        env.board = original_board  # Reset the board to its original state
    
    return best_action


# Load NEAT configuration
config = neat.config.Config(
    neat.DefaultGenome, 
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet, 
    neat.DefaultStagnation,
    config_path
)

# Save the winning network
import pickle

with open(best_model_path, "rb") as f:
    winner = pickle.load(f)

# Create the network from the winner genome
net = neat.nn.FeedForwardNetwork.create(winner, config)
observation = env.reset()
done = False

while not done:
    # features = get_board_features(observation)
    best_action = choose_best_action(observation, env, net)
    observation, reward, done, info = env.step(best_action)

    window.fill((0, 0, 0))
    env.render(surface=window, offset=(0, 0))
    pygame.display.flip()  # Update the full display Surface to the screen
    clock.tick(30)  # Control the game's frame rate

env.close()
