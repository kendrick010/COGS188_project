import gym
import gym_simpletetris
import pygame

# Initialize the environments
env_agent = gym.make('SimpleTetris-v0')
env_adversary = gym.make('SimpleTetris-v0')

# Window setup, adjust for two environments side by side
pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()

# Reset environments
state_agent = env_agent.reset()
state_adversary = env_adversary.reset()

episode = 0
while episode < 10:
    # Agent environment step
    action_agent = env_agent.action_space.sample()
    state_agent, reward_agent, done_agent, info_agent = env_agent.step(action_agent)

    # Adversary environment step
    action_adversary = env_adversary.action_space.sample()
    state_adversary, reward_adversary, done_adversary, info_adversary = env_adversary.step(action_adversary)
    
    if done_agent or done_adversary:
        print(f"Episode {episode + 1} has finished.")
        episode += 1

        state_agent = env_agent.reset()
        state_adversary = env_adversary.reset()

    # Rendering
    window.fill((0, 0, 0))
    env_agent.render(surface=window, offset=(0, 0))
    env_adversary.render(surface=window, offset=(window_size, 0))

    # Update the display
    pygame.display.update()
    clock.tick(30)

env_agent.close()
env_adversary.close()
pygame.quit()
