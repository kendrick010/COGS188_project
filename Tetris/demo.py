import gym
import gym_simpletetris
import pygame

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

# Window setup, adjust for two environments side by side
pygame.init()
window_size = 512
combined_window_size = (2 * window_size, window_size)
window = pygame.display.set_mode(combined_window_size)
clock = pygame.time.Clock()

# Reset environments
state_agent = env_agent.reset()
state_adversary = env_adversary.reset()

state_cache = {
    "agent": {},
    "adversary": {}
}

num_episodes = 10

def penalize(info, sending_agent="agent"):
    # Initialize state_cache if empty
    if not state_cache[sending_agent]: 
        state_cache[sending_agent] = info
        return

    delta = info["lines_cleared"] - state_cache[sending_agent]["lines_cleared"]
    if sending_agent == "agent":
        env_adversary.add_lines(delta)
    else:
        env_agent.add_lines(delta)

    state_cache[sending_agent] = info

if __name__ == "__main__":
    curr_episode = 0
    while curr_episode < num_episodes:
        # Agent environment step
        action_agent = env_agent.action_space.sample()
        state_agent, reward_agent, done_agent, info_agent = env_agent.step(action_agent)

        # Penalize adversary if cleared lines
        penalize(info_agent, "agent")

        # Adversary environment step
        action_adversary = env_adversary.action_space.sample()
        state_adversary, reward_adversary, done_adversary, info_adversary = env_adversary.step(action_adversary)

        # Penalize agent if cleared lines
        penalize(info_adversary, "adversary")
        
        if done_agent or done_adversary:
            print(f"Episode {curr_episode + 1} has finished.")
            curr_episode += 1

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
