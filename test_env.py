from env.generals_env import GeneralsEnv

# Create environment
env = GeneralsEnv()

# Reset the game
state = env.reset()
print("Environment reset successful!")
print("State shape:", state.shape)

# Get legal actions for first dice roll
actions = env.get_legal_actions(env.dice_value)
print("Legal actions:", actions)

# Take the first action (just for testing)
if len(actions) > 0:
    action = actions[0]["id"]
    print("Taking action:", action)
    
    next_state, reward, done, info = env.step(action)
    print("Next state shape:", next_state.shape)
    print("Reward:", reward)
    print("Game done:", done)
else:
    print("No legal actions found!")