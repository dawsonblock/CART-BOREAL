import gymnasium as gym
import math

env = gym.make("CartPole-v1")
env.unwrapped.theta_threshold_radians = 12 * 2 * math.pi / 360

env.reset()
# Force theta to 0.3
env.unwrapped.state = (0.0, 0.0, 0.3, 0.0)

obs, reward, terminated, truncated, info = env.step(1)
print(f"Test 1 - Force theta=0.3: Terminated={terminated}")

# Now reset and step again!
env.reset()
print(f"After Reset - Threshold is: {env.unwrapped.theta_threshold_radians}")
env.unwrapped.state = (0.0, 0.0, 0.3, 0.0)
obs, reward, terminated, truncated, info = env.step(1)
print(f"Test 2 - Force theta=0.3 after reset: Terminated={terminated}")
