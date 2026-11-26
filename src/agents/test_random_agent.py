import os
import sys

rt_ = os.path.dirname(os.path.abspath(__file__))
src_ = os.path.dirname(rt_)
if src_ not in sys.path:
  sys.path.insert(0, src_)

from environment.doom_env import DoomEnvironment
from random_agent import RandomAgent


env = DoomEnvironment()
agent = RandomAgent(action_size=4)

observation = env.reset()

total_reward = 0
step_count = 0

while True:

    action = agent.select_action(observation)

    observation, reward, done, info = env.step(action)

    total_reward += reward
    step_count += 1

    print(
        f"Step: {step_count} | "
        f"Action: {action} | "
        f"Reward: {reward} | "
        f"Total Reward: {total_reward}"
    )

    if done:
        break

print("\nEpisode finished!")
print("Total steps:", step_count)
print("Total reward:", total_reward)

env.close()
