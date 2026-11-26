from doom_env import DoomEnvironment

env = DoomEnvironment()

observation = env.reset()

print("Initial observation:", observation.shape)

for step in range(10):

    action = 3  # IDLE

    observation, reward, done, info = env.step(action)

    print(
        f"Step: {step + 1} | "
        f"Reward: {reward} | "
        f"Done: {done} | "
        f"Observation: {observation.shape}"
    )

    if done:
        break

env.close()