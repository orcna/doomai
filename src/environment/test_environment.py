import os
import vizdoom as vzd
from vizdoom import DoomGame, Mode

game = DoomGame()

#load basic config
cfg_path = os.path.join(vzd.scenarios_path, "basic.cfg")
game.load_config(cfg_path)

game.set_window_visible(True)
game.set_mode(Mode.PLAYER)
game.init()

buttons = game.get_available_buttons()
num_buttons = len(buttons)
print("available buttons:", buttons)

    
def make_one_hot(index: int, size: int) -> list[int]:
    vec = [0] * size
    if 0 <= index < size:
        vec[index] = 1
    return vec


#basic scenario order: MOVE_LEFT, MOVE_RIGHT, ATTACK
actions = {
    "LEFT": make_one_hot(0, num_buttons),
    "RIGHT": make_one_hot(1, num_buttons),
    "ATTACK": make_one_hot(2, num_buttons),
    "IDLE": [0] * num_buttons,
}

for name, action in actions.items():
    game.new_episode()
    print(f"\ntesting: {name}")

    for step in range(5):
        if game.is_episode_finished():
            print("episode finished!")
            break

        state = game.get_state()
        screen = state.screen_buffer if state is not None else None

        reward = game.make_action(action, 4)

        print(
            f"step: {step + 1} | "
            f"reward: {reward} | "
            f"total: {game.get_total_reward()} | "
            f"frame shape: {screen.shape if screen is not None else None}"
        )

    print("episode time:", game.get_episode_time())

game.close()