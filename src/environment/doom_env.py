
from vizdoom import DoomGame, Mode
import numpy as np

from preprocessing import preprocess_frame
from frame_stack import FrameStack


class DoomEnvironment:

    def __init__(self, config_path="basic.cfg", frame_skip=4):

        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()

        self.frame_stack = FrameStack(num_frames=4)

        self.frame_skip = frame_skip

        self.actions = [
            [True, False, False],   # LEFT
            [False, True, False],   # RIGHT
            [False, False, True],   # ATTACK
            [False, False, False]   # IDLE
        ]

    def reset(self):

        self.game.new_episode()

        state = self.game.get_state()
        frame = preprocess_frame(state.screen_buffer)

        observation = self.frame_stack.reset(frame)

        return observation

    def step(self, action):

        reward = self.game.make_action(
            self.actions[action],
            self.frame_skip
        )

        done = self.game.is_episode_finished()

        if done:
            observation = np.zeros((4, 84, 84), dtype=np.float32)
        else:
            state = self.game.get_state()

            if state is None:
                done = True
                observation = np.zeros((4, 84, 84), dtype=np.float32)
            else:
                frame = preprocess_frame(state.screen_buffer)
                observation = self.frame_stack.add_frame(frame)

        info = {}

        return observation, reward, done, info

    def close(self):

        self.game.close()
