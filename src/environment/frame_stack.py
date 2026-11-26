
from collections import deque
import numpy as np


class FrameStack:

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, initial_frame):

        self.frames.clear()

        for _ in range(self.num_frames):
            self.frames.append(initial_frame)

        return self.get_stack()

    def add_frame(self, frame):

        self.frames.append(frame)

        return self.get_stack()

    def get_stack(self):

        return np.stack(self.frames, axis=0)
