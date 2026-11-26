
import cv2
import numpy as np

from vizdoom import DoomGame, Mode

from preprocessing import preprocess_frame
from frame_stack import FrameStack


game = DoomGame()
game.load_config("basic.cfg")
game.set_window_visible(True)
game.set_mode(Mode.PLAYER)
game.init()
game.new_episode()

frame_stack = FrameStack(num_frames=4)

initialized = False

while not game.is_episode_finished():

    state = game.get_state()

    if state is None:
        break

    #get raw image
    screen = state.screen_buffer

    #preprocess image
    processed = preprocess_frame(screen)

    #initialize or update frame stack
    if not initialized:
        observation = frame_stack.reset(processed)
        initialized = True
    else:
        observation = frame_stack.add_frame(processed)

    #display current processed image
    cv2.imshow("Processed Vision", processed)

    #print observation information
    print(
        "Observation shape:", observation.shape,
        "| Dtype:", observation.dtype,
        "| Min:", observation.min(),
        "| Max:", observation.max()
    )

    #wait for action
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    game.make_action([False, False, False], 1)


game.close()
cv2.destroyAllWindows()
