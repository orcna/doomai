import os
import cv2
import vizdoom as vzd
from vizdoom import DoomGame, Mode

game = DoomGame()

cfg_path = os.path.join(vzd.scenarios_path, "basic.cfg")
game.load_config(cfg_path)
game.set_window_visible(True)
game.set_mode(Mode.PLAYER)
game.init()

num_buttons = game.get_available_buttons_size()
idle_action = [0] * num_buttons

game.new_episode()

try:
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        #(C, H, W) -> (H, W, C)
        screen = state.screen_buffer.transpose(1, 2, 0)

        
        #cv2.cvtColor(screen, cv2.COLOR_RGB2BGR) 
        cv2.imshow("ScreenPredator - Agent Vision", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        #idle for now
        game.make_action(idle_action, 1)
finally:
    game.close()
    cv2.destroyAllWindows()