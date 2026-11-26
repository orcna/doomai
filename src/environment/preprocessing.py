
import cv2
import numpy as np


def preprocess_frame(screen_buffer):

    #convert to numpy array
    frame = np.asarray(screen_buffer)

    #detect input layout and transpose to (H, W, C) if necessary
    if frame.shape[0] in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)
    elif frame.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"unsupported image shape: {frame.shape}")

    #convert to grayscale
    if frame.shape[-1] == 3:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.shape[-1] == 4:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    else:
        grayscale = frame.squeeze(axis=-1)

    #resize to 84x84
    resized = cv2.resize(
        grayscale,
        (84, 84),
        interpolation=cv2.INTER_AREA
    )

    #normalize pixel values
    normalized = resized.astype(np.float32) / 255.0

    return np.ascontiguousarray(normalized)
