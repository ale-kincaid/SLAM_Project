
import numpy as np
import imageio.v2 as imageio

def save_video(frames, filename="slam_video.mp4", fps=10):
    if not frames:
        print("No frames to save!")
        return

    writer = imageio.get_writer(filename, fps=fps)
    for fr in frames:
        writer.append_data(np.asarray(fr))
    writer.close()
    print("Saved video:", filename)
