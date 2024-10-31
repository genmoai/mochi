import os
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from genmo.lib.progress import get_new_progress_bar

class Timer:
    def __init__(self):
        self.times = {}  # Dictionary to store times per stage

    def __call__(self, name):
        print(f"Timing {name}")
        return self.TimerContextManager(self, name)

    def print_stats(self):
        total_time = sum(self.times.values())
        # Print table header
        print("{:<20} {:>10} {:>10}".format("Stage", "Time(s)", "Percent"))
        for name, t in self.times.items():
            percent = (t / total_time) * 100 if total_time > 0 else 0
            print("{:<20} {:>10.2f} {:>9.2f}%".format(name, t, percent))

    class TimerContextManager:
        def __init__(self, outer, name):
            self.outer = outer  # Reference to the Timer instance
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.outer.times[self.name] = self.outer.times.get(self.name, 0) + elapsed


def save_video(final_frames, output_path, fps=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(get_new_progress_bar(final_frames)):
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            frame_paths.append(frame_path)

        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p -preset veryfast {output_path}"
        try:
            subprocess.run(ffmpeg_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running ffmpeg:\n{e.stderr.decode()}")