#!/usr/bin/env python

"""
Visualize input/output simulation file combinations as a video using Pygame and MoviePy. The user must specify both the input and output file paths.
"""

import sys
import numpy as np
import argparse
import os

# Make PyGame headless
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ["SDL_VIDEO_X11_NO_MITSHM"] = "1"

import pygame
from moviepy.editor import VideoClip
from time import time


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Particle Simulation Visualizer")
    parser.add_argument("input_file", help="Path to the input (.in) file")
    parser.add_argument("output_file", help="Path to the output (.out) file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--output", type=str, help="Output MP4 filename, default is 'simulation.mp4'")
    parser.add_argument("--color_mode", choices=["speed", "index"], default="speed", help="Color particles by speed or index")
    parser.add_argument("--show_window", action="store_true", help="Enable full display of the Pygame window during rendering")
    args = parser.parse_args()

    if not args.output:
        args.output = "simulation.mp4"

    return args


def configure_pygame(show_window):
    """Configure Pygame settings, including headless mode if needed."""
    pygame.init()

    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    if not show_window:
        pygame.display.iconify()
    return screen, screen_size


def load_data(input_file, output_file):
    """Load simulation data from input and output files."""
    data_dict = {}
    with open(input_file, 'r', encoding="utf-8") as particle_in:
        # Number of particles
        N = int(particle_in.readline().strip())
        # Simulation area size
        L = float(particle_in.readline().strip())
        # Particle radius
        r = float(particle_in.readline().strip())
        # Number of steps
        steps = int(particle_in.readline().strip())

    with open(output_file, 'r', encoding="utf-8") as particle_out:
        prev_step = -1
        for line in particle_out.readlines():
            data = line.strip().split()
            step, index, x, y, v_x, v_y = map(float, data)
            step = int(step)
            if step < prev_step:
                print("Error: Steps are not in order.")
                sys.exit(1)
            index = int(index)
            v = (v_x**2 + v_y**2)**0.5

            if step not in data_dict:
                data_dict[step] = []
            data_dict[step].append((x, y, v, index))

    return data_dict, N, L, r, steps


def draw_frame(screen, screen_size, L, r, particles, color_mode):
    """Draw a single frame of the simulation."""
    screen.fill((255, 255, 255))  # Clear screen with white background

    positions = particles[:, :2]
    speeds = particles[:, 2]
    indices = particles[:, 3]

    if color_mode == "speed":
        max_speed = np.max(speeds)
        colors = np.column_stack([speeds / max_speed, np.zeros_like(speeds), 1 - speeds / max_speed])

    elif color_mode == "index":
        colors = np.column_stack([indices / len(indices), np.zeros_like(indices), 1 - indices / len(indices)])

    # Convert positions to pixel coordinates
    positions *= (screen_size / L)
    
    for pos, color in zip(positions, colors):
        pygame.draw.circle(screen, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), pos.astype(int), int(r * (screen_size / L)))

    pygame.display.flip()


def make_frame(t, screen, screen_size, L, r, data_dict, color_mode, fps):
    """Generate a frame for MoviePy."""
    frame = int(t * fps)
    if frame in data_dict:
        particles = np.array(data_dict[frame])
        draw_frame(screen, screen_size, L, r, particles, color_mode)
    surf = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
    return surf


def generate_video(data_dict, N, L, r, steps, output, fps, screen, screen_size, color_mode):
    """Generate MP4 video using MoviePy."""
    duration = (steps + 1) / fps
    animation = VideoClip(lambda t: make_frame(t, screen, screen_size, L, r, data_dict, color_mode, fps), duration=duration)
    animation.write_videofile(output, fps=fps, threads=os.cpu_count(), preset='ultrafast')


def main():
    args = parse_arguments()
    
    screen, screen_size = configure_pygame(args.show_window)
    
    data_dict, N, L, r, steps = load_data(args.input_file, args.output_file)
    
    start_time = time()
    generate_video(data_dict, N, L, r, steps, args.output, args.fps, screen, screen_size, args.color_mode)
    end_time = time()

    print(f"Animation saved to {args.output}!")
    print(f"Time taken: {end_time - start_time:.3f}s")
    
    pygame.quit()


if __name__ == "__main__":
    main()
