#! /usr/bin/env python3
from pathlib import Path
import shutil

import click
from moviepy.editor import VideoFileClip
from tqdm import tqdm


@click.command()
@click.argument("folder", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_folder", type=click.Path(dir_okay=True))
@click.option("--duration", "-d", type=float, default=5.4, help="Duration in seconds")
@click.option("--resolution", "-r", type=str, default="848x480", help="Video resolution")
def truncate_videos(folder, output_folder, duration, resolution):
    """Truncate all MP4 and MOV files in FOLDER to specified duration and resolution"""
    input_path = Path(folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse target resolution
    target_width, target_height = map(int, resolution.split("x"))

    # Find all MP4 and MOV files
    video_files = (
        list(input_path.rglob("*.mp4"))
        + list(input_path.rglob("*.MOV"))
        + list(input_path.rglob("*.mov"))
        + list(input_path.rglob("*.MP4"))
    )

    for file_path in tqdm(video_files):
        try:
            relative_path = file_path.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(".mp4")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            click.echo(f"Processing: {file_path}")
            video = VideoFileClip(str(file_path))

            # Skip if video is too short
            if video.duration < duration:
                click.echo(f"Skipping {file_path} as it is too short")
                continue

            # Skip if target resolution is larger than input
            if target_width > video.w or target_height > video.h:
                click.echo(
                    f"Skipping {file_path} as target resolution {resolution} is larger than input {video.w}x{video.h}"
                )
                continue

            # First truncate duration
            truncated = video.subclip(0, duration)

            # Calculate crop dimensions to maintain aspect ratio
            target_ratio = target_width / target_height
            current_ratio = truncated.w / truncated.h

            if current_ratio > target_ratio:
                # Video is wider than target ratio - crop width
                new_width = int(truncated.h * target_ratio)
                x1 = (truncated.w - new_width) // 2
                final = truncated.crop(x1=x1, width=new_width).resize((target_width, target_height))
            else:
                # Video is taller than target ratio - crop height
                new_height = int(truncated.w / target_ratio)
                y1 = (truncated.h - new_height) // 2
                final = truncated.crop(y1=y1, height=new_height).resize((target_width, target_height))

            # Set output parameters for consistent MP4 encoding
            output_params = {
                "codec": "libx264",
                "audio": False,  # Disable audio
                "preset": "medium",  # Balance between speed and quality
                "bitrate": "5000k",  # Adjust as needed
            }

            # Set FPS to 30
            final = final.set_fps(30)

            # Check for a corresponding .txt file
            txt_file_path = file_path.with_suffix('.txt')
            if txt_file_path.exists():
                output_txt_file = output_path / relative_path.with_suffix('.txt')
                output_txt_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(txt_file_path, output_txt_file)
                click.echo(f"Copied {txt_file_path} to {output_txt_file}")
            else:
                # Print warning in bold yellow with a warning emoji
                click.echo(f"\033[1;33m⚠️  Warning: No caption found for {file_path}, using an empty caption. This may hurt fine-tuning quality.\033[0m")
                output_txt_file = output_path / relative_path.with_suffix('.txt')
                output_txt_file.parent.mkdir(parents=True, exist_ok=True)
                output_txt_file.touch()

            # Write the output file
            final.write_videofile(str(output_file), **output_params)

            # Clean up
            video.close()
            truncated.close()
            final.close()

        except Exception as e:
            click.echo(f"\033[1;31m Error processing {file_path}: {str(e)}\033[0m", err=True)
            raise


if __name__ == "__main__":
    truncate_videos()
