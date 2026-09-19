"""Encode a directory of PNG frames into an MP4.

Runs on the Windows side, in labgen's separate tools venv, because the encoder
has no business inside the Isaac Lab environment -- that is a pinned beta and
CLAUDE.md rule 5 says anything extra goes in its own venv. The frames come
across from WSL as files, so nothing needs to be installed next to Newton.

    .venv-usd/Scripts/python scripts/encode_video.py out/frames_demo out/demo.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio

FPS = 60


def main() -> int:
    frames_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else FPS

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"no frames in {frames_dir}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    # quality=8 and yuv420p: yuv420p is what browsers, Slack, Keynote and
    # QuickTime will all actually play. The default pixel format encodes fine
    # and then refuses to open in half of them.
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for i, f in enumerate(frames):
        writer.append_data(imageio.imread(f))
        if i % 120 == 0:
            print(f"   {i}/{len(frames)}")
    writer.close()

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  {len(frames)} frames @ {fps} fps  "
          f"({len(frames) / fps:.1f} s, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
