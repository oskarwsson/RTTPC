"""Match ROS 2 echo dumps of odom -> base_link TFs to camera frames."""

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
import math
import re
from collections.abc import Iterator

import yaml


@dataclass
class TF:
    """Pose with quaternion (w, x, y, z), translation (x, y, z), and TF time."""

    q: list[float]
    xyz: list[float]
    timestamp: float


def sec_nanosec_to_float(sec: int, nanosec: int) -> float:
    """Return seconds; epoch-scale floats have sub-microsecond precision."""
    if not 0 <= nanosec < 1_000_000_000:
        raise ValueError("nanosec must be in [0, 1000000000)")
    return sec + nanosec * 1e-9


def _messages(file: str, *, header_only: bool = False) -> Iterator[dict]:
    """Read one YAML message at a time, ignoring terminal cursor escapes."""
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    rows = []
    skipping_data = False
    with open(file, encoding="utf-8") as source:
        for line in source:
            line = ansi.sub("", line)
            if line.strip() == "---":
                message = yaml.load("".join(rows), Loader=loader)
                if message is not None:
                    yield message
                rows = []
                skipping_data = False
            elif header_only and line.startswith("data:"):
                skipping_data = True
            elif not skipping_data:
                rows.append(line)
    message = yaml.load("".join(rows), Loader=loader)
    if message is not None:
        yield message


def _timestamp(header: dict) -> float:
    stamp = header["stamp"]
    return sec_nanosec_to_float(int(stamp["sec"]), int(stamp["nanosec"]))


def read_tf(file: str) -> list[TF]:
    """Extract only transforms with parent odom and child base_link."""
    out = []
    for message in _messages(file):
        for transform in message["transforms"]:
            header = transform["header"]
            if (header["frame_id"].lstrip("/") != "odom"
                    or transform["child_frame_id"].lstrip("/") != "base_link"):
                continue
            pose = transform["transform"]
            out.append(TF(
                q=[float(pose["rotation"][axis]) for axis in "wxyz"],
                xyz=[float(pose["translation"][axis]) for axis in "xyz"],
                timestamp=_timestamp(header),
            ))
    return out


def read_img(file: str) -> list[float]:
    """Extract image header timestamps in recorded frame order."""
    return [_timestamp(message["header"])
            for message in _messages(file, header_only=True)]


def sync_tf_img(
    tfs: list[TF], stamps: list[float], max_delta: float | None = None,
) -> list[TF]:
    """Return nearest TF per image, preserving image order and original TF time.

    Ties choose the earlier timestamp. TFs may be reused. Images outside the
    TF time range match an endpoint. max_delta optionally rejects matches
    farther away than that many seconds. No interpolation is performed.
    """
    if max_delta is not None and (not math.isfinite(max_delta) or max_delta < 0):
        raise ValueError("max_delta must be finite and non-negative")
    if not stamps:
        return []
    if not tfs:
        raise ValueError("No odom -> base_link transforms available to match images")
    if not all(math.isfinite(t.timestamp) for t in tfs) or not all(
        math.isfinite(stamp) for stamp in stamps
    ):
        raise ValueError("Timestamps must be finite")
    ordered = sorted(tfs, key=lambda tf: tf.timestamp)
    times = [tf.timestamp for tf in ordered]
    out = []
    for frame, stamp in enumerate(stamps):
        index = bisect_left(times, stamp)
        candidates = ordered[max(0, index - 1):min(len(ordered), index + 1)]
        match = min(candidates, key=lambda tf: (abs(tf.timestamp - stamp), tf.timestamp))
        delta = abs(match.timestamp - stamp)
        if max_delta is not None and delta > max_delta:
            raise ValueError(
                f"Image frame {frame}: nearest TF is {delta:.9f}s away "
                f"(maximum {max_delta}s)"
            )
        out.append(match)
    return out


def write_csv(file: str, tfs: list[TF]) -> None:
    """Write zero-based frame numbers, source TF times, and labeled pose values."""
    with open(file, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["frame", "timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for frame, tf in enumerate(tfs):
            writer.writerow([frame, tf.timestamp, *tf.xyz, *tf.q])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tf_file", nargs="?", default="example_TFs.txt")
    parser.add_argument("img_file", nargs="?", default="example_imgs.txt")
    parser.add_argument("-o", "--output", default="out.csv")
    parser.add_argument("--max-delta", type=float, help="Maximum allowed time gap in seconds")
    args = parser.parse_args()
    try:
        tfs = read_tf(args.tf_file)
        stamps = read_img(args.img_file)
        if not stamps:
            raise ValueError("No image timestamps found")
        synced = sync_tf_img(tfs, stamps, max_delta=args.max_delta)
        write_csv(args.output, synced)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")
    largest_gap = max(abs(tf.timestamp - stamp) for tf, stamp in zip(synced, stamps))
    print(f"Wrote {len(synced)} frames to {args.output}; largest time gap: {largest_gap:.9f}s")


if __name__ == "__main__":
    main()
