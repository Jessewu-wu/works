"""Analyze vertical jump ground reaction force data.

This script reads one-column ``.txt`` files where the first line contains a
signal name and the remaining lines provide vertical ground reaction force
samples recorded at 600 Hz.  For each file the script estimates the athlete's
body weight from an initial quiet standing phase, identifies the start of the
movement, take-off and landing, integrates the vertical acceleration to obtain
take-off velocity, and reports the flight characteristics of the jump.

The results are printed in a table and optionally written to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


GRAVITY = 9.80665  # m/s^2
SAMPLE_RATE = 600.0  # Hz


@dataclass
class JumpResult:
    file: str
    label: str
    weight_newton: float
    mass_kg: float
    movement_start_s: float
    takeoff_time_s: float
    landing_time_s: float
    flight_time_s: float
    takeoff_velocity_m_s: float
    jump_height_m: float


def read_signal(path: Path) -> Tuple[str, List[float]]:
    """Read a single-column ``.txt`` file.

    The first line is treated as a string label and the remaining lines as
    floating point samples.
    """

    with path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    if not lines:
        raise ValueError(f"File {path} is empty")

    label = lines[0]
    try:
        data = [float(value) for value in lines[1:]]
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Non-numeric data found in {path}") from exc

    if not data:
        raise ValueError(f"File {path} does not contain numeric samples")

    return label, data


def estimate_body_weight(samples: Sequence[float]) -> Tuple[float, int]:
    """Estimate body weight from a stable standing segment.

    The mean of the first 0.2 seconds (or as much data as available when the
    recording is shorter) is used to represent the quiet standing force, and the
    returned index marks the end of that baseline window.
    """

    total_samples = len(samples)
    if total_samples == 0:
        raise ValueError("No samples provided")

    window = min(total_samples, max(30, int(SAMPLE_RATE * 0.2)))
    weight = sum(samples[:window]) / window
    return weight, window


def find_first_departure(
    samples: Sequence[float],
    weight: float,
    start_index: int,
    threshold_ratio: float = 0.05,
    min_duration: float = 0.05,
) -> int:
    """Locate the first index where the signal departs from the baseline.

    Args:
        samples: Ground reaction force values.
        weight: Estimated body weight (baseline force).
        start_index: Index from which to begin searching.
        threshold_ratio: Fraction of the weight used as deviation threshold.
        min_duration: Minimum duration (seconds) that the deviation must last.

    Returns:
        The index of the first detected departure.  If no departure is found the
        provided ``start_index`` is returned.
    """

    threshold = max(abs(weight) * threshold_ratio, 20.0)
    min_samples = max(int(min_duration * SAMPLE_RATE), 1)

    size = len(samples)

    for idx in range(start_index, size):
        if abs(samples[idx] - weight) >= threshold:
            end = min(size, idx + min_samples)
            if all(abs(samples[j] - weight) >= threshold * 0.5 for j in range(idx, end)):
                return idx
    return start_index


def find_threshold_crossing(
    samples: Sequence[float],
    start_index: int,
    predicate,
    min_duration: float = 0.02,
) -> int:
    """Find the first index after ``start_index`` satisfying ``predicate``.

    The predicate is required to hold for at least ``min_duration`` seconds to
    avoid reacting to short spikes.
    """

    min_samples = max(int(min_duration * SAMPLE_RATE), 1)

    size = len(samples)

    for idx in range(start_index, size):
        if predicate(samples[idx]):
            end = min(size, idx + min_samples)
            if all(predicate(samples[j]) for j in range(idx, end)):
                return idx
    return size - 1


def compute_velocity(acceleration: Sequence[float], dt: float) -> List[float]:
    """Integrate acceleration using the trapezoidal rule to obtain velocity."""

    size = len(acceleration)
    if size == 0:
        return []

    velocity = [0.0] * size
    for idx in range(1, size):
        velocity[idx] = velocity[idx - 1] + 0.5 * (acceleration[idx - 1] + acceleration[idx]) * dt
    return velocity


def analyze_file(path: Path) -> JumpResult:
    label, samples = read_signal(path)

    weight, baseline_samples = estimate_body_weight(samples)
    mass = weight / GRAVITY

    movement_start_idx = find_first_departure(samples, weight, baseline_samples)

    flight_threshold = max(0.1 * abs(weight), 20.0)

    takeoff_idx = find_threshold_crossing(
        samples,
        movement_start_idx,
        predicate=lambda value: value <= flight_threshold,
    )

    landing_idx = find_threshold_crossing(
        samples,
        takeoff_idx,
        predicate=lambda value: value >= flight_threshold,
    )

    dt = 1.0 / SAMPLE_RATE
    accel_segment = [
        (samples[idx] - weight) / mass for idx in range(movement_start_idx, takeoff_idx)
    ]
    velocity = compute_velocity(accel_segment, dt)
    takeoff_velocity = float(velocity[-1]) if velocity else 0.0

    flight_time = max(landing_idx - takeoff_idx, 0) * dt
    jump_height = (takeoff_velocity ** 2) / (2 * GRAVITY)

    return JumpResult(
        file=str(path.name),
        label=label,
        weight_newton=weight,
        mass_kg=mass,
        movement_start_s=movement_start_idx * dt,
        takeoff_time_s=takeoff_idx * dt,
        landing_time_s=landing_idx * dt,
        flight_time_s=flight_time,
        takeoff_velocity_m_s=takeoff_velocity,
        jump_height_m=jump_height,
    )


def iter_txt_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*.txt")):
        if path.is_file():
            yield path


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def print_results(results: Sequence[JumpResult]) -> None:
    if not results:
        print("No .txt files were processed.")
        return

    headers = [
        "File",
        "Label",
        "Weight (N)",
        "Mass (kg)",
        "Move Start (s)",
        "Take-off (s)",
        "Landing (s)",
        "Flight Time (s)",
        "Take-off Velocity (m/s)",
        "Jump Height (m)",
    ]

    row_data: List[List[str]] = []
    for item in results:
        row_data.append(
            [
                item.file,
                item.label,
                format_float(item.weight_newton),
                format_float(item.mass_kg),
                format_float(item.movement_start_s),
                format_float(item.takeoff_time_s),
                format_float(item.landing_time_s),
                format_float(item.flight_time_s),
                format_float(item.takeoff_velocity_m_s),
                format_float(item.jump_height_m),
            ]
        )

    column_widths = [max(len(header), *(len(row[idx]) for row in row_data)) for idx, header in enumerate(headers)]

    def print_row(values: Sequence[str]) -> None:
        line = "  ".join(value.ljust(column_widths[idx]) for idx, value in enumerate(values))
        print(line)

    print_row(headers)
    print_row(["-" * width for width in column_widths])
    for row in row_data:
        print_row(row)


def write_csv(results: Sequence[JumpResult], destination: Path) -> None:
    def serialize(item: JumpResult) -> dict:
        data = asdict(item)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = f"{value:.6f}"
        return data

    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for item in results:
            writer.writerow(serialize(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .txt files exported from the force platform.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write the aggregated results as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder: Path = args.folder

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"{folder} is not a valid folder")

    txt_files = list(iter_txt_files(folder))
    results = [analyze_file(path) for path in txt_files]

    print_results(results)

    if args.csv and results:
        write_csv(results, args.csv)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()

