#!/usr/bin/env python3
"""Process vertical jump force plate data stored in plain text files.

The script reads all ``.txt`` files in a given folder, extracts
information about the jump phases, and computes take-off velocity,
jump height, and flight time.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import math

G = 9.80665  # gravitational acceleration (m/s^2)


@dataclass
class JumpResult:
    file: Path
    signal_name: str
    body_weight: float
    mass: float
    stable_start_s: float
    stable_end_s: float
    movement_start_s: float
    takeoff_time_s: float
    landing_time_s: float
    flight_time_s: float
    takeoff_velocity: float
    jump_height: float


@dataclass
class JumpPhases:
    weight: float
    mass: float
    stable_start: int
    stable_end: int
    movement_start: int
    takeoff: int
    landing: int


class JumpAnalysisError(RuntimeError):
    """Raised when the algorithm cannot identify a necessary phase."""


def read_force_signal(path: Path) -> Tuple[str, List[float]]:
    """Read the vertical force signal from ``path``.

    The first line contains the signal name, and the following lines contain
    one sample per row.
    """

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise JumpAnalysisError(f"{path} is empty")

    signal_name = lines[0]
    try:
        values = [float(v) for v in lines[1:]]
    except ValueError as exc:
        raise JumpAnalysisError(f"{path} contains non-numeric samples") from exc

    if not values:
        raise JumpAnalysisError(f"{path} does not contain any samples")

    return signal_name, values


def find_stable_segment(
    forces: Sequence[float],
    sample_rate: float,
    window_duration: float = 0.5,
    cv_threshold: float = 0.02,
) -> Tuple[float, int, int]:
    """Locate the earliest low-variance window and return its mean and bounds."""

    window_size = max(1, int(round(window_duration * sample_rate)))
    best_start: Optional[int] = None
    weight_estimate: Optional[float] = None

    total_samples = len(forces)
    for start in range(0, total_samples - window_size + 1):
        window = forces[start : start + window_size]
        mean = sum(window) / window_size
        if mean <= 0:
            continue
        variance = sum((value - mean) ** 2 for value in window) / window_size
        std = math.sqrt(variance)
        cv = std / mean if mean else math.inf
        if cv <= cv_threshold:
            best_start = start
            weight_estimate = mean
            break

    if best_start is None:
        # Fall back to the first window if no stable period was found.
        best_start = 0
        window = forces[:window_size]
        weight_estimate = sum(window) / window_size

    return weight_estimate, best_start, best_start + window_size


def find_movement_start(
    forces: Sequence[float],
    weight: float,
    start_index: int,
    deviation_ratio: float,
    absolute_threshold: float,
) -> int:
    threshold = max(weight * deviation_ratio, absolute_threshold)
    total_samples = len(forces)
    for idx in range(start_index, total_samples):
        if abs(forces[idx] - weight) >= threshold:
            return idx
    raise JumpAnalysisError("Unable to detect movement start")


def _find_transition(
    forces: Sequence[float],
    start_index: int,
    threshold: float,
    sample_rate: float,
    condition: str,
    min_duration: float = 0.015,
) -> int:
    consecutive = max(1, int(round(sample_rate * min_duration)))
    total_samples = len(forces)
    limit = total_samples - consecutive + 1

    if condition == "below":
        comparator = lambda value: value < threshold
    else:
        comparator = lambda value: value > threshold

    for idx in range(start_index, limit):
        window = forces[idx : idx + consecutive]
        if all(comparator(value) for value in window):
            return idx

    raise JumpAnalysisError(f"Unable to detect {condition} threshold crossing")


def find_takeoff(
    forces: Sequence[float],
    start_index: int,
    weight: float,
    sample_rate: float,
    ratio: float,
    absolute_threshold: float,
) -> int:
    threshold = max(weight * ratio, absolute_threshold)
    return _find_transition(forces, start_index, threshold, sample_rate, "below")


def find_landing(
    forces: Sequence[float],
    start_index: int,
    weight: float,
    sample_rate: float,
    ratio: float,
    absolute_threshold: float,
) -> int:
    threshold = max(weight * ratio, absolute_threshold)
    return _find_transition(forces, start_index, threshold, sample_rate, "greater")


def detect_jump_phases(forces: Sequence[float], sample_rate: float) -> JumpPhases:
    weight, stable_start, stable_end = find_stable_segment(forces, sample_rate)
    mass = weight / G
    movement_start = find_movement_start(
        forces,
        weight,
        stable_end,
        deviation_ratio=0.05,
        absolute_threshold=20.0,
    )

    takeoff = find_takeoff(
        forces,
        movement_start,
        weight,
        sample_rate,
        ratio=0.05,
        absolute_threshold=20.0,
    )

    landing = find_landing(
        forces,
        takeoff,
        weight,
        sample_rate,
        ratio=0.05,
        absolute_threshold=20.0,
    )

    if landing <= takeoff:
        raise JumpAnalysisError("Landing detection failed")

    return JumpPhases(
        weight=weight,
        mass=mass,
        stable_start=stable_start,
        stable_end=stable_end,
        movement_start=movement_start,
        takeoff=takeoff,
        landing=landing,
    )


def integrate_takeoff_velocity(
    forces: Sequence[float], phases: JumpPhases, sample_rate: float
) -> float:
    dt = 1.0 / sample_rate
    acceleration = [
        (value - phases.weight) / phases.mass for value in forces
    ]
    start = phases.movement_start
    stop = phases.takeoff
    if stop <= start:
        return 0.0
    segment = acceleration[start : stop + 1]
    if len(segment) < 2:
        return 0.0
    integral = 0.0
    for first, second in zip(segment[:-1], segment[1:]):
        integral += (first + second) * 0.5 * dt
    return integral


def compute_jump_height(takeoff_velocity: float) -> float:
    return max(0.0, (takeoff_velocity ** 2) / (2 * G))


def summarize_jump(
    file_path: Path, signal_name: str, forces: Sequence[float], sample_rate: float
) -> JumpResult:
    phases = detect_jump_phases(forces, sample_rate)
    takeoff_velocity = integrate_takeoff_velocity(forces, phases, sample_rate)
    height = compute_jump_height(takeoff_velocity)

    return JumpResult(
        file=file_path,
        signal_name=signal_name,
        body_weight=phases.weight,
        mass=phases.mass,
        stable_start_s=phases.stable_start / sample_rate,
        stable_end_s=phases.stable_end / sample_rate,
        movement_start_s=phases.movement_start / sample_rate,
        takeoff_time_s=phases.takeoff / sample_rate,
        landing_time_s=phases.landing / sample_rate,
        flight_time_s=(phases.landing - phases.takeoff) / sample_rate,
        takeoff_velocity=takeoff_velocity,
        jump_height=height,
    )


def process_directory(directory: Path, sample_rate: float) -> List[JumpResult]:
    results: List[JumpResult] = []
    for path in sorted(directory.glob("*.txt")):
        try:
            signal_name, forces = read_force_signal(path)
            results.append(summarize_jump(path, signal_name, forces, sample_rate))
        except JumpAnalysisError as exc:
            print(f"[warning] {exc}")
    return results


def write_results(results: Sequence[JumpResult], output_path: Path) -> None:
    if not results:
        return

    fieldnames = [
        "file",
        "signal_name",
        "body_weight_N",
        "mass_kg",
        "stable_start_s",
        "stable_end_s",
        "movement_start_s",
        "takeoff_time_s",
        "landing_time_s",
        "flight_time_s",
        "takeoff_velocity_m_s",
        "jump_height_m",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "file": result.file.name,
                    "signal_name": result.signal_name,
                    "body_weight_N": f"{result.body_weight:.2f}",
                    "mass_kg": f"{result.mass:.3f}",
                    "stable_start_s": f"{result.stable_start_s:.4f}",
                    "stable_end_s": f"{result.stable_end_s:.4f}",
                    "movement_start_s": f"{result.movement_start_s:.4f}",
                    "takeoff_time_s": f"{result.takeoff_time_s:.4f}",
                    "landing_time_s": f"{result.landing_time_s:.4f}",
                    "flight_time_s": f"{result.flight_time_s:.4f}",
                    "takeoff_velocity_m_s": f"{result.takeoff_velocity:.4f}",
                    "jump_height_m": f"{result.jump_height:.4f}",
                }
            )


def print_summary(results: Sequence[JumpResult]) -> None:
    if not results:
        print("No jump files were processed.")
        return

    for result in results:
        print(f"\nFile: {result.file.name}")
        print(f"Signal name: {result.signal_name}")
        print(f"Estimated body weight: {result.body_weight:.2f} N")
        print(f"Estimated mass: {result.mass:.3f} kg")
        print(f"Stable phase: {result.stable_start_s:.3f}s – {result.stable_end_s:.3f}s")
        print(f"Movement onset: {result.movement_start_s:.3f}s")
        print(f"Take-off time: {result.takeoff_time_s:.3f}s")
        print(f"Landing time: {result.landing_time_s:.3f}s")
        print(f"Flight time: {result.flight_time_s:.3f}s")
        print(f"Take-off velocity: {result.takeoff_velocity:.3f} m/s")
        print(f"Jump height: {result.jump_height:.3f} m")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Folder containing .txt files exported from the force plate",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=600.0,
        help="Sampling frequency of the recordings (Hz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a CSV summary (defaults to <directory>/jump_results.csv)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    directory = args.directory
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")

    results = process_directory(directory, args.sample_rate)
    print_summary(results)

    if args.output is not None:
        output_path = args.output
    else:
        output_path = directory / "jump_results.csv"

    if results:
        write_results(results, output_path)
        print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
