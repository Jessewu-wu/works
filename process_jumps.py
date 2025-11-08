#!/usr/bin/env python3
"""Process vertical jump force plate data stored in plain text files.

The script reads all ``.txt`` files in a given folder, extracts
information about the jump phases, and computes take-off velocity,
jump height, and flight time.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

G = 9.80665  # gravitational acceleration (m/s^2)


@dataclass
class JumpResult:
    file: Path
    signal_name: str
    body_weight: float
    mass: float
    sample_rate: float
    stable_start_idx: int
    stable_end_idx: int
    movement_start_idx: int
    takeoff_idx: int
    landing_idx: int
    stable_samples: int
    movement_samples: int
    flight_samples: int
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
    cv_threshold: float = 0.015,
    extend_ratio: float = 0.02,
    extend_absolute: float = 15.0,
) -> Tuple[float, int, int]:
    """Locate the initial quiet stance used to estimate body weight.

    The function searches the signal for the first low-variance window and then
    extends it forwards while the force stays close to the estimated body
    weight.  The returned ``stable_end`` index is exclusive so it can be used
    directly as the start of the movement phase.
    """

    window_size = max(1, int(round(window_duration * sample_rate)))
    total_samples = len(forces)
    if total_samples < window_size:
        raise JumpAnalysisError("Signal too short to locate a stable stance")

    best_start: Optional[int] = None
    best_cv = math.inf
    weight_estimate: Optional[float] = None

    search_limit = total_samples - window_size + 1
    for start in range(search_limit):
        window = forces[start : start + window_size]
        mean = sum(window) / window_size
        if mean <= 0:
            continue
        variance = sum((value - mean) ** 2 for value in window) / window_size
        std = math.sqrt(variance)
        cv = std / mean if mean else math.inf
        if cv < best_cv:
            best_cv = cv
            best_start = start
            weight_estimate = mean
        if cv <= cv_threshold:
            # Accept the first low-variance window we encounter.
            break

    if best_start is None or weight_estimate is None:
        raise JumpAnalysisError("Unable to determine stable stance")

    tolerance = max(weight_estimate * extend_ratio, extend_absolute)
    stable_end = best_start + window_size
    while stable_end < total_samples and abs(forces[stable_end] - weight_estimate) <= tolerance:
        stable_end += 1

    return weight_estimate, best_start, stable_end


def find_movement_start(
    forces: Sequence[float],
    weight: float,
    start_index: int,
    sample_rate: float,
    deviation_ratio: float,
    absolute_threshold: float,
    min_duration: float = 0.03,
) -> int:
    threshold = max(weight * deviation_ratio, absolute_threshold)
    consecutive = max(1, int(round(sample_rate * min_duration)))
    total_samples = len(forces)
    limit = total_samples - consecutive + 1
    for idx in range(start_index, limit):
        window = forces[idx : idx + consecutive]
        if all(abs(value - weight) >= threshold for value in window):
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
        sample_rate,
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
    start = phases.stable_end
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

    stable_samples = max(0, phases.stable_end - phases.stable_start)
    movement_samples = max(0, phases.takeoff - phases.movement_start)
    flight_samples = max(0, phases.landing - phases.takeoff)
    stable_end_idx = phases.stable_end - 1 if stable_samples else phases.stable_start
    return JumpResult(
        file=file_path,
        signal_name=signal_name,
        body_weight=phases.weight,
        mass=phases.mass,
        sample_rate=sample_rate,
        stable_start_idx=phases.stable_start,
        stable_end_idx=stable_end_idx,
        movement_start_idx=phases.movement_start,
        takeoff_idx=phases.takeoff,
        landing_idx=phases.landing,
        stable_samples=stable_samples,
        movement_samples=movement_samples,
        flight_samples=flight_samples,
        stable_start_s=phases.stable_start / sample_rate,
        stable_end_s=phases.stable_end / sample_rate,
        movement_start_s=phases.movement_start / sample_rate,
        takeoff_time_s=phases.takeoff / sample_rate,
        landing_time_s=phases.landing / sample_rate,
        flight_time_s=flight_samples / sample_rate,
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
        "sample_rate_Hz",
        "stable_start_idx",
        "stable_end_idx",
        "stable_samples",
        "movement_start_idx",
        "takeoff_idx",
        "movement_samples",
        "landing_idx",
        "flight_samples",
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
                    "sample_rate_Hz": f"{result.sample_rate:.1f}",
                    "stable_start_idx": result.stable_start_idx,
                    "stable_end_idx": result.stable_end_idx,
                    "stable_samples": result.stable_samples,
                    "movement_start_idx": result.movement_start_idx,
                    "takeoff_idx": result.takeoff_idx,
                    "movement_samples": result.movement_samples,
                    "landing_idx": result.landing_idx,
                    "flight_samples": result.flight_samples,
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


def _format_phase_line(name: str, start_idx: int, samples: int, sample_rate: float) -> str:
    if samples <= 0:
        return f"{name}：未能识别"
    end_idx = start_idx + samples - 1
    duration = samples / sample_rate
    return f"{name}：样本{start_idx}–{end_idx}（共{samples}点，{duration:.3f}s）"


def format_summary(results: Sequence[JumpResult]) -> str:
    if not results:
        return "未找到可用的纵跳数据，请确认文件夹内包含 .txt 数据文件。"

    lines: List[str] = []
    for result in results:
        takeoff_phase_samples = result.flight_samples
        flight_line = _format_phase_line(
            "滞空阶段",
            result.takeoff_idx,
            takeoff_phase_samples,
            result.sample_rate,
        )
        lines.extend(
            [
                f"文件：{result.file.name}",
                f"信号名称：{result.signal_name}",
                f"体重估计：{result.body_weight:.2f} N (≈ {result.mass:.3f} kg)",
                f"稳定区间：{result.stable_start_s:.3f}s – {result.stable_end_s:.3f}s",
                f"动作起始：{result.movement_start_s:.3f}s",
                f"起跳时刻：{result.takeoff_time_s:.3f}s",
                f"落地时刻：{result.landing_time_s:.3f}s",
                f"滞空时间：{result.flight_time_s:.3f}s",
                _format_phase_line(
                    "稳定阶段",
                    result.stable_start_idx,
                    result.stable_samples,
                    result.sample_rate,
                ),
                _format_phase_line(
                    "动作阶段",
                    result.movement_start_idx,
                    result.movement_samples,
                    result.sample_rate,
                ),
                flight_line,
                f"起跳速度：{result.takeoff_velocity:.3f} m/s",
                f"纵跳高度：{result.jump_height:.3f} m",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def print_summary(results: Sequence[JumpResult]) -> None:
    print(format_summary(results))


def run_analysis(
    directory: Path,
    sample_rate: float,
    output_path: Optional[Path] = None,
) -> Tuple[Sequence[JumpResult], Optional[Path]]:
    results = process_directory(directory, sample_rate)

    if output_path is None:
        output_path = directory / "jump_results.csv"

    if results:
        write_results(results, output_path)
        return results, output_path

    return results, None


def can_use_gui() -> bool:
    if os.environ.get("FORCE_CLI") == "1":
        return False

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False

    try:
        import tkinter  # noqa: F401
    except Exception:
        return False

    return True


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()

    directory = filedialog.askdirectory(title="请选择包含纵跳 TXT 文件的文件夹")
    if not directory:
        messagebox.showinfo("提示", "未选择文件夹，程序已退出。")
        return

    directory_path = Path(directory)
    try:
        results, saved_path = run_analysis(directory_path, sample_rate=600.0)
        summary = format_summary(results)

        if saved_path is not None:
            summary += f"\n\n结果已保存至：{saved_path}"

        messagebox.showinfo("纵跳分析结果", summary)
    except Exception as exc:
        messagebox.showerror("纵跳分析失败", str(exc))


def should_use_gui(argv: Optional[Sequence[str]]) -> bool:
    arguments = list(argv or [])
    return not arguments and can_use_gui()


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
    parsed_args = parse_args(argv)
    directory = parsed_args.directory
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")

    results, saved_path = run_analysis(
        directory, parsed_args.sample_rate, parsed_args.output
    )

    print_summary(results)

    if saved_path is not None:
        print(f"\nSummary saved to {saved_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if should_use_gui(argv):
        run_gui()
    else:
        main(argv)
