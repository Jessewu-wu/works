#!/usr/bin/env python3
"""Process vertical jump force plate data stored in plain text files.

The script reads all ``.txt`` files in a given folder, extracts
information about the jump phases, and computes take-off velocity,
jump height, and flight time.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
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


def robust_mean(values: Sequence[float], trim_ratio: float = 0.1) -> float:
    """Return a trimmed mean for robustness against outliers."""

    data = list(values)
    if not data:
        return 0.0
    if len(data) < 4:
        return statistics.fmean(data)

    trim = int(len(data) * trim_ratio)
    if trim == 0:
        return statistics.fmean(data)

    sorted_data = sorted(data)
    trimmed = sorted_data[trim:-trim] or sorted_data
    return statistics.fmean(trimmed)


def robust_noise(values: Sequence[float]) -> float:
    """Estimate noise using the median absolute deviation."""

    data = list(values)
    if not data:
        return 0.0

    median = statistics.median(data)
    deviations = [abs(value - median) for value in data]
    mad = statistics.median(deviations)
    if mad == 0.0:
        return statistics.pstdev(data) or 0.0
    # Consistent estimator for Gaussian noise.
    return mad * 1.4826


def find_stable_segment(
    forces: Sequence[float],
    sample_rate: float,
    min_duration: float = 0.2,
    search_duration: float = 0.4,
    window_samples: int = 20,
    rel_stable: float = 0.005,
    rel_change: float = 0.015,
    k_noise: float = 1.5,
    k_change: float = 3.0,
) -> Tuple[float, int, int]:
    """Locate the initial quiet stance used to estimate body weight.

    The detector analyses back-to-back window means within an early baseline
    segment. Differences that stay below a relative fraction of the baseline
    load (``rel_stable``) or a multiple of the baseline noise (``k_noise``)
    extend the stance, whereas deviations above ``rel_change``/``k_change``
    terminate it. The returned ``stable_end`` index is exclusive, allowing it to
    serve directly as the boundary before movement onset. If no stable run is
    found, a trimmed mean over the earliest samples is used as a fallback.
    """

    total_samples = len(forces)
    if total_samples == 0:
        raise JumpAnalysisError("Signal too short to locate a stable stance")

    window_size = max(1, window_samples)
    min_samples = max(window_size, int(round(min_duration * sample_rate)))
    search_samples = min(
        total_samples, max(2 * window_size, int(round(search_duration * sample_rate)))
    )
    if search_samples < 2 * window_size:
        raise JumpAnalysisError("Not enough data for sliding window comparison")

    baseline_segment = forces[:search_samples]
    baseline_mean = abs(robust_mean(baseline_segment)) or max(
        1.0, abs(statistics.fmean(baseline_segment))
    )
    noise_floor = robust_noise(baseline_segment)
    if noise_floor <= 0.0:
        noise_floor = max(1e-6, abs(baseline_mean) * 1e-3)

    stable_threshold = max(rel_stable * baseline_mean, k_noise * noise_floor)
    change_threshold = max(rel_change * baseline_mean, k_change * noise_floor)

    prefix = [0.0]
    for value in forces:
        prefix.append(prefix[-1] + value)

    def window_mean(start: int) -> float:
        end = start + window_size
        return (prefix[end] - prefix[start]) / window_size

    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_length = 0
    run_start: Optional[int] = None
    run_end: Optional[int] = None

    for start in range(0, search_samples - 2 * window_size + 1, window_size):
        mean1 = window_mean(start)
        mean2 = window_mean(start + window_size)
        diff = abs(mean2 - mean1)

        if diff <= stable_threshold:
            if run_start is None:
                run_start = start
            run_end = start + 2 * window_size
        else:
            if run_start is not None and run_end is not None:
                run_length = run_end - run_start
                if run_length > best_length:
                    best_start, best_end = run_start, run_end
                    best_length = run_length
            run_start = None
            run_end = None
            if diff >= change_threshold and best_start is not None:
                break

    if run_start is not None and run_end is not None:
        run_length = run_end - run_start
        if run_length > best_length:
            best_start, best_end = run_start, run_end
            best_length = run_length

    if best_start is None or best_end is None:
        fallback_end = min(total_samples, max(min_samples, window_size))
        fallback_samples = forces[:fallback_end]
        fallback_weight = robust_mean(fallback_samples)
        return fallback_weight, 0, fallback_end

    stable_start = best_start
    stable_end = best_end
    current = stable_end - window_size

    while current + 2 * window_size <= total_samples:
        mean1 = window_mean(current)
        next_start = current + window_size
        mean2 = window_mean(next_start)
        diff = abs(mean2 - mean1)

        if diff <= change_threshold:
            stable_end = next_start + window_size
            current = next_start
        else:
            break

    if stable_end - stable_start < min_samples:
        fallback_end = min(total_samples, max(min_samples, window_size))
        fallback_samples = forces[:fallback_end]
        fallback_weight = robust_mean(fallback_samples)
        return fallback_weight, 0, fallback_end

    refined_weight = robust_mean(forces[stable_start:stable_end])

    return refined_weight, stable_start, stable_end


def find_movement_start(
    forces: Sequence[float],
    weight: float,
    noise_floor: float,
    start_index: int,
    sample_rate: float,
    rel_dev: float,
    k_dev: float,
    min_duration: float = 0.03,
) -> int:
    threshold = max(weight * rel_dev, noise_floor * k_dev)
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
    noise_floor: float,
    sample_rate: float,
    rel_ratio: float,
    k_event: float,
    min_duration: float = 0.015,
) -> int:
    threshold = max(weight * rel_ratio, noise_floor * k_event)
    return _find_transition(
        forces, start_index, threshold, sample_rate, "below", min_duration=min_duration
    )


def find_landing(
    forces: Sequence[float],
    start_index: int,
    weight: float,
    noise_floor: float,
    sample_rate: float,
    rel_ratio: float,
    k_event: float,
    min_duration: float = 0.015,
) -> int:
    threshold = max(weight * rel_ratio, noise_floor * k_event)
    return _find_transition(
        forces, start_index, threshold, sample_rate, "greater", min_duration=min_duration
    )


def detect_jump_phases(forces: Sequence[float], sample_rate: float) -> JumpPhases:
    weight, stable_start, stable_end = find_stable_segment(forces, sample_rate)
    mass = weight / G
    stable_segment = forces[stable_start:stable_end] or forces[: max(1, int(sample_rate * 0.1))]
    noise_floor = robust_noise(stable_segment)
    if noise_floor <= 0.0:
        noise_floor = max(1e-6, abs(weight) * 1e-3)
    movement_start = find_movement_start(
        forces,
        weight,
        noise_floor,
        stable_end,
        sample_rate,
        rel_dev=0.025,
        k_dev=4.0,
    )

    takeoff = find_takeoff(
        forces,
        movement_start,
        weight,
        noise_floor,
        sample_rate,
        rel_ratio=0.05,
        k_event=5.0,
    )

    landing = find_landing(
        forces,
        takeoff,
        weight,
        noise_floor,
        sample_rate,
        rel_ratio=0.05,
        k_event=5.0,
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
