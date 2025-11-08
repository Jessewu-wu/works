"""基于加速度积分法的纵跳高度批量计算脚本。

该脚本自动扫描目标文件夹内的 txt / csv / Excel 文件，识别竖直地
面反作用力（GRF）序列，先通过稳定站立时段估算体重，再使用加速度
积分与速度法（v^2 / 2g）两种方式计算纵跳高度，并输出详细结果。

运行示例::

    python jump_height_gui.py --folder ./data --sample-rate 600 --export result.xlsx

脚本默认会把每个文件的处理结果打印在控制台，并在必要时提供错误
说明。若指定 ``--export``，结果会额外保存为 Excel。
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GRAVITY = 9.81


@dataclass
class JumpMetrics:
    """保存单个文件的纵跳分析结果。"""

    source: str
    integral_height: Optional[float] = None
    velocity_height: Optional[float] = None
    takeoff_velocity: Optional[float] = None
    note: str = ""


def read_grf_series(path: str) -> np.ndarray:
    """从 txt/csv/excel 文件中提取 GRF 数组。

    读取规则：
        * Excel: 默认读取首个工作表，挑选第一个包含有效数字的列。
        * txt/csv: 优先使用 pandas 解析；若失败则使用正则提取所有浮点数。
    """

    ext = os.path.splitext(path)[1].lower()
    if ext in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=0)
    elif ext in {".csv", ".txt", ".tsv"}:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = None
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    if df is not None:
        numeric_cols = []
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                numeric_cols.append(series.to_numpy(dtype=float))
        if numeric_cols:
            concatenated = np.concatenate(numeric_cols)
            if concatenated.size:
                return concatenated

    # pandas 解析失败时回退到正则提取
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    values = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if not values:
        raise ValueError("文件中未找到可用的 GRF 数据")
    return np.asarray(values, dtype=float)


def _find_stable_indices(
    grf: np.ndarray,
    sample_rate: float,
    window_sec: float,
    std_threshold: float,
) -> Optional[np.ndarray]:
    window = max(int(window_sec * sample_rate), 1)
    rolling = pd.Series(grf).rolling(window=window, center=True).std()
    stable = rolling[rolling < std_threshold].dropna().index.to_numpy(dtype=int)
    if stable.size == 0:
        return None

    groups: List[Sequence[int]] = []
    for _, group in groupby(enumerate(stable), key=lambda x: x[0] - x[1]):
        indices = [idx for _, idx in group]
        if len(indices) >= window:
            groups.append(indices)

    if not groups:
        return None
    return np.asarray(max(groups, key=len), dtype=int)


def integrate_jump(
    grf: np.ndarray,
    sample_rate: float,
    stability_window: float = 0.5,
    stability_std: float = 3.0,
    flight_force_ratio: float = 0.1,
    min_flight_time: float = 0.05,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """执行纵跳分析主流程。"""

    if grf.size < sample_rate:
        return None, None, None, "数据长度不足"

    stable_idx = _find_stable_indices(grf, sample_rate, stability_window, stability_std)
    if stable_idx is None:
        return None, None, None, "未检测到稳定站立段"

    bw = float(np.mean(grf[stable_idx]))
    mass = bw / GRAVITY if bw > 0 else 0.0
    if mass <= 0:
        return None, None, None, "体重计算失败"

    # 计算净加速度并寻找滞空段
    acc = (grf - bw) / mass
    flight_threshold = max(bw * flight_force_ratio, 30.0)
    flight_indices = np.where(grf < flight_threshold)[0]
    if flight_indices.size == 0:
        return None, None, None, "未检测到离地段"

    min_flight_samples = int(min_flight_time * sample_rate)
    groups: List[np.ndarray] = []
    for _, group in groupby(enumerate(flight_indices), key=lambda x: x[0] - x[1]):
        indices = np.asarray([idx for _, idx in group], dtype=int)
        if indices.size >= min_flight_samples:
            groups.append(indices)
    if not groups:
        return None, None, None, "滞空段持续时间不足"

    flight = groups[0]
    start = max(flight[0] - int(0.4 * sample_rate), 0)
    end = min(flight[-1] + int(0.2 * sample_rate), grf.size - 1)

    acc_segment = acc[start : end + 1]
    dt = 1.0 / sample_rate
    vel = np.cumsum(acc_segment) * dt
    drift = np.linspace(0, vel[-1], vel.size)
    vel_corrected = vel - drift
    disp = np.cumsum(vel_corrected) * dt

    height_integral = float(np.max(disp)) if disp.size else None

    takeoff_idx = flight[0] - start
    takeoff_idx = max(min(takeoff_idx, vel_corrected.size - 1), 0)
    v_takeoff = float(vel_corrected[takeoff_idx]) if vel_corrected.size else None
    height_velocity = None
    if v_takeoff is not None:
        height_velocity = max(v_takeoff ** 2 / (2 * GRAVITY), 0.0)

    return height_integral, height_velocity, v_takeoff, ""


def analyze_file(path: str, sample_rate: float) -> JumpMetrics:
    try:
        grf = read_grf_series(path)
    except Exception as exc:  # noqa: BLE001
        return JumpMetrics(source=os.path.basename(path), note=f"读取失败: {exc}")

    h_int, h_vel, v_takeoff, note = integrate_jump(grf, sample_rate)
    return JumpMetrics(
        source=os.path.basename(path),
        integral_height=None if h_int is None else round(h_int, 4),
        velocity_height=None if h_vel is None else round(h_vel, 4),
        takeoff_velocity=None if v_takeoff is None else round(v_takeoff, 3),
        note=note,
    )


def discover_files(folder: str) -> List[str]:
    supported = {".txt", ".csv", ".tsv", ".xls", ".xlsx", ".xlsm"}
    files = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in supported:
            files.append(path)
    return files


def build_dataframe(results: Iterable[JumpMetrics]) -> pd.DataFrame:
    data = [
        {
            "文件": r.source,
            "积分高度 (m)": r.integral_height,
            "v²/2g 高度 (m)": r.velocity_height,
            "起跳速度 (m/s)": r.takeoff_velocity,
            "备注": r.note,
        }
        for r in results
    ]
    return pd.DataFrame(data)


def run(folder: str, sample_rate: float, export_path: Optional[str]) -> pd.DataFrame:
    files = discover_files(folder)
    if not files:
        raise FileNotFoundError("指定文件夹中未找到可处理的文件")

    results = [analyze_file(path, sample_rate) for path in files]
    df = build_dataframe(results)

    print("分析结果：")
    print(df.to_string(index=False))

    if export_path:
        df.to_excel(export_path, index=False)
        print(f"\n结果已保存到: {export_path}")
    return df


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量计算纵跳高度")
    parser.add_argument("--folder", required=True, help="包含 GRF 文件的文件夹路径")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=600.0,
        help="采样频率 (Hz)，默认 600",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="可选的 Excel 输出路径",
    )
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    df = run(args.folder, args.sample_rate, args.export)
    error_rows = df[df["备注"] != ""]
    if not error_rows.empty:
        print("\n以下文件在处理过程中出现问题：")
        print(error_rows.to_string(index=False))


if __name__ == "__main__":
    main()
