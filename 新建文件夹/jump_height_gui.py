"""拖拽式图形界面程序：批量分析txt中的GRF跳跃高度。

该脚本提供一个基于 Tkinter 的桌面界面，支持将包含 GRF txt 文件的
文件夹拖入窗口或通过文件选择对话框加载，并批量计算跳跃高度。
结果支持两种高度估算方法（位移积分与 v²/2g），并将起跳速度列入
结果表格。同时提供结果导出、图表展示与错误弹窗机制。
"""

from __future__ import annotations

import os
import re
import traceback
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:  # 可选的拖拽支持
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND_AVAILABLE = True
except Exception:  # pragma: no cover - tkinterdnd2 为可选依赖
    TkinterDnD = tk.Tk  # type: ignore
    DND_FILES = "DND_Files"  # type: ignore
    _DND_AVAILABLE = False


@dataclass
class JumpResult:
    """保存每个文件的计算结果，统一用于表格展示与导出。"""

    file: str
    integral_height: Optional[float] = None
    velocity_height: Optional[float] = None
    takeoff_velocity: Optional[float] = None
    note: str = ""

    def as_row(self) -> List[str]:
        def _fmt(value: Optional[float], precision: int) -> str:
            if value is None:
                return ""
            return f"{value:.{precision}f}"

        return [
            self.file,
            _fmt(self.integral_height, 4),
            _fmt(self.velocity_height, 4),
            _fmt(self.takeoff_velocity, 3),
            self.note,
        ]


@dataclass
class PlotData:
    """存储绘图所需的关键信息。"""

    time: np.ndarray
    grf: np.ndarray
    displacement_time: np.ndarray
    displacement: np.ndarray
    takeoff_time: Optional[float]
    landing_time: Optional[float]
    peak_disp_time: Optional[float]
    peak_disp_value: Optional[float]


class JumpAnalyzerApp(TkinterDnD):  # type: ignore[misc]
    """基于 Tkinter 的 GRF 跳跃高度批量分析器。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("GRF 跳跃高度批量分析")
        self.geometry("980x640")
        self.minsize(900, 580)
        self.configure(bg="#f8f9fb")

        # 状态与缓存
        self.results: List[JumpResult] = []
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.analysis_cache: Dict[str, PlotData] = {}
        self.current_folder: Optional[str] = None
        self.status_var = tk.StringVar(value="请拖拽文件夹或点击按钮选择……")

        self._build_widgets()

        if _DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

    # ----------------- UI 构建 -----------------
    def _build_widgets(self) -> None:
        header = tk.Label(
            self,
            text="📁 GRF 跳跃高度批量分析",
            font=("Microsoft YaHei", 18, "bold"),
            bg="#f8f9fb",
        )
        header.pack(pady=(18, 4))

        sub_header = tk.Label(
            self,
            text="支持拖拽文件夹或选择文件夹，自动计算跳跃高度（积分法 + v²/2g）",
            font=("Microsoft YaHei", 11),
            bg="#f8f9fb",
            fg="#586069",
        )
        sub_header.pack(pady=(0, 16))

        self.drop_label = tk.Label(
            self,
            text=(
                "将包含 GRF txt 文件的文件夹拖拽到此处\n"
                "或点击下方“选择文件夹”按钮"
            ),
            relief="ridge",
            borderwidth=2,
            width=50,
            height=4,
            bg="#ffffff",
            font=("Microsoft YaHei", 11),
        )
        self.drop_label.pack(pady=(0, 12))

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=(0, 12))

        select_btn = ttk.Button(
            button_frame, text="选择文件夹", command=self._choose_folder
        )
        select_btn.grid(row=0, column=0, padx=6)

        clear_btn = ttk.Button(
            button_frame, text="清除结果", command=self._clear_results
        )
        clear_btn.grid(row=0, column=1, padx=6)

        save_btn = ttk.Button(
            button_frame, text="导出结果", command=self._save_results
        )
        save_btn.grid(row=0, column=2, padx=6)

        status_label = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 10),
            bg="#f8f9fb",
            fg="#2f363d",
        )
        status_label.pack(pady=(0, 10))

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        tree_frame = ttk.Frame(main_pane)
        self._build_tree(tree_frame)
        main_pane.add(tree_frame, weight=2)

        plot_frame = ttk.Frame(main_pane)
        self._build_plot(plot_frame)
        main_pane.add(plot_frame, weight=3)

    def _build_tree(self, parent: ttk.Frame) -> None:
        columns = ("文件", "积分高度 (m)", "v²/2g高度 (m)", "起跳速度 (m/s)", "备注")
        self.tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            height=16,
        )

        for col in columns:
            anchor = tk.W if col == "文件" else tk.CENTER
            width = 220 if col == "文件" else 120
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=anchor, width=width, stretch=True)

        y_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_select_result)

    def _build_plot(self, parent: ttk.Frame) -> None:
        self.figure: Figure = Figure(figsize=(6, 4), dpi=100)
        self.ax_grf = self.figure.add_subplot(211)
        self.ax_disp = self.figure.add_subplot(212, sharex=self.ax_grf)
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._reset_plot()

    # ----------------- 事件处理 -----------------
    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="请选择包含 GRF txt 文件的文件夹")
        if folder:
            self._analyze_folder(folder)

    def _on_drop(self, event: tk.Event) -> None:  # type: ignore[override]
        path = event.data.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        folder = path.strip()
        if folder:
            self._analyze_folder(folder)

    def _on_select_result(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        values = self.tree.item(item_id, "values")
        if not values:
            return
        filename = values[0]
        plot_data = self.analysis_cache.get(filename)
        if plot_data:
            self._update_plot(plot_data)
        else:
            self._reset_plot()

    # ----------------- 数据处理 -----------------
    def _analyze_folder(self, folder_path: str) -> None:
        if not os.path.isdir(folder_path):
            messagebox.showerror("路径无效", "请选择有效的文件夹。")
            return

        txt_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
        )
        if not txt_files:
            messagebox.showinfo("提示", "所选文件夹中未找到 txt 文件。")
            return

        self.status_var.set("正在分析，请稍候……")
        self.update_idletasks()

        dt = 1 / 600
        new_results: List[JumpResult] = []
        new_cache: Dict[str, PlotData] = {}

        for filename in txt_files:
            filepath = os.path.join(folder_path, filename)
            try:
                result, plot_data = self._process_file(filepath, dt)
                new_results.append(result)
                if plot_data is not None:
                    new_cache[result.file] = plot_data
            except Exception as exc:  # 捕获单个文件的错误
                new_results.append(
                    JumpResult(file=filename, note=f"处理失败：{exc}")
                )

        self.results = new_results
        self.analysis_cache = new_cache
        self.current_folder = folder_path
        self.results_df = pd.DataFrame(
            [
                {
                    "文件": r.file,
                    "积分高度 (m)": r.integral_height,
                    "v²/2g高度 (m)": r.velocity_height,
                    "起跳速度 (m/s)": r.takeoff_velocity,
                    "备注": r.note,
                }
                for r in self.results
            ]
        )

        self._refresh_tree()
        self._reset_plot()

        success_count = sum(1 for r in self.results if not r.note)
        self.status_var.set(
            f"分析完成，共处理 {len(self.results)} 个文件，其中 {success_count} 个成功。"
        )
        messagebox.showinfo(
            "分析完成",
            f"分析完成！共处理 {len(self.results)} 个文件，成功 {success_count} 个。",
        )

    def _process_file(
        self, filepath: str, dt: float
    ) -> Tuple[JumpResult, Optional[PlotData]]:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as handle:
            raw = handle.read()

        grf_values = [float(x) for x in re.findall(r"-?\d+\.\d+", raw)]
        if not grf_values:
            raise ValueError("文件中未提取到有效的 GRF 数值。")

        grf = np.array(grf_values)
        std_series = pd.Series(grf).rolling(window=30, center=True).std()
        stable_indices = std_series[std_series < 3].dropna().index.to_list()

        stable_groups: List[List[int]] = []
        for _, group in groupby(enumerate(stable_indices), lambda x: x[0] - x[1]):
            indices = list(map(itemgetter(1), group))
            if len(indices) >= 30:
                stable_groups.append(indices)

        if not stable_groups:
            return (
                JumpResult(file=filename, note="未找到稳定站立段"),
                None,
            )

        longest_stable = max(stable_groups, key=len)
        body_weight = float(np.mean(grf[longest_stable]))
        mass = body_weight / 9.81

        acc = (grf - body_weight) / mass
        air_indices = np.where(grf < 50)[0]
        if air_indices.size < 2:
            return (
                JumpResult(file=filename, note="未检测到滞空"),
                None,
            )

        start_idx = max(0, int(air_indices[0]) - 90)
        end_idx = min(len(acc), int(air_indices[-1]) + 60)
        if end_idx <= start_idx:
            return (
                JumpResult(file=filename, note="滞空段范围异常"),
                None,
            )

        acc_jump = acc[start_idx:end_idx]
        if acc_jump.size == 0:
            return (
                JumpResult(file=filename, note="滞空段数据为空"),
                None,
            )

        vel = np.cumsum(acc_jump) * dt
        drift = np.linspace(0, vel[-1], len(vel)) if vel.size else np.array([])
        vel_corr = vel - drift
        disp = np.cumsum(vel_corr) * dt if vel_corr.size else np.array([])

        h_integral = float(np.max(disp)) if disp.size else None
        takeoff_offset = int(air_indices[0] - start_idx)
        if takeoff_offset >= 2:
            takeoff_offset -= 2
        takeoff_velocity = (
            float(vel_corr[takeoff_offset])
            if 0 <= takeoff_offset < len(vel_corr)
            else None
        )
        h_v2 = (
            (takeoff_velocity ** 2) / (2 * 9.81)
            if takeoff_velocity is not None
            else None
        )

        time = np.arange(len(grf)) * dt
        disp_time = np.arange(start_idx, end_idx) * dt
        peak_index = int(np.argmax(disp)) if disp.size else None
        peak_time = disp_time[peak_index] if peak_index is not None else None
        peak_value = float(disp[peak_index]) if peak_index is not None else None

        plot_data = PlotData(
            time=time,
            grf=grf,
            displacement_time=disp_time,
            displacement=disp,
            takeoff_time=float(time[air_indices[0]]) if air_indices.size else None,
            landing_time=float(time[air_indices[-1]]) if air_indices.size else None,
            peak_disp_time=peak_time,
            peak_disp_value=peak_value,
        )

        result = JumpResult(
            file=filename,
            integral_height=h_integral,
            velocity_height=h_v2,
            takeoff_velocity=takeoff_velocity,
        )
        return result, plot_data

    # ----------------- 工具方法 -----------------
    def _refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for result in self.results:
            self.tree.insert("", tk.END, values=result.as_row())

    def _save_results(self) -> None:
        if self.results_df.empty:
            messagebox.showinfo("提示", "暂无可导出的结果。")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            title="保存结果为 Excel 文件",
        )
        if not save_path:
            return
        try:
            self.results_df.to_excel(save_path, index=False)
            messagebox.showinfo("导出完成", f"结果已保存至：\n{save_path}")
        except Exception as exc:
            messagebox.showerror("导出失败", f"导出 Excel 失败：{exc}")

    def _clear_results(self) -> None:
        self.results.clear()
        self.analysis_cache.clear()
        self.results_df = pd.DataFrame()
        self.current_folder = None
        self._refresh_tree()
        self._reset_plot()
        self.status_var.set("结果已清空。")

    def _reset_plot(self) -> None:
        self.ax_grf.clear()
        self.ax_disp.clear()
        self.ax_grf.set_title("垂直地面反作用力 (GRF)")
        self.ax_grf.set_ylabel("Force (N)")
        self.ax_disp.set_title("位移（积分结果）")
        self.ax_disp.set_ylabel("Displacement (m)")
        self.ax_disp.set_xlabel("Time (s)")
        self.ax_grf.grid(True, linestyle="--", alpha=0.3)
        self.ax_disp.grid(True, linestyle="--", alpha=0.3)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _update_plot(self, data: PlotData) -> None:
        self.ax_grf.clear()
        self.ax_disp.clear()

        self.ax_grf.plot(data.time, data.grf, color="#0070c0", label="GRF")
        if data.takeoff_time is not None:
            self.ax_grf.axvline(data.takeoff_time, color="#ff7f0e", linestyle="--", label="起跳")
        if data.landing_time is not None:
            self.ax_grf.axvline(data.landing_time, color="#2ca02c", linestyle="--", label="落地")
        self.ax_grf.set_ylabel("Force (N)")
        self.ax_grf.set_title("垂直地面反作用力 (GRF)")
        self.ax_grf.legend(loc="upper right")
        self.ax_grf.grid(True, linestyle="--", alpha=0.3)

        self.ax_disp.plot(
            data.displacement_time,
            data.displacement,
            color="#d62728",
            label="位移",
        )
        if data.peak_disp_time is not None and data.peak_disp_value is not None:
            peak_value = data.peak_disp_value
            self.ax_disp.axvline(
                data.peak_disp_time,
                color="#9467bd",
                linestyle=":",
                label="峰值位移",
            )
            self.ax_disp.annotate(
                f"最大位移: {peak_value:.4f} m",
                xy=(data.peak_disp_time, peak_value),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=9,
                color="#9467bd",
            )
        self.ax_disp.set_ylabel("Displacement (m)")
        self.ax_disp.set_xlabel("Time (s)")
        self.ax_disp.set_title("位移（积分结果）")
        self.ax_disp.legend(loc="upper right")
        self.ax_disp.grid(True, linestyle="--", alpha=0.3)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()


def main() -> None:
    app = JumpAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # 顶层异常，使用弹窗提示
        try:
            fallback_root = tk.Tk()
            fallback_root.withdraw()
            messagebox.showerror(
                "程序错误",
                f"发生未捕获的错误：\n{exc}\n\n详细信息：\n{traceback.format_exc()}",
            )
            fallback_root.destroy()
        except Exception:
            print("程序错误：", exc)
            print(traceback.format_exc())
