# 📁 拖拽式图形界面程序：批量分析txt中的GRF跳跃高度（双方法 + 报错弹窗）
# 使用 tkinter + filedialog + pandas + matplotlib 实现GUI界面

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import traceback
from itertools import groupby
from operator import itemgetter

# ========== 主程序入口 ==========
def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="请选择包含GRF跳跃txt文件的文件夹")

    if not folder_path:
        messagebox.showinfo("取消", "未选择文件夹，程序退出。")
        return

    results = []
    dt = 1 / 600  # 采样间隔（秒）

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()

            grf_values = [float(x) for x in re.findall(r"-?\d+\.\d+", raw)]
            GRF = np.array(grf_values)

            # ===== 自动寻找稳定站立段估算体重 =====
            std = pd.Series(GRF).rolling(window=30, center=True).std()
            stable_indices = std[std < 3].index

            stable_groups = []
            for k, g in groupby(enumerate(stable_indices), lambda x: x[0] - x[1]):
                group = list(map(itemgetter(1), g))
                if len(group) >= 30:  # 至少 0.5 秒稳定（600Hz）
                    stable_groups.append(group)

            if not stable_groups:
                results.append({"文件": filename, "跳跃高度": "未找到稳定站立段"})
                continue

            longest_stable = max(stable_groups, key=len)
            BW = np.mean(GRF[longest_stable])
            mass = BW / 9.81

            # ===== 计算加速度、寻找滞空段 =====
            acc = (GRF - BW) / mass
            air_idx = np.where(GRF < 50)[0]
            if len(air_idx) < 2:
                results.append({"文件": filename, "跳跃高度": "未检测到滞空"})
                continue

            start_idx = max(0, air_idx[0] - 90)
            end_idx = air_idx[-1] + 60
            acc_jump = acc[start_idx:end_idx]

            # ===== 积分 + 去漂移 + 再积分 =====
            vel = np.cumsum(acc_jump) * dt
            drift = np.linspace(0, vel[-1], len(vel))
            vel_corr = vel - drift
            disp = np.cumsum(vel_corr) * dt

            h_integral = np.max(disp)
            takeoff_index = air_idx[0] - start_idx - 2 if air_idx[0] - start_idx >= 2 else air_idx[0] - start_idx
            v_takeoff = vel_corr[takeoff_index] if 0 <= takeoff_index < len(vel_corr) else 0
            h_v2 = v_takeoff ** 2 / (2 * 9.81)

            results.append({
                "文件": filename,
                "积分高度 (m)": round(h_integral, 4),
                "v²/2g高度 (m)": round(h_v2, 4),
                "起跳速度 (m/s)": round(v_takeoff, 3)
            })

        except Exception as e:
            results.append({"文件": filename, "跳跃高度": f"处理失败: {str(e)}"})

    df = pd.DataFrame(results)
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", title="保存结果为Excel文件", filetypes=[("Excel文件", "*.xlsx")])
    if save_path:
        df.to_excel(save_path, index=False)
        messagebox.showinfo("完成", f"分析完成，结果已保存到：\n{save_path}")
    else:
        messagebox.showinfo("提示", "结果未保存。")

# ==== 启动主程序，并加上错误弹窗机制 ====
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        messagebox.showerror("程序错误", f"发生错误：\n{str(e)}\n\n详细：\n{traceback.format_exc()}")
