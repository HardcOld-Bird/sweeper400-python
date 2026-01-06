"""
手动测试Caliber类的新功能

这个脚本用于手动测试Caliber类的所有新功能，包括：
1. 新的重复测量机制
2. SweepData格式存储
3. 数据处理流程
4. 极坐标绘图
5. 双模式绘图
"""

import sys
from pathlib import Path

# 确保导入src目录下的sweeper400
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sweeper400.analyze import init_sampling_info, init_sine_args
from sweeper400.use import Caliber

# 配置参数
AI_CHANNEL = "PXI2Slot2/ai0"
AO_CHANNELS = (
    "PXI2Slot2/ao0",
    "PXI2Slot2/ao1",
    "PXI2Slot3/ao0",
    "PXI2Slot3/ao1",
    "PXI3Slot2/ao0",
    "PXI3Slot2/ao1",
    "PXI3Slot3/ao0",
    "PXI3Slot3/ao1",
)

# 创建采样信息和正弦波参数
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
sine_args = init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

# 创建输出目录
output_dir = Path("D:/EveryoneDownloaded/caliber_test")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("开始Caliber类新功能测试")
print("=" * 80)

# 创建Caliber对象
print("\n步骤1: 创建Caliber对象...")
caliber = Caliber(
    ai_channel=AI_CHANNEL,
    ao_channels=AO_CHANNELS,
    sampling_info=sampling_info,
    sine_args=sine_args,
    settle_time=0.5,
)
print("Caliber对象创建成功")

# 执行校准
print("\n步骤2: 执行校准...")
print("  启动次数: 2")
print("  每次启动chunk数: 4")
caliber.calibrate(
    starts_num=2,
    chunks_per_start=4,
    apply_filter=True,
    lowcut=100.0,
    highcut=20000.0,
)
print("校准完成")

# 保存校准结果
print("\n步骤3: 保存校准结果...")
calibration_file = output_dir / "calibration_results.pkl"
caliber.save_calib_data(calibration_file)
print(f"校准结果已保存到: {calibration_file}")

# 保存SweepData
print("\n步骤4: 保存SweepData...")
sweep_data_file = output_dir / "result_raw_sweep_data.pkl"
caliber.save_sweep_data(sweep_data_file)
print(f"SweepData已保存到: {sweep_data_file}")

# 绘制传递函数图（平均模式）
print("\n步骤5: 绘制传递函数图（平均模式）...")
plot_file_avg = output_dir / "transfer_functions_averaged.png"
caliber.plot_transfer_functions(mode="averaged", save_path=plot_file_avg)
print(f"传递函数图（平均模式）已保存到: {plot_file_avg}")

# 绘制传递函数图（详细模式）
print("\n步骤6: 绘制传递函数图（详细模式）...")
plot_file_det = output_dir / "transfer_functions_detailed.png"
caliber.plot_transfer_functions(mode="detailed", save_path=plot_file_det)
print(f"传递函数图（详细模式）已保存到: {plot_file_det}")

# 打印最终结果
print("\n步骤7: 打印最终结果...")
print("\n最终传递函数:")
if caliber.result_final_tf_list is not None:
    for channel_idx, tf in caliber.result_final_tf_list.items():
        import numpy as np

        print(
            f"  通道 {channel_idx} ({AO_CHANNELS[channel_idx]}): "
            f"幅值={np.abs(tf):.6f}, 相位={np.angle(tf):.6f}rad"
        )

print("\n" + "=" * 80)
print("Caliber类新功能测试完成")
print("=" * 80)
print(f"\n所有文件已保存到: {output_dir}")
