# sweeper400

协同控制NI数据采集卡和步进电机的自动化测量包

## 简介

sweeper400 是一个用于声学扫场测量的 Python 包，能够协同控制 NI 数据采集卡和步进电机，实现空间中不同位置的自动化信号采集和处理。

## 主要功能

- **自动化扫场测量**：在预定义的点阵中自动移动并采集声场信号
- **硬件协同控制**：同时控制 NI 数据采集卡（nidaqmx）和步进电机（MT_API.dll）
- **信号处理**：对采集的信号进行处理，获取空间分布信息
- **线程化设计**：后台执行测量任务，支持实时监控和中断

## 包结构

- **measure**：NI 数据采集相关功能
- **move**：步进电机控制相关功能
- **analyze**：信号和数据处理相关功能
- **use**：协同调用其他子包的专用对象
- **gui**：基于 flet 的图形用户界面（待定）

## 系统要求

- Python >= 3.12
- Windows 系统（用于步进电机控制）
- NI 数据采集硬件和相应驱动

## 依赖项

- nidaqmx-python
- numpy
- scipy
- matplotlib

## 快速开始

```python
from sweeper400 import Sweeper
from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles

# 创建输出波形
sampling_info = init_sampling_info(48000, 4800)
sine_args = init_sine_args(1000.0, 1.0, 0.0)
output_waveform = get_sine_cycles(sampling_info, sine_args, cycles=100)

# 创建扫场测量器
sweeper = Sweeper(
    ai_channel="400Slot2/ai0",
    ao_channel="400Slot2/ao0",
    output_waveform=output_waveform,
    point_list=grid
)

# 执行扫场测量
sweeper.sweep()
```

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 作者

400&402
