"""
测试nidaqmx read()方法返回的数据格式
"""
import nidaqmx
import numpy as np

# 测试PXIChassis2的4个AI通道
channels = [
    "PXI2Slot2/ai0",
    "PXI2Slot2/ai1",
    "PXI2Slot3/ai0",
    "PXI2Slot3/ai1",
]

print(f"测试 {len(channels)} 个AI通道的数据读取格式")
print(f"通道列表: {channels}")

# 创建任务
task = nidaqmx.Task("TestAIRead")

# 添加所有通道
for ch in channels:
    task.ai_channels.add_ai_microphone_chan(
        ch,
        units=nidaqmx.constants.SoundPressureUnits.PA,
        mic_sensitivity=0.004,
        max_snd_press_level=120.0,
        current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
        current_excit_val=0.004,
    )
    print(f"  已添加通道: {ch}")

# 配置采样
task.timing.cfg_samp_clk_timing(
    rate=48000.0,
    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
    samps_per_chan=100,  # 只读取100个样本
)

# 启动任务
task.start()

# 读取数据
data = task.read(number_of_samples_per_channel=100)

# 检查数据格式
print(f"\n数据类型: {type(data)}")
if isinstance(data, list):
    print(f"列表长度: {len(data)}")
    if data and isinstance(data[0], list):
        print(f"这是2D列表")
        print(f"  外层列表长度(通道数): {len(data)}")
        print(f"  内层列表长度(样本数): {len(data[0])}")
    else:
        print(f"这是1D列表")
        print(f"  列表长度(样本数): {len(data)}")

# 转换为numpy数组
data_array = np.array(data, dtype=np.float64)
print(f"\nnumpy数组shape: {data_array.shape}")
print(f"numpy数组ndim: {data_array.ndim}")

# 停止任务
task.stop()
task.close()

print("\n测试完成!")
