---
type: "manual"
---

<Measuring_Rules>
以下Rules适用于NI数据采集相关功能（"pysweep\measure"目录下）的开发。
- 我们正在python中使用nidaqmx包，控制NI数据采集卡进行数据采集工作。
- 如需查看nidaqmx包具体细节/官方例程，我已经将该包的完整代码仓库clone在本工作区的"参考资料\nidaqmx-python-master"处。（但你无需使用该目录中的文件，因为nidaqmx包已经安装在我们的Python环境中，你可以直接import。）
- 我们有一台型号为PXIe-1090的NI机箱（名称为"PXIChassis1"），但目前尚未连接至本机（因此在测试中，无法找到该硬件是正常的）。其上搭载一张PXIe-4468 DSA Analog I/O板卡，名称为"400Slot2"。板卡有"ai0"和"ai1"两个输入通道，以及"ao0""ao1"两个输出通道。板卡官方手册显示，其可以使用机箱中的PXIe_CLK100时钟作为硬件采样时钟源。
- 要求所有任务都使用PXIe_CLK100时钟作为硬件采样时钟源。对于成对的AI和AO任务（同时进行），默认使用触发器进行严格同时触发。在同步触发时，每张板卡都具有如下格式的时序引擎："……/400Slot2/te0/StartTrigger", "……te1/StartTrigger", "……te2/StartTrigger", "……te3/StartTrigger"，你可以按需求使用。
- 所有ai和ao通道的电压范围均为±10.0 V (true peak)，或RMS 7.07 V (Sine Wave)。为了设备的安全，任何情况下都不要令电压范围超过该值。（你可以默认使用该值作为AI/AO任务的相应参数）
- 我们目前的具体工作方式是，在"pysweep\measure\"目录中实现所有数据采集相关的函数/类/方法/属性，并在"tests"文件夹中调用相关功能。
</Measuring_Rules>
