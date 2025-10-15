---
type: "always_apply"
---

<Basic_Rules>

Environments:
- We are working with Python (miniconda).
- Python interpreter version: 3.12.11
- Some installed packages: nidaqmx, scipy, matplotlib, numpy, flet, pydantic, ruff, pre-commit, pytest
- If you need to install a new package, please stop responding and let me handle it manually. Do not install it by yourself.
- The "Type Checking Mode" option of "Pyright" has been set to "strict". Please comply with the relevant type checking specifications as strictly as possible.
- 禁止主动使用“# type: ignore”忽略类型检查错误，这会阻碍我进行手动检查和修复。如果你不能合理地解决类型检查问题，让错误保持原状即可。

---

Main Task:
- 我们正在开发 "sweeper400" package，它的主要功能是：协同控制NI数据采集卡（使用 "nidaqmx" package）和步进电机（使用"MT_API.dll"文件），自动化分步采集空间中不同位置的信号，并对信号进行处理，获取信号的空间分布等信息。
- "sweeper400" package 包含以下子包："measure"（包含NI数据采集相关module），"move"（包含步进电机控制相关module），"analyze"（包含信号和数据处理相关module），"use"（协同调用其他子包，将功能封装为适用于特定任务的专用对象，供外部以简洁的方式使用），"gui"（使用flet创建GUI应用程序，方便用户使用）

---

Detailed Rules:
- 本包配置了日志管理框架"sweeper400\sweeper400\logger.py"，请在开发中使用它统一进行日志管理，合理为我们的代码配置日志输出，方便你监测代码的运行情况。
- 请在开发中遵循以下方式："sweeper400" package的所有文件位于根目录的"sweeper400"目录中，测试代码则位于根目录的"tests"目录中。请在"sweeper400\sweeper400"目录中编写各子包/模块源代码（实现所有的函数/类/方法/属性），在"tests"目录中编写测试代码调用"sweeper400" package（已使用开发模式安装，可以直接import）。可以使用pytest运行测试。
- 开发过程中，请不要忘记适时更新各级"__init__.py"和package配置文件"sweeper400\pyproject.toml"。
- 你暂时不需要主动创建“使用示例/演示脚本”和“使用指南/说明文档”，只要将Docstring和代码注释写得清楚详细即可。

</Basic_Rules>
