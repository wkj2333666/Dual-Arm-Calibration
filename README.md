# dual-arm-calibration

面向**两条机械臂、两台腕相机、一台固定顶相机**的通用标定工具包。
左右机械臂可以不同品牌、不同关节数；核心不依赖任何机器人 SDK、ROS、远程服务或固定目录。

当前版本提供网页工作台、离线求解、手动采集 API、录像选帧、硬件适配协议、CLI 和 Codex Skill。
网页的硬件适配器、连接配置和标定板规格初始留空，明确标记为待接入。
真实硬件的驱动、相机预览、采集和自动运动入口需要接入具体设备后实现。
已有 [Piper / RoboBench 可选接入](docs/piper.md)，通过现场配置启用三相机预览、板位姿估算、轨迹预览和自动分阶段采集。
不配置硬件时仍保持空白；设备序列号与板规格不作为通用默认值。

## 流程

1. 绑定机械臂模型/安装版本、相机序列号、图像配置及标定板。
2. **可选**：每台相机独立拟合内参畸变，或复用厂家/已有参数。
3. 左右腕相机分别采集固定板观测，求解相机到法兰的变换。
4. 顶相机分别与左右腕相机配对观察移动标定板，连接两个基座。
5. 用独立三相机观测验证完整变换链。
6. **可选**：固定平放标定板，补偿板厚，定义桌面 `z=0`。
7. 保存冻结数据包，显式发布到本地标定文件。

跳过桌面标定时以顶相机光学坐标系为公共坐标系，桌面高度为 `null`。

**本工具包附带的内参拟合配置仅针对 RealSense D435i RGB：固定厂家焦距和主点，
仅拟合 k1、k2，令 k3、p1、p2 为零。** 不是通用相机内参完整重标定，也不标定深度尺度、
RGB–深度对齐或工具 TCP。其他相机可复用已经转换成支持的投影模型的参数。

## 安装与快速运行

Python 3.10+，无需机器人硬件：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

dual-arm-calibration web
```

浏览器打开 **http://127.0.0.1:8765**。网页支持查看流程、保存空白接入草稿、运行离线示例、
上传标准观测求解，以及下载标定 JSON 和完整数据包。填写草稿不会连接设备。
启动目录下的 `runs/web/` 保存草稿与任务；可通过 `--data-dir` 指定目录，`--port` 更换端口。
服务默认只监听本机地址，关闭终端中的服务后网页停止更新。

命令行也可以单独运行：

```bash
dual-arm-calibration demo --output runs/demo
dual-arm-calibration demo --table --output runs/demo-table
pytest -q
```

示例使用左侧六关节、右侧七关节数据，检查已知真值。它验证软件计算，不代表硬件精度。

真实数据使用同一入口：

```bash
dual-arm-calibration check-rig rig.json
dual-arm-calibration solve --rig rig.json --data observations.json --output runs/calibration-001
dual-arm-calibration publish runs/calibration-001 active-calibration.json
```

`solve` 不连接设备、不发送运动命令，也不覆盖已有运行目录。
`publish` 校验数据完整性，在替换目标 JSON 前保存旧版本。下游应用负责加载/激活结果。

## 数据与接入

- [统一标定流程](docs/workflow.md)：采集顺序、可选步骤、验收和失效规则。
- [数据格式](docs/data-format.md)：坐标约定、相机模型、观测结构和结果。
- [适配器接口](docs/adapters.md)：任意机械臂/相机如何接入。
- [网页工作台](docs/web.md)：启动、使用和待接入边界。
- [操作 Skill](skills/dual-arm-calibration/SKILL.md)：供 Codex 使用，可复制整个 Skill 目录到自己的技能目录。
- [代码来源](docs/provenance.json)：提取时的版本及工作区文件哈希。
- [验证记录](docs/validation.md)：合成测试、历史实测回放和安装包验证的范围。

安装 wheel 时，Skill 与上述文档也会安装到 Python 环境的 `share/dual-arm-calibration/`。
原始图像和录像应与运行记录一起归档；求解数据包保留角点和位姿，本版本不会自动复制外部媒体。

## 构建与 GitHub

```bash
python -m build
```

GitHub 仓库：[wkj2333666/Dual-Arm-Calibration](https://github.com/wkj2333666/Dual-Arm-Calibration)。

目前未指定开源许可证；公开发布前由仓库所有者选择许可证。

## 可选：IR 双目与深度高度验证

现在可在网页的“深度与双目标定”页面使用独立 IR 采集流程：

```bash
python -m pip install -e '.[depth]'
dual-arm-calibration web --depth-config runs/site/depth.json
```

现场配置必须显式提供 `serial`、`board`，不得把示例或其他机器的配置当作默认值。
采集固定相机的左右同步红外图，手持板改变位置、距离及两个方向倾角，停稳后保存；
网页提供左右角点预览、进度、停止恢复和可选提示音。不会移动机械臂或写相机固件标定。
该入口与 RGB/手眼标定独立，不能同时操作同一相机的其他采集入口。

候选对 SDK 已校正红外图做残差标定。比较固定厂家 K 与重新拟合 K/k1/k2 两种方案，
每五组留出一组；留出照片不参与拟合。结果为候选，仍需独立平放板的米制高度验证。
`installed=false`，不会替换机器人标定。官方 Tare 还需独立准确的光轴方向距离参考，
不能把待验证的深度值作为自己的真值。

静态高度对比可保存到数据目录 `depth/comparison.json`：显示原始、SDK 滤波、FFS 的
板面误差、帧间变化及衣服采样区域。必须区分“稳定”和“准确”；衣服无高度真值时不排名。
现场样本、设备序列号和服务配置只放在被忽略的 `runs/` 目录。
