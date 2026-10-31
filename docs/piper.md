# Piper / RoboBench 接入

此接入适用于已由 RoboBench 管理双 Piper 与三 RealSense 的机器。通用默认配置仍为空。
设备映射、标定板和厚度必须在当前机器现场确认，实际标识写入 runs/ 内的配置，不写进通用源码。

## 单一设备所有者

在现有 RoboBench 的 Python 环境和工作目录中，使用可选 ASGI 入口
`dual_arm_calibration.integrations.robobench_app:app`，并将本包 src/ 加入 PYTHONPATH。
该入口在原有 app 上附加设备状态和原始图像捕获接口，保留其余公开 API。
它不创建新的 CAN / USB 连接，不发送机器人运动命令。

桥接使用 RoboBench 的 OperationCoordinator 与标定硬件占用上下文。
捕获前后检查实测关节、末端反馈与 URDF FK，一次捕获约 0.6 秒，检查机械臂停稳、
相机帧递增、收帧时间差及图像新鲜度。图像通过无损 PNG 传输，使用对应的实际分辨率内参。

## 显式启用现场配置

配置 JSON 包含 provider（robobench）、base_url（同机已有 API）、identities.arms（两臂适配器序列号）、
identities.cameras（三相机序列号）、board（现场确认的完整规格）、board_thickness_mm。

```bash
dual-arm-calibration web --hardware-config runs/site/hardware-config.json --data-dir runs/site/web
```

网页增加“硬件标定”：三相机预览、实际身份/分辨率、参数来源选择、分阶段采集、计算与冻结、最终导出。
也可经 RoboBench 原有网页地址访问 `/api/dual-arm-calibration/workbench/`，
该入口复用现有 API 的端口转发；新增网页直接端口默认为 8765。
每台相机可复用厂家参数或单独拟合 D435i RGB 畸变。内参当前由停稳的独立图像采集；
录像选帧仍可用已有 CLI 离线入口。本接入不自动规划和执行 Piper 轨迹。

操作员手动调整姿态，按界面从独立内参（可选）、左右手眼、两侧顶相机桥接、独立验证，
到可选桌面步骤依次执行。每批腕外参板固定，桥接组间可以移动板，组内停稳。
已通过的阶段被冻结；重新标定时新建流程，历史会话和原始 PNG 保留在 hardware/ 下。

新流程不更改 RoboBench 原有样本和已安装外参，导出结果仍需明确发布。
最终求解失败的会话可以继续补采；成功结果需要新建会话才能重新标定。
重采未冻结阶段时保留旧观测 JSON 与原始 PNG；最终失败后恢复会重新开放独立验证和桌面批次。
