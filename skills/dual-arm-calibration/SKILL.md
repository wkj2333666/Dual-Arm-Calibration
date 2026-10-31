---
name: dual-arm-calibration
description: Calibrate two robot arms with one camera on each wrist and one fixed overhead camera. Use for capture planning, offline hand-eye and camera bridging, optional D435i RGB distortion fitting, optional tabletop coordinates, and calibration validation/export across robot brands.
---

# 双臂三相机标定

先读 [统一流程](../../docs/workflow.md)。接入硬件读 [适配器](../../docs/adapters.md)；导入或转换记录读 [数据格式](../../docs/data-format.md)。

此 Skill 随 dual-arm-calibration 仓库或 wheel 分发；使用当前 Python 环境中的
`dual-arm-calibration --help` 确认入口。没有硬件时用 `demo` 验证环境或用标准数据离线 `solve`。
CLI 不发送机器人运动命令。自动采集需已接入且有明确运动限制的外部适配器。

## 操作决策

1. 确认 left/right 机械臂、left/right 腕相机、top 固定相机角色及安装身份，不假定品牌或关节数。
2. 每台相机分别选择复用参数或重新拟合。仅 D435i RGB 使用现有 radial2 拟合：固定厂家 K，拟合 k1/k2，其他三项为零。
   其他相机应加载其适用模型，不把 D435i 限制当作通用规则。
3. 左右腕分别采固定板外参批次，板在每批内固定，两侧之间可移动；内参照片与外参照片不混用。
4. 顶相机分别与每侧腕相机观察同一静止板，组间改变板位置/倾角，独立连接两基座。
5. 用未拟合的三相机观测验收。桌面标定可选；跳过时公共坐标为 top_camera，不能假定桌面 z=0。
6. 要桌面坐标时，另采固定平放板观测，明确法向和图案到桌面的厚度，再求解。
7. 将输入、验证、结果保存在新 run 目录；只有通过验证的数据包才能 publish。

## 失败处理与报告

按覆盖报告补采缺失边缘、距离或倾角；单轴姿态不足、板移动、图像过期、配置不匹配时修正采集原因。
不要降低质量阈值、重复照片或改 ID 来制造通过结果。冻结结果变化后重新求解依赖阶段。
无可靠初始手眼外参时用人工引导采集；有可靠外参的单图板定位只能提供规划初值。
复用用户已明确的操作范围；离线求解、打包和文档操作不意味着要让机器人运动。

报告实际完成的阶段、是否拟合相机参数、是否建立桌面、拟合/留出误差、独立验证、数据包路径和仍未接入的硬件能力。
区分合成数据验证、历史实测回放和新硬件现场验证，不把其中一种说成另一种。
