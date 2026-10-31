# 硬件接入

核心只接受标准观测；没有硬件适配器仍能执行全部离线求解。
左右机器人可以不同厂家、不同关节数，代码不调用品牌 SDK 或假定 Euler 角顺序。

`dual_arm_calibration.adapters` 提供 ArmState、CameraFrame、ArmReader、CameraReader、MotionAdapter 协议。

- ArmReader.read_state：实测 base_from_flange、joint_angles_rad、stationary、host_monotonic_s。
- CameraReader.capture：BGR 图像、序列号、递增帧号、图像配置和 host_monotonic_s。
- 可选 MotionAdapter：plan、execute、stop；规划、可达性、碰撞检查、速度/加速度限制及控制器异常处理属于适配器职责。

协议本身不实现驱动。注册可安装插件后由 `load_adapter(name, **config)` 加载：

```toml
[project.entry-points."dual_arm_calibration.adapters"]
my_robot = "my_robot_adapter:Robot"
my_camera = "my_camera_adapter:Camera"
```

调用 `capture.capture_stationary(arms, cameras, models, board_spec)` 被动采集。
所有适配器时间戳必须转换到当前采集进程的 `time.monotonic()` 时基；设备本地时间不可直接混用。
它检查帧/状态新鲜度、两次状态间位姿漂移、0.25 秒板稳定窗口、帧更新和相机收帧间隔 ≤50 ms。
这是软件收帧配对，不是硬件曝光同步，适用于停稳采集；适配器应保证时标和 stationary 的可信性。
接口调用的超时由适配器实现，以免底层 SDK 阻塞。

函数返回 `(sample, images)`，sample 是三相机格式的子集，images 为按角色索引的 BGR 原图。
上层需立即归档原图。转换到腕批次时取 sample.cameras[arm] 并加入对应 base_from_flange；
转换到桥接批次时取 sample.cameras.top、wrist 和对应 base_from_flange；保留捕获 ID 和配对证据。

人工示教时只需要读取接口。自动采集还需要机器人特定的轨迹规划器；本版本没有跨品牌自动轨迹生成器。
现有外参用于估计标定板位置只是规划初值，不绕过适配器的路径检查。
法兰位姿由 SDK 或 FK 提供，工具偏移、单位和轴约定在接入时验证；本核心不假定某一厂家的末端字段。

视频：`dual-arm-calibration select-video recording.avi --rig rig.json --camera top --output selected.json`。
输出原始帧索引、角点、覆盖报告，采集端必须保证整个视频来自该相机与图像配置。
选择结果不足覆盖要求时补录；不要靠重复相同帧凑数。把 samples 合并到独立内参批次前分配唯一捕获 ID，
保留视频文件身份与原始帧索引，避免不同视频的同名帧 ID 冲突。
