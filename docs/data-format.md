# 数据格式 v1

所有输入为 JSON。矩阵字段 `a_from_b` 把 b 坐标中的点变换到 a；齐次 4×4、行优先嵌套数组，
右手坐标系、平移米、关节弧度。相机使用光学坐标（x 向右、y 向下、z 向前）。
先执行 `dual-arm-calibration demo --output runs/example` 可得到完整有效的 rig 和 observations 示例。

## rig.json

- `schema_version: 1`。
- `board`：squares_x、squares_y、square_length_m、marker_length_m、dictionary。
- `arms.left/right.model_id`：可追踪的运动学与安装身份，推荐模型和安装参数哈希。
- `cameras.left/right/top`：`mount` 分别为 left_flange、right_flange、fixed。
- 每台相机的 `intrinsics_mode` 是 reuse 或 fit，缺省 reuse。
- fit 时必须提供 `family: realsense_d435i` 和 `stream: rgb`。
- `table.enabled` 缺省 false。启用时要求 `thickness_mm` 和
  `board_z_direction: into_table | out_of_table`。

每台相机的 `model` 包含：

```json
{
  "serial": "camera-unique-id",
  "source": "factory",
  "image_rotation_deg": 0,
  "factory_intrinsics": {
    "width": 1280, "height": 720,
    "fx": 850, "fy": 850, "ppx": 640, "ppy": 360,
    "model": "opencv_brown_conrady", "coeffs": [0, 0, 0, 0, 0]
  },
  "intrinsics": {
    "width": 1280, "height": 720,
    "fx": 850, "fy": 850, "ppx": 640, "ppy": 360,
    "model": "opencv_brown_conrady", "coeffs": [0, 0, 0, 0, 0]
  }
}
```

source 可为 factory、reused、independent_intrinsics。factory_intrinsics 描述原始采集图像，
intrinsics 是后续投影使用的模型。两者不能混淆或对图像重复去畸变。
接受 OpenCV 前向 Brown 模型，以及所有系数为零的 inverse Brown（数学上无畸变）。
非零 inverse Brown、鱼眼等需在适配器侧显式转换/实现，不可仅改模型名称。
旋转后的图像必须同时配套变换后的 K 和光学坐标约定；核心不会自动旋转图像。

## observations.json

- `schema_version: 1`。
- `rig_sha256`：使用 `workflow.digest(rig)` 或 `check-rig` 输出；它包含精确配置绑定。
- `intrinsics.left/right/top`：仅拟合的相机需要，值为观测数组。
- `wrists.left/right`：各至少 20 条腕相机观测，额外包含 base_from_flange；可附带任意长度的 joint_angles_rad。
- `bridges.left/right`：各至少 12 条顶相机观测，额外包含 base_from_flange 和 wrist（对应腕相机观测）。
- `validation`：至少 3 组三相机观测。
- `table`：启用桌面时至少 3 组三相机观测，板在全部组间保持不动。

普通观测包含 id、camera_frame（serial、image_rotation_deg，可附时间戳/帧号）、intrinsics（原始参数）、
charuco_ids（一维整数索引）、charuco_corners（N×2 像素）、camera_from_target（PnP 初值），
可附 reprojection_error_px、图像路径、配对和静止证据。求解会从原始角点重新计算 PnP。

三相机观测包含 id、cameras（left/right/top 普通观测）、base_from_flange（left/right）。
所有顶层观测 ID 跨批次唯一；不要为同一照片生成不同 ID 来绕过批次独立性。
ID/配置校验不能鉴别内容伪造；采集端需保存原图、时间证据，并确保验证数据未参与拟合。

## 输出

运行目录包括 rig.json、observations.json、calibration.json、manifest.json。
结果包含 world_from_top、world_from_base.left/right、各腕 flange_from_camera、相机模型、
可选 table、各阶段验证及源数据哈希。桌面不可用时 table_z_m 为 null。
末端运动时：`world_from_wrist = world_from_base @ base_from_flange @ flange_from_camera`。
结果的 `tcp_verified=false` 表示此流程没有完成工具中心标定。

本地原始媒体放在运行记录旁，并记录来源路径/哈希。当前 CLI 保存观测，不复制媒体。
