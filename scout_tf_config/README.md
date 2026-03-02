# scout_tf_config (Phase 2)

Phase 2 artifact set for TF chain + timestamp checks in OmniNav real-world pipeline.

## Scope

- Build a single TF chain for demo runtime:
  - `map -> base_link -> lidar_frame -> cam_front/left/right_color_optical_frame`
- Use FAST-LIO `/Odometry` as pose authority via bridge node.
- Keep OmniNav inference logic unchanged (Phase 3 item is separate).

## Files

- `config/sensor_tf.yaml`: all frame/topic/transform parameters
- `launch/sensor_tf_launch.py`: starts static TF publishers + odom TF bridge
- `scripts/odom_tf_bridge.py`: `/Odometry -> map->base_link` TF
- `scripts/verify_tf.py`: TF and timestamp verification

## Required pre-change (Scout)

Disable wheel odom TF authority first:

- file (inside runtime container):
  - `/workspace/install/scout_mini_base/share/scout_mini_base/config/scout_mini.yaml`
- change:
  - `enable_odom_tf: false`

Without this, you can get dual-parent conflicts for `base_link`.

## Base-to-LiDAR measurement usage

Configured value in `sensor_tf.yaml`:

- `x=0.09`, `y=0.0`, `z=0.242`

Reason:

- Measured `ground->lidar` = `0.43m`
- Scout `base_link` is not at ground level.
- URDF wheel geometry implies `base_link->ground ~= 0.188m`.
- Therefore `base_link->lidar.z ~= 0.43 - 0.188 = 0.242m`.

## Run

```bash
python3 scout_tf_config/launch/sensor_tf_launch.py
```

Run with verification in one shot:

```bash
python3 scout_tf_config/launch/sensor_tf_launch.py --run-verify
```

Standalone verification:

```bash
python3 scout_tf_config/scripts/verify_tf.py
```

## Notes

- FAST-LIO native TF uses `camera_init -> body`; bridge keeps this untouched and publishes `map -> base_link` from `/Odometry` stamps.
- Phase 3 task (`run_infer_online_panorama.py` pose injection) is intentionally out of scope here.
