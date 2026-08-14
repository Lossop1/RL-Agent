# Taili MuJoCo Stair Scenes

Taili is integrated into the unified `rl_sar` project on the MuJoCo host.
The canonical project root is `/root/rl_sar`.

The staircase generator is:

```text
/root/rl_sar/scripts/generate_taili_mujoco_stairs.py
```

It writes scenes into:

```text
/root/rl_sar/src/rl_sar_zoo/taili_description/mjcf/
```

`--height` is required and is expressed in meters. Other staircase geometry
parameters are also runtime options; the generator does not require a fixed
20 cm XML scene.

Example commands from the project root:

```bash
cd /root/rl_sar
python3 scripts/generate_taili_mujoco_stairs.py \
  --direction up --height 0.18 --depth 0.36 --steps 6 \
  --width 2.4 --approach 1.2 --landing 2.0 --launch
```

For a descending staircase, change `--direction up` to `--direction down`.
The unified runner remains:

```bash
./cmake_build/bin/rl_sim_mujoco taili <scene_name>
```

The generator preserves the Taili model, policy, observation history, PD
settings, and control period. It only creates the requested environment scene.
