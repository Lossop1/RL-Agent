# Taili blind runtime payload

This directory defines the clean training runtime payload for Taili blind
locomotion. It is intentionally a build recipe, not a second source tree.

The deployment model is:

1. Build a timestamped tarball from the manifest.
2. Upload it to `/root/gpufree-data/training_payloads/`.
3. Extract it to an immutable timestamped directory.
4. Put that directory on `PYTHONPATH`.
5. Let `sitecustomize.py` import `taili_blind_runtime`, which registers the Gym
   task before IsaacLab's `train.py` resolves it.
6. Launch through `python -m taili_blind_runtime.launch_taili_train`.

This replaces the old strategy of copying files into the remote task-source tree.

## Runtime package contents

The tarball contains:

- `sitecustomize.py`
- `taili_blind_runtime/__init__.py`
- `taili_blind_runtime/blind_tp_env.py`
- `taili_blind_runtime/blind_tp_env_cfg.py`
- `taili_blind_runtime/taili_blind_env_cfg.py`
- `taili_blind_runtime/taili_amp_env.py`
- `taili_blind_runtime/taili_amp_env_cfg.py`
- `taili_blind_runtime/terrain_perceiver_policy.py`
- `taili_blind_runtime/terrain_perceiver_aux_patch.py`
- `taili_blind_runtime/telemetry_emit.py`
- `taili_blind_runtime/launch_taili_train.py`
- `taili_blind_runtime/train_taili.py`
- `taili_blind_runtime/taili_blind_config.py`
- `taili_blind_runtime/taili_blind_config.yaml`
- `taili_blind_runtime/multi_motion_loader.py`
- `taili_blind_runtime/motions.py`
- `taili_blind_runtime/parametric_ref.py`
- `taili_blind_runtime/taili_core/*.py`
- `taili_blind_runtime/motions/clips/*.npz`
- `taili_blind_runtime/assets/taili.py`
- `taili_blind_runtime/assets/robots/taili-dog/robot.urdf`
- `taili_blind_runtime/assets/robots/taili-dog/meshes/*.STL`

The remaining external dependencies are the remote Python/IsaacLab/skrl/PyTorch
environment. The runtime package must not
depend on remote RobotLab task source directories or RobotLab Python modules.

## Run output contract

The launcher writes every run under the data disk by default:

`/root/gpufree-data/taili_runs/<run_id>/`

- `run.json`: launch metadata and paths.
- `taili_blind_config.yaml`: run-local copy of the single editable Taili blind
  configuration.
- `effective_config.yaml`: resolved config after launcher overrides such as
  total steps and checkpoint/write intervals.
- `agent.skrl.yaml`: generated skrl runtime config with `experiment.directory` set to
  `/root/gpufree-data/taili_runs` and `experiment_name` set to `<run_id>`.
- `train.log`: semantic `[TPPATH]`, `[TPSTAT]`, `[TPREW]`, `[TPCURR]` telemetry.
- `train.telemetry.jsonl`: structured telemetry for the console and LLM.
- `console.log`: complete stdout/stderr capture from the train process.
- `checkpoints/`: skrl checkpoints.
- `events.out.tfevents.*`: TensorBoard event files when skrl writes them.
