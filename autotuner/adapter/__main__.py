"""CLI for the Taili-only framework-adaptation backend.

Default mode is a local dry-run:

    python -m autotuner.adapter [taili] [--composition ID] [--work-dir DIR]

The outward-facing path still requires explicit confirmation:

    python -m autotuner.adapter taili --execute --confirm [--launch]

This adapter is kept as a read-only/dry-run support surface for the console.
The active training deployment direction is the Taili blind runtime payload,
not robot_lab source-tree compatibility.
"""
from __future__ import annotations

import argparse
import datetime
import tempfile

from autotuner.adapter.pipeline import (
    ConfigSet,
    consistency_report,
    deploy_readiness,
    plan_adaptation,
    render_bundle,
)


_REMOTE_TAILI = "/root/gpufree-data/robot_lab/source/robot_lab/robot_lab/tasks/direct/taili_amp"

# Current tree is Taili-only. Unitree/B2 proof fixtures were archived under
# archive/2026-07-cleanup/b2-removed/.
PRESETS = {
    "taili": dict(
        urdf="assets/robots/taili-dog/robot.urdf",
        env_cfg_src="autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py",
        asset_src="autotuner/blind_locomotion/assets/taili.py",
        remote_env_cfg=f"{_REMOTE_TAILI}/taili_amp_env_cfg.py",
        remote_asset="/root/gpufree-data/robot_lab/.../asset/taili.py",
        framework_composition="taili_amp_proven",
        launch_cmd="/opt/conda/envs/isaaclab/bin/python3 /root/tp_train_real.py",
    ),
}


def build_config_set(robot: str, composition: str | None = None) -> ConfigSet:
    if robot not in PRESETS:
        raise ValueError(f"unknown robot {robot!r}; known: {', '.join(PRESETS)}")
    cs = ConfigSet(**PRESETS[robot])
    if composition:
        cs.framework_composition = composition
    return cs


def run_cs(cs: ConfigSet, work_dir: str | None = None, stamp: str = "dryrun"):
    """Dry-run the offline chain for a ConfigSet; returns (bundle, consistency_report)."""
    wd = work_dir or tempfile.mkdtemp(prefix="adapt_")
    bundle = plan_adaptation(cs, wd, stamp=stamp)
    return bundle, consistency_report(bundle)


def run(
    robot: str = "taili",
    work_dir: str | None = None,
    stamp: str = "dryrun",
    composition: str | None = None,
):
    """Dry-run the offline chain from a preset; returns (bundle, consistency_report)."""
    return run_cs(build_config_set(robot, composition), work_dir, stamp)


def main() -> None:
    parser = argparse.ArgumentParser(prog="autotuner.adapter")
    parser.add_argument("robot", nargs="?", default="taili", choices=list(PRESETS))
    parser.add_argument("--composition", default=None, help="override framework composition id")
    parser.add_argument("--work-dir", default=None, help="staging dir for materialized copies")
    parser.add_argument("--execute", action="store_true", help="OUTWARD-FACING: deploy the plan to remote")
    parser.add_argument("--confirm", action="store_true", help="required with --execute")
    parser.add_argument("--launch", action="store_true", help="with --execute --confirm: also start training")
    parser.add_argument("--force", action="store_true", help="deploy even if the readiness gate has blockers")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate AMP reference clips into the work_dir before deploy-readiness check",
    )
    parser.add_argument(
        "--clip-remote-dir",
        default=None,
        dest="clip_remote_dir",
        help="remote dir for regenerated reference clips",
    )
    parser.add_argument("--list", action="store_true", help="list stored ConfigSets and exit")
    parser.add_argument("--load", default=None, help="load a stored ConfigSet by id")
    parser.add_argument("--save", default=None, help="save the resolved ConfigSet under this id")
    parser.add_argument("--audit", default=None, help="write the full adaptation audit record JSON")
    args = parser.parse_args()

    from autotuner.adapter.config_store import ConfigSetStore

    store = ConfigSetStore()
    if args.list:
        rows = store.list()
        print(f"stored ConfigSets ({len(rows)}):")
        for row in rows:
            print(f"  {row['id']:20} fw={row['framework']:18} updated={row['updated']}  {row['note']}")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cs = store.load(args.load) if args.load else build_config_set(args.robot, args.composition)
    if args.regenerate:
        cs.regenerate_clips = True
    if args.clip_remote_dir:
        cs.clip_remote_dir = args.clip_remote_dir
    if args.save:
        path = store.save(args.save, cs, stamp, note=f"robot={args.robot}")
        print(f"saved ConfigSet {args.save!r} -> {path}")

    bundle, rep = run_cs(cs, args.work_dir, stamp)
    print(render_bundle(bundle))
    print(
        f"\nconsistency [{rep['verdict']}]: {rep['n_agree']} agree, "
        f"{rep['n_corrected']} corrected, {rep['n_flagged']} flagged advisory"
    )
    for hazard in rep["sim2real_hazards_fixed"]:
        print(f"   ! sim2real hazard fixed: {hazard}")

    readiness = deploy_readiness(bundle)
    print(f"\ndeploy readiness: {'READY' if readiness['ready'] else 'NOT READY'}")
    for blocker in readiness["blockers"]:
        print(f"   - {blocker}")

    if args.audit:
        from autotuner.adapter.audit import build_audit_record, write_audit

        record = build_audit_record(bundle, rep, readiness, stamp=stamp)
        print(f"\naudit record -> {write_audit(record, args.audit)}")

    if not args.execute:
        print("\nDRY-RUN. Add --execute --confirm to deploy.")
        return
    if not args.confirm:
        print("\nREFUSED: --execute requires --confirm.")
        return
    if not readiness["ready"] and not args.force:
        print("\nREFUSED: deploy readiness gate has blockers. Resolve them or pass --force.")
        return

    from autotuner.adapter.deploy import execute, restore_cmds
    from autotuner.adapter.remote_deploy import from_ssh_json

    ssh = from_ssh_json()
    print("\n[execute] deploying over SSH (backup-then-put + checksum verify)")
    result = execute(bundle.plan, ssh, confirm=True, do_launch=args.launch)
    print(f"[execute] ok={result.ok} launched={result.launched}")
    for item in result.items:
        print(f"   {item.remote}: verified={item.verified} backed_up={item.backed_up} {item.detail}")
    if not result.ok:
        print("   rollback with:")
        for cmd in restore_cmds(bundle.plan):
            print("     " + cmd)


if __name__ == "__main__":
    main()
