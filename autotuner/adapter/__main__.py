"""产品合同驱动的机器人适配命令行入口。"""
from __future__ import annotations

import argparse
import datetime
import tempfile

from autotuner.adapter.pipeline import (
    ConfigSet,
    build_config_set,
    consistency_report,
    deploy_readiness,
    plan_adaptation,
    render_bundle,
)
from autotuner.product import ProductRegistry


def available_products() -> list[str]:
    """返回注册表中的产品标识；无效清单不会被伪装成可用 preset。"""
    return [product.product_id for product in ProductRegistry().list(valid_only=True)]


def run_cs(cs: ConfigSet, work_dir: str | None = None, stamp: str = "dryrun"):
    """执行纯本地适配预览，不写远端、不启动训练。"""
    directory = work_dir or tempfile.mkdtemp(prefix="adapt_")
    bundle = plan_adaptation(cs, directory, stamp=stamp)
    return bundle, consistency_report(bundle)


def run(
    product_id: str | None = None,
    work_dir: str | None = None,
    stamp: str = "dryrun",
    composition: str | None = None,
):
    """从产品清单构造配置并执行本地适配预览。"""
    return run_cs(build_config_set(product_id, composition), work_dir, stamp)


def main() -> None:
    parser = argparse.ArgumentParser(prog="autotuner.adapter")
    parser.add_argument("product", nargs="?", default=None, help="product id; optional when only one is registered")
    parser.add_argument("--composition", default=None, help="override framework composition id")
    parser.add_argument("--work-dir", default=None, help="staging directory for materialized copies")
    parser.add_argument("--execute", action="store_true", help="deploy the legacy file plan")
    parser.add_argument("--confirm", action="store_true", help="required with --execute")
    parser.add_argument("--launch", action="store_true", help="also launch the declared legacy command")
    parser.add_argument("--force", action="store_true", help="bypass legacy deploy-readiness blockers")
    parser.add_argument("--regenerate", action="store_true", help="regenerate product reference artifacts")
    parser.add_argument("--clip-remote-dir", default=None, dest="clip_remote_dir")
    parser.add_argument("--list-products", action="store_true", help="list registered products and exit")
    parser.add_argument("--list", action="store_true", help="list stored adaptation ConfigSets and exit")
    parser.add_argument("--load", default=None, help="load a stored ConfigSet by id")
    parser.add_argument("--save", default=None, help="save the resolved ConfigSet under this id")
    parser.add_argument("--audit", default=None, help="write the adaptation audit record JSON")
    args = parser.parse_args()

    if args.list_products:
        for product_id in available_products():
            print(product_id)
        return

    from autotuner.adapter.config_store import ConfigSetStore

    store = ConfigSetStore()
    if args.list:
        rows = store.list()
        print(f"stored ConfigSets ({len(rows)}):")
        for row in rows:
            print(f"  {row['id']:20} fw={row['framework']:18} updated={row['updated']}  {row['note']}")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_set = store.load(args.load) if args.load else build_config_set(args.product, args.composition)
    if args.regenerate:
        config_set.regenerate_clips = True
    if args.clip_remote_dir:
        config_set.clip_remote_dir = args.clip_remote_dir
    if args.save:
        path = store.save(args.save, config_set, stamp, note=f"product={config_set.product_id}")
        print(f"saved ConfigSet {args.save!r} -> {path}")

    bundle, report = run_cs(config_set, args.work_dir, stamp)
    print(render_bundle(bundle))
    print(
        f"\nconsistency [{report['verdict']}]: {report['n_agree']} agree, "
        f"{report['n_corrected']} corrected, {report['n_flagged']} flagged advisory"
    )
    readiness = deploy_readiness(bundle)
    print(f"\ndeploy readiness: {'READY' if readiness['ready'] else 'NOT READY'}")
    for blocker in readiness["blockers"]:
        print(f"   - {blocker}")

    if args.audit:
        from autotuner.adapter.audit import build_audit_record, write_audit

        record = build_audit_record(bundle, report, readiness, stamp=stamp)
        print(f"\naudit record -> {write_audit(record, args.audit)}")

    if not args.execute:
        print("\nDRY-RUN. Product payload deployment is the authoritative execution path.")
        return
    if not args.confirm:
        raise PermissionError("--execute requires --confirm")
    if not readiness["ready"] and not args.force:
        raise RuntimeError("legacy deployment is not ready; resolve blockers or pass --force")

    from autotuner.adapter.deploy import execute
    from autotuner.adapter.remote_deploy import from_ssh_json

    ssh = from_ssh_json()
    try:
        result = execute(bundle.plan, ssh, confirm=True, do_launch=args.launch)
    finally:
        ssh.close()
    if not result.ok:
        raise RuntimeError("legacy deployment failed; consult the audit record and rollback plan")


if __name__ == "__main__":
    main()


__all__ = ["available_products", "build_config_set", "main", "run", "run_cs"]
