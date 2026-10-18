"""Build (or tear down) the CloudGuard sandbox in YOUR OWN free Azure subscription.

Dry run by default: prints the az commands and changes nothing.

  python sandbox/seed.py                       # dry run, prints commands
  python sandbox/seed.py --apply               # creates resources (az login first)
  python sandbox/seed.py --destroy --apply     # removes locks, deletes the 6 groups
  python sandbox/seed.py --mock-only           # regenerate mock snapshot + mock ground truth

After --apply it writes ground_truth.live.json for your real subscription ID.
Approximate cost: two 4 GiB Standard HDD disks plus two empty storage accounts, a few
dollars a month at most. Run --destroy when you're done.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloudguard import sandbox  # noqa: E402

SPEC = ROOT / "sandbox" / "seed_spec.json"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def current_subscription() -> str:
    out = subprocess.run(["az", "account", "show", "--query", "id", "-o", "tsv"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def az_show_min_tls_version(resource_group: str, name: str) -> str:
    """Read back what Azure actually stored, since a successful `create` does not guarantee
    the platform honored every requested property (e.g. retired TLS versions)."""
    out = subprocess.run(
        ["az", "storage", "account", "show", "-g", resource_group, "-n", name,
         "--query", "minimumTlsVersion", "-o", "tsv"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def verify_weak_tls(
    spec: dict,
    sa_created: dict[str, str],
    suffix: str,
    show=az_show_min_tls_version,
) -> bool:
    """Ground truth for weak_tls must reflect what Azure actually stored, not what we asked
    for or whether the `create` call exited 0. Returns whether every spec'd weak account is
    still weak on the live platform."""
    weak_tls_seeded = True
    for sa in spec["storage_accounts"]:
        name = sandbox.sa_name(sa, suffix)
        rg = sa_created.get(name)
        if rg is None:
            if sa["weak"]:
                weak_tls_seeded = False
            continue
        actual = show(rg, name)
        if actual != sa["min_tls"]:
            print(f"!! {name}: requested min TLS {sa['min_tls']} but Azure stored {actual!r}.",
                  file=sys.stderr)
        if sa["weak"] and actual not in ("TLS1_0", "TLS1_1"):
            weak_tls_seeded = False
    return weak_tls_seeded


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually run the commands")
    p.add_argument("--destroy", action="store_true", help="tear the sandbox down instead")
    p.add_argument("--mock-only", action="store_true", help="only regenerate mock files")
    a = p.parse_args()

    spec = sandbox.load_spec(SPEC)

    if a.mock_only:
        write_json(ROOT / "mock" / "subscription_snapshot.json", sandbox.build_mock_snapshot(spec))
        write_json(ROOT / "ground_truth.json", sandbox.build_ground_truth(spec, sandbox.PLACEHOLDER_SUB))
        return 0

    cmds = sandbox.az_destroy_commands(spec) if a.destroy else sandbox.az_seed_commands(spec)
    if not a.apply:
        print("# DRY RUN - nothing changed. Re-run with --apply to execute.\n")
        for c in cmds:
            print(shlex.join(c))
        return 0

    sub = current_subscription()
    print(f"Target subscription: {sub}")
    if input("Type the subscription ID to confirm: ").strip() != sub:
        print("Confirmation did not match; aborting.")
        return 1

    sa_created: dict[str, str] = {}  # storage account name -> resource group, for accounts that actually got created
    for c in cmds:
        print("+", shlex.join(c))
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr)
            if not a.destroy:
                return r.returncode
            continue
        if c[:4] == ["az", "storage", "account", "create"]:
            sa_created[c[c.index("-n") + 1]] = c[c.index("-g") + 1]

    if not a.destroy:
        # Azure can accept a storage account `create` (exit 0) while silently storing a
        # different minimumTlsVersion than requested (e.g. TLS 1.0/1.1 were retired), so
        # ground truth must come from reading the platform back, not from the exit code.
        weak_tls_seeded = verify_weak_tls(spec, sa_created, spec["suffix"])
        write_json(ROOT / "ground_truth.live.json", sandbox.build_ground_truth(spec, sub, weak_tls_seeded))
        print("\nNext: CLOUDGUARD_LIVE_TESTS=1 AZURE_SUBSCRIPTION_ID=<id> pytest -m live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
