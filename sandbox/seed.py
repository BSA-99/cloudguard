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

    weak_tls_seeded = True
    for c in cmds:
        print("+", shlex.join(c))
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr)
            if "--min-tls-version" in c and "TLS1_0" in c:
                # Azure Storage retired TLS 1.0/1.1 on 2026-02-03; the platform may refuse this setting.
                print("!! weak-TLS account was rejected by Azure; it will be excluded from live ground truth.")
                weak_tls_seeded = False
                continue
            if not a.destroy:
                return r.returncode

    if not a.destroy:
        write_json(ROOT / "ground_truth.live.json", sandbox.build_ground_truth(spec, sub, weak_tls_seeded))
        print("\nNext: CLOUDGUARD_LIVE_TESTS=1 AZURE_SUBSCRIPTION_ID=<id> pytest -m live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
