import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_seed():
    spec = importlib.util.spec_from_file_location("sandbox_seed", ROOT / "sandbox" / "seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = _load_seed()

SPEC = {
    "suffix": "sfx",
    "storage_accounts": [
        {"name_prefix": "cgweaktls", "resource_group": "cg-legacy-app", "min_tls": "TLS1_0", "weak": True},
        {"name_prefix": "cgstrongtls", "resource_group": "cg-legacy-app", "min_tls": "TLS1_2", "weak": False},
    ],
}


def test_az_show_min_tls_version_parses_stdout(monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="TLS1_2\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert seed.az_show_min_tls_version("cg-legacy-app", "cgweaktlssfx") == "TLS1_2"
    assert captured["cmd"] == [
        "az", "storage", "account", "show", "-g", "cg-legacy-app", "-n", "cgweaktlssfx",
        "--query", "minimumTlsVersion", "-o", "tsv",
    ]


def test_verify_weak_tls_drops_finding_when_azure_overrode_the_request(capsys):
    sa_created = {"cgweaktlssfx": "cg-legacy-app", "cgstrongtlssfx": "cg-legacy-app"}

    def fake_show(rg, name):
        return {"cgweaktlssfx": "TLS1_2", "cgstrongtlssfx": "TLS1_2"}[name]

    result = seed.verify_weak_tls(SPEC, sa_created, "sfx", show=fake_show)

    assert result is False
    err = capsys.readouterr().err
    assert "cgweaktlssfx" in err and "TLS1_0" in err and "TLS1_2" in err
    assert "cgstrongtlssfx" not in err


def test_verify_weak_tls_keeps_finding_when_azure_honored_the_request(capsys):
    sa_created = {"cgweaktlssfx": "cg-legacy-app", "cgstrongtlssfx": "cg-legacy-app"}

    def fake_show(rg, name):
        return {"cgweaktlssfx": "TLS1_0", "cgstrongtlssfx": "TLS1_2"}[name]

    result = seed.verify_weak_tls(SPEC, sa_created, "sfx", show=fake_show)

    assert result is True
    assert capsys.readouterr().err == ""


def test_verify_weak_tls_false_when_weak_account_was_never_created():
    result = seed.verify_weak_tls(SPEC, {}, "sfx", show=lambda rg, name: "TLS1_2")
    assert result is False
