import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STUB = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["NEWS_TEST_STATE"])
log_path = Path(os.environ["NEWS_TEST_LOG"])
state = json.loads(state_path.read_text())
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with log_path.open("a") as log:
    log.write(json.dumps([name, *args]) + "\\n")
if name == "systemd-run":
    unit = next(a.split("=", 1)[1] for a in args if a.startswith("--unit="))
    timed = "--on-active=1s" in args
    timer_only = timed and "--stage" not in args
    service = unit + ".service"
    if not timer_only and service in state["units"]:
        sys.exit("Unit was already loaded or has a fragment file")
    if timer_only and service not in state["units"]:
        sys.exit("Missing timer target service")
    if timed:
        state["units"][unit + ".timer"] = "active"
    if not timer_only:
        state["units"][service] = "active"
elif "set-property" in args:
    unit = args[3]
    state.setdefault("quotas", {})[unit] = args[4]
elif "show-environment" in args:
    sys.exit(0)
elif "list-units" in args:
    if "oil-news-download-*.service" in args:
        for unit in state["legacy"]:
            print(unit, "loaded active running legacy worker")
elif "stop" in args:
    targets = args[2:]
    if any(unit in state["frozen"] for unit in targets):
        sys.exit("Cannot perform operation on frozen unit")
    state["legacy"] = [unit for unit in state["legacy"] if unit not in targets]
elif "thaw" in args:
    if os.environ.get("NEWS_TEST_THAW_FAIL"):
        sys.exit("Cannot thaw unit")
    state["frozen"] = [unit for unit in state["frozen"] if unit not in args[2:]]
elif "is-active" in args:
    status = state["units"].get(args[-1], "inactive")
    print(status)
    sys.exit(0 if status == "active" else 3)
elif "show" in args:
    if "--property=FreezerState" in args:
        print("frozen" if args[2] in state["frozen"] else "running")
    else:
        print("loaded" if args[2] in state["units"] else "not-found")
elif "start" in args:
    state["units"][args[-1]] = "active"
state_path.write_text(json.dumps(state))
"""


@pytest.mark.parametrize("frozen", [False, True])
def test_launcher_retires_legacy_workers_and_downloads_one_sequential_queue(
    tmp_path: Path,
    *,
    frozen: bool,
) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "scripts" / "start_news.sh"
    launcher = scripts / "start_news.sh"
    shutil.copyfile(source, launcher)
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("systemctl", "systemd-run"):
        stub = binaries / name
        stub.write_text(STUB)
        stub.chmod(0o755)
    state_path, log_path = tmp_path / "state.json", tmp_path / "calls.jsonl"
    legacy = [f"oil-news-download-{letter}.service" for letter in "abcde"]
    legacy += [
        "oil-news-process.timer",
        "oil-news-process.service",
        "oil-news-us.service",
    ]
    frozen_units = (
        [unit for unit in legacy if unit.endswith(".service")] if frozen else []
    )
    state_path.write_text(
        json.dumps(
            {
                "legacy": legacy,
                "frozen": frozen_units,
                "units": {"oil-news-awake.service": "active"},
            }
        )
    )
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}")
    env.update(NEWS_TEST_STATE=str(state_path), NEWS_TEST_LOG=str(log_path))
    env.pop("OIL_NEWS_CPU_QUOTA", None)

    def invoke() -> None:
        subprocess.run(  # noqa: S603 -- only the copied launcher with stubbed systemctl
            ["/usr/bin/bash", str(launcher)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    if frozen:
        # A failed thaw must abort before any replacement downloader starts.
        env["NEWS_TEST_THAW_FAIL"] = "1"
        with pytest.raises(subprocess.CalledProcessError):
            invoke()
        calls = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert not any(call[0] == "systemd-run" for call in calls)
        del env["NEWS_TEST_THAW_FAIL"]
        log_path.write_text("")

    invoke()
    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    stops = [call for call in calls if "stop" in call]
    remaining = legacy if not frozen else [u for u in legacy if u.endswith(".service")]
    assert {unit for call in stops for unit in call[3:]} == set(remaining)
    thaws = [call for call in calls if "thaw" in call]
    assert {call[-1] for call in thaws} == set(frozen_units)
    for thaw in thaws:
        assert calls[calls.index(thaw) + 1] == ["systemctl", "--user", "stop", thaw[-1]]
    jobs = [call for call in calls if call[0] == "systemd-run"]
    downloads = [call for call in jobs if "download" in call]
    assert len(downloads) == 1
    assert downloads[0][-5:] == [
        "run",
        "--stage",
        "download",
        "--only",
        "us,sa,ae,it,es,om,az",
    ]
    assert "--unit=oil-news-index" in downloads[0]
    assert "-u" in downloads[0]
    assert "MEDIACLOUD_API_KEY" in downloads[0]
    assert not any("EnvironmentFile=" in arg for call in jobs for arg in call)
    processing = [call for call in jobs if "--stage" in call and "download" not in call]
    assert sorted(call[-1] for call in processing) == [
        "assemble",
        "filter",
        "fulltext",
    ]
    for job in [downloads[0], *processing]:
        assert "--property=CPUQuota=85%" in job
    assert all(calls.index(stop) < calls.index(jobs[0]) for stop in stops)

    # A downloader waiting for its hourly retry counts as already running.
    state = json.loads(state_path.read_text())
    state["units"]["oil-news-index.service"] = "activating"
    state["units"]["oil-news-filter.timer"] = "inactive"
    state_path.write_text(json.dumps(state))
    env["OIL_NEWS_CPU_QUOTA"] = "70%"
    invoke()
    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len([call for call in calls if call[0] == "systemd-run"]) == len(jobs)
    assert ["systemctl", "--user", "start", "oil-news-filter.timer"] in calls
    state = json.loads(state_path.read_text())
    assert state["quotas"] == {
        f"oil-news-{worker}.service": "CPUQuota=70%"
        for worker in ("index", "filter", "texts", "snapshot")
    }

    # Pausing can unload the timers while the frozen services survive. Resume
    # must thaw those services and create only the timers, without new commands.
    for worker in ("filter", "texts", "snapshot"):
        del state["units"][f"oil-news-{worker}.timer"]
    state["frozen"] = ["oil-news-filter.service", "oil-news-texts.service"]
    state_path.write_text(json.dumps(state))
    before_count = len(calls)
    invoke()
    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    resumed_jobs = [call for call in calls[before_count:] if call[0] == "systemd-run"]
    assert len(resumed_jobs) == 3
    assert {
        next(arg for arg in job if arg.startswith("--unit=")) for job in resumed_jobs
    } == {f"--unit=oil-news-{worker}" for worker in ("filter", "texts", "snapshot")}
    for job in resumed_jobs:
        assert "--stage" not in job
        assert "/usr/bin/env" not in job
    state = json.loads(state_path.read_text())
    assert state["frozen"] == []
    assert state["units"]["oil-news-filter.service"] == "active"
    assert state["units"]["oil-news-texts.service"] == "active"
    assert all(
        state["units"][f"oil-news-{worker}.timer"] == "active"
        for worker in ("filter", "texts", "snapshot")
    )

    # Invalid input must fail before changing any systemd state.
    before = log_path.read_text()
    env["OIL_NEWS_CPU_QUOTA"] = "0%"
    with pytest.raises(subprocess.CalledProcessError):
        invoke()
    assert log_path.read_text() == before
