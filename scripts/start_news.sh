#!/usr/bin/env bash
# One Media Cloud key: US, Saudi Arabia, UAE, then incomplete IT, ES, OM, AZ.
# Classification, article fetching, and dataset snapshots run independently
# against data already on disk.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY="$REPO/.venv/bin/python"
# systemd percentages are relative to ONE logical CPU, not the entire machine.
# The classifier measured ~108% CPU before limiting; 85% is about 80% of that.
CPU_QUOTA=${OIL_NEWS_CPU_QUOTA:-85%}
if [[ ! $CPU_QUOTA =~ ^[1-9][0-9]*%$ ]]; then
  echo "OIL_NEWS_CPU_QUOTA must be a positive integer followed by %, e.g. 85%." >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY; run uv sync --all-groups in $REPO first." >&2
  exit 1
fi

# Fail before making changes if the user bus is unavailable.
systemctl --user show-environment >/dev/null

thaw_if_frozen() {
  local freezer
  freezer=$(systemctl --user show "$1" --property=FreezerState --value)
  case "$freezer" in
    frozen|freezing|thawing) systemctl --user thaw "$1" ;;
  esac
}

# Retire every legacy multi-key downloader and the old combined worker.
# Stop the old timer first, then explicitly thaw and stop each legacy service.
# Some systemd versions refuse stop on a frozen unit. Handle services one at a
# time so the old multi-key downloaders are not all resumed together.
mapfile -t obsolete < <(
  systemctl --user list-units --all --plain --no-legend \
    'oil-news-download-*.service' 'oil-news-us.service' 'oil-news-process.service' \
    'oil-news-process.timer' | awk '{print $1}'
)
if ((${#obsolete[@]})); then
  for unit in "${obsolete[@]}"; do
    if [[ $unit == *.timer ]]; then
      systemctl --user stop "$unit"
    fi
  done
  for unit in "${obsolete[@]}"; do
    [[ $unit == *.service ]] || continue
    thaw_if_frozen "$unit"
    systemctl --user stop "$unit"
  done
fi
systemctl --user reset-failed 'oil-news-*' 2>/dev/null || true

missing() {
  case "$(systemctl --user is-active "$1" || true)" in
    active|activating|reloading) return 1 ;;
    *) return 0 ;;
  esac
}
loaded() {
  [[ $(systemctl --user show "$1" --property=LoadState --value) == loaded ]]
}
run() {
  systemd-run --user -q --working-directory="$REPO" \
    --property=Environment=PYTHONUNBUFFERED=1 "$@"
}
periodic() { # unit, interval, stage
  # Update existing services in place, including workers already running.
  # Runtime overrides also apply to later activations of these loaded units.
  if loaded "$1.service"; then
    thaw_if_frozen "$1.service"
    systemctl --user set-property --runtime "$1.service" "CPUQuota=$CPU_QUOTA"
  fi
  missing "$1.timer" || return 0
  if loaded "$1.timer"; then
    systemctl --user start "$1.timer"
    return 0
  fi
  if loaded "$1.service"; then
    # A stopped transient timer may have been unloaded while its frozen service
    # survived. With no command, systemd-run creates only the missing timer.
    systemd-run --user -q --unit="$1" \
      --on-active=1s --on-unit-inactive="$2"
  else
    run --unit="$1" --on-active=1s --on-unit-inactive="$2" \
      --property="CPUQuota=$CPU_QUOTA" \
      --property=Environment=FULLTEXT_REGION_LIMIT=5000 \
      /usr/bin/env -u MEDIACLOUD_API_KEY -u VIRTUAL_ENV \
      "$PY" -m news run --stage "$3"
  fi
}

if missing oil-news-awake.service; then
  if loaded oil-news-awake.service; then
    systemctl --user start oil-news-awake.service
  else
    run --unit=oil-news-awake \
      systemd-inhibit --what=sleep:idle:handle-lid-switch --who=oil-news-graph \
      --why="News index download and processing" sleep infinity
  fi
fi

if loaded oil-news-index.service; then
  thaw_if_frozen oil-news-index.service
  systemctl --user set-property --runtime oil-news-index.service "CPUQuota=$CPU_QUOTA"
fi
if missing oil-news-index.service; then
  # Completed windows are skipped. The same process and .env key handle all
  # queued regions sequentially; a quota failure resumes this queue hourly.
  if loaded oil-news-index.service; then
    systemctl --user start oil-news-index.service
  else
    run --unit=oil-news-index -p Restart=on-failure -p RestartSec=1h \
      --property="CPUQuota=$CPU_QUOTA" \
      /usr/bin/env -u MEDIACLOUD_API_KEY -u VIRTUAL_ENV \
      "$PY" -m news run --stage download --only us,sa,ae,it,es,om,az
  fi
fi

# CUDA is selected automatically when available. The text worker uses the CPU
# for extraction concurrently and never loads the encoder or calls Media Cloud.
# Fetch at most 5000 new pages per region per pass so one large region cannot
# hold up datasets for every other region. Each pass visits the US first.
periodic oil-news-filter 30min filter
periodic oil-news-texts 5min fulltext
periodic oil-news-snapshot 15min assemble

systemctl --user list-units 'oil-news-*' --all --no-pager

# Pause everything:
# systemctl --user stop oil-news-{filter,texts,snapshot}.timer \
#   oil-news-{index,filter,texts,snapshot,awake}.service
# Resume: ~/Projects/oil-news-start.sh
