#!/usr/bin/env bash
set -euo pipefail

RAY_BIN=""
RAY_ROLE=""

usage() {
    cat <<'EOF'
Usage:
  scripts/run_ray_cluster.sh [options] -- <command> [args...]

Purpose:
  Start a static Ray cluster, then run <command> on node-rank 0.

Options:
  --node-count N              Total cluster nodes. Default: 1.
  --node-rank N               Current node rank. Default: 0.
  --head-addr HOST            Address workers use to reach Ray head.
  --head-node-ip HOST         Node IP passed to ray start --head. Default: head addr.
  --ray-port PORT             Ray head port. Default: 6379.
  --ray-bin PATH              Ray executable. Default: ray, then ./.venv/bin/ray.
  --python-bin PATH           Python executable for readiness checks. Default: ./.venv/bin/python, python3, python.
  --wait-timeout SECONDS      Startup wait timeout. Default: 600.
  --worker-poll-interval SEC  Worker polling interval. Default: 10.
  --include-dashboard BOOL    Enable Ray dashboard. Default: false.
  --dashboard-port PORT       Ray dashboard port. Default: 8265.
  --metrics-port PORT         Ray metrics export port. Default: 8080.
  --client-port PORT          Optional Ray Client server port.
EOF
}

log() {
    echo "Ray Cluster: $*" >&2
}

fail_usage() {
    echo "Error: $*" >&2
    usage >&2
    exit 2
}

is_uint() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

cleanup_ray() {
    local status=$?
    if [[ -n "${RAY_BIN}" ]]; then
        log "stopping Ray ${RAY_ROLE:-runtime}"
        "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
    exit "${status}"
}

main() {
    local node_count="1"
    local node_rank="0"
    local head_addr=""
    local head_node_ip=""
    local ray_port="6379"
    local ray_bin_arg=""
    local python_bin=""
    local wait_timeout="600"
    local worker_poll_interval="10"
    local include_dashboard="false"
    local dashboard_port="8265"
    local metrics_port="8080"
    local client_port=""
    local command=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --node-count)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                node_count="$2"
                shift 2
                ;;
            --node-rank)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                node_rank="$2"
                shift 2
                ;;
            --head-addr)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                head_addr="$2"
                shift 2
                ;;
            --head-node-ip)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                head_node_ip="$2"
                shift 2
                ;;
            --ray-port)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                ray_port="$2"
                shift 2
                ;;
            --ray-bin)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                ray_bin_arg="$2"
                shift 2
                ;;
            --python-bin)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                python_bin="$2"
                shift 2
                ;;
            --wait-timeout)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                wait_timeout="$2"
                shift 2
                ;;
            --worker-poll-interval)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                worker_poll_interval="$2"
                shift 2
                ;;
            --include-dashboard)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                include_dashboard="$2"
                shift 2
                ;;
            --dashboard-port)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                dashboard_port="$2"
                shift 2
                ;;
            --metrics-port)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                metrics_port="$2"
                shift 2
                ;;
            --client-port)
                [[ $# -ge 2 ]] || fail_usage "$1 requires a value"
                client_port="$2"
                shift 2
                ;;
            --)
                shift
                command=("$@")
                break
                ;;
            *)
                fail_usage "unknown option before --: $1"
                ;;
        esac
    done

    [[ ${#command[@]} -gt 0 ]] || fail_usage "missing command."
    is_uint "${node_count}" && [[ "${node_count}" -ge 1 ]] || fail_usage "invalid node count: ${node_count}"
    is_uint "${node_rank}" || fail_usage "invalid node rank: ${node_rank}"
    is_uint "${ray_port}" || fail_usage "invalid Ray port: ${ray_port}"
    is_uint "${wait_timeout}" || fail_usage "invalid wait timeout: ${wait_timeout}"
    is_uint "${worker_poll_interval}" || fail_usage "invalid worker poll interval: ${worker_poll_interval}"
    is_uint "${dashboard_port}" || fail_usage "invalid dashboard port: ${dashboard_port}"
    is_uint "${metrics_port}" || fail_usage "invalid metrics port: ${metrics_port}"
    [[ -z "${client_port}" || "${client_port}" =~ ^[0-9]+$ ]] || fail_usage "invalid client port: ${client_port}"

    if [[ "${node_count}" -eq 1 ]]; then
        log "single-node job detected; running command unchanged."
        exec "${command[@]}"
    fi

    [[ -n "${head_addr}" ]] || fail_usage "--head-addr is required for multi-node Ray jobs."
    [[ -n "${head_node_ip}" ]] || head_node_ip="${head_addr}"

    if [[ -n "${ray_bin_arg}" ]]; then
        if command -v "${ray_bin_arg}" >/dev/null 2>&1 || [[ -x "${ray_bin_arg}" ]]; then
            RAY_BIN="${ray_bin_arg}"
        else
            echo "Error: ray executable is not runnable: ${ray_bin_arg}" >&2
            exit 1
        fi
    elif command -v ray >/dev/null 2>&1; then
        RAY_BIN="$(command -v ray)"
    elif [[ -x "./.venv/bin/ray" ]]; then
        RAY_BIN="./.venv/bin/ray"
    else
        echo "Error: ray executable not found. Use --ray-bin." >&2
        exit 1
    fi

    if [[ -n "${python_bin}" ]]; then
        if ! command -v "${python_bin}" >/dev/null 2>&1 && [[ ! -x "${python_bin}" ]]; then
            echo "Error: python executable is not runnable: ${python_bin}" >&2
            exit 1
        fi
    elif [[ -x "./.venv/bin/python" ]]; then
        python_bin="./.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        python_bin="$(command -v python)"
    else
        echo "Error: python executable not found. Use --python-bin." >&2
        exit 1
    fi

    local ray_address="${head_addr}:${ray_port}"
    export RAY_ADDRESS="${ray_address}"
    export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
    export NO_PROXY="${head_addr},127.0.0.1,localhost,${NO_PROXY:-}"
    export no_proxy="${head_addr},127.0.0.1,localhost,${no_proxy:-}"

    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true

    if [[ "${node_rank}" -eq 0 ]]; then
        RAY_ROLE="head"
        trap cleanup_ray EXIT INT TERM

        local ray_start_args=(
            start
            --head
            --node-ip-address="${head_node_ip}"
            --port="${ray_port}"
            --dashboard-host=0.0.0.0
            --dashboard-port="${dashboard_port}"
            --metrics-export-port="${metrics_port}"
            --include-dashboard="${include_dashboard}"
        )
        [[ -z "${client_port}" ]] || ray_start_args+=(--ray-client-server-port="${client_port}")

        log "starting Ray head on ${head_node_ip}:${ray_port}"
        "${RAY_BIN}" "${ray_start_args[@]}"

        local deadline=$((SECONDS + wait_timeout))
        local alive_count="0"
        while [[ "${SECONDS}" -lt "${deadline}" ]]; do
            alive_count="$(RAY_ADDRESS="${ray_address}" "${python_bin}" - <<'PY' 2>/dev/null | tail -n 1 || true
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR", log_to_driver=False)
try:
    print(sum(1 for node in ray.nodes() if node.get("Alive")))
finally:
    ray.shutdown()
PY
)"
            if is_uint "${alive_count}" && [[ "${alive_count}" -ge "${node_count}" ]]; then
                log "Ray cluster has ${alive_count}/${node_count} alive nodes"
                log "running command with RAY_ADDRESS=${RAY_ADDRESS}: ${command[*]}"
                "${command[@]}"
                return
            fi
            log "waiting for Ray nodes: ${alive_count:-0}/${node_count} alive"
            sleep 2
        done
        echo "Error: timed out waiting for ${node_count} Ray nodes at ${ray_address}; last alive count=${alive_count:-0}" >&2
        exit 1
    else
        RAY_ROLE="worker"
        trap cleanup_ray EXIT INT TERM

        local deadline=$((SECONDS + wait_timeout))
        while [[ "${SECONDS}" -lt "${deadline}" ]]; do
            if "${RAY_BIN}" status --address="${ray_address}" >/dev/null 2>&1; then
                log "Ray head is reachable at ${ray_address}"
                break
            fi
            log "waiting for Ray head at ${ray_address}"
            sleep 2
        done
        [[ "${SECONDS}" -lt "${deadline}" ]] || {
            echo "Error: timed out waiting for Ray head at ${ray_address}" >&2
            exit 1
        }

        log "starting Ray worker for ${ray_address}"
        "${RAY_BIN}" start --address="${ray_address}"

        log "Ray worker joined; waiting for head shutdown"
        while "${RAY_BIN}" status --address="${ray_address}" >/dev/null 2>&1; do
            sleep "${worker_poll_interval}"
        done
        log "Ray head is no longer reachable; worker exiting"
    fi
}

main "$@"
