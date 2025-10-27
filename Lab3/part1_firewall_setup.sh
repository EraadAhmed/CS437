#!/usr/bin/env bash
set -euo pipefail

# Lab 3 - Step 1 Firewalling helper script
# This script installs and configures UFW on Raspberry Pi OS (or other Debian-based systems)
# while capturing command output for later documentation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/part1_firewall_outputs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/ufw_setup_${TIMESTAMP}.log"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
ALLOWED_RULES_FILE="${OUTPUT_DIR}/allowed_rules_${TIMESTAMP}.txt"

# List of firewall rules to allow. Update this list to match the services you actually need.
# Supported formats include application profiles (e.g., "ssh"), port/protocol pairs (e.g., "1883/tcp"),
# or full rule specifications accepted by UFW (e.g., "proto tcp from any to any port 8883").
readarray -t ALLOWED_RULES <<'EOF'
ssh
1883/tcp
8883/tcp
5000/tcp
EOF

RESET_UFW="${RESET_UFW:-false}"
ALLOW_VNC="${ALLOW_VNC:-false}"
VNC_PORT="${VNC_PORT:-5900}"

mkdir -p "${OUTPUT_DIR}"

echo "Lab 3 Part 1 firewall setup run at ${TIMESTAMP}" | tee "${SUMMARY_FILE}"

run_cmd() {
    local description="$1"
    shift
    local -a cmd=("$@")
    {
        echo "===== ${description} ====="
        echo "+ ${cmd[*]}"
        "${cmd[@]}"
        echo
    } | tee -a "${LOG_FILE}"
}

append_heading() {
    local heading="$1"
    {
        echo "===== ${heading} ====="
    } | tee -a "${LOG_FILE}"
}

collect_system_context() {
    append_heading "System context"
    run_cmd "Date" date --iso-8601=seconds
    run_cmd "Hostname" hostnamectl
    run_cmd "Kernel" uname -a
    if command -v lsb_release >/dev/null 2>&1; then
        run_cmd "Distribution" lsb_release -a
    fi
    run_cmd "Network interfaces" ip -brief addr
}

install_ufw() {
    append_heading "Install and enable UFW"
    run_cmd "Package index refresh" sudo apt-get update
    run_cmd "Install ufw" sudo apt-get install -y ufw
    if [[ "${RESET_UFW}" == "true" ]]; then
        run_cmd "Reset existing UFW rules" sudo ufw --force reset
    fi
}

snapshot_pre_state() {
    append_heading "Pre-change firewall state"
    run_cmd "Current UFW status" sudo ufw status verbose
    if command -v iptables >/dev/null 2>&1; then
        run_cmd "iptables filter table" sudo iptables -L -n -v
    fi
    if command -v nft >/dev/null 2>&1; then
        run_cmd "nftables ruleset" sudo nft list ruleset
    fi
}

configure_defaults() {
    append_heading "Configure default policies"
    run_cmd "Allow outgoing by default" sudo ufw default allow outgoing
    run_cmd "Deny incoming by default" sudo ufw default deny incoming
}

apply_allowed_rules() {
    append_heading "Apply allow rules"
    printf "Allowed rules (edit the ALLOWED_RULES array in the script):\n" > "${ALLOWED_RULES_FILE}"
    local rule
    for rule in "${ALLOWED_RULES[@]}"; do
        # Skip commented or empty lines.
        if [[ -z "${rule}" || "${rule}" == \#* ]]; then
            continue
        fi
        printf -- "- %s\n" "${rule}" | tee -a "${ALLOWED_RULES_FILE}"
        run_cmd "Allow ${rule}" sudo ufw allow ${rule}
    done

    if [[ "${ALLOW_VNC}" == "true" ]]; then
        printf -- "- VNC (port %s/tcp)\n" "${VNC_PORT}" | tee -a "${ALLOWED_RULES_FILE}"
        run_cmd "Allow VNC" sudo ufw allow ${VNC_PORT}/tcp
    fi
}

finalize_firewall() {
    append_heading "Enable firewall"
    run_cmd "Enable UFW" sudo ufw --force enable
    append_heading "Post-change firewall state"
    run_cmd "UFW status" sudo ufw status numbered
    if command -v ufw >/dev/null 2>&1; then
        run_cmd "UFW app list" sudo ufw app list
    fi
    if command -v iptables >/dev/null 2>&1; then
        run_cmd "iptables after changes" sudo iptables -L -n -v
    fi
}

collect_verification() {
    append_heading "Verification checks"
    run_cmd "Listening sockets" sudo ss -tulpen
}

collect_system_context
install_ufw
snapshot_pre_state
configure_defaults
apply_allowed_rules
finalize_firewall
collect_verification

echo "Outputs stored in ${OUTPUT_DIR}" | tee -a "${SUMMARY_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
echo "Allowed rules record: ${ALLOWED_RULES_FILE}" | tee -a "${SUMMARY_FILE}"

echo "Run complete. Review the log and summary files for documentation." | tee -a "${SUMMARY_FILE}"
