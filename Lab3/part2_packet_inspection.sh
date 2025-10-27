#!/usr/bin/env bash
set -euo pipefail

# Lab 3 - Part 2 Packet Inspection helper script
# Installs Wireshark/tshark, performs a capture on the chosen interface,
# generates traffic for evidence, and collates logs for the lab report.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/part2_packet_outputs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/packet_inspection_${TIMESTAMP}.log"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
PCAP_FILE="${OUTPUT_DIR}/capture_${TIMESTAMP}.pcapng"
PCAP_TMP="/tmp/packet_capture_${TIMESTAMP}.pcapng"
CAP_STDOUT="${OUTPUT_DIR}/tshark_capture_${TIMESTAMP}.txt"
TEXT_REPORT="${OUTPUT_DIR}/analysis_${TIMESTAMP}.txt"

NET_IFACE="${NET_IFACE:-wlan0}"
PING_TARGET="${PING_TARGET:-8.8.8.8}"
HTTP_URL="${HTTP_URL:-https://example.com}"

mkdir -p "${OUTPUT_DIR}"

echo "Lab 3 Part 2 packet inspection run at ${TIMESTAMP}" | tee "${SUMMARY_FILE}"

die() {
    echo "Error: $*" >&2
    exit 1
}

run_cmd() {
    local description="$1"
    shift
    local -a cmd=("$@")
    {
        echo "===== ${description} ====="
        echo "+ ${cmd[*]}"
        if ! "${cmd[@]}"; then
            echo "Command failed with exit code $?" >&2
            exit 1
        fi
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

install_tools() {
    append_heading "Install Wireshark toolchain"
    run_cmd "Add wireshark setuid preseed" bash -c "echo 'wireshark-common wireshark-common/install-setuid boolean true' | sudo debconf-set-selections"
    run_cmd "Install wireshark packages" sudo DEBIAN_FRONTEND=noninteractive apt-get install -y wireshark tshark
    run_cmd "Reconfigure dumpcap permissions" sudo dpkg-reconfigure -f noninteractive wireshark-common
    run_cmd "Ensure dumpcap executable" sudo chmod +x /usr/bin/dumpcap
    run_cmd "Add current user to wireshark group" sudo usermod -aG wireshark "${USER}"
}

list_capture_interfaces() {
    append_heading "Available capture interfaces"
    run_cmd "tshark interface list" sudo tshark -D
}

run_capture_session() {
    append_heading "Packet capture session"
    {
        echo "Capturing on interface ${NET_IFACE} for 20 seconds"
        echo "PCAP destination: ${PCAP_FILE}"
    } | tee -a "${LOG_FILE}"

    sudo tshark -i "${NET_IFACE}" -a duration:20 -w "${PCAP_TMP}" >"${CAP_STDOUT}" 2>&1 &
    local capture_pid=$!
    echo "tshark PID ${capture_pid}" | tee -a "${LOG_FILE}"

    sleep 3
    run_cmd "Ping ${PING_TARGET}" ping -c 6 "${PING_TARGET}"
    run_cmd "HTTP HEAD ${HTTP_URL}" curl -fsSI --max-time 10 "${HTTP_URL}"

    wait "${capture_pid}"
    echo "tshark capture output" | tee -a "${LOG_FILE}"
    cat "${CAP_STDOUT}" | tee -a "${LOG_FILE}"

    if [[ ! -s "${PCAP_TMP}" ]]; then
        die "Capture file ${PCAP_TMP} is empty"
    fi

    sudo chown "${USER}:${USER}" "${PCAP_TMP}"
    mv "${PCAP_TMP}" "${PCAP_FILE}"
}

analyze_capture() {
    append_heading "Capture analysis"
    run_cmd "capinfos summary" capinfos "${PCAP_FILE}"
    run_cmd "Top talkers" sudo tshark -r "${PCAP_FILE}" -q -z endpoints,ip
    run_cmd "Protocol hierarchy" sudo tshark -r "${PCAP_FILE}" -q -z io,phs
    run_cmd "First 10 packets" sudo tshark -r "${PCAP_FILE}" -c 10

    {
        echo "Capture: ${PCAP_FILE}"
        echo "Interface: ${NET_IFACE}"
        echo "Ping target: ${PING_TARGET}"
        echo "HTTP URL: ${HTTP_URL}"
        echo
        echo "Protocol hierarchy (tshark -z io,phs):"
        sudo tshark -r "${PCAP_FILE}" -q -z io,phs
        echo
        echo "Endpoint summary (tshark -z endpoints,ip):"
        sudo tshark -r "${PCAP_FILE}" -q -z endpoints,ip
    } | tee "${TEXT_REPORT}"
}

collect_system_context
install_tools
list_capture_interfaces
run_capture_session
analyze_capture

echo "Outputs stored in ${OUTPUT_DIR}" | tee -a "${SUMMARY_FILE}"
echo "Capture file: ${PCAP_FILE}" | tee -a "${SUMMARY_FILE}"
echo "Analysis notes: ${TEXT_REPORT}" | tee -a "${SUMMARY_FILE}"

echo "Run complete. Log available at ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
echo "Note: if you want to run capture without sudo, log out/in after the wireshark group change." | tee -a "${SUMMARY_FILE}"
