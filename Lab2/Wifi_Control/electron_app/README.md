# IoT Car Control Electron App

This desktop UI provides real-time telemetry and remote driving controls for the Raspberry Pi car used in Lab 2. It connects to the WiFi socket server running on the Pi (`wifi_server.py`) and exchanges JSON messages over a raw TCP socket.

## Features

- One-click connect/disconnect to the Pi over TCP (defaults to `192.168.125.22:65432`).
- Live telemetry updates (temperature, speed, battery level, power output, steering angle).
- On-screen drive controls and keyboard bindings (arrow keys / WASD / space for stop).
- Event log for acknowledgements and connection transitions.
- Safe command queueing and graceful disconnect handling.

## Prerequisites

- Node.js 18+ (Electron 31 requires a modern Node runtime).
- npm (bundled with Node.js).
- The Raspberry Pi server from `../wifi_server.py` running and listening on the configured host/port.

## Installation

From the `electron_app` directory:

```bash
npm install
```

## Usage

1. Start the WiFi server on the Raspberry Pi:
   ```bash
   python3 wifi_server.py
   ```
2. On your laptop/desktop, launch the Electron client:
   ```bash
   npm start
   ```
3. Enter the Pi's IP address and port if they differ from the defaults, then click **Connect**.
4. Use the on-screen buttons or the keyboard shortcuts to drive the car. Telemetry will update once per second.

## Keyboard Shortcuts

| Keys            | Action    |
|-----------------|-----------|
| Arrow Up / W    | Forward   |
| Arrow Down / S  | Reverse   |
| Arrow Left / A  | Turn left |
| Arrow Right / D | Turn right|
| Space           | Stop      |

## Configuration

- Override the default host/port by setting environment variables before launching the app:
  ```bash
  CAR_HOST=192.168.1.42 CAR_PORT=65432 npm start
  ```
- The UI will remember the last values entered in the form for the current session.

## Message Format

The client expects newline-delimited JSON payloads from the Pi. Examples:

```json
{"type":"telemetry","temperature_c":37.12,"speed_cm_s":12.5,"battery_percent":88.3,"power_percent":40,"steering_deg":5,"timestamp":1730599463.12}
{"type":"ack","command":"FWD","status":"accelerating","power_percent":50,"steering_deg":0,"timestamp":1730599464.21}
```

Unknown payloads are echoed to the log for inspection, which helps diagnose firmware/backend issues without crashing the UI.

## Troubleshooting

- If the **Connect** button fails immediately, verify that the Pi is reachable (e.g., `ping <pi-ip>`).
- Ensure the Pi's firewall allows inbound TCP connections on the chosen port.
- If telemetry stops updating, the UI will show a disconnect warning; reconnect once the Pi server is running again.
- Electron may show a content-security-policy reminder in the console. The renderer already avoids `eval`, but you can silence the warning by adding a strict `Content-Security-Policy` meta tag in `public/index.html` (for example, `default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'`) if you prefer hardened logs.
