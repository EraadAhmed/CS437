# WiFi Control Stack – Runbook

This guide walks you through running the Raspberry Pi control server and the Electron desktop dashboard so you can steer the car and monitor telemetry over WiFi.

## Overview

Component | Role | Location
---------|------|---------
`wifi_server.py` | Python socket server that runs on the Raspberry Pi and directly controls the PicarX hardware. | `Wifi_Control/wifi_server.py`
Electron dashboard | Desktop UI that connects to the Pi, streams telemetry, and issues drive commands. | `Wifi_Control/electron_app`

## Prerequisites

### Raspberry Pi

- Raspberry Pi with Lab 2 hardware (PicarX) assembled.
- Raspberry Pi OS with Python 3.9+.
- `robot-hat` and `picarx` libraries installed (per Lab 1/2 setup).
- The car and your laptop must be on the same WiFi network.

### Laptop/Desktop

- macOS/Windows/Linux with Node.js 18+ and npm.
- Git working copy of this repository.

> **Tip:** If you already ran Lab 1, your Pi should have the required Python drivers. Otherwise follow the SunFounder PicarX setup instructions first.

## 1. Configure the Raspberry Pi Server

1. SSH into the Pi or open a terminal directly on it.
2. Navigate to the project folder (adjust the path if your checkout differs):

   ```bash
   cd ~/CS_437/Github/Lab2/Wifi_Control
   ```
3. (Optional) Update the host/port constants in `wifi_server.py` if the defaults do not match your network. By default the script binds to the Pi’s IP address and port `65432`.
4. Start the server:


   ```bash
   python3 wifi_server.py
   ```

What to expect:

- The script prints `Connected` once a desktop client connects.

- Telemetry is streamed as newline-delimited JSON once per second.
- Acknowledgements are sent after each command the client issues.

To stop the server, press `Ctrl+C`. The vehicle will automatically brake and straighten the wheels on shutdown.

## 2. Launch the Electron Dashboard

1. On your laptop/desktop, open a terminal and go to the Electron app folder:

   ```bash
   cd /Users/saad/Documents/School/CS/CS_437/Github/Lab2/Wifi_Control/electron_app
   ```

2. Install dependencies (first time only):

   ```bash
   npm install
   ```

3. Start the desktop app:

   ```bash
   npm start
   ```

The Electron window should appear. Enter the Pi’s IP address and port (defaults to `192.168.125.22:65432`) and click **Connect**. Once connected you can drive using the on-screen buttons or keyboard shortcuts (WASD / arrow keys / space for stop). Telemetry tiles update once per second.

## 3. Keyboard Shortcuts

Key(s) | Action
-------|-------
W / ↑ | Accelerate forward
S / ↓ | Reverse / reduce forward speed
A / ← | Steer left
D / → | Steer right
Space | Immediate stop

## 4. Customising Host/Port

- Change the defaults in the UI form each time you launch, **or**
- Set environment variables before `npm start` to persist ephemeral defaults:

  ```bash
  CAR_HOST=192.168.1.42 CAR_PORT=65432 npm start
  ```

## 5. Troubleshooting

Symptom | Possible Fix
-------|--------------
Cannot connect | Verify the Pi’s IP with `hostname -I`, ensure both devices are on the same network, and confirm port `65432` is open.
Immediate disconnects | Check `wifi_server.py` logs for errors; hardware issues (dead battery) will trigger shutdown.
No telemetry updates | Ensure the server is running after you connect. Restart the Electron app if the Pi rebooted.
Keyboard controls unresponsive | The Electron window must be focused; click inside the window before using shortcuts.

## 6. Shutdown & Cleanup

1. From the Electron app click **Disconnect**; the car will maintain its last command until told to stop.
2. In the Pi terminal press `Ctrl+C` to stop `wifi_server.py`; the car will brake and straighten automatically.
3. Close the Electron window or stop the npm process with `Ctrl+C`.

You are now ready to record demo videos or extend the telemetry packets with additional sensor data.
