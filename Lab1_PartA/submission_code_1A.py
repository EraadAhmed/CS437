#!/usr/bin/env python3
from picamera2 import Picamera2
from picarx import Picarx
import time

# -------------------- Tunables --------------------
# Steering
MAX_STEER_DEG = 30.0
STEER_RATE_DEG_PER_S = 200.0
SERVO_DIR = 1                  # flip to -1 if steering is reversed

# Speed
BASE_SPEED = 5                 # forward speed on straight
MIN_SPEED = 2
ACCEL_PER_S = 15.0             # speed ramp limit (units/s)

# PD gains (corridor centering using side sensors)
KP = 20.0
KD = 4.0

# Filters
ERR_EMA_ALPHA = 0.35           # error smoothing
ERROR_DEADBAND = 0.04          # |-err| below this treated as 0

# Grayscale normalization (black line is dark)
ADC_MAX = 4095.0
LINE_IS_DARK = True
HIT_THRESH = 0.65              # immediate push-away when side this dark
WEAK_SUM_THRESH = 0.06         # sL+sR below this => no boundaries seen

# Ultrasonic (cm)
DANGER_DIST = 25.0             # enter blocked state below this
SAFE_DIST   = 30.0             # leave blocked state above this
US_SAMPLES  = 5                # median over N samples to reduce spikes

# Camera
SNAP_PATH   = "/home/desi/picar-x/class_work/latest.jpg"
SNAP_PERIOD = 0.3

CONTROL_HZ  = 25.0
# --------------------------------------------------

px = Picarx(servo_pins=['P0','P1','P3'])

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(1.5)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

def read_boundary_signals():
    """Return raw [L,M,R] and normalized 0..1 signals (0=floor, 1=line)."""
    raw = px.get_grayscale_data()
    L, M, R = [float(v) for v in raw]

    # --- Empirical floor vs line ranges (tune these!) ---
    floor_val = 1050.0   # typical reading on white floor
    line_val  = 750.0    # typical reading on black tape

    def normalize(v):
        # 0 on floor, 1 on line
        return clamp((floor_val - v) / (floor_val - line_val + 1e-6), 0.0, 1.0)

    sL, sM, sR = normalize(L), normalize(M), normalize(R)
    return raw, sL, sM, sR


# State
prev_time = time.monotonic()
err_f = 0.0
prev_err = 0.0
steer_deg = 0.0
speed_cmd = 0.0
last_snap = 0.0
blocked = False                 # hysteresis state for ultrasonic

try:
    while True:
        now = time.monotonic()
        dt = max(1.0/CONTROL_HZ, now - prev_time)
        prev_time = now

        # --- Sensors ---
        dist = round(px.ultrasonic.read(), 2)
        raw, sL, sM, sR = read_boundary_signals()
        side_sum = sL + sR

        # --- Obstacle hysteresis (stable behavior) ---
        if blocked:
            if dist == -1 or dist > SAFE_DIST:
                blocked = False
        else:
            if dist != -1 and dist < DANGER_DIST:
                blocked = True

        if blocked:
            # Stop & back up gently while pointing straight
            px.stop()
            px.set_dir_servo_angle(0)
            px.backward(BASE_SPEED)
            time.sleep(0.25)
            px.stop()
            # Skip lane control this cycle
            continue

        # --- Corridor control (stay between black side lines) ---
        # Error: positive when right boundary darker -> steer LEFT.
        # Use sR - sL and invert with SERVO_DIR below to match hardware.
        err = sR - sL

        # Deadband for jitter
        if abs(err) < ERROR_DEADBAND:
            err = 0.0

        # If a side is *very* dark, force a strong push-away regardless of PD
        force_away = None
        if sL >= HIT_THRESH and sR < HIT_THRESH:
            force_away = +0.9 * MAX_STEER_DEG  # steer right
        elif sR >= HIT_THRESH and sL < HIT_THRESH:
            force_away = -0.9 * MAX_STEER_DEG  # steer left

        if force_away is not None:
            raw_cmd = force_away
        else:
            # If no boundaries sensed, creep straight and try to reacquire
            if side_sum < WEAK_SUM_THRESH:
                raw_cmd = 0.0
            else:
                # PD on filtered error
                err_f = ema(err_f, err, ERR_EMA_ALPHA)
                derr = (err_f - prev_err) / dt
                prev_err = err_f
                raw_cmd = KP * err_f + KD * derr

        # Apply steering direction, clamp, and rate limit
        raw_cmd *= SERVO_DIR
        raw_cmd = clamp(raw_cmd, -MAX_STEER_DEG, MAX_STEER_DEG)
        max_delta = STEER_RATE_DEG_PER_S * dt
        steer_deg += clamp(raw_cmd - steer_deg, -max_delta, +max_delta)
        steer_deg = clamp(steer_deg, -MAX_STEER_DEG, MAX_STEER_DEG)
        px.set_dir_servo_angle(int(round(steer_deg)))

        # Speed: slow down with large steering, but don't stall
        turn_factor = 1.0 - 0.6 * (abs(steer_deg) / MAX_STEER_DEG)   # softer penalty
        target_speed = clamp(BASE_SPEED * turn_factor, MIN_SPEED, BASE_SPEED)

        # Smooth speed ramp
        if target_speed > speed_cmd:
            speed_cmd = min(target_speed, speed_cmd + ACCEL_PER_S * dt)
        else:
            speed_cmd = max(target_speed, speed_cmd - ACCEL_PER_S * dt)

        # If lost both lines, creep straight to search
        if side_sum < WEAK_SUM_THRESH:
            px.set_dir_servo_angle(0)
            px.forward(int(round(max(MIN_SPEED, 0.6 * BASE_SPEED))))
        else:
            px.forward(int(round(speed_cmd)))

        # Optional snapshot (throttled)
        if (now - last_snap) >= SNAP_PERIOD:
            try:
                picam2.capture_file(SNAP_PATH)
            except Exception:
                pass
            last_snap = now

        # Pace the loop
        sleep_left = (1.0/CONTROL_HZ) - (time.monotonic() - now)
        if sleep_left > 0:
            time.sleep(sleep_left)

except KeyboardInterrupt:
    print("\nCtrl+C detected, stopping...")
finally:
    try: px.stop()
    except: pass
    try: picam2.stop()
    except: pass
    print("Stopped cleanly")
