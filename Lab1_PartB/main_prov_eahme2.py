import asyncio
from navigation import Navigator
from picarx import Picarx

if __name__ == "__main__":
    nav = Navigator(Picarx(servo_pins=["P0", "P1", "P3"]))
    try:
        asyncio.run(nav.start())
    except KeyboardInterrupt:
        print("Stopping...")
        nav.stop()