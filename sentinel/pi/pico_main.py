"""
Pico firmware: report DS18B20 temperatures and door states over USB serial.

Wiring:
  DS18B20 data -> GP4, with a 4.7k pull-up resistor between data and 3V3.
  Door switch -> GP5 and GND, using the Pico's internal pull-up.

Output format (one comma-separated record per line):
  DATA,<sensor_rom_id>,<temp_celsius>      -- temperature probe
  DOOR,<door_id>,<open|closed>             -- door state
"""

import time
import machine
import onewire
import ds18x20

# ---- Configuration: match your actual wiring ----
TEMP_DATA_PIN = 4          # DS18B20 data GPIO.
DOOR_PIN = 5                # Door switch signal GPIO.
DOOR_ID = "door-gp5"         # Door ID used by probeconfig.json.

TEMP_READ_INTERVAL = 5       # Temperature reading interval in seconds.
DOOR_POLL_INTERVAL = 1       # Door reporting interval in seconds.
# -----------------------------------------------

ow_pin = machine.Pin(TEMP_DATA_PIN)
ow = onewire.OneWire(ow_pin)
ds = ds18x20.DS18X20(ow)

door = machine.Pin(DOOR_PIN, machine.Pin.IN, machine.Pin.PULL_UP)


def rom_to_id(rom):
    return "-".join(["{:02x}".format(b) for b in rom])


def door_state_str():
    # Internal pull-up: closed circuit -> 0 -> "closed".
    # Open circuit -> 1 -> "open".
    return "closed" if door.value() == 0 else "open"


def read_and_report_temps():
    roms = ds.scan()
    if not roms:
        print("# NO_SENSORS")
        return
    ds.convert_temp()
    time.sleep_ms(750)  # DS18B20 conversion requires at least 750 ms.
    for rom in roms:
        try:
            temp = ds.read_temp(rom)
            print("DATA,{},{:.2f}".format(rom_to_id(rom), temp))
        except Exception as e:
            print("# READ_ERROR,{}".format(e))


def main():
    if not ds.scan():
        print("# ERROR: no DS18B20 found, check wiring/pullup resistor")

    last_temp_read = time.time() - TEMP_READ_INTERVAL  # Read temperatures on the first iteration.
    last_door_state = None

    while True:
        now = time.time()

        # ---- Read and report the door state on each iteration. ----
        state = door_state_str()
        print("DOOR,{},{}".format(DOOR_ID, state))
        if state != last_door_state:
            last_door_state = state

        # ---- Read temperatures at the configured interval. ----
        if now - last_temp_read >= TEMP_READ_INTERVAL:
            read_and_report_temps()
            last_temp_read = now

        time.sleep(DOOR_POLL_INTERVAL)


if __name__ == "__main__":
    main()
