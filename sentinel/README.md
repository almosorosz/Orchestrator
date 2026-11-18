# Sentinel: Connecting Pi and PC over Ethernet or Wi-Fi

**History, automatic PC saving, and cloud sync folder copies**: see [Update and usage instructions for both devices](pi/HISTORY.md).
The Pi records every 5 seconds by default; the PC saves every 60 seconds and catches up after a disconnection.
Copies in the cloud sync folder are uploaded by the PC's sync client. The page distinguishes local file saving from remote upload status.

Sentinel's data acquisition, device configuration, event CSV, and camera services run on the Pi. The PC's Sentinel page displays the Pi's web interface over HTTP. Shutting down or disconnecting the PC does not stop data acquisition on the Pi.

The Sentinel page provides two connection options: `Ethernet` and `Wi-Fi`. Each option remembers its own Pi address. Clicking Connect saves the selected mode and address. Switching modes stops the old connection; enter the new address and click Connect. Each mode's **Help** button provides the corresponding setup steps in English.

This selection controls the address used to reach the Pi. Configure the Pi's Wi-Fi network and campus account in the Pi's own system network settings. The PC can reach the Pi over Wi-Fi or the campus wired network if the campus network allows communication between the devices. The application does not change network adapter settings on either device.

The page contains a title, connection controls, status, and dashboard. Setup instructions are available under Help. Project files now use `sentinel/`, and saved connection settings from the previous name are migrated automatically.

**TOPDON TC001 thermal camera integration**: see [Pi wiring, update, and startup instructions](pi/TC001.md).
This feature requires updating the Pi's `app.py` and web page and adding `tc001_reader.py`. Running `python app.py` now automatically detects the TC001 by default; set `THERMAL_DEVICE=off` to disable it. The PC's Sentinel connection settings remain the same. For automatic startup at boot, see [Pi autostart](pi/AUTOSTART.md).

## Files and changes

- `gui/pages/sentinel_page.py`: embedded web page, separate Ethernet/Wi-Fi addresses, connection mode selection and help, asynchronous sensor API checks, timeout messages, reconnection every 5 seconds, refresh, external browser access, and file downloads.
- `MAIN_WINDOW.py`: stops Sentinel client timers, requests, and downloads before exiting.
- `sentinel/pi/`: Pi deployment version based on the user-provided files. `fumehood_dashboard.html` no longer generates random demo data when requests fail. It instead shows the disconnection, last update time, and offline status, and resumes updates automatically after reconnection. `app.py` adds TC001 endpoints and distinguishes the cameras; `tc001_reader.py` is the new reader module. All Pico serial connection, retry, disconnection, and parsing error messages are now in English; firmware behavior is unchanged. `install_autostart.py` can install the Pi boot service using the current conda environment.
- `sentinel/requirements-desktop.txt`, `sentinel/pi/requirements.txt`: dependencies for both devices.
- `tests/test_sentinel_page.py`, `tests/test_sentinel_dashboard.cjs`: verification using local mock services.

These changes were made directly in the current project without generating a backup. Switching between Wi-Fi and Ethernet alone does not require a Pi update. To enable the TC001, deploy the additional files as described in `pi/TC001.md`. No Pico firmware update is required.

## 1. Wiring and static IP addresses

Connect the Pi's RJ45 Ethernet port to the PC's Ethernet port using a standard Ethernet cable. Power the Pi separately with a suitable power supply. If the PC's existing Ethernet port is needed for the campus network, add a USB-to-Ethernet adapter dedicated to the Pi.

Apply the following addresses only to these two directly connected interfaces:

| Setting | PC | Pi |
| --- | --- | --- |
| IPv4 | `192.168.77.1` | `192.168.77.2` |
| Subnet mask | `255.255.255.0` | `255.255.255.0` |
| Gateway, DNS | Leave blank | Leave blank |

First confirm that `192.168.77.0/24` does not conflict with the campus network, a VPN, or another interface. If it does, choose an unused private subnet and update the address on the Sentinel page accordingly.

Windows: run `ncpa.cpl` → find the adapter connected to the Pi → Properties → Internet Protocol Version 4 (TCP/IPv4) → enter the settings above. Do not change the campus network adapter. A network bridge and Internet Connection Sharing are not required.

Pi: edit the wired connection in network settings, select manual IPv4 configuration, enter `192.168.77.2/24`, and leave gateway and DNS blank. When using the NetworkManager command line, first run `nmcli device status` to identify the actual wired interface, then configure its connection. Do not assume an interface name.

Both machines can access the internet through other interfaces at the same time. Sentinel's local communication does not require internet access. Weather and Google Fonts on the web page require external internet access, but losing it does not affect the local sensor API.

## 2. Wi-Fi connection setup

### If the Pi is already connected to campus Wi-Fi

Keep the existing `192.168.77.2/24` address on `eth0`. Set the Pi's Wi-Fi IPv4 configuration to **Automatic (DHCP)**; do not assign the wired address to the wireless interface as well.

1. Run this command on the Pi to find the wireless interface address:

   ```bash
   ip -4 addr show wlan0
   ```

   Find the IPv4 address after `inet`, omitting suffixes such as `/20` or `/24`. Do not use `127.0.0.1` from `lo` or `192.168.77.2` from `eth0`. If `wlan0` does not exist, run `nmcli device status`, find the actual device with TYPE `wifi`, and substitute its name for `wlan0` in the command.

2. Keep the existing Sentinel service running. Run `python app.py` from the directory containing `app.py`, using the Python environment that previously worked. If the service is already running, do not start a second instance. The existing `app.py` listens on `0.0.0.0:8000` and supports both wired and wireless access.

3. Connect the PC to a laboratory/campus network that can reach the Pi. Wi-Fi or the campus wired network can be used; device-to-device access depends on the school's network configuration.

4. Open `http://<Pi-wireless-IP>:8000` in the PC browser. For example, use `http://10.186.99.100:8000` **only if the Pi actually reports `10.186.99.100`**. This is an example address, not necessarily your Pi's current address.

5. Open Orchestrator → Sentinel → select **Wi-Fi** as the connection mode → enter the same complete URL in the address field → **Connect**.

### If the Pi has not joined campus Wi-Fi

Use the network icon at the top right of the Pi desktop to select the school's Wi-Fi, then configure the account and certificates as required by the school. The existing instructions recommend the school's [SecureW2 setup portal](https://service.purdue.edu/TDClient/32/Purdue/KB/Article/399/How-to-connect-to-Wi-Fi-at-Purdue-West-Lafayette-including-Purdue-Indianapolis) for Purdue PAL3.0/eduroam. After joining Wi-Fi, follow the address lookup and connection steps above.

### If the wireless connection fails

In **PC PowerShell**, replace the example IP with the Pi's current wireless IP and run:

```powershell
Test-NetConnection 10.186.99.100 -Port 8000
```

`TcpTestSucceeded : True` means the TCP port is reachable; check the web page/API next. If it is False, check whether the Pi service is running, whether its wireless IP has changed, the Pi firewall's inbound TCP 8000 rule, and whether the campus network restricts communication between devices. Sharing a Wi-Fi network name does not guarantee device-to-device access. The Pi having internet access does not establish that the PC can connect to it.

If the school network blocks communication between devices, ask school IT to adjust access permissions, or switch Sentinel back to **Ethernet** and use `http://192.168.77.2:8000`. The application does not bypass campus network isolation or automatically switch to another address after disconnection.

The DHCP address on campus Wi-Fi may change after reconnection. If a connection fails, look up the address again and update Sentinel. For a long-term fixed wireless address, request a suitable address assignment from school IT.

## 3. Pi deployment

If Sentinel is already running on the Pi, replace its existing web page with `sentinel/pi/fumehood_dashboard.html` from this repository, then refresh the page. Enabling the TC001 also requires updating `app.py` and adding the reader module; see the dedicated instructions above. Preserve the existing `userinfo.json`, `probeconfig.json`, and `event_log.csv`.

For a first deployment, copy the contents of `sentinel/pi/` into a dedicated directory on the Pi and run the following there:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

If using a camera, also install `opencv-python-headless` and confirm that the Pi can read the corresponding V4L2 camera. The Pi's 1-Wire configuration and USB serial access permissions must match the original system.

`pico_main.py` is **MicroPython firmware for the Pico**. Copy it to the Pico as `main.py` only during initial installation or when updating the Pico. Do not run it with ordinary Python on the PC or Pi.

Keep the `python app.py` process running so the PC can access the page and data. For automatic startup at boot, install the systemd service as described in `pi/AUTOSTART.md`. The service runs the same main entry point using the current user and Python/conda environment. Background acquisition and logging threads start in the main entry point of `app.py`. Do not substitute `flask run` or a standard WSGI import, because those threads will not start.

## 4. PC setup

Install the full PySide6 package in the same Python environment used to launch Orchestrator:

```powershell
python -m pip install -r sentinel/requirements-desktop.txt
python MAIN_WINDOW.py
```

You can also open Sentinel by itself to verify the connection without initializing experimental equipment:

```powershell
python -m gui.pages.sentinel_page
```

Wired connection: open Sentinel → select **Ethernet** → enter `http://192.168.77.2:8000` in the address field → Connect. Wireless connection: select **Wi-Fi**, enter the Pi's wireless address, and connect.

Both addresses are stored in the current PC user's Qt settings (on Windows, `HKEY_CURRENT_USER\Software\Crystalloop\Orchestrator`). The next time Sentinel is first opened, it restores the mode and address most recently used with Connect and connects automatically. Disconnect stops automatic reconnection for the current session; saved addresses remain. Changing the dropdown without clicking Connect does not replace the last connected mode. Address edits for either mode are temporarily retained within the current page until Connect is clicked.

The old `sentinel/url` setting is migrated to the Ethernet address. Subsequent settings use `sentinel/ethernet_url`, `sentinel/wifi_url`, and `sentinel/connection_mode`. On a new installation, the Wi-Fi address is blank and must be entered; it does not automatically reuse the wired address.

Connection checks call `/api/sensors` every 5 seconds with a 4-second timeout, without blocking the desktop interface. A connected service status means only that the API is reachable; check the web page for each probe's online status. After disconnection, the old page and video are hidden and stopped, and the last successful check time is shown. The page reloads after recovery. Clicking Download CSV on the web page opens the PC's Save dialog; canceling does not download a file.

## 5. Troubleshooting and acceptance checks

1. Open `http://localhost:8000/api/sensors` on the Pi and confirm that it returns a JSON array.
2. Open `http://192.168.77.2:8000` in the PC browser and confirm that the page and actual data update.
3. If the browser cannot open the page, check the static IP addresses, cable, service process, and the Pi's inbound TCP 8000 rule.
4. After the Sentinel page connects successfully, unplug the cable. It should show the service as unavailable within roughly one check interval plus the timeout (usually no more than 9 seconds), without presenting old readings as live values.
5. Reconnect the cable and confirm automatic recovery. Test temperature, door contact, video (if configured), and CSV download.
6. When accessing the Pi page directly in a browser, API failures must also show a disconnection and must never generate random demo data.
7. Connect to the Pi's wireless address in Wi-Fi mode and confirm that actual sensor readings update. To test wireless communication, temporarily unplug the direct PC-to-Pi Ethernet cable while keeping the service running, and verify that the wireless address remains accessible.
8. Switch between Wi-Fi and Ethernet and confirm that each restores its own address. Connection starts after Connect is clicked. After a simulated disconnection, retries should use the same mode and must not switch to the other address.

## Local tests

The following tests use mock HTTP services and do not connect to experimental hardware. The Python tests require the full PySide6 package. The old `tests/test_controller.py` refers to the missing `app_control.Controller`, so run these tests separately:

```powershell
python -m unittest discover -s tests -p test_sentinel_page.py -v
node tests/test_sentinel_dashboard.cjs
```

TC001 decoding and Pi API tests require the dependencies in `sentinel/pi/requirements-thermal.txt`:

```powershell
python -m unittest discover -s tests -p test_tc001.py -v
python -m unittest discover -s tests -p test_pi_autostart.py -v
python -m unittest discover -s tests -p test_sentinel_history.py -v
python -m unittest discover -s tests -p test_sentinel_history_ui.py -v
```

These tests use constructed raw frames. They do not establish that a physical TC001 has been calibrated or verified on your Pi.

References: [Qt WebEngine](https://doc.qt.io/qtforpython-6/PySide6/QtWebEngineWidgets/QWebEngineView.html), [Flask deployment](https://flask.palletsprojects.com/en/stable/deploying/), [Raspberry Pi network configuration](https://www.raspberrypi.com/documentation/computers/configuration.html#networking).
