# Start Sentinel automatically when the Pi boots

TC001 detection is now enabled by default with `python app.py`. The following setup starts the entire Pi service automatically at boot, without opening a terminal or logging in through remote desktop. It restarts automatically if the program exits with an error.

## One-time installation

Place the update package files in the original program directory. Preserve `userinfo.json`, `probeconfig.json`, `event_log.csv`, and any existing `sentinel_history.sqlite3`. No Pico firmware update is required. Make sure the new `history_store.py` is also in the same directory. First press Ctrl+C in the old service terminal to stop any manually running `app.py`, avoiding duplicate use of port 8000.

Using the directory and environment shown in your screenshot, run the following in the **Pi terminal**:

```bash
conda activate envmo
cd /home/oybc1999/Desktop/Pi/Pi
python install_autostart.py
```

If the directory or environment name has changed, use the actual values. Do not add `sudo` to the last line. The installer invokes sudo itself when writing the system service; enter the Pi's system password if prompted.

The installer:

- Checks Flask, pyserial, NumPy, OpenCV, and required program files in the current environment.
- Automatically records the current user, program directory, Python interpreter, and conda environment location.
- Uses `conda run --no-capture-output --prefix ...` for conda environments to preserve environment activation behavior.
- Installs `/etc/systemd/system/sentinel.service`, running as the current regular user.
- Enables startup at boot, starts the service immediately, and checks whether the local API is ready.

On success, it displays `Sentinel is running and will start automatically at boot.` This means the service has started. Check `available` in `/api/thermal` to determine whether thermal acquisition is working. Closing the terminal or disconnecting remote desktop does not stop the service. Ethernet and Wi-Fi access remain unchanged.

If the port is reported as occupied, first stop the old manually running `app.py`. After installation, do not manually start a second `python app.py` instance.

To preview the generated system service without installing it:

```bash
python install_autostart.py --dry-run
```

## Check status, update the program, and disable automatic startup

```bash
systemctl status sentinel.service --no-pager
journalctl -u sentinel.service -n 50 --no-pager
```

Live logs: `journalctl -u sentinel.service -f`. Pressing Ctrl+C exits only the log viewer.

Restart after updating program files:

```bash
sudo systemctl restart sentinel.service
```

After moving the program directory or changing conda environments, enter the new directory, activate the new environment, and rerun the installer. There is no need to reinstall Python packages if they have not changed.

Stop the service temporarily:

```bash
sudo systemctl stop sentinel.service
```

Disable automatic startup and stop the service:

```bash
sudo systemctl disable --now sentinel.service
```

Enable it again: `sudo systemctl enable --now sentinel.service`.

After rebooting the Pi, check the service status and `http://127.0.0.1:8000/api/thermal`. Missing devices do not prevent the web page from starting. Pico and TC001 connections retry in the background; a successful service start does not establish that devices are online.

Local simulated tests have verified configuration generation and the installation workflow. Actual installation and reboot verification on your Pi are still required.

References: [systemd services](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html), [conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html).
