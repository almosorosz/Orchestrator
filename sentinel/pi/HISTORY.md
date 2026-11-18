# History: Pi recording, PC saving, and cloud sync folder copies

## What this version provides

- The Pi saves temperature, door state, and TC001 maximum/minimum/mean/center temperatures every 5 seconds by default.
- Events records alarm level changes, door state changes, devices coming online/going offline, and normal service starts/stops.
- History provides Events / Readings views, date and sensor filters, pagination, and CSV export of the entire selection.
- The PC saves incrementally every 60 seconds by default, resuming from the last record ID after reconnection or restarting the application.
- An additional copy can be written to a cloud sync folder on the PC and uploaded by your cloud sync client.
- Ethernet, Wi-Fi, automatic TC001 detection by default, the English interface, and optional automatic startup are retained.

This version does not log in to cloud services directly, requires no cloud account or API key, and does not upload directly from the Pi to the cloud. Cloud backups require the PC, Orchestrator, and cloud sync client to remain running.

## 1. Update the Pi

Stop the old program first. For a manually started program, press Ctrl+C in its terminal. For an installed system service, run `sudo systemctl stop sentinel.service`.

Copy the files from the Pi update package into the existing directory. The new `history_store.py` file must be included. Preserve the following runtime data:

- `userinfo.json`, `probeconfig.json`
- `event_log.csv`
- If the new version has already been run, preserve `sentinel_history.sqlite3`. Do not delete its associated `sentinel_history.sqlite3-wal` and `sentinel_history.sqlite3-shm` files after stopping the service either.

The first run imports the old CSV into the database once, without modifying the old CSV. Malformed legacy rows are skipped, and History displays the number skipped. The original CSV is retained; use Export CSV on the page for exports.

No new third-party dependencies are required. SQLite uses Python's built-in `sqlite3`. Continue using the conda environment that already runs the TC001 successfully.

For manual startup, use your current environment and directory:

```bash
conda activate envmo
cd /home/oybc1999/Desktop/Pi/Pi
python app.py
```

If you already use the system service, run this instead after updating the files:

```bash
sudo systemctl restart sentinel.service
```

Do not run a manual instance and the system service at the same time. Automatic startup remains optional; see `AUTOSTART.md`. No Pico firmware update is required.

## 2. Update and run the PC application

Extract the complete Orchestrator update package and run the following in the existing PC Python environment:

```powershell
python MAIN_WINDOW.py
```

You can also run Sentinel alone: `python -m gui.pages.sentinel_page`. It continues to use PySide6 without new third-party dependencies. The two new PC modules are in `gui/`; replacing only `sentinel_page.py` is insufficient.

Select Ethernet or Wi-Fi in Sentinel, connect to the Pi, and open History. The default PC directory is `Sentinel Logs` under the current user's Documents folder. Click **Log settings** or **PC → Configure** in History to change the directory, disable saving, or adjust the interval (10–3600 seconds). Changing the directory causes all records still retained on the Pi to be transferred again into the new directory.

The directory structure is as follows, with dates grouped by UTC:

```text
Sentinel Logs/
  <Pi dataset ID>/
    archive.sqlite3
    2026-10-21-events.csv
    2026-10-21-readings.csv
```

The database is the complete local PC copy; CSV files can be analyzed with Excel and other tools. Changing the same Pi's IP address or switching between Ethernet / Wi-Fi retains the same database ID and does not create duplicate records. Do not delete the Pi database when updating code; rebuilding it creates a new ID and a separate PC archive directory.

Synchronization stops when Orchestrator closes or Disconnect is clicked, but the Pi continues recording. Reconnecting transfers the missing records. Saving runs in the background; leaving the Sentinel page does not stop automatic saving for an established connection.

## 3. Cloud sync folder copy

In **Log settings** on the PC:

1. Check **Also save a copy to a cloud sync folder**.
2. Use **Browse…** to select an existing directory managed by your cloud sync client.
3. Save. Sentinel writes the first copy immediately, then refreshes it through the PC saving task approximately every 5 minutes. If the PC saving interval exceeds 5 minutes, it uses that longer interval instead.
4. Click **Sync now** in History to request an immediate refresh.

The cloud sync folder contains:

```text
<Selected folder>/Sentinel/<Pi dataset ID>/
  history.sqlite3
  2026-10-21-events.csv
  2026-10-21-readings.csv
  manifest.json
```

The database is generated using a consistent SQLite snapshot; the database file being written is not copied directly. CSV files and snapshots are written to temporary files before replacement. Failed operations retry during subsequent saving tasks.

**Folder saved · upload not verified** means only that files have been written to the PC's cloud sync folder. Check the cloud sync client for actual upload status; Sentinel does not report a successful local write as a successful cloud upload. If the sync client is missing, signed out, or not syncing the directory, the files remain on the PC only.

If the cloud copy is enabled while ordinary PC saving is disabled, the application still uses a cache database in the local application data directory to track transfer progress. Cloud account credentials are not placed on the Pi web page.

## 4. History page settings

- **Events**: state changes and system events.
- **Readings**: sensor data at each sampling time; thermal readings do not include full video or screenshots.
- **Today / Last 7 days / All time / Custom dates**: filter by the browser's local dates.
- **Export CSV**: export the entire selection, not just the 200 rows displayed on the page. CSV timestamps use UTC.
- **Load more**: view older records. Automatic list replacement pauses while browsing older records; click **Refresh latest** to resume live updates.
- **Settings**: adjust the sampling interval and Pi retention period. The default of 0 days disables automatic deletion.

When retention is greater than 0 days, records older than that period are deleted from the Pi database during acquisition. The page requests confirmation when this setting changes. Records already saved on the PC and the original CSV are unaffected. If the PC stays disconnected too long, some records may expire before they can be transferred. The PC displays a missing-data warning; deleted original Pi data cannot be recovered.

Events compare states at sampling times, so changes shorter than the sampling interval may not be recorded. Shorten the interval if shorter changes need to be captured. A power failure cannot write a normal “Service stopped” event; a “Service started” event appears after restart.

## 5. Verification

In the Pi's local browser, open `http://127.0.0.1:8000/api/history/status`. `recording` should be true, and `record_count` should increase as samples are collected.

After connecting the PC, confirm that the PC status in History is Saved and that the selected directory contains the database and the day's CSV files. Disconnect, wait for the Pi to record several samples, then reconnect. Confirm that the CSV files catch up without duplicate records. After an IP change, data should still be written to the same Pi dataset ID subdirectory.

Local tests cover real HTTP / SQLite operations, page filtering and downloads, Qt background tasks, transfer resumption after disconnection, export failure retries, and cloud folder snapshots. Verification on your actual Pi, PC, and cloud sync client is still required.
