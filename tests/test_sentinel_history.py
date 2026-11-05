"""Pi persistence, API selection, and durable PC/cloud-folder replication."""

import csv
from contextlib import closing
from datetime import datetime, timezone, timedelta
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

PI_DIR = Path(__file__).resolve().parents[1] / "sentinel/pi"
sys.path.insert(0, str(PI_DIR))
from history_store import HistoryStore, COLUMNS, utc_timestamp
from gui.sentinel_archive import ArchiveReplica, replicate


def reading(sensor_id="probe-1", value=24.5, **extra):
    return {"sensor_id": sensor_id, "sensor_name": "Inlet", "type": "temp", "source": "pico", "level": "ok",
            "status": "ok", "detail": "24.50 °C", "value": value, "unit": "°C", "state": None,
            "min_c": None, "max_c": None, "mean_c": None, "center_c": None, **extra}


class HistoryFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sentinel-history-")
        self.root = Path(self.temp.name)
        self.store = HistoryStore(self.root / "pi/history.sqlite3")

    def tearDown(self):
        self.temp.cleanup()


class HistoryTests(HistoryFixture, unittest.TestCase):
    def test_capture_persists_readings_without_repeating_normal_events(self):
        self.store.capture([reading()], "2026-10-20T10:00:00Z")
        self.store.capture([reading(value=25)], "2026-10-20T10:00:05Z")
        self.assertEqual(len(self.store.query(kind="events")["records"]), 1)
        self.assertEqual(len(self.store.query(kind="readings")["records"]), 2)
        reopened = HistoryStore(self.store.path)
        self.assertEqual(reopened.status()["dataset_id"], self.store.status()["dataset_id"])
        self.assertEqual(reopened.status()["record_count"], 3)

    def test_door_open_close_are_events_even_when_both_are_normal(self):
        for state in ("closed", "open", "closed"):
            self.store.capture([reading("door", None, type="door", state=state, detail="Door " + state)])
        self.assertEqual([row["state"] for row in self.store.query(kind="events")["records"]], ["closed", "open", "closed"])

    def test_missing_sensor_is_recorded_offline_without_fake_thermal_values(self):
        self.store.capture([reading(type="thermal", max_c=44, min_c=20, mean_c=30, center_c=25)])
        self.store.capture([])
        self.store.capture([])
        row = self.store.query(kind="readings")["records"][0]
        self.assertEqual(row["level"], "offline")
        self.assertTrue(all(row[key] is None for key in ("value", "max_c", "min_c", "mean_c", "center_c")))
        self.assertEqual(len(self.store.query(kind="events")["records"]), 2)

    def test_restart_recovers_prior_states_and_detects_disconnection(self):
        self.store.capture([reading()])
        reopened = HistoryStore(self.store.path)
        events = reopened.capture([])
        self.assertEqual(events[0]["level"], "offline")

    def test_legacy_csv_imports_once_and_preserves_the_original(self):
        legacy = self.root / "events.csv"
        legacy.write_text('timestamp,sensor_id,sensor_name,type,level,detail\n2026-10-19 12:00:00,p1,Old probe,temp,warn,30 C\ninvalid,p2,Broken,temp,warn,unknown\n', encoding="utf-8")
        original = legacy.read_bytes()
        store = HistoryStore(self.root / "import.sqlite3", legacy)
        self.assertEqual(store.status()["record_count"], 1)
        self.assertEqual(store.status()["legacy_skipped"], 1)
        self.assertEqual(HistoryStore(store.path, legacy).status()["record_count"], 1)
        self.assertEqual(legacy.read_bytes(), original)
        self.assertEqual(store.status()["sensors"], [{"id": "p1", "name": "Old probe"}])

    def test_date_sensor_selection_and_export_include_every_page(self):
        for second in range(15):
            self.store.capture([reading(), reading("other")], f"2026-10-20T12:00:{second:02d}Z")
        filters = dict(kind="readings", sensor_id="probe-1", start="2026-10-20T12:00:05Z", end="2026-10-20T12:00:10Z")
        first = self.store.query(limit=2, **filters)
        second = self.store.query(limit=2, before_seq=first["before_seq"], **filters)
        self.assertTrue(first["has_more"])
        self.assertFalse(set(row["seq"] for row in first["records"]) & set(row["seq"] for row in second["records"]))
        exported = "".join(self.store.export_csv(**filters)).lstrip("\ufeff")
        rows = list(csv.DictReader(io.StringIO(exported)))
        self.assertEqual(len(rows), 5)
        self.assertEqual(set(rows[0]), set(COLUMNS))
        self.assertTrue(all(row["sensor_id"] == "probe-1" for row in rows))

    def test_sync_cursor_resumes_and_keeps_event_and_reading_ids_distinct(self):
        for _ in range(5):
            self.store.capture([reading()])
        first = self.store.sync_batch(limit=2)
        second = self.store.sync_batch(first["next_cursor"], limit=100)
        sequences = [row["seq"] for row in first["records"] + second["records"]]
        self.assertEqual(sequences, list(range(1, 7)))
        self.assertFalse(second["has_more"])
        self.assertEqual(self.store.sync_batch(second["next_cursor"])["records"], [])

    def test_retention_is_opt_in_and_marks_gaps_for_pc(self):
        self.store.capture([reading()], utc_timestamp(datetime.now(timezone.utc) - timedelta(days=100)))
        self.store.prune()
        self.assertGreater(self.store.status()["record_count"], 0)
        self.store.save_settings({"record_interval_seconds": 10, "retention_days": 7})
        self.store.prune()
        batch = self.store.sync_batch()
        self.assertEqual(batch["records"], [])
        self.assertGreater(batch["pruned_through"], 0)
        self.assertGreater(batch["next_cursor"], 0)

    def test_invalid_settings_and_filters_are_rejected(self):
        for settings in ({}, {"record_interval_seconds": True, "retention_days": 0}, {"record_interval_seconds": 0, "retention_days": -1}):
            with self.assertRaises(ValueError):
                self.store.save_settings(settings)
        for filters in ({"kind": "sql"}, {"start": "bad"}, {"before_seq": -1}, {"start": "2026-10-21", "end": "2026-10-20"}):
            with self.assertRaises(ValueError):
                self.store.query(**filters)


class HistoryAPITests(HistoryFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("sentinel_history_backend_test", PI_DIR / "app.py")
        cls.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.backend)

    def setUp(self):
        super().setUp()
        self.patch_store = patch.object(self.backend, "history_store", self.store)
        self.patch_store.start()
        self.client = self.backend.app.test_client()

    def tearDown(self):
        self.patch_store.stop()
        super().tearDown()

    def test_endpoints_export_more_than_visible_page_and_report_errors(self):
        for _ in range(210):
            self.store.capture([reading()])
        data = self.client.get("/api/history?kind=readings&limit=200").get_json()
        self.assertEqual(len(data["records"]), 200)
        self.assertTrue(data["has_more"])
        response = self.client.get("/api/history/export.csv?kind=readings")
        rows = list(csv.DictReader(io.StringIO(response.data.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 210)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertEqual(self.client.get("/api/history?kind=wrong").status_code, 400)
        self.assertEqual(self.client.get("/api/history/export.csv?start=bad").status_code, 400)
        self.assertEqual(self.client.get("/api/history/sync?after=-1").status_code, 400)
        self.assertEqual(self.client.post("/api/history/settings", json={}).status_code, 400)
        response = self.client.post("/api/history/settings", json={"record_interval_seconds": 15, "retention_days": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.settings()["record_interval_seconds"], 15)

    def test_logger_records_thermal_statistics_and_legacy_events(self):
        sensor = {"id": "thermal:tc001", "type": "thermal", "status": "ok", "temp_c": 35,
                  "min_c": 20, "max_c": 35, "mean_c": 27, "center_c": 25}
        with patch.object(self.backend, "read_all_sensors", return_value=[sensor]), \
                patch.object(self.backend, "load_probeconfig", return_value={}), \
                patch.object(self.backend, "EVENTLOG_PATH", str(self.root / "legacy.csv")), \
                patch.object(self.backend, "_history_stop") as stop:
            stop.is_set.side_effect = [False, True]
            self.backend._event_logger_loop()
            stop.wait.assert_called_with(5)
        row = self.store.query(kind="readings")["records"][0]
        self.assertEqual((row["value"], row["max_c"], row["center_c"]), (35, 35, 25))
        self.assertEqual(row["level"], "alert")
        self.assertTrue((self.root / "legacy.csv").is_file())

    def api_fetch(self, origin, path):
        response = self.client.get(path)
        if response.status_code != 200:
            raise ValueError(response.get_json()["error"])
        return response.get_json()

    def test_pc_reconnect_deduplicates_and_cloud_snapshot_contains_committed_wal(self):
        pc, cloud = self.root / "pc", self.root / "cloud"
        cloud.mkdir()
        with patch("gui.sentinel_archive.fetch_json", side_effect=self.api_fetch):
            self.store.capture([reading()], "2026-10-20T12:00:00Z")
            first = replicate("http://pi", pc, cloud, cloud_due=True)
            self.store.capture([reading(value=29)], "2026-10-21T12:00:00Z")
            second = replicate("http://pi-wifi", pc, cloud, cloud_due=True)
            third = replicate("http://pi", pc)
        self.assertEqual(first["pc"], "Saved")
        self.assertTrue(second["cloud_saved"])
        self.assertEqual(second["cursor"], third["cursor"])
        dataset = self.store.status()["dataset_id"]
        with closing(sqlite3.connect(pc / dataset / "archive.sqlite3")) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM records").fetchone()[0], 3)
        with closing(sqlite3.connect(cloud / "Sentinel" / dataset / "history.sqlite3")) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM records").fetchone()[0], 3)
        csv_file = pc / dataset / "2026-10-21-readings.csv"
        with csv_file.open(encoding="utf-8-sig", newline="") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)
        manifest = json.loads((cloud / "Sentinel" / dataset / "manifest.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(manifest["through_seq"], third["cursor"])
        self.assertIn("upload not verified", second["cloud"])

    def test_export_failure_retries_from_durable_dirty_days(self):
        pc = self.root / "pc"
        self.store.capture([reading()])
        with patch("gui.sentinel_archive.fetch_json", side_effect=self.api_fetch):
            with patch.object(ArchiveReplica, "_export_day", side_effect=OSError("Disk unavailable")):
                with self.assertRaises(OSError):
                    replicate("http://pi", pc)
            replica = ArchiveReplica(pc, self.store.status()["dataset_id"])
            cursor = replica.cursor()
            result = replicate("http://pi", pc)
        self.assertEqual(result["cursor"], cursor)
        self.assertTrue(list(replica.directory.glob("*-readings.csv")))

    def test_cloud_failure_keeps_pc_records_and_can_retry(self):
        self.store.capture([reading()])
        cloud = self.root / "cloud"
        with patch("gui.sentinel_archive.fetch_json", side_effect=self.api_fetch):
            failed = replicate("http://pi", self.root / "pc", cloud, cloud_due=True)
            cloud.mkdir()
            recovered = replicate("http://pi", self.root / "pc", cloud, cloud_due=True)
        self.assertEqual(failed["pc"], "Saved")
        self.assertEqual(failed["cloud"], "Folder write failed")
        self.assertTrue(recovered["cloud_saved"])

    def test_connection_loss_mid_transfer_resumes_from_committed_cursor(self):
        with self.store.connection() as con:
            for _ in range(1200):
                self.store._insert(con, {**reading(), "kind": "readings", "timestamp": utc_timestamp()})
        requests = []
        def flaky(origin, path):
            requests.append(path)
            if "after=500&" in path:
                raise ConnectionError("Pi disconnected")
            return self.api_fetch(origin, path)
        pc = self.root / "pc"
        with patch("gui.sentinel_archive.fetch_json", side_effect=flaky):
            with self.assertRaises(ConnectionError):
                replicate("http://pi", pc)
        replica = ArchiveReplica(pc, self.store.status()["dataset_id"])
        self.assertEqual(replica.cursor(), 500)
        with patch("gui.sentinel_archive.fetch_json", side_effect=self.api_fetch):
            result = replicate("http://pi", pc)
        self.assertEqual(result["cursor"], 1200)
        with replica.connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM records").fetchone()[0], 1200)

    def test_replay_batch_is_idempotent_and_invalid_batch_rolls_back(self):
        self.store.capture([reading()])
        batch = self.store.sync_batch()
        replica = ArchiveReplica(self.root / "pc", batch["dataset_id"])
        replica.ingest(batch)
        replica.ingest(batch)
        cursor = replica.cursor()
        invalid = {**batch, "records": [dict(batch["records"][0], detail="Changed")]}
        invalid["next_cursor"] = invalid["records"][0]["seq"]
        invalid["has_more"] = invalid["next_cursor"] < invalid["latest_seq"]
        with self.assertRaises(ValueError):
            replica.ingest(invalid)
        self.assertEqual(replica.cursor(), cursor)

    def test_new_pi_database_has_a_separate_archive(self):
        self.store.capture([reading()])
        old_id = self.store.status()["dataset_id"]
        with patch("gui.sentinel_archive.fetch_json", side_effect=self.api_fetch):
            replicate("http://pi", self.root / "pc")
            self.store = HistoryStore(self.root / "replacement.sqlite3")
            self.backend.history_store = self.store
            self.store.capture([reading()])
            replicate("http://pi", self.root / "pc")
        new_id = self.store.status()["dataset_id"]
        self.assertNotEqual(old_id, new_id)
        self.assertTrue((self.root / "pc" / old_id / "archive.sqlite3").is_file())
        self.assertTrue((self.root / "pc" / new_id / "archive.sqlite3").is_file())


if __name__ == "__main__":
    unittest.main()
