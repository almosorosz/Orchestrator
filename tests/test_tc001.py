"""TC001 decoding and Pi HTTP integration using synthetic raw frames, never hardware."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

PI_DIR = Path(__file__).resolve().parents[1] / "sentinel/pi"
sys.path.insert(0, str(PI_DIR))
from tc001_reader import TC001Reader, decode_tc001_frame, render_thermal_jpeg, video_devices


def raw_frame():
    frame = np.zeros((384, 256, 2), dtype=np.uint8)
    frame[:192] = 200  # Image-plane pixels must not influence temperatures.
    words = np.full((192, 256), 0x4A80, dtype=np.uint16)
    words[0, 0], words[1, 1] = 0x4A01, 0x4AFF
    frame[192:, :, 0] = words & 255
    frame[192:, :, 1] = words >> 8
    return frame


class TC001Tests(unittest.TestCase):
    def setUp(self):
        self.frame = raw_frame()
        self.temperatures, self.valid, self.stats = decode_tc001_frame(self.frame)

    def reader(self, stale_seconds=3):
        reader = TC001Reader("/dev/video-test", stale_seconds=stale_seconds)
        reader._publish(b"jpeg", self.stats, "/dev/video-test")
        return reader

    def test_exact_little_endian_min_max_and_center(self):
        self.assertAlmostEqual(self.stats["min_c"], round(0x4A01 / 64 - 273.15, 2))
        self.assertAlmostEqual(self.stats["max_c"], round(0x4AFF / 64 - 273.15, 2))
        self.assertAlmostEqual(self.stats["center_c"], round(0x4A80 / 64 - 273.15, 2))
        self.assertEqual(self.temperatures.shape, (192, 256))
        self.assertEqual(self.stats["invalid_pixels"], 0)

    def test_flat_and_uint16_raw_buffers(self):
        for frame in (self.frame.reshape(1, -1), self.frame.reshape(-1), self.frame.view("<u2").reshape(384, 256)):
            with self.subTest(shape=frame.shape):
                self.assertEqual(decode_tc001_frame(frame)[2], self.stats)

    def test_rejects_rgb_and_missing_temperature_plane(self):
        for frame in (np.zeros((384, 256, 3), np.uint8), self.frame[:192], np.zeros((4,), np.uint8)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                decode_tc001_frame(frame)

    def test_rejects_invalid_plane_instead_of_generating_temperatures(self):
        for value in (0, 255):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_tc001_frame(np.full_like(self.frame, value))

    def test_invalid_pixel_is_excluded_from_statistics(self):
        self.frame[192 + 96, 128] = 0
        _, valid, stats = decode_tc001_frame(self.frame)
        self.assertFalse(valid[96, 128])
        self.assertEqual(stats["invalid_pixels"], 1)
        self.assertIsNone(stats["center_c"])
        self.assertEqual(stats["max_c"], self.stats["max_c"])

    def test_jpeg_is_generated_from_temperature_matrix(self):
        jpeg = render_thermal_jpeg(self.temperatures, self.valid, self.stats)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (384, 512, 3))
        self.assertGreater(int(image.max()) - int(image.min()), 50)

    def test_disabled_reader_does_not_add_phantom_sensor(self):
        reader = TC001Reader()
        self.assertIsNone(reader.sensor())
        self.assertFalse(reader.snapshot()["available"])

    def test_stale_data_is_cleared_and_video_stops(self):
        reader = self.reader(stale_seconds=0.01)
        time.sleep(0.02)
        self.assertEqual(reader.sensor()["status"], "offline")
        for key in ("temp_c", "min_c", "max_c", "mean_c", "center_c"):
            self.assertIsNone(reader.sensor()[key])
        self.assertEqual(list(reader.frames()), [])

    def test_multiple_clients_share_the_same_cached_frame(self):
        reader = self.reader()
        first, second = reader.frames(), reader.frames()
        self.assertEqual(next(first), next(second))
        self.assertEqual(reader.sensor()["temp_c"], self.stats["max_c"])
        first.close()
        second.close()

    def test_disconnect_removes_cached_readings_immediately(self):
        reader = self.reader()
        reader._unavailable("USB disconnected")
        self.assertIsNone(reader.sensor()["max_c"])
        self.assertEqual(reader.snapshot()["error"], "USB disconnected")
        reader._publish(b"new-frame", self.stats, "/dev/video-new")
        self.assertTrue(reader.snapshot()["available"])
        self.assertEqual(reader.snapshot()["device"], "/dev/video-new")

    def test_worker_negotiates_raw_format_and_releases_device(self):
        frame = self.frame

        class FakeCapture:
            released = False
            settings = {}

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.settings[prop] = value
                return True

            def read(self):
                time.sleep(0.01)
                return True, frame.copy()

            def release(self):
                self.released = True

        capture = FakeCapture()
        reader = TC001Reader("/dev/video-test")
        with patch.object(cv2, "VideoCapture", return_value=capture) as opened:
            try:
                reader.start()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not reader.snapshot()["available"]:
                    time.sleep(0.01)
                self.assertTrue(reader.snapshot()["available"], reader.snapshot())
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(capture.settings[cv2.CAP_PROP_FRAME_HEIGHT], 384)
                self.assertEqual(capture.settings[cv2.CAP_PROP_CONVERT_RGB], 0.0)
            finally:
                reader.stop()
        self.assertTrue(capture.released)
        self.assertFalse(reader.snapshot()["available"])

    def test_sysfs_identifies_camera_without_opening_it(self):
        with tempfile.TemporaryDirectory() as directory:
            usb = Path(directory)
            (usb / "idVendor").write_text("0bda\n")
            (usb / "idProduct").write_text("5830\n")
            root = usb / "video4linux"
            (root / "video2").mkdir(parents=True)
            (root / "video2/name").write_text("USB Camera\n")
            devices = video_devices(root)
            self.assertEqual(devices[0]["path"], "/dev/video2")
            self.assertTrue(devices[0]["tc001"])

    def test_worker_recovers_when_camera_reappears_with_new_device_number(self):
        frame = self.frame
        disconnected = False

        class FakeCapture:
            def __init__(self, path):
                self.path = path

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                time.sleep(0.01)
                return (False, None) if self.path == "/dev/video0" and disconnected else (True, frame.copy())

            def release(self):
                pass

        def devices():
            return [{"path": "/dev/video2" if disconnected else "/dev/video0", "tc001": True}]

        def wait_for(condition, timeout=6):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if condition():
                    return
                time.sleep(0.01)
            self.fail("Capture did not recover: " + str(reader.snapshot()))

        reader = TC001Reader("auto")
        with patch("tc001_reader.video_devices", side_effect=devices), patch.object(cv2, "VideoCapture", side_effect=lambda path, backend: FakeCapture(path)):
            try:
                reader.start()
                wait_for(lambda: reader.snapshot()["available"])
                disconnected = True
                wait_for(lambda: not reader.snapshot()["available"])
                wait_for(lambda: reader.snapshot()["available"] and reader.snapshot()["device"] == "/dev/video2")
            finally:
                reader.stop()


class ThermalAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("sentinel_pi_backend_test", PI_DIR / "app.py")
        cls.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.backend)

    def setUp(self):
        _, _, stats = decode_tc001_frame(raw_frame())
        self.reader = TC001Reader("/dev/video-test")
        self.reader._publish(b"synthetic-jpeg", stats, "/dev/video-test")
        self.patch_reader = patch.object(self.backend, "thermal_reader", self.reader)
        self.patch_reader.start()
        self.client = self.backend.app.test_client()

    def tearDown(self):
        self.patch_reader.stop()

    def test_thermal_api_and_sensor_share_the_same_maximum(self):
        data = self.client.get("/api/thermal").get_json()
        self.assertTrue(data["available"])
        self.assertIn("no-store", self.client.get("/api/thermal").headers["Cache-Control"])
        with patch.object(self.backend, "find_sensors", return_value=[]), patch.object(self.backend, "read_all_doors", return_value=[]):
            sensors = self.client.get("/api/sensors").get_json()
        thermal = next(s for s in sensors if s["id"] == "thermal:tc001")
        self.assertEqual(thermal["temp_c"], data["max_c"])
        self.assertEqual(thermal["type"], "thermal")

    def test_stream_uses_cached_jpeg_and_offline_returns_503(self):
        response = self.client.get("/thermal_feed", buffered=False)
        self.assertIn(b"synthetic-jpeg", next(response.response))
        response.close()
        self.reader._unavailable("Unplugged")
        self.assertEqual(self.client.get("/thermal_feed").status_code, 503)
        self.assertIsNone(self.client.get("/api/thermal").get_json()["max_c"])

    def test_normal_camera_does_not_open_thermal_node(self):
        devices = [{"path": "/dev/video0", "tc001": True, "vendor": "0bda"}, {"path": "/dev/video2", "tc001": False, "vendor": "1234"}]
        with patch.object(self.backend, "video_devices", return_value=devices), patch.object(self.backend, "CAMERA_DEVICE", ""), patch.dict(os.environ):
            os.environ.pop("CAMERA_INDEX", None)
            self.assertEqual(self.backend.visible_camera_candidates(), ["/dev/video2"])
            self.backend.CAMERA_DEVICE = "/dev/video0"
            self.assertEqual(self.backend.visible_camera_candidates(), [])

    def test_alarm_thresholds_use_thermal_maximum(self):
        sensor = self.reader.sensor()
        config = {sensor["id"]: {"caution": 20, "alert": 25}}
        self.assertEqual(self.backend.classify_temp_level(sensor, config), "alert")


class ThermalStartupTests(unittest.TestCase):
    def load_backend(self, value=None):
        with patch.dict(os.environ):
            if value is None:
                os.environ.pop("THERMAL_DEVICE", None)
            else:
                os.environ["THERMAL_DEVICE"] = value
            spec = importlib.util.spec_from_file_location("sentinel_startup_test", PI_DIR / "app.py")
            backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend)
        return backend

    def test_normal_start_enables_detection_without_opening_hardware_on_import(self):
        with patch.object(cv2, "VideoCapture") as opened:
            backend = self.load_backend()
        self.assertTrue(backend.thermal_reader.enabled)
        self.assertEqual(backend.thermal_reader.device, "auto")
        self.assertIsNone(backend.thermal_reader._thread)
        opened.assert_not_called()
        self.assertTrue(backend.app.test_client().get("/api/thermal").get_json()["enabled"])

    def test_explicit_disable_omits_thermal_sensor(self):
        backend = self.load_backend("off")
        self.assertFalse(backend.thermal_reader.enabled)
        self.assertIsNone(backend.thermal_reader.sensor())

    def test_manual_camera_path_still_overrides_detection(self):
        backend = self.load_backend(" /dev/video4 ")
        self.assertTrue(backend.thermal_reader.enabled)
        self.assertEqual(backend.thermal_reader._candidates(), ["/dev/video4"])


if __name__ == "__main__":
    unittest.main()
