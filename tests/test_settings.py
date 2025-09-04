import os
import unittest
import tempfile
from unittest import mock
from chat_app.settings import Settings, save_settings, load_settings


class TestSettingsRoundtrip(unittest.TestCase):
    def test_round_trip_with_tomli_w(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.toml")
            original = Settings()
            original.app.port = 1234
            save_settings(original, path)
            loaded = load_settings(path)
            self.assertEqual(loaded, original)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(os.path.splitext(path)[0] + ".json"))

    def test_round_trip_without_tomli_w(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.toml")
            original = Settings()
            original.app.port = 4321
            from chat_app import settings as cfg
            with mock.patch.object(cfg, "tomli_w", None):
                save_settings(original, path)
                loaded = load_settings(path)
            self.assertEqual(loaded, original)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.splitext(path)[0] + ".json"))

if __name__ == '__main__':
    unittest.main()
