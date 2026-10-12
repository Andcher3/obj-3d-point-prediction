"""CPU-only checks for resume/completion evidence, no network or model."""
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from resume_cross_point_assets import completed


class CompletionTests(unittest.TestCase):
    def test_missing_and_aria2_partial_are_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.safetensors'
            row = {'path': str(path), 'kind': 'trackcraft3r/checkpoint'}
            self.assertFalse(completed(row))
            path.write_bytes(b'test')
            Path(str(path)+'.aria2').write_bytes(b'control')
            self.assertFalse(completed(row))

    def test_lfs_size_and_hash_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.safetensors'
            path.write_bytes(b'test')
            row = {'path': str(path), 'kind': 'trackcraft3r/checkpoint',
                   'bytes': 4, 'sha256': hashlib.sha256(b'test').hexdigest()}
            self.assertTrue(completed(row))
            row['bytes'] = 5
            self.assertFalse(completed(row))
            row['bytes'] = 4
            row['sha256'] = '0'*64
            self.assertFalse(completed(row))

    def test_small_json_requires_valid_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            row = {'path': str(path), 'kind': 'depth-anything/DA3-BASE'}
            path.write_text('{')
            self.assertFalse(completed(row))
            path.write_text('{}')
            self.assertTrue(completed(row))

    def test_source_zip_crc_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('README.md', 'source')
            self.assertTrue(completed({'path': str(path), 'kind': 'source'}))


if __name__ == '__main__':
    unittest.main()
