"""Verify the authorized public release transfer; no checkpoint deserialization."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--expected-sha256', required=True)
    args = ap.parse_args()
    archive = args.base/'weights/motionforesight-release.zip'
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != args.expected_sha256.lower():
        raise ValueError('Transfer SHA256 mismatch')
    destination = args.base/'weights/motionforesight'
    destination.mkdir(exist_ok=True)
    names = ['trainedon_ss100k/best.pt', 'trainedon_ss100k/config.json']
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad:
            raise ValueError('ZIP CRC failed: '+bad)
        for name in names:
            if name not in handle.namelist():
                raise ValueError('Missing official runtime file')
            handle.extract(name, destination)
    config = json.loads((destination/names[1]).read_text())
    result = {'fine_checkpoint_files_ready': True,
              'official_source': 'MotionForesight README linked Google Drive release',
              'official_drive_id': '1isAB_TnNW8gbMlgs5yVnPBKZSR85-JyZ',
              'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest.hexdigest(),
              'publisher_sha256_available': False,
              'local_remote_sha256_match': True, 'zip_crc_pass': True,
              'extracted_files': names, 'config': config,
              'model_loaded': False, 'forward_called': False, 'gpu_used': False}
    (args.base/'logs/fine-checkpoint-complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
