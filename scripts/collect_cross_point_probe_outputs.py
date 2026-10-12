"""Collect hashes for compact experiment evidence; does not copy model assets."""
import argparse
import hashlib
import json
from pathlib import Path

FILES = [
    'reader-results-g1.1/results.json',
    'reader-results-g1.1/paired-summary.json',
    'reader-results-g1.1/paired-summary-reviewed.json',
    'logs/gpu-interface-first-status.json',
    'logs/gpu-geometry-all-status.json',
    'logs/gpu-forecast-all-status.json',
    'logs/reader-stage-status.json',
    'logs/observed-calibration-complete.json',
    'logs/calibrated-protocol.json',
    'logs/first-model-coordinate-audit.json',
    'logs/reader-training.log',
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--verify-local',action='store_true')
    args=ap.parse_args()
    manifest_path=args.base/'evidence-manifest.json'
    if args.verify_local:
        manifest=json.loads(manifest_path.read_text())
        for row in manifest['files']:
            path=args.base/row['local_name']
            if path.stat().st_size!=row['bytes']:
                raise ValueError('Size mismatch: '+row['local_name'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Hash mismatch: '+row['local_name'])
        print(json.dumps({'verified_files':len(manifest['files']),
                          'bytes':sum(r['bytes'] for r in manifest['files'])}))
        return
    rows=[]
    for name in FILES:
        path=args.base/name
        rows.append({'remote_relative':name,'local_name':path.name,
                     'bytes':path.stat().st_size,
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={'files':rows,'weights_included':False}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
