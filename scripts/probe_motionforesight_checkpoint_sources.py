"""Read-only, bounded public metadata probes; never download model weights."""
import concurrent.futures
import json
import os
import time
from pathlib import Path

import requests


REPO = 'brains-bots-n-behavior/motionforesight'
DRIVE_ID = '1isAB_TnNW8gbMlgs5yVnPBKZSR85-JyZ'


def probe(label, url, direct=False):
    start = time.monotonic()
    session = requests.Session()
    session.trust_env = not direct
    try:
        with session.get(url, timeout=(10, 15), stream=True) as response:
            data = next(response.iter_content(65536), b'')
            row = {'label': label, 'status': response.status_code,
                   'content_type': response.headers.get('Content-Type'),
                   'content_length': response.headers.get('Content-Length'),
                   'sample_bytes': len(data), 'seconds': round(time.monotonic()-start, 2)}
            if response.status_code == 200 and 'json' in response.headers.get('Content-Type', ''):
                # API pages are small; cap even those to avoid unrelated large transfers.
                for chunk in response.iter_content(65536):
                    data += chunk
                    if len(data) > 1024*1024:
                        break
                try:
                    obj = json.loads(data)
                    if label.startswith('hf_'):
                        row['models'] = [x.get('id') for x in obj] if isinstance(obj, list) else obj
                    elif label == 'github_releases':
                        row['releases'] = [{'tag': x.get('tag_name'), 'assets':
                            [a.get('name') for a in x.get('assets', [])]} for x in obj]
                    elif label == 'github_forks':
                        row['forks'] = [x.get('full_name') for x in obj]
                    elif label.endswith('_releases'):
                        row['releases'] = [{'tag': x.get('tag_name'), 'assets':
                            [a.get('name') for a in x.get('assets', [])]} for x in obj]
                    else:
                        row['metadata'] = obj
                except (ValueError, TypeError):
                    row['json_sample_complete'] = False
            if label.startswith('drive_'):
                text = data.decode('utf-8', errors='replace')
                row['download_form_present'] = '<form' in text
                row['zip_signature'] = data[:4] == b'PK\x03\x04'
                row['permission_notice'] = any(x in text.lower() for x in
                    ['request access', 'you need access', 'access denied'])
            return row
    except requests.RequestException as exc:
        # Exception strings can contain proxy credentials. Emit only class name.
        return {'label': label, 'error_type': type(exc).__name__,
                'seconds': round(time.monotonic()-start, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    checks = [
        ('hf_models', 'https://huggingface.co/api/models?search=motionforesight&limit=100', False),
        ('hf_spaces', 'https://huggingface.co/api/spaces?search=motionforesight&limit=100', False),
        ('github_releases', 'https://api.github.com/repos/'+REPO+'/releases', False),
        ('github_forks', 'https://api.github.com/repos/'+REPO+'/forks', False),
        ('drive_accelerated', 'https://drive.google.com/uc?export=download&id='+DRIVE_ID, False),
        ('drive_direct', 'https://drive.usercontent.google.com/download?export=download&confirm=t&id='+DRIVE_ID, True),
        ('drive_docs_direct', 'https://docs.google.com/uc?export=download&id='+DRIVE_ID, True),
    ]
    for owner in ['flyingGH', 'ggred0123', 'CloudEngineHub']:
        checks.append((owner+'_releases', 'https://api.github.com/repos/'+owner+'/motionforesight/releases', False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(checks)) as executor:
        rows = list(executor.map(lambda x: probe(*x), checks))
    result = {'max_sample_bytes_per_download': 65536, 'model_loaded': False,
              'weight_downloaded': False, 'checks': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
