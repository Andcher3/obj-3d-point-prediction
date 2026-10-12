"""Bounded remote route probes. No credentials, proxies or signed URLs in output."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def sample(label, url, trust_env, resolve_only=False):
    start = time.monotonic()
    session = requests.Session()
    session.trust_env = trust_env
    total = 0
    try:
        if resolve_only:
            response = session.get(url, allow_redirects=False, stream=True, timeout=(8, 10))
            if response.is_redirect:
                url = response.headers['Location']
                response.close()
                session.trust_env = False
            else:
                status = response.status_code
                response.close()
                return {'route': label, 'resolver_status': status, 'redirect': False}
        with session.get(url, headers={'Range': 'bytes=1073741824-1074790399'},
                         stream=True, timeout=(8, 10)) as response:
            response.raise_for_status()
            for block in response.iter_content(65536):
                total += len(block)
                if total >= 1048576 or time.monotonic()-start > 22:
                    break
            elapsed = time.monotonic()-start
            return {'route': label, 'status': response.status_code,
                    'content_range': response.headers.get('Content-Range'),
                    'bytes_sampled': total, 'seconds': round(elapsed, 2),
                    'KiB_s_including_connect': round(total/1024/elapsed, 1)}
    except requests.RequestException as error:
        return {'route': label, 'error_type': type(error).__name__,
                'seconds': round(time.monotonic()-start, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.base/'logs/asset-manifest.json').read_text())
    asset = next(x for x in manifest['assets'] if x['kind'] == 'trackcraft3r/checkpoint')
    suffix = '/trackcraft3r/checkpoint/resolve/'+asset['revision']+'/model.safetensors'
    fresh = '?download=true&route_refresh='+str(time.time_ns())
    routes = [
        ('mirror_cached_direct', 'https://hf-mirror.com'+suffix, False, False),
        ('mirror_fresh_direct', 'https://hf-mirror.com'+suffix+fresh, False, False),
        ('official_academic', 'https://huggingface.co'+suffix+fresh, True, False),
        ('official_resolver_academic_cdn_direct', 'https://huggingface.co'+suffix+fresh, True, True),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda x: sample(*x), routes))
    output = {'routes': result, 'sample_only': True, 'model_forward': False}
    (args.base/'logs/trackcraft-route-probe.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
