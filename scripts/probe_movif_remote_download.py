"""Bounded public-GCS transport comparison. Never prints proxy settings."""
import json
import time
import requests

name = "tfds/movi_f/256x256/1.0.0/movi_f-validation.tfrecord-00000-of-00032"
urls = {
    "gcs_virtual_host": "https://kubric-public.storage.googleapis.com/" + name,
    "gcs_json_media": "https://storage.googleapis.com/download/storage/v1/b/kubric-public/o/" + name.replace("/", "%2F") + "?alt=media",
}
for label, url in urls.items():
    for accelerated in (False, True):
        session = requests.Session()
        session.trust_env = accelerated
        start = time.monotonic()
        size = 0
        try:
            with session.get(url, headers={"Range": "bytes=0-1048575"}, timeout=(10, 15), stream=True) as r:
                status = r.status_code
                r.raise_for_status()
                for chunk in r.iter_content(65536):
                    size += len(chunk)
                    if size >= 1048576 or time.monotonic() - start > 20:
                        break
            elapsed = time.monotonic() - start
            print(json.dumps({"endpoint": label, "academic_acceleration": accelerated, "status": status,
                              "bytes": size, "seconds": round(elapsed, 2), "kib_per_sec": round(size/1024/elapsed, 1)}), flush=True)
        except Exception as exc:
            print(json.dumps({"endpoint": label, "academic_acceleration": accelerated,
                              "error_type": type(exc).__name__, "bytes": size}), flush=True)
