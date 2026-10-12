"""Bounded remote throughput samples; never logs proxies or redirect URLs."""
import json
import time
import requests

for endpoint,accelerated in [('hf-mirror.com',False),('huggingface.co',False),('huggingface.co',True)]:
    session=requests.Session(); session.trust_env=accelerated
    started=time.monotonic(); size=0
    try:
        url='https://'+endpoint+'/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth'
        with session.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(8,12)) as r:
            r.raise_for_status()
            for chunk in r.iter_content(65536):
                size+=len(chunk)
                if size>=1048576 or time.monotonic()-started>18: break
            elapsed=time.monotonic()-started
            print(json.dumps({'endpoint':endpoint,'acceleration':accelerated,'status':r.status_code,
                'bytes':size,'seconds':round(elapsed,2),'KiB_s':round(size/1024/elapsed,1)}),flush=True)
    except Exception as e:
        print(json.dumps({'endpoint':endpoint,'acceleration':accelerated,'error_type':type(e).__name__}),flush=True)
