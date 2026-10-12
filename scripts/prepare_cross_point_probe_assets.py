"""Prepare frozen-model source/weights on data disk. No Torch/model forward."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--skip-drive',action='store_true',help='Download other official assets; explicitly retain missing fine checkpoint status')
    ap.add_argument('--hf-endpoint',choices=['official','mirror'],default='official')
    args=ap.parse_args(); base=args.base
    for name in ['source','weights','logs']: (base/name).mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    repo='brains-bots-n-behavior/motionforesight'
    r=session.get('https://api.github.com/repos/'+repo+'/commits/main',timeout=45); r.raise_for_status()
    revision=r.json()['sha']
    source=base/'source'/('motionforesight-'+revision)
    assets=[{'kind':'source','url':'https://codeload.github.com/'+repo+'/zip/'+revision,'path':base/'source/motionforesight.zip','sha256':None}]
    repositories=[('trackcraft3r/checkpoint',['model.safetensors'],'trackcraft3r'),
        ('Wan-AI/Wan2.1-T2V-1.3B',['Wan2.1_VAE.pth','diffusion_pytorch_model.safetensors'],'Wan-AI/Wan2.1-T2V-1.3B'),
        ('depth-anything/DA3-BASE',['model.safetensors','config.json'],'DA3-BASE')]
    for name,wanted,local in repositories:
        r=session.get('https://huggingface.co/api/models/'+name+'?blobs=true',timeout=45); r.raise_for_status(); info=r.json()
        files={f['rfilename']:f for f in info['siblings']}
        if name.startswith('Wan-AI'):
            # Loaders may initialize the tokenizer even with cached empty text.
            wanted += [p for p in files if p.startswith('google/umt5-xxl/')]
        for filename in wanted:
            if filename not in files: raise ValueError('Official model file missing: '+name+'/'+filename)
            f=files[filename]
            assets.append({'kind':name,'revision':info['sha'],'url':'https://huggingface.co/'+name+'/resolve/'+info['sha']+'/'+filename,
                'path':base/'weights'/local/filename,'sha256':f.get('lfs',{}).get('sha256'),'bytes':f.get('size') or f.get('lfs',{}).get('size')})
    # Official MotionForesight README links this Drive file. Resolve the public
    # confirmation form only; never save cookies or log signed download URLs.
    drive='1isAB_TnNW8gbMlgs5yVnPBKZSR85-JyZ'
    if not args.skip_drive:
        r=session.get('https://drive.usercontent.google.com/download?export=download&confirm=t&id='+drive,timeout=45,stream=True); r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''):
            form=BeautifulSoup(r.text,'html.parser').find('form')
            if form is None: raise ValueError('Drive public download form unavailable')
            params={i.get('name'):i.get('value','') for i in form.find_all('input') if i.get('name')}
            fine_url=urllib.parse.urljoin(r.url,form.get('action'))+'?'+urllib.parse.urlencode(params)
        else: fine_url=r.url
        r.close()
        assets.append({'kind':'MotionForesight_finetuned','url':fine_url,'path':base/'weights/motionforesight-release.zip','sha256':None,'official_drive_id':drive})
    total=sum(a.get('bytes') or 0 for a in assets)
    if shutil.disk_usage(base).free<total+6*1024**3: raise ValueError('Insufficient data disk reserve')
    manifest=[]; queue=[]
    for a in assets:
        a['path'].parent.mkdir(parents=True,exist_ok=True)
        manifest.append({k:str(v) if isinstance(v,Path) else v for k,v in a.items() if k!='url'})
        if a['kind']=='source' and a['path'].exists() and not Path(str(a['path'])+'.aria2').exists():
            try:
                with zipfile.ZipFile(a['path']) as z:
                    if z.testzip() is None: continue
            except zipfile.BadZipFile: pass
        if a['path'].exists() and a.get('sha256') and sha(a['path'])==a['sha256']: continue
        if args.hf_endpoint=='mirror': a['url']=a['url'].replace('https://huggingface.co/','https://hf-mirror.com/')
        queue.append(a['url']+'\n  dir='+str(a['path'].parent)+'\n  out='+a['path'].name+'\n'+('  checksum=sha-256='+a['sha256']+'\n' if a.get('sha256') else ''))
    (base/'logs/asset-manifest.json').write_text(json.dumps({'assets':manifest,'repo_revision':revision},indent=2))
    queue_path=base/'logs/model-aria2-queue.private.txt'; queue_path.write_text(''.join(queue)); queue_path.chmod(0o600)
    env=os.environ.copy()
    if args.hf_endpoint=='mirror':
        for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']: env.pop(key,None)
    with (base/'logs/models-aria2.log').open('a') as log:
        subprocess.run(['aria2c','-c','-j','3','-x','16','-s','16','--min-split-size=1M','--file-allocation=none',
            '--console-log-level=error','--summary-interval=15','--max-tries=3','--auto-file-renaming=false',
            '--allow-overwrite=true','--timeout=45','-i',str(queue_path)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    rows=[]
    for a in assets:
        p=a['path']; digest=sha(p)
        if a.get('sha256') and digest!=a['sha256']: raise ValueError('Official SHA256 mismatch: '+p.name)
        if a.get('bytes') and p.stat().st_size!=a['bytes']: raise ValueError('Official size mismatch: '+p.name)
        rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'official_sha256_available':a.get('sha256') is not None})
    archives=[(base/'source/motionforesight.zip',base/'source')]
    if not args.skip_drive: archives.append((base/'weights/motionforesight-release.zip',base/'weights/motionforesight'))
    for archive,dest in archives:
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip()
            if bad: raise ValueError('Archive CRC failure')
            for item in z.infolist():
                resolved=(dest/item.filename).resolve()
                if not resolved.is_relative_to(dest.resolve()): raise ValueError('Unsafe archive path')
            z.extractall(dest)
    configs=list((base/'weights/motionforesight').rglob('config.json'))
    for p in configs: print('FINE_CONFIG',p.read_text()[:3000],flush=True)
    (base/'logs/assets-complete.json').write_text(json.dumps({'assets_ready':not args.skip_drive,'source':str(source),'files':rows,
        'missing_assets':['MotionForesight_finetuned'] if args.skip_drive else [],
        'model_loaded':False,'forward_called':False,'gpu_used':False,'fine_archive_crc_pass':not args.skip_drive},indent=2))
    print('ASSETS_DOWNLOAD_COMPLETE_NO_MODEL_FORWARD',flush=True)


if __name__=='__main__': main()
