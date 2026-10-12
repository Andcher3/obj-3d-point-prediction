"""Verify a transferred review archive and extract only safe regular files."""
import argparse
import hashlib
import tarfile
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    args=ap.parse_args()
    if sha(args.archive)!=args.sha256:
        raise ValueError('Archive SHA256 mismatch; nothing was extracted')
    base=args.base.resolve()
    with tarfile.open(args.archive,'r:gz') as tar:
        members=tar.getmembers()
        seen=set()
        for member in members:
            target=(base/member.name).resolve()
            if not member.isfile() or base not in target.parents or target in seen:
                raise ValueError('Unsafe or duplicate archive member: '+member.name)
            seen.add(target)
            if target.exists():
                with tar.extractfile(member) as data:
                    if hashlib.sha256(data.read()).hexdigest()!=sha(target):
                        raise FileExistsError('Conflicting existing artifact: '+str(target))
        tar.extractall(base,members=members)
    print('Archive SHA256 passed; safe regular files extracted:',len(members))


if __name__=='__main__':
    main()
