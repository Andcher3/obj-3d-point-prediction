"""Pinned SSH inspection alongside the running model upload; no credentials saved."""
import argparse
import getpass
import hashlib
import json
import sys
import time
from pathlib import Path

import paramiko


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint

    def missing_host_key(self, client, hostname, key):
        if hashlib.sha256(key.asbytes()).hexdigest() != self.fingerprint:
            raise paramiko.SSHException("Unexpected SSH host key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host-key-sha256-hex", required=True)
    args = ap.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(PinnedHostKey(args.host_key_sha256_hex))
    password = getpass.getpass("SSH password (hidden): ")
    client.connect(args.host, port=args.port, username="root", password=password,
                   look_for_keys=False, allow_agent=False, timeout=30,
                   auth_timeout=30, banner_timeout=30)
    password = None
    transport = client.get_transport()
    actual = hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest()
    if actual != args.host_key_sha256_hex:
        client.close()
        raise ValueError("Unexpected SSH host key")
    transport.set_keepalive(20)
    print(json.dumps({"connected": True, "host_key_verified": True}), flush=True)
    try:
        for line in sys.stdin:
            try:
                job = json.loads(line)
                if job["op"] == "close":
                    break
                if job["op"] == "upload":
                    with client.open_sftp() as sftp:
                        sftp.put(job["local"], job["remote"])
                    print(json.dumps({"uploaded": Path(job["local"]).name}), flush=True)
                elif job["op"] == "download":
                    with client.open_sftp() as sftp:
                        sftp.get(job["remote"], job["local"])
                    print(json.dumps({"downloaded": job["remote"]}), flush=True)
                elif job["op"] == "exec":
                    channel = transport.open_session()
                    channel.exec_command(job["command"])
                    deadline = time.monotonic() + job.get("timeout", 60)
                    while True:
                        if channel.recv_ready():
                            print(channel.recv(65536).decode("utf-8", "replace"), end="", flush=True)
                        if channel.recv_stderr_ready():
                            print(channel.recv_stderr(65536).decode("utf-8", "replace"), end="", flush=True)
                        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                            break
                        if time.monotonic() > deadline:
                            channel.close()
                            raise TimeoutError("SSH command timed out")
                        time.sleep(0.05)
                    print(json.dumps({"returncode": channel.recv_exit_status()}), flush=True)
                else:
                    raise ValueError("Unsupported operation")
            except Exception as exc:
                print(json.dumps({"error": str(exc), "type": type(exc).__name__}), flush=True)
            print("READY", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
