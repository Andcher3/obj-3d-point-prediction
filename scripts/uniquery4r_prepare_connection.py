"""Interactive SSH transport; password stays in process memory, never on disk."""
import argparse
import getpass
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    password = getpass.getpass("SSH password (hidden): ")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username="root", password=password,
                   look_for_keys=False, allow_agent=False, timeout=30,
                   auth_timeout=30, banner_timeout=30)
    password = None
    transport = client.get_transport()
    transport.set_keepalive(20)
    transport.channel_timeout = 60
    print(json.dumps({"connected": True, "host_key_sha256_hex": hashlib.sha256(
        transport.get_remote_server_key().asbytes()).hexdigest()}), flush=True)
    try:
        for line in sys.stdin:
            try:
                job = json.loads(line)
                op = job["op"]
                if op == "close":
                    break
                if op == "upload":
                    with client.open_sftp() as sftp:
                        sftp.put(job["local"], job["remote"])
                    print(json.dumps({"uploaded": Path(job["local"]).name,
                                      "remote": job["remote"]}), flush=True)
                elif op == "download":
                    with client.open_sftp() as sftp:
                        sftp.get_channel().settimeout(60)
                        sftp.get(job["remote"], job["local"])
                    print(json.dumps({"downloaded": job["remote"]}), flush=True)
                elif op == "download_batch":
                    with client.open_sftp() as sftp:
                        sftp.get_channel().settimeout(60)
                        for item in job["files"]:
                            sftp.get(item["remote"], item["local"])
                            print(json.dumps({"downloaded": item["remote"]}), flush=True)
                elif op == "download_stream":
                    local_path = Path(job["local"])
                    offset = local_path.stat().st_size if job.get("resume") and local_path.exists() else 0
                    channel = transport.open_session(window_size=32*1024*1024,
                                                     max_packet_size=64*1024)
                    channel.settimeout(60)
                    command = ("tail -c +" + str(offset+1) + " -- " if offset else "cat -- ")
                    channel.exec_command(command + shlex.quote(job["remote"]))
                    received = offset
                    threshold = ((offset // (1024*1024)) + 1) * 1024*1024
                    with local_path.open("ab" if offset else "wb") as output:
                        while True:
                            block = channel.recv(256*1024)
                            if not block:
                                break
                            output.write(block)
                            received += len(block)
                            if received >= threshold:
                                print(json.dumps({"transfer_bytes": received}), flush=True)
                                threshold += 1024*1024
                    code = channel.recv_exit_status()
                    if code != 0:
                        raise RuntimeError("Remote file stream exited " + str(code))
                    channel.close()
                    print(json.dumps({"downloaded": job["remote"], "bytes": received,
                                      "resumed_from": offset,
                                      "transport": "single SSH stdout stream"}), flush=True)
                elif op == "exec":
                    channel = transport.open_session()
                    channel.exec_command(job["command"])
                    deadline = time.monotonic() + job.get("timeout", 180)
                    while True:
                        if channel.recv_ready():
                            print(channel.recv(65536).decode("utf-8", "replace"),
                                  end="", flush=True)
                        if channel.recv_stderr_ready():
                            print(channel.recv_stderr(65536).decode("utf-8", "replace"),
                                  end="", flush=True)
                        if (channel.exit_status_ready() and not channel.recv_ready()
                                and not channel.recv_stderr_ready()):
                            break
                        if time.monotonic() > deadline:
                            channel.close()
                            raise TimeoutError("SSH command channel timed out")
                        time.sleep(0.05)
                    print(json.dumps({"returncode": channel.recv_exit_status()}), flush=True)
                else:
                    raise ValueError("Unknown operation")
            except Exception as exc:
                print(json.dumps({"error": str(exc), "type": type(exc).__name__}), flush=True)
            print("READY", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
