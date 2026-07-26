#!/usr/bin/env python3
"""
Minimal reproduction of the vLLM TCP delay issue.

Hypothesis: The EngineCore process inherits the pre-bound server socket via
fork(), and the dual fd reference in parent+child causes the kernel to delay
accepting external TCP connections.

This script mimics the vLLM startup flow:
  1. Create + bind socket (like api_server.py:setup_server)
  2. Fork child process (like EngineCoreProc via multiprocessing fork)
  3. Parent: listen() + start accept loop
  4. Test self-connect (same process) vs external-connect (subprocess)

Usage:
  python3 scripts/diagnose_fork_socket.py [fork|spawn|spawn-fork]
"""

import multiprocessing
import os
import socket
import subprocess
import sys
import time


def create_server_socket(host: str, port: int) -> socket.socket:
    """Mimic vLLM's create_server_socket()."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Also set SO_REUSEPORT to match vLLM behavior
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    print("[PARENT {os.getpid()}] Socket bound to {host}:{port}, fd={sock.fileno()}")
    return sock


def child_process(sock_fd: int):
    """Simulate EngineCore process — inherits fds but does NOT use the server socket."""
    pid = os.getpid()
    print("[CHILD {pid}] Started (inherited fd={sock_fd})")

    # Print all open fds to verify the socket is inherited
    try:
        fd_dir = f"/proc/{pid}/fd"
        fds = os.listdir(fd_dir)
        print("[CHILD {pid}] Open fds: {sorted(int(f) for f in fds)}")
        for fd_name in sorted(fds, key=int):
            try:
                link = os.readlink(os.path.join(fd_dir, fd_name))
                if "socket" in link:
                    print("[CHILD {pid}]   fd {fd_name} -> {link}")
            except OSError:
                pass
    except Exception:
        print("[CHILD {pid}] Cannot list fds: {e}")

    # Sleep to keep the process alive (like EngineCore does)
    print("[CHILD {pid}] Sleeping (simulating EngineCore busy loop)...")
    time.sleep(600)  # 10 min, longer than the TCP delay


def external_connect(host: str, port: int, label: str, timeout: float = 5.0):
    """Try TCP connect from a subprocess (like curl does)."""
    code = f"""
import socket, time, sys
s = socket.socket()
s.settimeout({timeout})
t0 = time.monotonic()
try:
    s.connect(('{host}', {port}))
    dt = time.monotonic() - t0
    print("[{label}] CONNECTED after {{dt:.3f}}s")
    s.close()
    sys.exit(0)
except Exception as e:
    dt = time.monotonic() - t0
    print("[{label}] FAILED after {{dt:.3f}}s: {{e}}")
    sys.exit(1)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print("  stderr: {proc.stderr.strip()}")
    return proc.returncode == 0


def self_connect(host: str, port: int, timeout: float = 5.0):
    """Try TCP connect from the SAME process (like vLLM's SELF_PROBE)."""
    s = socket.socket()
    s.settimeout(timeout)
    t0 = time.monotonic()
    try:
        s.connect((host, port))
        time.monotonic() - t0
        print("[SELF_CONNECT] CONNECTED after {dt:.3f}s")
        s.close()
        return True
    except Exception:
        time.monotonic() - t0
        print("[SELF_CONNECT] FAILED after {dt:.3f}s: {e}")
        return False


def test_fork_mode():
    """Test with fork() — mimics the current vLLM behavior on Ascend."""
    HOST, PORT = "127.0.0.1", 19991
    sock = create_server_socket(HOST, PORT)
    sock_fd = sock.fileno()

    print("\n=== Forking child process ===")
    pid = os.fork()
    if pid == 0:
        # Child: simulate EngineCore
        sock.close()  # Child closes its copy — test WITHOUT socket inheritance
        child_process(sock_fd)
        os._exit(0)
    else:
        # Parent: start listening
        print("[PARENT {os.getpid()}] Child PID = {pid}")
        sock.listen(128)
        print("[PARENT {os.getpid()}] Socket listening, entering accept loop...")

        # Give child time to start
        time.sleep(1)

        # Test 1: Self-connect (same process)
        print("\n--- Test 1: Self-connect ---")
        self_connect(HOST, PORT)

        # Test 2: External connect (from subprocess)
        print("\n--- Test 2: External connect (every 2s, max 30s) ---")
        for i in range(15):
            time.sleep(2)
            if external_connect(HOST, PORT, f"EXT_iter{i}"):
                break
        else:
            print("[RESULT] External connect FAILED — reproducing the issue!")

        # Cleanup
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        sock.close()


def test_fork_with_inheritance():
    """Test with fork() where child does NOT close the socket — exact vLLM behavior."""
    HOST, PORT = "127.0.0.1", 19992
    sock = create_server_socket(HOST, PORT)
    sock_fd = sock.fileno()

    print("\n=== Forking child (KEEPING inherited socket) ===")
    pid = os.fork()
    if pid == 0:
        # Child: keep the inherited socket (like EngineCore does)
        child_process(sock_fd)
        os._exit(0)
    else:
        print("[PARENT {os.getpid()}] Child PID = {pid}")
        sock.listen(128)
        print("[PARENT {os.getpid()}] Socket listening, entering accept loop...")

        time.sleep(1)

        print("\n--- Test 1: Self-connect ---")
        self_connect(HOST, PORT)

        print("\n--- Test 2: External connect (every 2s, max 60s) ---")
        success = False
        for i in range(30):
            time.sleep(2)
            if external_connect(HOST, PORT, f"EXT_iter{i}"):
                success = True
                break
        if not success:
            print("[RESULT] External connect FAILED! Socket fd inheritance IS the root cause!")

        os.kill(pid, 9)
        os.waitpid(pid, 0)
        sock.close()
        return success


def test_spawn_mode():
    """Test simulating spawn — child does NOT inherit any fds."""
    HOST, PORT = "127.0.0.1", 19993
    sock = create_server_socket(HOST, PORT)

    print("\n=== Spawning child via multiprocessing (no fd inheritance) ===")

    # Use spawn to launch child — no fd inheritance
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=child_process, args=(-1,))
    proc.start()

    print("[PARENT {os.getpid()}] Child PID = {proc.pid}")
    sock.listen(128)
    print("[PARENT {os.getpid()}] Socket listening, entering accept loop...")

    time.sleep(1)

    print("\n--- Test 1: Self-connect ---")
    self_connect(HOST, PORT)

    print("\n--- Test 2: External connect (every 2s, max 30s) ---")
    success = False
    for i in range(15):
        time.sleep(2)
        if external_connect(HOST, PORT, f"EXT_iter{i}"):
            success = True
            break
    if success:
        print("[RESULT] External connect OK — spawn mode fixes it!")

    proc.kill()
    proc.join()
    sock.close()
    return success


def check_engine_core_fds():
    """Check if any running engine core process has inherited the server socket."""
    print("\n=== Checking running processes for inherited sockets ===")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "EngineCore|engine_core"],
            capture_output=True, text=True,
        )
        engine_pids = result.stdout.strip().split()
        if not engine_pids:
            print("No EngineCore processes found.")
            return

        for pid_str in engine_pids:
            pid = int(pid_str)
            fd_dir = f"/proc/{pid}/fd"
            if not os.path.isdir(fd_dir):
                continue
            fds = os.listdir(fd_dir)
            sockets = []
            for fd_name in sorted(fds, key=int):
                try:
                    link = os.readlink(os.path.join(fd_dir, fd_name))
                    if "socket" in link:
                        # Get more info from /proc/net/tcp
                        sockets.append((fd_name, link))
                except OSError:
                    pass
            print("  PID {pid}: {len(sockets)} socket fds")
            for fd_name, link in sockets[:20]:
                print("    fd {fd_name}: {link}")
    except Exception:
        print("Error: {e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"

    if mode == "fork":
        test_fork_mode()
    elif mode == "fork-inherit":
        test_fork_with_inheritance()
    elif mode == "spawn":
        test_spawn_mode()
    elif mode == "check":
        check_engine_core_fds()
    elif mode == "compare":
        # Run both and compare
        print("=" * 60)
        print("TEST A: fork WITHOUT socket inheritance (child closes socket)")
        print("=" * 60)
        test_fork_mode()

        print("\n\n" + "=" * 60)
        print("TEST B: fork WITH socket inheritance (exact vLLM behavior)")
        print("=" * 60)
        test_fork_with_inheritance()

        print("\n\n" + "=" * 60)
        print("TEST C: spawn mode (child does NOT inherit fds)")
        print("=" * 60)
        test_spawn_mode()

        print("\n\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("  Test B (fork + inherit): {'FAIL (reproduced!)' if not result_b else 'PASS'}")
        print("  Test C (spawn):          {'PASS (fixed!)' if result_c else 'FAIL'}")
    else:
        print("Unknown mode: {mode}")
        print("Usage: python3 {sys.argv[0]} [fork|fork-inherit|spawn|check|compare]")


if __name__ == "__main__":
    main()
