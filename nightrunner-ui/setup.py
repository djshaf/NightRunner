#!/usr/bin/env python3
"""
night-runner-ui setup script.

Checks Docker is installed and running, builds and starts the full stack
(Valhalla + safety/lighting reranker + web-app), waits for it to actually
be ready, then opens it in your browser.

Safe to run more than once - it won't reinstall or rebuild things that
are already fine, and it never tries to install Docker itself (that
needs your manual OK, since it's a system-level install).
"""
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# When PyInstaller bundles this into a single .exe, __file__ no longer
# points to where the .exe actually sits - it resolves to a temporary
# extraction folder instead. sys.executable is the correct thing to use
# in that case. This check keeps the script working identically whether
# it's run as plain setup.py or as the compiled .exe.
if getattr(sys, "frozen", False):
    SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    SCRIPT_DIR = Path(__file__).resolve().parent

SAFE_ROUTING_DIR = SCRIPT_DIR / "safe-routing"
WEB_APP_DIR = SCRIPT_DIR / "web-app"
APP_URL = "http://localhost:3000"

DOCKER_INSTALL_URL = "https://www.docker.com/products/docker-desktop/"


def _print_step(step_num: int, total: int, message: str) -> None:
    print(f"\n[{step_num}/{total}] {message}")


def _fail(message: str) -> None:
    print(f"\n{'=' * 60}")
    print("SETUP STOPPED")
    print('=' * 60)
    print(message)
    print()
    input("Press Enter to close this window...")
    sys.exit(1)


def check_folders_present() -> None:
    missing = []
    if not SAFE_ROUTING_DIR.exists():
        missing.append("safe-routing/")
    if not WEB_APP_DIR.exists():
        missing.append("web-app/")
    if missing:
        _fail(
            "This setup script expects to sit alongside both of these folders:\n"
            f"  {', '.join(missing)}\n\n"
            "Make sure you downloaded/extracted the WHOLE night-runner-ui folder, "
            "not just this script on its own."
        )


def check_docker_installed() -> None:
    if shutil.which("docker") is None:
        _fail(
            "Docker Desktop doesn't seem to be installed.\n\n"
            f"1. Download and install it from:\n   {DOCKER_INSTALL_URL}\n"
            "2. Open Docker Desktop and wait for it to say it's running.\n"
            "3. Re-run this setup script."
        )


def check_docker_running() -> None:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        result = None

    if result is None or result.returncode != 0:
        _fail(
            "Docker is installed, but it doesn't look like it's running.\n\n"
            "Please open Docker Desktop (check for its icon in your taskbar/menu "
            "bar), wait until it says it's ready, then re-run this setup script."
        )


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) != 0


def check_ports_available() -> None:
    busy = [p for p in (3000, 5050, 8002) if not _port_is_free(p)]
    if busy:
        print(
            f"NOTE: port(s) {', '.join(str(p) for p in busy)} already have something "
            "running on them. If that's an earlier run of this same app, that's "
            "fine - we'll just reuse it. If it's something else on your machine, "
            "you may need to close it first."
        )


def build_and_start() -> None:
    print("Building the app - this can take a few minutes the first time "
          "(downloading images, installing packages)...")
    build = subprocess.run(
        ["docker", "compose", "build"],
        cwd=SAFE_ROUTING_DIR,
    )
    if build.returncode != 0:
        _fail(
            "The build step failed - scroll up to see the actual error from "
            "Docker above. Common causes: no internet connection, or Docker "
            "Desktop needs more disk space allocated in its settings."
        )

    up = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=SAFE_ROUTING_DIR,
    )
    if up.returncode != 0:
        _fail("Starting the containers failed - scroll up to see the error from Docker above.")


def wait_until_ready(timeout_s: int = 180) -> bool:
    import urllib.request

    print("Waiting for the app to finish starting up...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(APP_URL, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print()
    return False


def main() -> None:
    total_steps = 6
    print("=" * 60)
    print("night-runner-ui setup")
    print("=" * 60)

    _print_step(1, total_steps, "Checking everything's in the right place...")
    check_folders_present()

    _print_step(2, total_steps, "Checking Docker is installed...")
    check_docker_installed()

    _print_step(3, total_steps, "Checking Docker is running...")
    check_docker_running()

    _print_step(4, total_steps, "Checking required ports are free...")
    check_ports_available()

    _print_step(5, total_steps, "Building and starting the app...")
    build_and_start()

    _print_step(6, total_steps, "Waiting for it to be ready...")
    ready = wait_until_ready()

    print("\n" + "=" * 60)
    if ready:
        print(f"All set! Opening {APP_URL} in your browser now.")
        print("=" * 60)
        webbrowser.open(APP_URL)
    else:
        print(
            "The app is taking longer than expected to start. It may still be\n"
            f"finishing up - try opening {APP_URL} in your browser in a minute,\n"
            "or run 'docker compose logs' inside the safe-routing folder to see\n"
            "what's happening."
        )
        print("=" * 60)

    input("\nPress Enter to close this window...")


if __name__ == "__main__":
    main()
