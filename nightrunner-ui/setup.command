#!/bin/bash
# Double-click this file in Finder to set up and launch night-runner-ui.
#
# FIRST TIME ONLY: macOS will likely refuse to open this with a message like
# "cannot be opened because it is from an unidentified developer". If that
# happens: right-click (or Control-click) this file, choose "Open" from the
# menu, then click "Open" again in the dialog that appears. You only need
# to do this once - after that, double-clicking works normally.

cd "$(dirname "$0")"

# Prefer python3 (standard on modern macOS); fall back to python if needed.
if command -v python3 &> /dev/null; then
    python3 setup.py
elif command -v python &> /dev/null; then
    python setup.py
else
    echo "Python doesn't seem to be installed on this Mac."
    echo "Please install it from https://www.python.org/downloads/ and try again."
    read -p "Press Enter to close this window..."
fi
