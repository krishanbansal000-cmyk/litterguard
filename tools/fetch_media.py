#!/usr/bin/env python
"""Fetch sample test videos (litter-drop scenes + street negatives)."""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "media")

# LitterBug project sample videos (GPL repo, direct raw links)
CANDIDATES = [
    ("litter1.mp4", "https://raw.githubusercontent.com/LitterBugCam/LitterBug-Algorithm/Desktop_Litterbug/litter1.mp4"),
    ("Litter2.avi", "https://raw.githubusercontent.com/LitterBugCam/LitterBug-Algorithm/Desktop_Litterbug/Litter2.avi"),
    ("litter3.mp4", "https://raw.githubusercontent.com/LitterBugCam/LitterBug-Algorithm/Desktop_Litterbug/litter3.mp4"),
]


def main():
    os.makedirs(MEDIA, exist_ok=True)
    got = []
    for name, url in CANDIDATES:
        dest = os.path.join(MEDIA, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print(f"[skip] {name}")
            got.append(dest)
            continue
        try:
            print(f"[get ] {name}")
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 50_000:
                os.remove(tmp)
                print(f"[miss] {name} (too small / 404)")
                continue
            os.replace(tmp, dest)
            print(f"[done] {name} ({os.path.getsize(dest)/1e6:.1f} MB)")
            got.append(dest)
        except Exception as e:
            print(f"[miss] {name}: {e}")
    if not got:
        print("No videos fetched — check network or add URLs to CANDIDATES.", file=sys.stderr)
        return 1
    print("\nAvailable test media:")
    for p in got:
        print("  " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
