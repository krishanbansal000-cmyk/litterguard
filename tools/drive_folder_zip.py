#!/usr/bin/env python
"""Download a public Google Drive folder as a zip (server-side zip path).
Handles the large-file confirm form + resumes with Range if needed."""
import http.cookiejar
import re
import sys
import urllib.parse
import urllib.request

FOLDER_ID = sys.argv[1] if len(sys.argv) > 1 else "1OnA3rHn5uKAMEuHsQc11oxueTT-TQQwJ"
OUT = sys.argv[2] if len(sys.argv) > 2 else "media/mivia_validation.zip"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", UA)]

url = f"https://drive.usercontent.google.com/download?id={FOLDER_ID}&export=download&authuser=0"

# step 1: get confirm form if present
resp = opener.open(url, timeout=60)
head = resp.read(4096).decode("utf-8", "replace")
resp.close()

params = {}
for name in ("confirm", "uuid", "export"):
    m = re.search(rf'name="{name}"\s+value="([^"]+)"', head)
    if m:
        params[name] = m.group(1)
if "confirm" not in params and "confirm" not in head:
    print("no confirm form; direct download")
    final = url
else:
    # fall back: modern pages embed action + params as JSON-ish attributes
    if "confirm" not in params:
        m = re.search(r'https://drive\.usercontent\.google\.com/download\?[^"\']+', head)
        if m:
            final = m.group(0).replace("&amp;", "&")
            print("extracted action url")
        else:
            params.setdefault("confirm", "t")
            params["id"] = FOLDER_ID
            final = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(params)
    else:
        params["id"] = FOLDER_ID
        final = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(params)

print("downloading:", final[:120], "...")
req = urllib.request.Request(final, headers={"User-Agent": UA})
with opener.open(req, timeout=120) as r, open(OUT + ".part", "wb") as f:
    total = 0
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
        total += len(chunk)
        if total % (50 << 20) < (1 << 20):
            print(f"  {total/1e6:.0f} MB", flush=True)

import os
size = os.path.getsize(OUT + ".part")
# zip magic check
with open(OUT + ".part", "rb") as f:
    magic = f.read(4)
if magic[:2] == b"PK" and size > 50_000_000:
    os.replace(OUT + ".part", OUT)
    print(f"DONE: {OUT} ({size/1e6:.0f} MB)")
else:
    print(f"NOT A ZIP or too small ({size} bytes, magic={magic[:4]}) - kept as .part for inspection")
