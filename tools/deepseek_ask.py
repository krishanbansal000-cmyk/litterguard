#!/usr/bin/env python
"""Send a prompt file to DeepSeek v4.1-flash via the OpenCode Go endpoint."""
import json
import sys
import urllib.request

prompt_path = sys.argv[1]
with open(prompt_path, encoding="utf-8") as f:
    prompt = f.read()

body = json.dumps({"model": "deepseek-v4.1-flash",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 2200, "temperature": 0.3}).encode()
req = urllib.request.Request(
    "https://opencode.ai/zen/go/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer oc_sk_662eaa1edb43_3VSgzBF0uB649nmB7GEHlVG7wR6KA13D",
        "Content-Type": "application/json", "User-Agent": "litterguard/0.1",
        "x-opencode-session": "lg-recall-1"})
out = json.load(urllib.request.urlopen(req, timeout=300))
msg = out["choices"][0]["message"]
print(msg.get("content") or msg.get("reasoning_content"))
