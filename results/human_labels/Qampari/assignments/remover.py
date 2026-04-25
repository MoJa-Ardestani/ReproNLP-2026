import json
import glob
import os

base = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(base, "**/*.json"), recursive=True)

for path in files:
    with open(path) as f:
        data = json.load(f)
    changed = False
    tasks = data.get("tasks", [])
    for task in tasks:
        if "context_segments" in task:
            del task["context_segments"]
            changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated: {os.path.relpath(path, base)}")
    else:
        print(f"No change: {os.path.relpath(path, base)}")
