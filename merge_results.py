# merge_results.py
import json, glob, sys

output = sys.argv[1]          # e.g. results/all_results.json
files  = sys.argv[2:]         # e.g. results/results_zero_*.json results/results_pp_*.json

all_records = []
for f in files:
    with open(f) as fh:
        all_records.extend(json.load(fh))

with open(output, "w") as fh:
    json.dump(all_records, fh, indent=2)

print(f"Merged {len(all_records)} records into {output}") 