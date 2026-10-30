"""Verify published resources against the versioned SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.data / 'checksums/sha256.json').read_text())
    for name, expected in manifest['files'].items():
        path = (args.data / name).resolve()
        assert path.is_relative_to(args.data.resolve()), 'invalid manifest path'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    print(f"PASSED: {len(manifest['files'])} SHA-256 entries")


if __name__ == '__main__':
    main()
