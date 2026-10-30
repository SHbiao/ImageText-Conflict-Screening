"""Small CSV helpers shared by result reconstruction commands."""
import csv
import gzip
from pathlib import Path


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
