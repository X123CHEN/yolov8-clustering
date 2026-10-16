"""Restore all losslessly archived per-image evaluation files."""
import argparse
import csv
from pathlib import Path
import tarfile


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=project)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    destination = args.destination.resolve()
    with (project/'snapshot/evaluation_archives.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        archive = project/row['archive']
        if args.list:
            print(row['archive'], row['files'])
            continue
        (destination/row['restored_directory']).mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as packed:
            packed.extractall(destination)
    print(f'{len(rows)} evaluation archives; {sum(int(r["files"]) for r in rows)} files')


if __name__ == '__main__':
    main()
