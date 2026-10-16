"""Prepare relocated annotation copies without editing archived records."""
import csv
from pathlib import Path


def relocate_image(value, dataset):
    marker = '/datasets/RDD2022/'
    if marker not in value:
        raise ValueError(f'Expected an RDD2022 image path: {value}')
    return str(dataset / value.split(marker, 1)[1])


def prepare(project, dataset, output):
    project, dataset, output = map(lambda p: Path(p).resolve(), (project, dataset, output))
    for source, name in [
        (project/'datasets/RDD2022/ImageSets', 'country'),
        (project/'stage0-latent_domain_analysis/domain_splits', 'latent'),
    ]:
        dest = output/name
        dest.mkdir(parents=True, exist_ok=True)
        for text in sorted(source.glob('*.txt')):
            lines = []
            for line in text.read_text().splitlines():
                if not line.strip():
                    continue
                parts = []
                for part in line.split('&'):
                    fields = part.split()
                    fields[0] = relocate_image(fields[0], dataset)
                    parts.append(' '.join(fields))
                lines.append('&'.join(parts))
            (dest/text.name).write_text('\n'.join(lines)+'\n')
    source = project/'stage0-latent_domain_analysis/results/final_cluster_labels.csv'
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row['image_path'] = relocate_image(row['image_path'], dataset)
    with (output/'final_cluster_labels.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def check(project, dataset, inputs):
    def records(path):
        result, domains = {}, {}
        for line in path.read_text().splitlines():
            for domain, part in enumerate(line.split('&')):
                fields = part.split()
                image = str(Path(fields[0]).resolve())
                record = fields[1:]
                if image in result and (result[image] != record or domains[image] != domain):
                    raise ValueError(f'Conflicting image record: {image}')
                result[image], domains[image] = record, domain
        return result, domains

    country, _ = records(inputs/'country/251030_train.txt')
    latent, domain_ids = records(inputs/'latent/combined_train.txt')
    country_val, _ = records(inputs/'country/251030_val.txt')
    latent_val, _ = records(inputs/'latent/combined_val.txt')
    assert country == latent and country_val == latent_val
    assert len(country) == 12742 and len(country_val) == 1592
    assert not country.keys() & country_val.keys()
    with (inputs/'final_cluster_labels.csv').open() as stream:
        labels = {str(Path(r['image_path']).resolve()): int(r['latent_domain']) for r in csv.DictReader(stream)}
    assert labels == domain_ids
    assert [list(labels.values()).count(i) for i in (0, 1)] == [5913, 6829]
    missing = [p for p in set(country) | set(country_val) if not Path(p).is_file()]
    domains = ['Japan', 'India', 'United_States', 'Norway', 'new_Czech', 'new_China_Drone', 'new_China_MotorBike']
    test_counts = {}
    for domain in domains:
        ids = (project/'datasets/RDD2022'/domain/'ImageSets/Main/test_all.txt').read_text().split()
        test_counts[domain] = len(ids)
        for image in ids:
            for folder, suffix in [('JPEGImages', '.jpg'), ('Annotations', '.xml')]:
                path = dataset/domain/folder/(image+suffix)
                if not path.is_file():
                    missing.append(str(path))
    assert sum(test_counts.values()) == 9433
    print(f'Annotation protocol OK: train={len(country)}, validation={len(country_val)}, test={test_counts}')
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing data files; first paths: {missing[:5]}')
    for name in ['Japan', 'India', 'United_States', 'Cluster_0', 'Cluster_1']:
        for filename in ['best_epoch_weights.pth', 'best_checkpoint.json']:
            path = project/'snapshot_dependencies/experts'/name/filename
            if not path.is_file():
                raise FileNotFoundError(path)
    if not (project/'model_data/yolov8_s.pth').is_file():
        raise FileNotFoundError('model_data/yolov8_s.pth')
    print('Image files, test XML, initialization and five expert best checkpoints are present.')
