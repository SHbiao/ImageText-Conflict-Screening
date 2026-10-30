"""Native strict caption selection; all planned pairs stay in the denominator."""
from collections import defaultdict
from evaluation.data import read


def evaluate(data):
    groups = defaultdict(list)
    for row in read(data / 'predictions/external_scores/sugarcrepe.csv.gz'):
        for category in [row['category'], 'all']:
            groups[row['model'], category].append(row)
    result = []
    for (model, category), rows in groups.items():
        correct = sum(r['status'] == 'SCORED' and float(r['positive_score']) < float(r['negative_score'])
                      for r in rows)
        result.append(dict(model=model, category=category, planned_pairs=len(rows),
            image_groups=len({r['image_id'] for r in rows}), correct=correct,
            native_accuracy_full_denominator=correct / len(rows),
            failures=sum(r['status'] != 'SCORED' for r in rows),
            ties=sum(r['tie'] == 'True' for r in rows)))
    return result
