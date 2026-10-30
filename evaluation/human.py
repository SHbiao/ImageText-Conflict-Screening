"""Identical-mask source/human evaluation, with model-specific and shared exclusions."""
from collections import defaultdict
from evaluation.data import read
from evaluation.metrics import metrics, pair_metrics


def evaluate(data):
    human = {r['case_id']: r for r in read(data / 'data/human_reference.csv')}
    source = {r['case_id']: r for r in read(data / 'data/source_labels.csv')}
    by_model = defaultdict(list)
    for row in read(data / 'predictions/readout_scores/human_predictions.csv'):
        by_model[row['model']].append(row)
    out = []
    for model, rows in by_model.items():
        for cohort in ['random', 'hard']:
            for scope in ['all', 'model_training_dev_excluded', 'common_training_dev_excluded']:
                planned = [r for r in rows if r['cohort'] == cohort and
                    (scope == 'all' or r['training_dev_excluded' if scope.startswith('model_')
                                         else 'common_training_dev_excluded'] == 'True')]
                valid = [r for r in planned if human[r['case_id']]['human_reference'] != 'INSUFFICIENT'
                         and r['scoring_status'] == 'SCORED']
                for field in ['source_label', 'human_reference']:
                    labels = [(source if field == 'source_label' else human)[r['case_id']][field] == 'CONFLICT'
                              for r in valid]
                    scores = [float(r['score']) for r in valid]
                    threshold = float(rows[0]['frozen_threshold'])
                    out.append(dict(model=model, cohort=cohort, scope=scope, labels=field,
                        planned_conditions=len(planned), planned_images=len({r['image_id'] for r in planned}),
                        human_insufficient=sum(human[r['case_id']]['human_reference'] == 'INSUFFICIENT'
                                               for r in planned), scoring_failure=len([r for r in planned
                                                  if r['scoring_status'] != 'SCORED']),
                        **metrics(labels, scores, threshold),
                        **pair_metrics([r['image_id'] for r in valid], labels, scores)))
    return out
