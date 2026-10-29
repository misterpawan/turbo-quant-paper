"""Check every main quality-grid cell, per-example arrays, duplicates, and failures."""
import argparse
import json
from collections import Counter
from pathlib import Path

METHODS = ('bf16', 'w3', 'w4', 'gptq_diag_4', 'awq_4', 'qdiff4', 'turboldm_3p1', 'turboldm_4p1')
TASKS = ('mmlu', 'arc_challenge', 'hellaswag', 'gsm8k')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/overnight'))
    args = parser.parse_args()
    expected = {(m, d, s) for m in METHODS for d in TASKS for s in (42, 43, 44)}
    counts, invalid = Counter(), []
    path = args.runs / 'phase34_main_ci_n500.jsonl'
    if not path.exists():
        raise SystemExit(f'Missing {path}; run codes/run_phase34.py first.')
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row.get('method'), row.get('dataset'), row.get('seed'))
            extra = row.get('extra', {})
            if isinstance(extra, str):
                extra = json.loads(extra)
            correct = extra.get('correct_per_example', [])
            if key not in expected or row.get('model') != 'llada_8b_instruct':
                raise ValueError('unexpected method/task/seed/model')
            if row.get('n_examples') != 500 or len(correct) != 500:
                raise ValueError('expected 500 evaluated examples and correctness entries')
            if any(v not in (0, 1, False, True) for v in correct):
                raise ValueError('correctness entries must be binary')
            if abs(sum(correct) / 500 - row['score_value']) > 1e-6:
                raise ValueError('accuracy does not match per-example correctness')
            counts[key] += 1
        except (ValueError, TypeError, KeyError) as exc:
            invalid.append(f'line {i}: {exc}')
    missing = sorted(expected - counts.keys())
    duplicates = {str(k): v for k, v in counts.items() if v > 1}
    print(json.dumps({'valid_cells': len(counts), 'expected_cells': len(expected),
                      'missing': missing, 'duplicates': duplicates, 'invalid': invalid}, indent=2))
    failures = args.runs / 'failures.jsonl'
    if failures.exists() and failures.stat().st_size:
        print(f'Inspect {failures}: it contains failure history, including possibly recovered cells.')
    raise SystemExit(1 if missing or duplicates or invalid else 0)


if __name__ == '__main__':
    main()
