## Heuristics

```bash
# These scripts accept the same arguments.
# heuristics/random_unweighted.py
# heuristics/random_weighted.py
# heuristics/most_common_answer.py

python heuristics/random_unweighted.py --aokvqa-dir ${AOKVQA_DIR} --split val --mc --out ${PREDS_DIR}/random-unweighted_val-mc.json
# Exclude --mc for the direct answer setting
```
