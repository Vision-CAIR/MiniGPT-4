## Querying GPT-3

To follow our experiments which use GPT-3, you must have access to the [OpenAI API](https://openai.com/api/) (at cost). Please retrieve your [organization](https://beta.openai.com/account/org-settings) and [API](https://beta.openai.com/account/api-keys) keys and set them in your environment variables.

```bash
export OPENAI_ORG=....
export OPENAI_API_KEY=...
```

For producing predictions for both DA and MC settings, run:
```bash
python gpt3/query_gpt3.py --aokvqa-dir ${AOKVQA_DIR} --split val --out ${PREDS_DIR}/gpt3_val-da.json
python remap_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --pred ${PREDS_DIR}/gpt3_val-da.json --out ${PREDS_DIR}/gpt3_val-mc.json
```
