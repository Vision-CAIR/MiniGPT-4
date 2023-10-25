## Transfer Learning Experiments

We use the following training/prediction scripts for the classifier, zero-shot, and contrastive experiments in Table 3.

```bash
## Training
python transfer_experiments/train.py --aokvqa-dir ${AOKVQA_DIR} --vocab ${AOKVQA_DIR}/large_vocab_train.csv --log-dir ${LOG_DIR}

--backbone clip --clip-model-type ViT-B/32 --train-features ${FEATURES_DIR}/clip-ViT-B-32_train.pt --val-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt
--inputs question # OR --inputs image  # OR --inputs question image
# OR
--backbone resnet --train-features ${FEATURES_DIR}/resnet_train.pt --val-features ${FEATURES_DIR}/resnet_val.pt --inputs image
# OR
--backbone bert --train-features ${FEATURES_DIR}/bert_train.pt --val-features ${FEATURES_DIR}/bert_val.pt --inputs question

--objective classifier
# OR
--objective contrastive --vocab-features ${FEATURE_DIR}/clip-ViT-B-32_large_vocab.pt
```

You can make predictions for CLIP zero-shot or from a classifier/contrastive checkpoint trained above.

```bash
## Predicting
python transfer_experiments/predict.py --aokvqa-dir ${AOKVQA_DIR} --out ${PREDS_DIR}/clip-classifier_val-mc.json

--split val  # or test
--features ${FEATURE_DIR}/clip-ViT-B-32_val.pt  # adjust for backbone and eval split

--ckpt path/to/model.ckpt
# OR
--zero-shot --clip-model-type ViT-B/32
--inputs question  # OR --inputs image  # OR --inputs question image

--mc  # Multiple-choice. Exclude for direct-answer.

# IF classifier OR direct-answer
--vocab ${AOKVQA_DIR}/large_vocab_train.csv
# IF contrastive/zero-shot AND direct-answer
--vocab-features ${FEATURES_DIR}/clip-ViT-B-32_large_vocab.pt
```
