import os
import sys
import json
import argparse
import pathlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import sentencepiece; import pytorch_lightning as pl

import torchmetrics.functional as MF

from load_aokvqa import load_aokvqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--vocab', type=argparse.FileType('r'), required=True)
    parser.add_argument('--log-dir', type=pathlib.Path, dest='log_dir', required=True)
    #
    parser.add_argument('--backbone', type=str, choices=['clip', 'resnet', 'bert'], required=True)
    parser.add_argument('--clip-model-type', type=str,
                        choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                        dest='clip_model_type', required=('clip' in sys.argv))
    parser.add_argument('--train-features', type=pathlib.Path, required=True, dest='train_features')
    parser.add_argument('--val-features', type=pathlib.Path, required=True, dest='val_features')
    parser.add_argument('--vocab-features', type=pathlib.Path, required=('contrastive' in sys.argv), dest='vocab_features')
    #
    parser.add_argument('--objective', type=str, choices=['classifier', 'contrastive'], required=True)
    parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], required=True)
    # Defaults
    parser.add_argument('--bs', type=int, default=128, dest='batch_size')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    pl.seed_everything(1)
    vocab = args.vocab.read().splitlines()

    ## Data loading

    dm = AokvqaEmbeddingsDataModule(
        args.aokvqa_dir,
        args.train_features,
        args.val_features,
        args.objective,
        args.backbone,
        args.inputs,
        vocab,
        args.vocab_features,
        batch_size=args.batch_size,
        num_workers=16
    )

    ## Model definition

    model = LinearClassifier(
        args.objective,
        args.backbone,
        args.clip_model_type,
        args.inputs,
        len(vocab),
        args.lr
    )

    ## Training and testing loops

    logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=f'{args.backbone}-{args.objective}',
        version=f"inputs:{'+'.join(args.inputs)}"
    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                filename="{epoch:02d}-{val_acc:.2f}",
                mode="max"
            )
        ],
    )

    trainer.fit(model, dm)


class AokvqaEmbeddingsDataset(Dataset):
    def __init__(self, aokvqa_dir, split, input_features, objective, backbone, inputs, vocab, vocab_features):

        aokvqa_set = load_aokvqa(aokvqa_dir, split)

        assert ( backbone == 'resnet' and inputs == ['image'] and objective == 'classifier' ) \
            or ( backbone == 'bert' and inputs == ['question'] and objective == 'classifier' ) \
            or ( backbone == 'clip' )

        embeddings = torch.load(input_features)
        if backbone == 'clip':
            for q in embeddings.keys():
                embeddings[q]['question'] /= embeddings[q]['question'].norm(dim=-1, keepdim=True)
                embeddings[q]['image'] /= embeddings[q]['image'].norm(dim=-1, keepdim=True)
            if objective == 'contrastive':
                vocab_embeddings = torch.load(vocab_features)
                vocab_embeddings /= vocab_embeddings.norm(dim=-1, keepdim=True)

        self.objective = objective
        self.vocab_len = len(vocab)

        self.embeddings = []
        self.answers = []

        for o in aokvqa_set:
            correct_answers = set([o['choices'][o['correct_choice_idx']]] + o['direct_answers'])
            correct_answers = [vocab.index(a) for a in correct_answers if a in vocab]
            if self.objective == 'contrastive':
                correct_answers = [vocab_embeddings[a] for a in correct_answers]
            if len(correct_answers) == 0: continue
            self.answers.append(correct_answers)

            q = o['question_id']
            if 'question' in inputs and 'image' in inputs:
                e = torch.cat((embeddings[q]['question'], embeddings[q]['image']))
            elif 'question' in inputs and 'image' not in inputs:
                e = embeddings[q]['question']
            elif 'question' not in inputs and 'image' in inputs:
                e = embeddings[q]['image']
            self.embeddings.append(e)

    def __getitem__(self, index):
        e = self.embeddings[index]
        a = self.answers[index]
        if self.objective == 'classifier':
            a = torch.sum(F.one_hot(torch.tensor(a), num_classes=self.vocab_len), dim=0)
        elif self.objective == 'contrastive':
            a = random.sample(a, 1)[0]
        return e, a

    def __len__(self):
        return len(self.embeddings)


class AokvqaEmbeddingsDataModule(pl.LightningDataModule):

    def __init__(self, aokvqa_dir, train_features, val_features, objective, backbone, inputs, vocab, vocab_features, batch_size=1, num_workers=0):
        super().__init__()
        self.aokvqa_dir = aokvqa_dir
        self.train_features = train_features
        self.val_features = val_features
        self.objective = objective
        self.backbone = backbone
        self.inputs = inputs
        self.vocab = vocab
        self.vocab_features = vocab_features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'train', self.train_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )
        self.val_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'val', self.val_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=int(0.8 * self.num_workers)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=int(0.2 * self.num_workers)
        )


class LinearClassifier(pl.LightningModule):
    def __init__(self, objective, backbone, clip_model_type, inputs, vocab_len, lr=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr

        if self.hparams.backbone == 'clip':
            clip_dim = {
                'RN50' : 1024,
                'RN50x4' : 640,
                'RN50x16' : 768,
                'RN50x64' : 1024,
                'RN101' : 512,
                'ViT-B/32' : 512,
                'ViT-B/16' : 512,
                'ViT-L/14' : 768,
                'ViT-L/14@336px' : 768,
            }[clip_model_type]
            emb_dim = clip_dim * len(inputs)
        elif self.hparams.backbone == 'resnet':
            emb_dim = 2048
        elif self.hparams.backbone == 'bert':
            emb_dim = 768

        if self.hparams.objective == 'classifier':
            out_dim = vocab_len
        elif self.hparams.objective == 'contrastive':
            out_dim = clip_dim

        self.linear = nn.Linear(emb_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        if self.hparams.objective == 'classifier':
            x = torch.sigmoid(x)
        return x

    def compute_loss(self, batch):
        x, y = batch

        y_pred = self.forward(x)

        if self.hparams.objective == 'classifier':
            loss = F.binary_cross_entropy(y_pred, y.float())
        elif self.hparams.objective == 'contrastive':
            indices = torch.arange(0, x.shape[0], dtype=torch.int64, device=self.device)
            sim = (y_pred @ y.T).softmax(dim=-1)
            loss = F.cross_entropy(sim, indices)

        if self.hparams.objective == 'classifier':
            acc = MF.f1_score(y_pred, y)
        elif self.hparams.objective == 'contrastive':
            acc = torch.mean(sim[indices, indices])

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    main()
