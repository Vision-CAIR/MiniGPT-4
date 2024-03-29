"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch.distributed as dist
from collections import defaultdict
from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
import minigpt4.common.dist_utils as dist_utils
from minigpt4.common.logger import MetricLogger
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.common.dist_utils import is_dist_avail_and_initialized
from minigpt4.common.vqa_tools.vqa import VQA
from minigpt4.common.vqa_tools.vqa_eval import VQAEval
from minigpt4.common.caption_tools.caption_utils import coco_caption_eval, textcaps_caption_eval


@registry.register_task("instruction_tuning")
class InstructionTask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = defaultdict(dict)
        self.anno_files = defaultdict(dict)

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 30)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )
    
    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                source = dataset[split].source
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split][source] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split][source] = dataset[split].coco_fmt_anno_file

                # try:
                #     self.answer_list = dataset[split].answer_list
                # except AttributeError:
                #     # if answer_list is not provided, then set it to None
                #     pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
    
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        text_inputs = samples["text_input"]

        sources = samples["source"]
        source = samples["source"][0]

        if source in ['vqav2','okvqa','gqa']:
            sample_ids = [int(sample_id.item()) for sample_id in samples["question_id"]]
        elif source in ['aokvqa']:
            sample_ids = [sample_id for sample_id in samples["question_id"]]
        elif source in ['coco_cap', 'text_cap', 'text_vqa']:
            sample_ids = samples["image_id"]

        # For GQA
        full_answers = samples.get("fullAnswer", ["" for i in range(len(sample_ids))])

        # For AOKVQA & GQA & TextVQA
        gt_answers = samples.get("gt_answers", ["" for i in range(len(sample_ids))])

        # For AOKVQA
        choices = samples.get("choices", ["" for i in range(len(sample_ids))])

        for answer, sample_id, text_input, full_answer, gt_answer, choice, source in zip(answers, sample_ids, text_inputs, full_answers, gt_answers, choices, sources):
            pred_qa_pairs.append({
                "question_id": sample_id,
                "question": text_input,
                "full_answer": full_answer,
                "answer": answer,
                "gt_ans": gt_answer,
                "choice": choice,
                "source": source})
        return pred_qa_pairs

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        total_results = list()
        for sub_data_loader in  data_loader.loaders:
            results = []
            for samples in metric_logger.log_every(sub_data_loader, print_freq, header):

                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)

                results.extend(eval_output)

            total_results.append(results)
        
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        return total_results


    def after_evaluation(self, val_result, split_name, **kwargs):

        final_metrics = dict()
        for i in range(len(val_result)):
            source = val_result[i][0]["source"]
            result_file = self.save_result(
                val_result[i],
                result_dir=registry.get_path("result_dir"),
                filename=f"{split_name}_vqa_result_{source}",
                remove_duplicate="question_id",
            )

            if source in ['vqav2','okvqa']:
                try:
                    metrics = self._report_metrics_coco_vqa(result_file=result_file, split=split_name, source=source)
                except Exception as e:
                    metrics = None
                    print(f"Report Metrics {source} Error: {e}")
            elif source in ['gqa','aokvqa','text_vqa']:
                try:
                    metrics = self._report_metrics_gqa_aokvqa_textvqa(result_file=result_file, source=source)
                except Exception as e:
                    metrics = None
                    print(f"Report Metrics {source} Error: {e}")
            elif source in ['coco_cap','text_cap']:
                try:
                    metrics = self._report_metrics_caption(result_file=result_file, split_name=split_name, source=source)
                except Exception as e:
                    metrics = None
                    print(f"Report Metrics {source} Error: {e}")
            else:
                metrics = None
            final_metrics[source] = metrics

        try:
            agg_metrics_lst = [v["agg_metrics"] for k,v in final_metrics.items()]
            final_metrics["agg_metrics"] = sum(agg_metrics_lst)/len(agg_metrics_lst)
        except Exception as e:
            print("Calculate agg metrics error... ", e)
            final_metrics = None
            
        return final_metrics

    @dist_utils.main_process
    def _report_metrics_coco_vqa(self, result_file, split, source='vqav2'):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split][source], self.ques_files[split][source])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split][source]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), f"evaluate_{source}.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics


    @dist_utils.main_process
    def _report_metrics_gqa_aokvqa_textvqa(self, result_file, source='gqa'):
        """
        Validation of GQA & aokvqa
        source = 'gqa' / 'aokvqa'
        """
        # measuring accuracy compared to answer
        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:

            gt_ans = res["gt_ans"]
            pred = res["answer"]

            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            # vqa_acc = 1 if pred == gt_ans else 0
            vqa_acc = 1 if gt_ans in pred else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), f"evaluate_{source}.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
 

    @dist_utils.main_process
    def _report_metrics_caption(self, result_file, split_name, source='coco_cap'):
        """
        Use official COCO Cap evaluation script to report metrics.
        """
        if source == 'coco_cap':
            coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
            eval = coco_caption_eval(coco_gt_root, result_file, split_name)
        elif source == 'text_cap':
            annotaion_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/TextCap/TextCaps_0.1_val.json"
            eval = textcaps_caption_eval(annotaion_file, result_file)

        agg_metrics = eval.eval["CIDEr"] + eval.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in eval.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), f"evaluate_{source}.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        result = {k: v for k, v in eval.eval.items()}
        result["agg_metrics"] = agg_metrics

        return result
