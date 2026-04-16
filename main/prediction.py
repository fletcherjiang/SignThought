#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List, Dict
from torchtext.data import Dataset
from main.loss import XentLoss
from main.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from main.metrics import bleu, chrf, rouge, wer_list
from main.model import build_model, SignModel
from main.batch import Batch
from main.data import load_data, make_data_iter
from main.vocabulary import PAD_TOKEN
from main.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data: Dataset,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    txt_pad_index: int,
    translation_beam_size: int = 7,
    translation_beam_alpha: int = 4,
    batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
    frame_subsampling_ratio: int = None,
) -> Dict:
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param translation_max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param translation_loss_function: translation loss function (XEntropy)
    :param translation_loss_weight: Translation loss weight
    :param txt_pad_index: txt padding token index
    :param sgn_dim: Feature dimension of sgn frames
    :param translation_beam_size: beam size for validation (translation).
        If 0 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty (translation),
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param dataset_version: phoenix_2014 or phoenix_2014_trans
    :param frame_subsampling_ratio: frame subsampling ratio

    :return: dictionary with validation results
    """
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_txt_outputs = []
        all_attention_scores = []
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_seqs = 0
        for valid_batch in iter(valid_iter):
            batch = Batch(
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()

            batch_translation_loss = model.get_loss_for_batch(
                batch=batch,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=translation_loss_weight,
            )
            total_translation_loss += batch_translation_loss
            total_num_txt_tokens += batch.num_txt_tokens
            total_num_seqs += batch.num_seqs

            (
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
                translation_beam_size=translation_beam_size,
                translation_beam_alpha=translation_beam_alpha,
                translation_max_output_length=translation_max_output_length,
            )

            # sort outputs back to original order
            all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
            all_attention_scores.extend(
                batch_attention_scores[sort_reverse_index]
                if batch_attention_scores is not None
                else []
            )

        assert len(all_txt_outputs) == len(data)
        if (
            translation_loss_function is not None
            and translation_loss_weight != 0
            and total_num_txt_tokens > 0
        ):
            # total validation translation loss
            valid_translation_loss = total_translation_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_translation_loss / total_num_txt_tokens)
        else:
            valid_translation_loss = -1
            valid_ppl = -1
        # decode back to symbols
        decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        # Construct text sequences for metrics
        txt_ref = [join_char.join(t) for t in data.txt]
        txt_hyp = [join_char.join(t) for t in decoded_txt]
        # post-process
        if level == "bpe":
            txt_ref = [bpe_postprocess(v) for v in txt_ref]
            txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
        assert len(txt_ref) == len(txt_hyp)

        # TXT Metrics
        txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
        txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
        txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores = {}

        valid_scores["bleu"] = txt_bleu["bleu4"]
        valid_scores["bleu_scores"] = txt_bleu
        valid_scores["chrf"] = txt_chrf
        valid_scores["rouge"] = txt_rouge

    results = {
        "valid_scores": valid_scores,
        "all_attention_scores": all_attention_scores,
    }

    results["valid_translation_loss"] = valid_translation_loss
    results["valid_ppl"] = valid_ppl
    results["decoded_txt"] = decoded_txt
    results["txt_ref"] = txt_ref
    results["txt_hyp"] = txt_hyp

    return results


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    _, dev_data, test_data, txt_vocab = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    multimodal = cfg["data"].get("multimodal", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        multimodal=multimodal,
        thinking_cfg=cfg.get("thinking", {}),
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]




    translation_loss_function = XentLoss(
        pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
    )
    if use_cuda:
        translation_loss_function.cuda()

    


    logger.info("=" * 60)
    dev_translation_results = {}
    dev_best_bleu_score = float("-inf")
    dev_best_translation_beam_size = 1
    dev_best_translation_alpha = 1
    for tbw in translation_beam_sizes:
        dev_translation_results[tbw] = {}
        for ta in translation_beam_alphas:
            dev_translation_results[tbw][ta] = validate_on_data(
                model=model,
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                level=level,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                batch_type=batch_type,
                dataset_version=dataset_version,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=1,
                translation_max_output_length=translation_max_output_length,
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                translation_beam_size=tbw,
                translation_beam_alpha=ta,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )

            if (
                dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                > dev_best_bleu_score
            ):
                dev_best_bleu_score = dev_translation_results[tbw][ta][
                    "valid_scores"
                ]["bleu"]
                dev_best_translation_beam_size = tbw
                dev_best_translation_alpha = ta
                dev_best_translation_result = dev_translation_results[tbw][ta]
                logger.info(
                    "[DEV] partition [Translation] results:\n\t"
                    "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                    "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                    "CHRF %.2f\t"
                    "ROUGE %.2f",
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    dev_best_translation_result["valid_scores"]["bleu"],
                    dev_best_translation_result["valid_scores"]["bleu_scores"][
                        "bleu1"
                    ],
                    dev_best_translation_result["valid_scores"]["bleu_scores"][
                        "bleu2"
                    ],
                    dev_best_translation_result["valid_scores"]["bleu_scores"][
                        "bleu3"
                    ],
                    dev_best_translation_result["valid_scores"]["bleu_scores"][
                        "bleu4"
                    ],
                    dev_best_translation_result["valid_scores"]["chrf"],
                    dev_best_translation_result["valid_scores"]["rouge"],
                )
                logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Translation] results:\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_translation_beam_size,
        dev_best_translation_alpha,
        dev_best_translation_result["valid_scores"]["bleu"],
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"],
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"],
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"],
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"],
        dev_best_translation_result["valid_scores"]["chrf"],
        dev_best_translation_result["valid_scores"]["rouge"],
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        translation_loss_function=translation_loss_function,
        translation_loss_weight=1,
        translation_max_output_length=translation_max_output_length,
        level=level,
        translation_beam_size=dev_best_translation_beam_size,
        translation_beam_alpha=dev_best_translation_alpha,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Translation] results:\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "BLEU %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_translation_beam_size,
        dev_best_translation_alpha,
        test_best_result["valid_scores"]["bleu"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"],
        test_best_result["valid_scores"]["chrf"],
        test_best_result["valid_scores"]["rouge"],
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:

        if dev_best_translation_beam_size > -1:
            dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                output_path,
                dev_best_translation_beam_size,
                dev_best_translation_alpha,
                "dev",
            )
            test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                output_path,
                dev_best_translation_beam_size,
                dev_best_translation_alpha,
                "test",
            )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "translation_results": dev_translation_results,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)
