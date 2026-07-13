# coding: utf-8
"""
Data module
"""
import glob
import os
from typing import Tuple

import torch
from torchtext import data
from torchtext.data import Field, RawField


_FEATURE_CACHE_DEVICE = None


def feature_cache_device():
    global _FEATURE_CACHE_DEVICE
    if _FEATURE_CACHE_DEVICE is None:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            _FEATURE_CACHE_DEVICE = torch.device(f"cuda:{device_index}")
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
            device_name = torch.cuda.get_device_name(device_index)
            print(
                "[Data] Feature cache device: "
                f"{_FEATURE_CACHE_DEVICE} ({device_name}); "
                f"CUDA_VISIBLE_DEVICES={visible_devices}",
                flush=True,
            )
        else:
            _FEATURE_CACHE_DEVICE = torch.device("cpu")
            print("[Data] Feature cache device: cpu (CUDA unavailable)", flush=True)
    return _FEATURE_CACHE_DEVICE


def load_split_dir(dirname):
    if not os.path.isdir(dirname):
        raise ValueError(
            "Expected a directory containing per-video .pt files. "
            "Legacy .train/.dev/.test split files are no longer supported: "
            f"{dirname}"
        )

    sample_files = sorted(glob.glob(os.path.join(dirname, "*.pt")))
    if not sample_files:
        raise ValueError(f"No .pt samples found in dataset directory: {dirname}")

    samples = []
    cache_device = feature_cache_device()
    for sample_file in sample_files:
        sample = torch.load(sample_file, map_location=cache_device)
        missing = {"name", "text"} - set(sample)
        if missing:
            raise ValueError(f"{sample_file} is missing required fields: {missing}")
        samples.append(sample)
    return samples


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("txt", fields[3]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for split_dir in path:
            tmp = load_split_dir(split_dir)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)


class SignTranslationDataset3D(data.Dataset):
    """Defines a dataset for machine translation based on 3D keypoint data."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, Field, Field],
        **kwargs
    ):

        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("name", fields[0]),   # Name of the sample
                ("sgn", fields[1]),    # Keypoint data
                ("txt", fields[2]),    # Text
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for split_dir in path:
            tmp = load_split_dir(split_dir)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["text"] == s["text"]
                    # Concatenate keypoint data along the num_frames axis (0)
                    samples[seq_id]["keypoint"] = torch.cat(
                        [samples[seq_id]["keypoint"], s["keypoint"]], axis=0
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "text": s["text"],
                        "keypoint": s["keypoint"],  # Store 3D keypoint data
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        # Adding small value for numerical stability
                        sample["keypoint"] + 1e-8,  # Keypoint data (num_frames, 133, 3)
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
