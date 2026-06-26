from collections import Counter, defaultdict
from typing import Dict, List
import numpy as np
import logging
import hydra
import os

# vllm env vars must be set BEFORE importing vllm. spawn avoids fork+CUDA issues
# in vllm's v1 engine; disabling deep_gemm warmup avoids a crash on bf16 models
# when the optional `deep_gemm` package isn't installed.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, Features, Value, Sequence
from vllm import LLM, SamplingParams
import torch
import json
import math
import unicodedata

import regex
from helpers import print_example, compute_pass_at_k


_OUTPUT_PATTERN = r"""\\boxed\{
        (?P<content>
            (?:
                [^{}]
                | \{ (?:[^{}] | \{[^{}]*\})* \}
            )*
        )
    \}"""


def _get_cfg_value(cfg, key, default):
    if cfg is None:
        return default
    value = getattr(cfg, key, default)
    if isinstance(value, str) and not value.strip():
        return default
    return value


def _get_local_eval_output_dir(args, dataset_state_path: str) -> str:
    eval_output_dir = _get_cfg_value(getattr(args, "eval_pars", None), "output_dir", "")
    if eval_output_dir:
        return eval_output_dir

    model_dir = args.model_pars.model_dir
    if model_dir and os.path.exists(model_dir):
        return model_dir

    if dataset_state_path:
        dataset_state_dir = os.path.dirname(dataset_state_path)
        if dataset_state_dir:
            return dataset_state_dir

    return os.path.join("exp", args.model_pars.hf_model_id)


def _infer_vllm_max_model_len(llm_fn, tokenizer) -> int | None:
    candidates = [
        getattr(getattr(getattr(llm_fn, "llm", None), "llm_engine", None), "model_config", None),
        getattr(getattr(llm_fn, "llm", None), "model_config", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            if 0 < candidate < 10**9:
                return candidate
            continue
        max_len = getattr(candidate, "max_model_len", None)
        if isinstance(max_len, int) and 0 < max_len < 10**9:
            return max_len
        max_len = getattr(candidate, "max_position_embeddings", None)
        if isinstance(max_len, int) and 0 < max_len < 10**9:
            return max_len
    return None


def _validate_prompt_lengths(tokenizer, prompts: list[str], model_max_len: int | None, dataset_id: str) -> list[int]:
    prompt_lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    if model_max_len is not None and prompt_lengths:
        max_prompt_len = max(prompt_lengths)
        if max_prompt_len > model_max_len:
            raise ValueError(
                f"Prompt length overflow for {dataset_id}: longest prompt is "
                f"{max_prompt_len} tokens, but model max length is {model_max_len}. "
                f"For graph train-ICL, reduce num_worlds or sample_ratio_from_worlds."
            )
    return prompt_lengths


def _make_json_serializable(obj):
    """Recursively convert numpy / tensor-like values into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, set):
        return [_make_json_serializable(v) for v in sorted(obj, key=repr)]

    if hasattr(obj, "detach") and callable(obj.detach):
        return _make_json_serializable(obj.detach().cpu().tolist())
    if hasattr(obj, "tolist") and callable(obj.tolist) and not isinstance(obj, str):
        return _make_json_serializable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _make_json_serializable(obj.item())
        except (TypeError, ValueError):
            pass

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


class LLMInference:
    def __init__(self, num_datapoints, tokenizer, vocab_size: int, model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", 
                 max_tokens=1024, n=16, temperature=0.8, logging_interval=100, tensor_parallel_size=1, pipeline_parallel_size=1):
        # NOTE: forcing TRITON_ATTN avoids vllm's bundled FlashAttention-3 .so,
        # which is built against a CUDA toolkit too new for the current driver
        # (575.x) on this cluster. Triton kernels are JIT-compiled at runtime
        # against the local CUDA, so they're driver-version-tolerant.
        # max_num_seqs lowered from vllm's default of 1024 to fit within the
        # Mamba/SSM cache block budget on hybrid architectures like Qwen3.6.
        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
                       max_num_seqs=256)
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(max_tokens=max_tokens, n=n, temperature=temperature, logprobs=100)
        self._logging_counter = 0
        self.logging_interval = logging_interval
        self.num_datapoints = num_datapoints

    def __call__(self, batch: Dict[str, torch.LongTensor]) -> Dict[str, list]:
        outputs = self.llm.generate(batch["prompt"], self.sampling_params)
        # One per input in the batch
        prompt: List[str] = []
        # Multiple per input in the batch (num_samples)
        generated_text: List[List[str]] = []
        # Multiple per input in the batch (num_samples) and multiple per sample (num_timesteps), dict with target_token_id -> logprob
        token_ids: List[List[int]] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append([self.tokenizer.decode(o.token_ids) for o in output.outputs])
            token_ids.append([o.token_ids for o in output.outputs])
            all_output_target_logprobs = []
            for o in output.outputs:
                logprobs = o.logprobs

        if self._logging_counter % self.logging_interval == 0:
            logging.info(f"Working on batch {self._logging_counter}/{self.num_datapoints // len(outputs)}")
            logging.info(f"Generated texts 1: {generated_text[0]}")
        self._logging_counter += 1
        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "answer": batch["answer"],
            "example_id": batch["example_id"],
            "token_ids": token_ids,
            "query_entity": batch["query_entity"]
        }
        # Pass through optional fields if present in the batch
        for key in ["primary_inference_type", "explanation_type"]:
            if key in batch:
                result[key] = batch[key]
        return result


def _canonicalize_explanation_prediction(pred: str) -> str:
    norm = " ".join(unicodedata.normalize("NFKC", pred.strip()).split())
    compact = norm.lower().strip()
    mc_match = regex.match(r"^(?:option\s*)?\(?(?P<label>[abcde])\)?[).:]?$", compact)
    if mc_match is not None:
        return mc_match.group("label")
    compact = compact.rstrip(". )")
    if compact in {"yes", "no"}:
        return compact
    independent_match = regex.match(
        r"^(?P<a>\S+)\s+and\s+(?P<b>\S+)\s+are independent$",
        compact,
    )
    if independent_match is not None:
        a, b = sorted((independent_match.group("a"), independent_match.group("b")))
        return f"{a} and {b} are independent"
    no_relation_match = regex.match(
        r"^there is no relation between\s+(?P<a>\S+)\s+and\s+(?P<b>\S+)$",
        compact,
    )
    if no_relation_match is not None:
        a, b = sorted((no_relation_match.group("a"), no_relation_match.group("b")))
        return f"{a} and {b} are independent"
    return norm


def extract_output(completion: str):
    """Return last \\boxed{} content in completion."""
    last = None
    for m in regex.finditer(_OUTPUT_PATTERN, completion,
                            flags=regex.DOTALL | regex.VERBOSE):
        last = m.group('content')
    return last


def eval_outputs(outputs, pass_at_k: int):
    """Accuracy + probability calibration metrics."""

    accuracy = 0
    pass_at_k_scores = {k: 0 for k in range(1, pass_at_k + 1)}
    all_predictions = []

    for out in outputs:
        texts = out["generated_text"]
        token_ids = out["token_ids"]
        example_id = out["example_id"]
        ground_truth = out["answer"]
        is_explanation = True
        canonical_ground_truth = _canonicalize_explanation_prediction(ground_truth)

        preds = []
        canonicalized_preds = []
        num_correct = 0

        for text, tok_ids in zip(
            texts, token_ids
        ):
            pred = extract_output(text)
            preds.append(pred)
            canonicalized_pred = _canonicalize_explanation_prediction(pred)
            canonicalized_preds.append(canonicalized_pred)
            if (
                pred is not None
                and canonicalized_pred == canonical_ground_truth
            ):
                num_correct += 1

        example_accuracy = num_correct / len(texts)
        accuracy += example_accuracy

        current_pass_at_k = {}
        for k in range(1, pass_at_k + 1):
            p_at_k = compute_pass_at_k(len(texts), num_correct, k)
            pass_at_k_scores[k] += p_at_k
            current_pass_at_k[k] = p_at_k

        pred_entry = {
            "example_id": example_id,
            "answer": ground_truth,
            "canonicalized_ground_truth": canonical_ground_truth,
            "accuracy": example_accuracy,
            "pass_at_k": current_pass_at_k,
            "predictions": preds,
            "canonicalized_predictions": canonicalized_preds,
            "prompt": out["prompt"],
            "texts": texts,
            "primary_inference_type": out.get("primary_inference_type"),
            "inference_types": out.get("inference_types"),
            "explanation_type": out.get("explanation_type"),
            "query_entity": out.get("query_entity"),
        }
        all_predictions.append(pred_entry)

    n = len(outputs)
    accuracy /= n
    for k in pass_at_k_scores:
        pass_at_k_scores[k] /= n

    extra_metrics = {}

    # Per-type accuracy
    type_correct = defaultdict(list)
    for pred in all_predictions:
        ptype = pred.get("primary_inference_type", "unknown")
        type_correct[ptype].append(pred["accuracy"])
    for ptype, accs in sorted(type_correct.items()):
        extra_metrics[f"acc_{ptype}"] = float(np.mean(accs))

    return (
        accuracy,
        pass_at_k_scores,
        all_predictions,
        extra_metrics,
    )



def load_sft_dataset(val_path: str, test_path: str, tokenizer):
    features = Features({
        "answer": Value("string"),
        "completion": Value("string"),
        "explanation_type": Value("string"),
        "graph_idx": Value("int64"),
        "inference_types": Sequence(Value("string")),
        "kind": Value("string"),
        "observed_entities": Sequence(Value("string")),
        "observed_values": Sequence(Value("string")),
        "primary_inference_type": Value("string"),
        "prompt": Value("string"),
        "query_entity": Value("string"),
        "split": Value("string"),
    })

    ds = load_dataset(
        "json",
        data_files={
            "validation": val_path,
            "test": test_path,
        },
        features=features,
    )

    def to_chat_text(example, idx):
        raw_prompt = example["prompt"]
        raw_completion = example["completion"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": raw_prompt},
                {"role": "assistant", "content": raw_completion},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        completion_text = full_text[len(prompt_text):]

        return {
            "example_id": idx,
            "raw_prompt": raw_prompt,
            "raw_completion": raw_completion,
            "prompt": prompt_text,
            "completion": completion_text,
            "text": full_text,
        }

    formatted = DatasetDict({
        split: ds[split].map(
            to_chat_text,
            with_indices=True
        )
        for split in ds
    })

    return formatted


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args):

    if "allenai" in args.model_pars.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir, fix_mistral_regex=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir, fix_mistral_regex=True)
    dataset_dict = load_sft_dataset(os.path.join(args.dataset_pars.dataset_folder_path, "val.jsonl"),
                                    os.path.join(args.dataset_pars.dataset_folder_path, "test.jsonl"),
                                    tokenizer=tokenizer)

    for split_name, split_ds in dataset_dict.items():
        example = split_ds[0]
        logging.info("=" * 80)
        logging.info("Example from split: %s", split_name)
        logging.info("Prompt with chat template:\n%s", example["prompt"])
        logging.info("Completion with chat template:\n%s", example["completion"])
        logging.info("=" * 80)
    local_output_dir = _get_local_eval_output_dir(args, args.dataset_pars.dataset_folder_path)
    logging.info("Resolved dataset_state_path to %s", args.dataset_pars.dataset_folder_path)
    logging.info("Writing evaluation artifacts to %s", local_output_dir)

    if args.method in ["eval"]:
        datasets_dict = {}
        for inf_type, ds_split in dataset_dict.items():
            datasets_dict[f"{inf_type}"] = ds_split

        for dataset_id, dataset in datasets_dict.items():
            logging.info(f"Evaluation dataset size ({dataset_id}): {len(dataset)}")
            if len(dataset) > 0:
                print_example(dataset)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.method == "eval":
        # Create LLMInference once and reuse across all splits to avoid GPU OOM
        # (vLLM does not release GPU memory when an LLM instance is replaced in a loop)
        llm_fn = LLMInference(
            num_datapoints=1, tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            model_name_or_path=args.model_pars.model_dir,
            max_tokens=args.eval_pars.max_tokens, n=args.eval_pars.num_samples,
            temperature=args.eval_pars.temperature,
            logging_interval=args.eval_pars.logging_interval,
            tensor_parallel_size=args.eval_pars.tensor_parallel_size,
            pipeline_parallel_size=args.eval_pars.pipeline_parallel_size,
        )
        model_max_len = _infer_vllm_max_model_len(llm_fn, tokenizer)
        if model_max_len is not None:
            logging.info("Resolved model max context length: %s tokens", model_max_len)
        for dataset_id, dataset in datasets_dict.items():
            if not len(dataset):
                logging.info(f"Skipping dataset {dataset_id} as it is empty.")
                continue
            logging.info(f"Working on dataset {dataset_id}")
            # num_datapoints = len(dataset)
            logging.info(f"Evaluating on max {args.eval_pars.max_to_eval} datapoints.")
            num_datapoints = min(len(dataset), args.eval_pars.max_to_eval)
            is_explanation_split = "rule_explanation" in dataset_id
            # Do inference
            llm_fn.num_datapoints = num_datapoints
            df = dataset.to_pandas()
            if num_datapoints < len(df):
                # Sample deterministically to decide *which* rows to keep (so
                # the cap doesn't systematically clip the first N rows when the
                # data is in group/pool/aug order), then sort the kept rows
                # back into example_id order so the output JSON stays grouped
                # by (group, pool) and is easy to inspect.
                df = df.sample(n=num_datapoints, random_state=args.seed)
                if "example_id" in df.columns:
                    df = df.sort_values("example_id")
                df = df.reset_index(drop=True)
            else:
                df = df[:num_datapoints]
            outputs = []
            for start in range(0, len(df), args.eval_pars.batch_size):
                batch_df = df.iloc[start:start + args.eval_pars.batch_size]
                batch = {col: batch_df[col].tolist() for col in batch_df.columns}
                prompt_lengths = _validate_prompt_lengths(
                    tokenizer=tokenizer,
                    prompts=batch["prompt"],
                    model_max_len=model_max_len,
                    dataset_id=dataset_id,
                )
                if prompt_lengths:
                    logging.info(
                        "Prompt lengths for %s batch starting at %s: min=%s, max=%s, mean=%.1f",
                        dataset_id,
                        start,
                        min(prompt_lengths),
                        max(prompt_lengths),
                        sum(prompt_lengths) / len(prompt_lengths),
                    )
                result = llm_fn(batch)
                n = len(result["prompt"])
                outputs.extend([{k: v[i] for k, v in result.items()} for i in range(n)])
            output_path = os.path.join(local_output_dir, f"{dataset_id}_generated_outputs.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    _make_json_serializable(
                        outputs
                    ),
                    f,
                    indent=4,
                )
            accuracy, pass_at_k, all_predictions, extra_metrics = eval_outputs(outputs, pass_at_k=args.eval_pars.pass_at_k)
            logging.info(f"Results on dataset {dataset_id}:")
            results_path = os.path.join(local_output_dir, f"{dataset_id}_results.json")
            results = {
                "accuracy": accuracy,
                "pass_at_k": pass_at_k,
                **extra_metrics,
                "all_predictions": all_predictions,
            }
            with open(results_path, "w") as f:
                json.dump(_make_json_serializable(results), f, indent=4)
            logging.info(f"Saved results to {results_path}.")
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
