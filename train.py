from transformers import AutoTokenizer, TrainerCallback, AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
from datasets import load_dataset, DatasetDict, Features, Value, Sequence
from collections import defaultdict
from copy import deepcopy
from abc import ABC
import unicodedata
import tempfile
import logging
import random
import numpy as np
import wandb
import torch
import hydra
import math
import os
import json
import shutil
import time

from callbacks import SaveCheckpointCallback, SanityManualGreedyOnce


def log_config(arguments):
    logging.info("Logging used config:")
    logging.info("-" * 50)
    for argument, value in arguments.items():
        logging.info("{}: {}".format(argument, value))
    logging.info("-" * 50)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_process_rank():
    return _env_int("RANK", _env_int("LOCAL_RANK", 0))


def get_world_size():
    return _env_int("WORLD_SIZE", 1)


def is_main_process():
    return get_process_rank() == 0


def report_to_wandb():
    return "wandb" if is_main_process() else "none"


def _safe_metric_fragment(value):
    text = str(value or "unknown").strip().lower()
    chars = [c if c.isalnum() else "_" for c in text]
    safe = "_".join(part for part in "".join(chars).split("_") if part)
    return safe or "unknown"


def add_rule_explanation_type_eval_datasets(eval_dataset_dict):
    """Add per-explanation-type eval splits for graph rule explanations.

    These are intentionally extra eval datasets rather than a custom loss:
    Trainer will compute the same CE it reports for validation_rule_explanation,
    but separately for relationship, common_cause, retrieval, rule, etc.
    """
    rule_ds = eval_dataset_dict.get("validation_rule_explanation")
    if rule_ds is None or len(rule_ds) == 0:
        return eval_dataset_dict

    augmented = dict(eval_dataset_dict)
    explanation_types = sorted({
        row.get("explanation_type") or "unknown"
        for row in rule_ds
    })

    for explanation_type in explanation_types:
        safe_type = _safe_metric_fragment(explanation_type)
        split_name = f"validation_rule_explanation_type_{safe_type}"
        if split_name in augmented:
            continue

        type_ds = rule_ds.filter(
            lambda row, et=explanation_type: (row.get("explanation_type") or "unknown") == et
        )
        if len(type_ds) == 0:
            continue

        augmented[split_name] = type_ds
        logging.info(
            "Val dataset size (%s): %s examples",
            split_name,
            len(type_ds),
        )

    return augmented


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=repr)]
    if hasattr(value, "detach") and callable(value.detach):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist) and not isinstance(value, str):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _clean_markdown_text(value):
    if value is None:
        return ""
    return str(value).replace("``", "` `")

def wandb_meta_path(args):
    job_key = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or "single"
    )
    safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(job_key))
    return os.path.join(args.save_dir, f".wandb_meta_{safe_key}.json")


def _dtype_from_finetuning_args(args):
    if getattr(args.finetuning_pars, "bf16", False):
        return torch.bfloat16
    if getattr(args.finetuning_pars, "fp16", False):
        return torch.float16
    return "auto"


def _log_length_stats(dataset, tokenizer, name="dataset", sample=2000):
    if dataset is None or len(dataset) == 0:
        return
    cols = set(dataset.column_names)
    has_prompt = "prompt" in cols
    has_completion = "completion" in cols
    if not (has_prompt or has_completion):
        return
    n = len(dataset)
    if n > sample:
        idx = list(range(0, n, max(1, n // sample)))[:sample]
        ds_view = dataset.select(idx)
    else:
        ds_view = dataset
    prompt_lens, comp_lens, total_lens = [], [], []
    for ex in ds_view:
        p = len(tokenizer.encode(ex["prompt"], add_special_tokens=False)) if has_prompt else 0
        c = len(tokenizer.encode(ex["completion"], add_special_tokens=False)) if has_completion else 0
        if has_prompt:
            prompt_lens.append(p)
        if has_completion:
            comp_lens.append(c)
        total_lens.append(p + c)

    def _fmt(xs):
        if not xs:
            return "n/a"
        xs_sorted = sorted(xs)
        return (
            f"min={xs_sorted[0]} avg={sum(xs)/len(xs):.1f} "
            f"p50={xs_sorted[len(xs)//2]} p95={xs_sorted[int(0.95*len(xs))-1]} "
            f"max={xs_sorted[-1]}"
        )

    logging.info(f"[length stats] {name} ({len(total_lens)} sampled of {n}):")
    if prompt_lens:
        logging.info(f"  prompt:     {_fmt(prompt_lens)}")
    if comp_lens:
        logging.info(f"  completion: {_fmt(comp_lens)}")
    logging.info(f"  total:      {_fmt(total_lens)}")


def global_setup(args, wandb_run_id=None):
    """
    Rank-0: real wandb.init; writes {id, name} to wandb_meta.json + exports env.
    Workers: DO NOT init; they poll the file and set WANDB_RUN_NAME/ID locally.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    meta_path = wandb_meta_path(args)
    if not is_main_process():
        deadline = time.time() + 600
        while not os.path.exists(meta_path):
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for rank-0 W&B metadata at {meta_path}")
            time.sleep(1)
        with open(meta_path) as f:
            meta = json.load(f)
        run_id = meta["id"]
        run_name = meta["name"]
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RUN_NAME"] = run_name
        logging.info(f"Using rank-0 W&B run {run_name} ({run_id}) on rank {get_process_rank()}.")
        return run_id, run_name

    if args.resume:
        run = wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            id=wandb_run_id,
            resume="must",
            group=str(args.wandb.group),
            config=vars(args)["_content"],
            settings=wandb.Settings(start_method="thread", _service_wait=240),
        )
        run_id = run.id
        run_name = run.name
    else:
        run = wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            id=wandb_run_id,
            resume="allow",
            group=str(args.wandb.group),
            config=vars(args)["_content"],
            settings=wandb.Settings(start_method="thread", _service_wait=240),
        )
        run_id = wandb.run.id
        run_name = wandb.run.name

    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RUN_NAME"] = run_name
    tmp_meta_path = f"{meta_path}.tmp"
    with open(tmp_meta_path, "w") as f:
        json.dump({"id": run_id, "name": run_name}, f)
    os.replace(tmp_meta_path, meta_path)
                
    return run_id, run_name


def log_metadata(metadata):
    logging.info(f"--- Graph parameters ---")
    logging.info(f"Num graphs: {metadata['num_graphs']}")
    logging.info(f"Num entities: {metadata['num_entities']}")
    logging.info(f"Num worlds: {metadata['num_worlds']}")
    logging.info(f"Sample ratio from worlds: {metadata['sample_ratio_from_worlds']}")
    logging.info(f"Root prior: {metadata['root_prior']}")
    logging.info(f"Edge p: {metadata['edge_prob']}")
    logging.info(f"Max obs size: {metadata['max_obs_size']}")
    logging.info(f"Node OR p: {metadata['node_or_prob']}")
    
    logging.info(f"--- Dataset parameters ---")
    logging.info(f"Num copies explanations: {metadata['num_copies_explanations']}")
    logging.info(f"Num instruction datapoints: {metadata['add_instruction_datapoints']}")
    logging.info(f"Examples per type:")
    for key, value in metadata['kind_counts'].items():
        logging.info(f"{value} x {key}")
    logging.info(f"Examples per split")
    for key, value in metadata['split_counts'].items():
        logging.info(f"{value} x {key}")
    
    logging.info(f"--- Generalization parameters ---")
    logging.info(f"Heldout scope: {metadata['heldout_scope']}")
    num_entities_train = sum([val == "train" for key, val in metadata['explanation_split_by_entity'].items()])
    num_entities_val = sum([val == "val" for key, val in metadata['explanation_split_by_entity'].items()])
    num_entities_test = sum([val == "test" for key, val in metadata['explanation_split_by_entity'].items()])
    logging.info(f"Num explained nodes in train: {num_entities_train}")
    logging.info(f"Num explained nodes in val: {num_entities_val}")
    logging.info(f"Num explained nodes in test: {num_entities_test}")


def load_sft_dataset(train_path: str, val_path: str, test_path: str, tokenizer, seed):
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
            "train": train_path,
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
            "example_idx": idx,
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

    formatted["train"] = formatted["train"].shuffle(seed=seed)

    return formatted


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args):

    log_config(args)

    assert args.method in ["SFT"]
    
    if not args.resume:
        logging.info(f"Loading model from scratch: {args.model_pars.hf_model_id}")
        wandb_run_id, wandb_run_name = global_setup(args)
        output_dir = os.path.join(args.save_dir, f"{wandb_run_name}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "dataset_dir.txt"), "w") as outfile:
                outfile.write(f"Trained on dataset in path {args.dataset_pars.dataset_folder_path}.\n")
        if "allenai" in args.model_pars.hf_tokenizer_id:
            tokenizer = AutoTokenizer.from_pretrained(args.model_pars.hf_tokenizer_id, fix_mistral_regex=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_pars.hf_tokenizer_id)
    else:
        logging.info(f"Loading trained model from and resuming training: {args.model_pars.model_dir}")
        model_dir = args.model_pars.model_dir
        output_dir = os.path.dirname(model_dir)
        resume_step = int(model_dir.split("-")[-1])
        wandb_run_id = None
        with open(os.path.join(model_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read().strip()
        global_setup(args, wandb_run_id)
        if "allenai" in args.model_pars.model_dir:
            tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir, fix_mistral_regex=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir)

    with open(os.path.join(args.dataset_pars.dataset_folder_path, "metadata.json"), "r") as infile:
        metadata = json.load(infile)
    
    log_metadata(metadata)

    dataset_dict = load_sft_dataset(os.path.join(args.dataset_pars.dataset_folder_path, "train.jsonl"),
                                    os.path.join(args.dataset_pars.dataset_folder_path, "val.jsonl"),
                                    os.path.join(args.dataset_pars.dataset_folder_path, "test.jsonl"),
                                    tokenizer=tokenizer,
                                    seed=args.seed)

    for split_name, split_ds in dataset_dict.items():
        example = split_ds[0]
        logging.info("=" * 80)
        logging.info("Example from split: %s", split_name)
        logging.info("Prompt with chat template:\n%s", example["prompt"])
        logging.info("Completion with chat template:\n%s", example["completion"])
        logging.info("=" * 80)

    if args.method in ["SFT"]:
        # InductiveDataset / GraphDataset: split by inference type for eval
        eval_dataset_dict = {"validation": dataset_dict["validation"]}

        logging.info(f"Training dataset size: {len(dataset_dict['train'])}")
        logging.info("Training inference type distribution:")
        graph_train_counts = defaultdict(int)
        graph_train_expl_counts = defaultdict(int)
        for x in dataset_dict["train"]:
            graph_train_counts[x['primary_inference_type']] += 1
            if x['primary_inference_type'] == 'rule_explanation':
                expl_type = x.get('explanation_type') or 'unknown'
                graph_train_expl_counts[expl_type] += 1
        for inf_type, example_count in sorted(graph_train_counts.items()):
            logging.info(f"  {inf_type}: {example_count} examples")
            if inf_type == 'rule_explanation' and graph_train_expl_counts:
                for expl_type, expl_count in sorted(graph_train_expl_counts.items()):
                    logging.info(f"    {expl_type}: {expl_count} examples")

        for inf_type, ds_split in eval_dataset_dict.items():
            total_count = len(ds_split)
            if total_count == 0:
                continue
            logging.info(f"Val dataset size ({inf_type}): {total_count}")
            val_expl_counts = defaultdict(int)
            for x in ds_split:
                expl_type = x.get('explanation_type') or 'unknown'
                val_expl_counts[expl_type] += 1
            for expl_type, expl_count in sorted(val_expl_counts.items()):
                logging.info(f"  {expl_type}: {expl_count} examples")

        if is_main_process():
            _log_length_stats(dataset_dict['train'], tokenizer, name="train")
            for split_name, ds_split in eval_dataset_dict.items():
                _log_length_stats(ds_split, tokenizer, name=split_name)

        trainer_eval_dataset_dict = eval_dataset_dict
        if getattr(args.finetuning_pars, "log_rule_explanation_type_losses", True):
            trainer_eval_dataset_dict = add_rule_explanation_type_eval_datasets(
                eval_dataset_dict
            )

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch_size = args.finetuning_pars.per_device_train_batch_size * max(1, args.finetuning_pars.gradient_accumulation_steps) * world_size
        steps_per_epoch = len(dataset_dict['train']) // effective_batch_size
        save_every_n_steps = steps_per_epoch // args.finetuning_pars.save_n_per_epoch if args.finetuning_pars.save_n_per_epoch > 0 else 0
        eval_every_n_steps = steps_per_epoch // args.finetuning_pars.eval_n_per_epoch if args.finetuning_pars.eval_n_per_epoch > 0 else args.finetuning_pars.eval_every_n_steps
        logging_steps = max(steps_per_epoch // args.finetuning_pars.log_n_per_epoch, 1) if args.finetuning_pars.log_n_per_epoch > 0 else args.finetuning_pars.logging_steps
        logging.info(f"Effective batch size: {effective_batch_size}, steps per epoch: {steps_per_epoch}, saving every {save_every_n_steps} steps, evaluating every {eval_every_n_steps} steps, logging steps {logging_steps}.")
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    if args.method == "SFT":
        training_args = SFTConfig(
            output_dir=output_dir,
            report_to=report_to_wandb(),
            logging_strategy="steps",
            logging_steps=logging_steps,
            num_train_epochs=args.finetuning_pars.num_train_epochs,
            completion_only_loss=not getattr(args.finetuning_pars, 'prompt_loss', False),
            per_device_train_batch_size=args.finetuning_pars.per_device_train_batch_size,
            save_steps=save_every_n_steps,
            save_strategy="steps",
            save_total_limit=args.finetuning_pars.save_total_limit,
            eval_steps=eval_every_n_steps,
            eval_strategy="steps" if eval_every_n_steps > 0 else "no",
            gradient_accumulation_steps=args.finetuning_pars.gradient_accumulation_steps,
            learning_rate=args.finetuning_pars.reference_learning_rate,
            lr_scheduler_type=args.finetuning_pars.lr_scheduler_type,
            weight_decay=args.finetuning_pars.weight_decay,
            warmup_ratio=args.finetuning_pars.warmup_ratio,
            max_grad_norm=args.finetuning_pars.max_grad_norm,
            packing=args.finetuning_pars.packing,
            max_length=args.finetuning_pars.max_length,
            label_smoothing_factor=args.finetuning_pars.label_smoothing_factor,
            deepspeed=args.finetuning_pars.deepspeed if args.finetuning_pars.deepspeed != "" else None,
            ddp_find_unused_parameters=args.finetuning_pars.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=5,
            bf16=args.finetuning_pars.bf16,
            fp16=args.finetuning_pars.fp16,
            prediction_loss_only=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=args.finetuning_pars.gradient_checkpointing,
            dataloader_num_workers=args.finetuning_pars.dataloader_num_workers,
            dataloader_pin_memory=args.finetuning_pars.dataloader_pin_memory,
            dataloader_prefetch_factor=args.finetuning_pars.dataloader_prefetch_factor,
            dataloader_persistent_workers=args.finetuning_pars.dataloader_persistent_workers,
        )
        if not args.model_pars.model_dir:
            model = args.model_pars.hf_model_id
        else:
            model = args.model_pars.model_dir
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset_dict["train"],
            eval_dataset=trainer_eval_dataset_dict,
            args=training_args,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    trainer.add_callback(SanityManualGreedyOnce(trainer, tokenizer))
    trainer.add_callback(SaveCheckpointCallback(trainer, base_model_id=args.model_pars.hf_model_id))

    logging.info(f"Starting training on {args.model_pars.hf_model_id} using {args.method} (output dir {output_dir}).")
    if not args.resume:
        trainer.train()
    else:
        logging.info(f"Resuming training from checkpoint at {model_dir} (output dir {output_dir}).")
        trainer.train(resume_from_checkpoint=model_dir)


if __name__ == "__main__":
    main()
