#!/usr/bin/env python3
"""Interactive prompting for one or two vLLM-backed checkpoints."""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

# vLLM env vars must be set before importing vLLM. These mirror evaluate.py.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")


MODEL_DIR_MARKERS = (
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
)


@dataclass
class LoadedModel:
    label: str
    path: Path
    tokenizer: object
    llm: object
    raw_prompt: bool
    system_prompt: Optional[str]

    def format_prompt(self, user_prompt: str) -> str:
        if self.raw_prompt:
            return user_prompt

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            print(
                f"[{self.label}] Could not apply chat template ({exc}); "
                "using the raw prompt.",
                file=sys.stderr,
            )
            return user_prompt

    def complete(
        self,
        user_prompt: str,
        sampling_params: object,
    ) -> list[str]:
        formatted_prompt = self.format_prompt(user_prompt)
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        return [completion.text for completion in outputs[0].outputs]


def _path_from_user(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in MODEL_DIR_MARKERS)


def _checkpoint_sort_key(path: Path) -> Tuple[int, Union[int, str]]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def _discover_checkpoint_dirs(checkpoint_dir: Path) -> list[Path]:
    if _looks_like_model_dir(checkpoint_dir):
        return [checkpoint_dir]

    checkpoint_dirs = sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ],
        key=_checkpoint_sort_key,
    )
    if checkpoint_dirs:
        return checkpoint_dirs

    return sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and _looks_like_model_dir(path)
        ],
        key=lambda path: path.name,
    )


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    supplied = list(args.checkpoints)
    supplied.extend(args.checkpoint or [])

    checkpoint_dir = _path_from_user(args.checkpoint_dir) if args.checkpoint_dir else None

    if not supplied:
        if checkpoint_dir is None:
            raise ValueError(
                "Pass one or two checkpoint paths, or pass --checkpoint-dir."
            )
        resolved = _discover_checkpoint_dirs(checkpoint_dir)
        if len(resolved) > 2:
            names = ", ".join(path.name for path in resolved[:5])
            raise ValueError(
                f"Found {len(resolved)} checkpoints under {checkpoint_dir}. "
                f"Pass the one or two you want with -c/--checkpoint. "
                f"First matches: {names}"
            )
    else:
        resolved = []
        for value in supplied:
            path = _path_from_user(value)
            if checkpoint_dir is not None and not path.is_absolute():
                path = checkpoint_dir / path
            resolved.append(path)

    if not 1 <= len(resolved) <= 2:
        raise ValueError(f"Expected one or two checkpoints, got {len(resolved)}.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint path(s): " + ", ".join(missing))

    not_dirs = [str(path) for path in resolved if not path.is_dir()]
    if not_dirs:
        raise NotADirectoryError("Checkpoint path(s) must be directories: " + ", ".join(not_dirs))

    return resolved


def build_sampling_params(args: argparse.Namespace) -> object:
    from vllm import SamplingParams

    kwargs = {
        "max_tokens": args.max_tokens,
        "n": args.num_completions,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.stop:
        kwargs["stop"] = args.stop
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return SamplingParams(**kwargs)


def load_models(args: argparse.Namespace, checkpoint_paths: Sequence[Path]) -> list[LoadedModel]:
    from transformers import AutoTokenizer
    from vllm import LLM

    labels = args.label or []
    if labels and len(labels) != len(checkpoint_paths):
        raise ValueError("Pass exactly one --label per checkpoint, or omit labels.")

    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None and len(checkpoint_paths) == 2:
        gpu_memory_utilization = 0.45
        print(
            "Loading two models with gpu_memory_utilization=0.45 each. "
            "Override with --gpu-memory-utilization if needed."
        )

    models = []
    for index, checkpoint_path in enumerate(checkpoint_paths):
        label = labels[index] if labels else checkpoint_path.name
        tokenizer_path = _path_from_user(args.tokenizer) if args.tokenizer else checkpoint_path
        print(f"Loading {label}: {checkpoint_path}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            fix_mistral_regex=True,
            trust_remote_code=args.trust_remote_code,
        )

        llm_kwargs = {
            "model": str(checkpoint_path),
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "max_num_seqs": args.max_num_seqs,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

        llm = LLM(**llm_kwargs)
        models.append(
            LoadedModel(
                label=label,
                path=checkpoint_path,
                tokenizer=tokenizer,
                llm=llm,
                raw_prompt=args.raw_prompt,
                system_prompt=args.system_prompt,
            )
        )
    return models


def read_multiline_prompt() -> str:
    print("Enter prompt. Finish with a line containing only EOF.")
    lines = []
    while True:
        line = input()
        if line == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)


def print_completion(label: str, completions: Sequence[str]) -> None:
    print(f"\n===== {label} =====")
    for index, completion in enumerate(completions, start=1):
        if len(completions) > 1:
            print(f"--- completion {index} ---")
        print(completion.strip())
    print()


def run_interactive_loop(
    models: Sequence[LoadedModel],
    sampling_params: object,
) -> None:
    print("\nInteractive session ready.")
    print("Type a prompt and press Enter. Commands: :multi, :quit")

    while True:
        try:
            user_prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_prompt:
            continue
        if user_prompt in {":q", ":quit", "quit", "exit"}:
            break
        if user_prompt == ":multi":
            user_prompt = read_multiline_prompt().strip()
            if not user_prompt:
                continue

        for model in models:
            try:
                completions = model.complete(user_prompt, sampling_params)
            except KeyboardInterrupt:
                print()
                raise
            print_completion(model.label, completions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start an interactive comparison session for one or two model "
            "checkpoints."
        )
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="Checkpoint paths, or names relative to --checkpoint-dir.",
    )
    parser.add_argument(
        "-d",
        "--checkpoint-dir",
        help=(
            "Directory containing checkpoints. If no checkpoint names are passed, "
            "the script uses this directory if it is itself a model directory, or "
            "auto-discovers one/two checkpoint-* children."
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        action="append",
        help="Checkpoint name/path. Can be passed once or twice.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Display label for a checkpoint. Pass once per checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer path/id to use for all checkpoints. Defaults to each checkpoint.",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="Do not apply a chat template.")
    parser.add_argument("--system-prompt", help="Optional system message for chat templates.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-completions", "-n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", help="Stop string. Can be passed multiple times.")
    parser.add_argument("--seed", type=int, help="Optional vLLM sampling seed.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "vLLM GPU memory fraction per loaded model. Defaults to vLLM's "
            "default for one model, and 0.45 when loading two models."
        ),
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        checkpoint_paths = resolve_checkpoint_paths(args)
        sampling_params = build_sampling_params(args)
        models = load_models(args, checkpoint_paths)
        run_interactive_loop(models, sampling_params)
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
