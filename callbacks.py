
from transformers import AutoTokenizer, TrainerCallback
import torch

import logging
import wandb
import json
import os


class SaveCheckpointCallback(TrainerCallback):
    PROCESSOR_CONFIG_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")

    def __init__(self, trainer, base_model_id=None):
        self.trainer = trainer
        self.tokenizer = trainer.processing_class
        self.base_model_id = base_model_id

    def _copy_processor_configs(self, output_dir):
        # Multimodal checkpoints (e.g. Qwen3.5/3.6 VL) are saved via AutoModelForCausalLM
        # without the image/video processor configs. vLLM eval then fails because it
        # detects the multimodal arch and looks for preprocessor_config.json.
        # Copy them from the HF base repo so the saved checkpoint is self-contained.
        if not self.base_model_id:
            return
        try:
            with open(os.path.join(output_dir, "config.json")) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return
        is_multimodal = "vision_config" in cfg or "image_token_id" in cfg
        if not is_multimodal:
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return
        for fn in self.PROCESSOR_CONFIG_FILES:
            dst = os.path.join(output_dir, fn)
            if os.path.exists(dst):
                continue
            try:
                src = hf_hub_download(repo_id=self.base_model_id, filename=fn)
                shutil.copy(src, dst)
                logging.info(f"Copied {fn} from {self.base_model_id} to {output_dir}.")
            except Exception as e:
                logging.warning(f"Could not fetch {fn} from {self.base_model_id}: {e}")

    def _is_main(self, state=None):
        if state is not None and hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)
        acc = getattr(self.trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "is_main_process"):
            return bool(acc.is_main_process)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def on_save(self, args, state, control, **kwargs):
        if not self._is_main(state):
            return control
        # call this is training is done
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if getattr(wandb, "run", None) is not None:
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"checkpoint-{state.global_step}"})
        os.makedirs(output_dir, exist_ok=True)
        run_id = getattr(getattr(wandb, "run", None), "id", None) or os.environ.get("WANDB_RUN_ID")
        if run_id:
            with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
                f.write(run_id)
        self._copy_processor_configs(output_dir)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_main(state):
            return control
        logging.info("Training ended.")
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if getattr(wandb, "run", None) is not None:
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"checkpoint-{state.global_step}"})
        os.makedirs(output_dir, exist_ok=True)
        run_id = getattr(getattr(wandb, "run", None), "id", None) or os.environ.get("WANDB_RUN_ID")
        if run_id:
            with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
                f.write(run_id)
        self._copy_processor_configs(output_dir)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control


class SanityManualGreedyOnce(TrainerCallback):
    """
    One-time manual greedy decode at training start (no .generate()).
    - Works with ZeRO-3: all ranks run forward; only rank-0 prints.
    - Uses use_cache=False to avoid KV/cache issues.
    - Keeps decode short (few tokens) to avoid memory spikes.
    """
    def __init__(self, trainer, tokenizer, prompt="The quick brown fox", max_new_tokens=16):
        self.trainer = trainer
        self.tok = tokenizer
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.done = False

    def _is_main(self, trainer):
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "is_local_main_process"):
            return bool(acc.is_local_main_process)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _run_once(self, trainer):
        if self.done:
            return
        self.done = True

        model = trainer.model
        device = getattr(trainer.accelerator, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        device = next(trainer.model.parameters()).device
        enc = self.tok(self.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if isinstance(attn, torch.Tensor) else torch.ones_like(input_ids, device=device)

        was_training = model.training
        model.eval()

        # Manual greedy decode: append one token at a time; no caches or extra kwargs.
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # keep attention mask in lockstep
                one = torch.ones_like(next_token, device=device)
                attn = torch.cat([attn, one], dim=1)

        # sync & print only on rank-0
        try:
            trainer.accelerator.wait_for_everyone()
        except Exception:
            pass
        if self._is_main(trainer):
            text = self.tok.decode(input_ids[0], skip_special_tokens=True)
            print("[Check model loaded correctly] prompt:", repr(self.prompt))
            print("[Check model loaded correctly] output:", text)

        # restore training state
        model.train(was_training)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kw):
        tr = getattr(self, "trainer", None)
        self._run_once(tr)
        return control
