# Simple graph synthetic dataset

```bash
uv venv graph_synth_data --python 3.10
source graph_synth_data/bin/activate
uv pip install matplotlib networkx datasets
python3 simple_graph_dataset.py --visualize
```

# New requirements for train

# First get on a gpu
```
srun --account=lingo --partition=lingo-h100 --qos=lingo-main --gpus=1 --time=4:00:00 --mem=80G --cpus-per-task=16 --pty /bin/bash
```

Then make a training env:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# Below command can take ~5-10 minutes
uv pip install transformers[torch] datasets accelerate bitsandbytes tokenizers peft tf-keras trl wandb fire hydra-core pebble timeout_decorator
uv pip install flash-attn --no-build-isolation
hf auth login
wandb login
```

Check that the GPU is available in Torch
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Output should be something like:

```bash
2.6.0+cu124
12.4
True
NVIDIA H100 NVL
```

Make a separate eval env.
Make sure you're still on the GPU node
We need a separate env because we need the package VLLM for evaluation, which has strict dependencies that don't play nicely with the training env above.

```bash
uv venv eval_env --python 3.12
source eval_env/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install transformers[torch] hydra-core datasets
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

To run training:
```
python train.py +experiment=sft ++wandb.entity=<your_wandb_org>
```
Or submit a job to slurm (see train.sh)

To run eval:
```
python evaluate.py +experiment=evaluate
```
Or submit a job to slurm (see eval.sh)
