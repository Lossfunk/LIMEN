# LIMEN: Discovering Reinforcement Learning Interfaces with Large Language Models

[[Paper]](https://arxiv.org/abs/TODO) [[Website]](https://lossfunk.com/limen)

Manually designing observation and reward functions for RL agents is tedious, requires domain expertise, and often leads to suboptimal performance. **LIMEN** automates this process — an LLM generates Python code defining what the agent sees (`get_observation`) and what signal it learns from (`compute_reward`), trains a PPO agent to evaluate each candidate, and evolves better designs over time using MAP-Elites + island model evolution.

LIMEN achieves **99% / 99% / 85% / 67% / 55%** success rates across Easy / Medium / Hard / Panda / Go1 tasks at a total LLM cost of **$42** in **36 hours on a single GPU**.

## Key Features

- **Automated MDP interface design** — LLM generates both observation and reward functions as JAX-compatible Python code
- **Cascade evaluation** — 3-stage crash filter (syntax, import, JIT dry-run) followed by short training to filter hopeless candidates before expensive full training
- **MAP-Elites + island model** — maintains behavioral diversity across reward complexity and observation dimensionality
- **Multi-environment support** — XLand-MiniGrid (discrete grid worlds), MuJoCo Playground (Panda arm, Go1 quadruped), and Brax (Ant, Humanoid locomotion)
- **Model ensemble** — weighted random selection across multiple LLMs via any OpenAI-compatible endpoint (OpenRouter, OpenAI, Together, Groq, local vLLM/Ollama, …)
- **Stochastic prompt engineering** — randomized improvement guidance prevents the LLM from getting stuck in local optima

## Installation

LIMEN requires Python >= 3.10 and a CUDA-capable GPU. Tested on Ubuntu 20.04 and 22.04.

We recommend [`uv`](https://docs.astral.sh/uv/) for installation (much faster than pip). Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it. Plain `pip` works too — just substitute `pip install` for `uv pip install` below.

1. **Clone the repo:**

```bash
git clone https://github.com/lossfunk/limen.git
cd limen
```

2. **Create a virtual environment and install LIMEN:**

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

3. **Install JAX with CUDA support** (adjust for your CUDA version):

```bash
uv pip install -U "jax[cuda12]"
```

4. **For MuJoCo tasks** (Panda, Go1), also install MuJoCo Playground:

```bash
git clone https://github.com/google-deepmind/mujoco_playground.git
uv pip install -e mujoco_playground
```

5. **Set up your LLM API key.** LIMEN uses an OpenAI-compatible client, so it works with OpenRouter, OpenAI, Together, Groq, Fireworks, Azure OpenAI, vLLM, Ollama, LM Studio, etc. Default config points at OpenRouter.

   Copy the example env file and add your key:

   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=sk-or-...
   ```

   To use a different provider, set `llm.api_base` in your YAML config (or pass `--api-base`):

   | Provider   | `api_base`                          |
   |------------|-------------------------------------|
   | OpenRouter | `https://openrouter.ai/api/v1`      |
   | OpenAI     | `https://api.openai.com/v1`         |
   | Together   | `https://api.together.xyz/v1`       |
   | Groq       | `https://api.groq.com/openai/v1`    |
   | vLLM/local | `http://localhost:8000/v1`          |

## Getting Started

Navigate to the LIMEN directory and run:

```bash
python run.py --config configs/easy_pickup.yaml \
              --task "Pick up the blue pyramid." \
              --iterations 50
```

### Example Commands

```bash
# Easy task: pick up an object (XLand-MiniGrid, ~30 min)
python run.py --config configs/easy_pickup.yaml \
              --task "Pick up the blue pyramid."

# Medium task: place object near target (XLand-MiniGrid, ~2 hours)
python run.py --config configs/medium_place_near.yaml \
              --task "Place the yellow pyramid adjacent to the green square."

# Hard task: multi-step rule chain (XLand-MiniGrid, ~8 hours)
python run.py --config configs/hard_rule_chain.yaml \
              --task "Pick up the blue pyramid (transforms to green ball), place near yellow hex."

# Panda arm tracking (MuJoCo, ~4 hours)
python run.py --config configs/panda_pick_and_track.yaml

# Go1 quadruped push recovery (MuJoCo, ~6 hours)
python run.py --config configs/go1_push_recovery.yaml

# Brax Ant locomotion (Brax, ~2 hours)
python run.py --config configs/brax_ant.yaml
```

Each run creates a timestamped directory in `runs/` containing:
- `best_interface.py` — the best evolved observation + reward functions
- `evolution_trace.jsonl` — full evolution history
- `database/` — MAP-Elites population checkpoint
- `evolution.log` — detailed training logs

### Key Command Line Options

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--task` | Natural language task description |
| `--iterations` | Number of evolution iterations |
| `--mode` | Evolution mode: `full`, `reward_only`, `obs_only`, `default`, `random` |
| `--model` | LLM model override (e.g., `anthropic/claude-sonnet-4`, `openai/gpt-4o-mini`) |
| `--api-base` | OpenAI-compatible API base URL (e.g., `https://openrouter.ai/api/v1`) |
| `--api-key` | API key override (else read from `OPENAI_API_KEY`) |
| `--timesteps` | Short training timesteps |
| `--timesteps-full` | Full training timesteps |
| `--num-seeds` | Seeds for multi-seed averaging |
| `--resume` | Resume from a checkpoint directory |

Full config options are in `configs/easy_pickup.yaml`.

## What LIMEN Produces

LIMEN evolves Python functions that define the RL agent's interface. See `examples/evolved_interfaces/` for the best interfaces discovered in our experiments. For example, the Go1 push recovery interface (55% success) looks like:

```python
import jax
import jax.numpy as jnp

def get_observation(state):
    # Extracts gyroscope, gravity, joint states, push forces, position errors...
    return jnp.concatenate([gyro, gravity, joint_offsets, ...])

def compute_reward(state, action, next_state):
    # Dense reward: position recovery + upright bonus + velocity settling + ...
    return position_reward + upright_bonus - action_penalty
```

The LLM discovers these designs automatically through evolutionary search — no human reward engineering required.

## Running on a New Environment

1. **Create an environment adapter** by subclassing `limen.adapters.base.EnvAdapter`:

```python
from limen.adapters.base import EnvAdapter

class MyAdapter(EnvAdapter):
    def get_dummy_state(self):
        """Return a real env state for crash-filter validation."""
        ...

    def get_default_obs_fn(self):
        """Baseline observation function (for reward_only ablation)."""
        ...

    def get_default_reward_fn(self):
        """Baseline reward function (None = use env built-in)."""
        ...
```

2. **Register the adapter:**

```python
from limen.adapters import register_adapter
register_adapter("my_env", "my_module.MyAdapter")
```

3. **Write a context file** (e.g., `contexts/my_env.md`) describing the state object, action space, and JAX constraints. This is the library reference the LLM uses to write code. See `contexts/xminigrid.md` for an example.

4. **Create a config YAML** (copy from `configs/easy_pickup.yaml` and modify) with your environment settings.

5. **Run LIMEN:**

```bash
python run.py --config configs/my_env.yaml --task "Description of the task."
```

## Project Structure

```
limen/                    # Core package
  config.py               # Hierarchical dataclass config with YAML loading
  controller.py           # Main evolution loop
  evaluator.py            # Cascade evaluation pipeline
  crash_filter.py         # 3-stage crash filter (syntax -> import -> JIT)
  database.py             # MAP-Elites + island model database
  prompts.py              # LLM prompt construction with stochastic guidance
  llm_client.py           # OpenAI-compatible LLM client with model ensemble
  interface.py            # MDP interface loader and validator
  nn.py                   # MLP + GRU actor-critic (Flax/JAX)
  train.py                # PPO training for discrete envs (xminigrid)
  train_brax.py           # Brax PPO training for continuous control
  adapters/               # Environment adapters
    xminigrid.py           # XLand-MiniGrid adapter
    mujoco.py              # MuJoCo Playground adapter (Panda, Go1)
    brax.py                # Native Brax adapter (Ant, Humanoid)
  tasks/                  # MuJoCo task environments
configs/                  # YAML experiment configs
contexts/                 # LLM context documents per environment
rulesets/                 # XLand-MiniGrid task rulesets
examples/                 # Best evolved interfaces from paper results
run.py                    # CLI entry point
```

## Acknowledgement

- Our grid-world environments are from [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid)
- MuJoCo tasks use [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- Locomotion environments are from [Brax](https://github.com/google/brax)
- Architecture inspired by [OpenEvolve](https://github.com/codelion/openevolve)
- Design philosophy influenced by [Eureka](https://github.com/eureka-research/Eureka) (ICLR 2024)

## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{jaswal2025limen,
  title   = {LIMEN: Discovering Reinforcement Learning Interfaces with Large Language Models},
  author  = {Akshat Singh Jaswal and Ashish Baghel and Paras Chopra},
  year    = {2025},
  journal = {arXiv preprint arXiv: TODO}
}
```
