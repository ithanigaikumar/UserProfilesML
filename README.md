# Hidden User Models

Replication and extension of [Chen et al. (2024)](https://arxiv.org/abs/2406.07882) — linear probing of user attributes in LLaMA-2-13B-Chat, extended to five dynamic cognitive/emotional states.

## Key results
- All five dynamic attributes (confusion, knowledge level, emotional state, intent, formality) are linearly decodable at **>90% accuracy**.
- Dynamic attributes peak at **earlier layers** than static demographics (two-tier structure).
- Activation-space causal interventions achieve a **96.7% mean flip rate** vs. 0% for a random-direction control.

## Structure
```
src/
  probes.py            # probe training & evaluation
  probe_score_eval.py  # activation-space causal intervention eval
  run_intervention.py  # LLM-based intervention (archived)
  config.py            # paths and attribute definitions
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

Model: `meta-llama/Llama-2-13b-chat-hf` on CUDA (A100 recommended).  
Data and activations are not tracked — generate with the scripts in `src/`.
