# 🎯 Meta Ads Attribution Recovery Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**An OpenEnv-compliant reinforcement learning environment that models Meta Ads attribution recovery under iOS tracking constraints, narrow attribution windows, and incomplete conversion signals.**

> 🚨 **The Problem**: Meta advertisers lose significant revenue because iOS privacy changes, narrow attribution windows, and browser tracking restrictions leave **40-70% of conversions** untracked. As a result, Meta's optimization system learns from incomplete signals, often overvaluing short-lag outcomes while undervaluing high-performing ads with delayed conversions.

---

## 🔥 The Attribution Crisis Explained

### What's Breaking Attribution?

1. **Narrow Attribution Windows**: Defaults shifted from 28-day to 7-day (or 1-day), so later conversions are excluded.
2. **iOS 14.5+ Privacy**: Apple ATT suppresses a large share of iOS conversion tracking via Meta Pixel.
3. **Browser Restrictions**: Safari ITP, Firefox protections, and ad blockers further reduce signal quality.
4. **Missing Server-Side Tracking**: Many advertisers still lack Conversions API (CAPI), limiting server-side recovery.

### The Impact

**Example Campaign:**
- 📊 **Reported Metrics**: 59 conversions, $76 CPA, 0.98x ROAS → *Appears unprofitable*
- ✅ **True Performance**: 180 conversions, $25 CPA, 3.0x ROAS → *Actually highly profitable!*
- ❌ **Attribution Gap**: **67% of conversions untracked**

**Result**: The optimization loop can pause profitable inventory and over-allocate spend to weaker ad sets.

### This Environment Teaches Agents To:
1. Diagnose attribution failures from campaign-level and ad set-level signals.
2. Apply technical remediations (window expansion, CAPI, AEM).
3. Reallocate budget using true performance rather than biased observed metrics.
4. Recover signal quality so optimization decisions become reliable again.

---

## 🎮 Environment Overview

### OpenEnv Compliance ✅
- **Typed Pydantic models** for Observation, Action, Reward, State
- **Standard API**: `reset()`, `step(action)`, `state()`
- **Three difficulty levels** with programmatic graders (0.0–1.0 scoring)
- **Realistic simulator** modeling attribution degradation
- **Multi-component rewards** that capture incremental progress

### Key Metrics
- **Episode Length**: 5-10 steps
- **Success Threshold**: Score ≥0.60
- **Action Space**: 10 discrete actions (window adjustment, CAPI, budget optimization)
- **Observation Space**: Campaign metrics + diagnostic signals + natural language context

---

## 🎯 Action Space

| Action | Parameters | Use Case |
|--------|-----------|----------|
| `adjust_attribution_window` | `{"window": "7d_click"}` | When window is too narrow (1d_click) |
| `enable_conversions_api` | `{}` | When iOS >40% and Pixel signal <60% |
| `enable_aggregated_event_measurement` | `{}` | After CAPI, for additional iOS recovery |
| `add_utm_tracking` | `{}` | Improve cross-domain attribution |
| `pause_underperforming_adsets` | `{"roas_threshold": 1.0}` | When true ROAS <1.0 |
| `reallocate_to_top_performers` | `{"amount": 2000}` | Shift budget to high-ROAS adsets |
| `adjust_budget_allocation` | `{"shifts": {...}}` | Fine-grained budget control |
| `change_bid_strategy` | `{"strategy": "value_optimisation"}` | Optimize for ROAS vs CPA |
| `segment_audience` | `{}` | Create better-targeted segments |
| `no_op` | `{}` | No action needed |

---

## 👁️ Observation Space

```python
{
  # Campaign Performance
  "reported_conversions": 59,        # What Meta sees
  "true_conversions": 180,            # Ground truth (hidden from algorithm)
  "attribution_gap_pct": 0.672,       # 67% untracked!
  "reported_roas": 0.98,              # Appears unprofitable
  "true_roas": 3.0,                   # Actually profitable
  
  # Attribution Setup
  "attribution_window": "1d_click",   # Too narrow
  "pixel_signal_quality": 0.86,       # 14% signal loss
  "ios_traffic_pct": 0.25,            # 25% iOS users
  "conversions_api_enabled": false,   # Missing CAPI
  "aem_enabled": false,               # Missing AEM
  
  # Natural Language Context (for LLM agents)
  "context": "Campaign 'Spring Sale' | Objective: CONVERSIONS\n..."
}
```

---

## 📊 Tasks & Difficulty

### 🟢 Easy: Attribution Window Fix
**Problem**: A 1-day attribution window excludes most delayed conversions.  
**Solution**: Adjust to 7-day click window  
**Baseline Score**: 0.893 ✅

### 🟡 Medium: iOS Signal Recovery
**Problem**: High iOS share without CAPI/AEM causes substantial signal loss.  
**Solution**: Enable CAPI → Enable AEM  
**Baseline Score**: 0.850 ✅

### 🔴 Hard: Full Attribution Audit
**Problem**: Narrow window, high iOS exposure, missing tracking stack, and misallocated budget.  
**Solution**: Multi-step optimization (5+ actions)  
**Baseline Score**: 0.794 ✅

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/meta-ads-openenv.git
cd meta-ads-openenv

# Install dependencies
pip install -r requirements.txt

# Set up API key (copy .env.example to .env and add your key)
cp .env.example .env
# Edit .env with required values:
# API_BASE_URL=https://router.huggingface.co/v1
# MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# HF_TOKEN=hf_your_token_here
```

### Run Baseline Agent

```bash
# Required (for LLM-backed paths):
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here

# Run baseline across all 3 tasks
python baseline/run_baseline.py
```

**Expected Output:**
```
TASK: EASY_ATTRIBUTION_WINDOW
Score: 0.8926 (PASS ✅) | Steps: 5/5

TASK: MEDIUM_PIXEL_RECOVERY  
Score: 0.8500 (PASS ✅) | Steps: 4/7

TASK: HARD_FULL_ATTRIBUTION_AUDIT
Score: 0.7942 (PASS ✅) | Steps: 7/10

Average Score: 0.8456
```

### Launch Demo UI

```bash
python app/app.py
# Open browser to http://127.0.0.1:7860
```

### Use Programmatically

```python
from meta_ads_env import MetaAdsAttributionEnv
from meta_ads_env.models import Action

# Initialize
env = MetaAdsAttributionEnv(task_id="easy_attribution_window")
obs = env.reset()

# Check initial state
print(f"Attribution gap: {obs.attribution_gap_pct:.1%}")
print(f"Reported ROAS: {obs.roas_reported:.2f}x")
print(f"True ROAS: {obs.roas_true:.2f}x")

# Take action
action = Action(
    action_type="adjust_attribution_window",
    parameters={"window": "7d_click"}
)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.total:.4f}")
print(f"New gap: {obs.attribution_gap_pct:.1%}")

# Grade episode
if done:
    result = env.grade_episode()
    print(f"Score: {result.score:.4f} - {'PASS' if result.passed else 'FAIL'}")
```

---

## 📈 Baseline Results

**Model**: Qwen/Qwen2.5-72B-Instruct (OpenAI-compatible interface) | **Temperature**: 0.0

| Task | Score | Pass | Steps | Key Actions |
|------|-------|------|-------|-------------|
| Easy | 0.893 | ✅ | 5/5 | Investigate + window fix + convergence handling |
| Medium | 0.850 | ✅ | 4/7 | Investigate + CAPI + AEM + modeled reporting |
| Hard | 0.794 | ✅ | 7/10 | Investigate + window + CAPI + AEM + pause + reallocate |
| **Average** | **0.846** | **100%** | - | **All passing!** 🎉 |

---

## 🎁 Reward Function

Multi-component reward designed to reward meaningful progress:

```python
reward = (
    0.35 × attribution_accuracy     # Gap closure
  + 0.25 × roas_improvement         # True ROAS increase
  + 0.25 × signal_quality_gain      # Pixel recovery
  + 0.10 × action_validity          # Right action for context
  + 0.05 × step_efficiency          # Fewer steps bonus
  - trajectory_penalty              # Harmful action penalty
)
```

**Range**: -1.0 to 1.0 per step

---

## 🐳 Docker Deployment

### Build and Run Locally
```bash
docker build -t meta-ads-env .
docker run -p 7860:7860 -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct -e HF_TOKEN=hf_your_token_here meta-ads-env
```

### Deploy to Hugging Face Spaces
1. Create new Space (Docker SDK)
2. Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` as Space secrets
3. Push code to the Space repository
4. Space auto-builds and deploys

---

## 🧪 Inference Workflow

### Inference Script

Use `inference.py` at the repository root to run standardized task inference with structured logs.

**Set environment variables:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
```

**Run inference:**
```bash
python inference.py
```

**Output format:**
```
[START] task=easy_attribution_window env=meta_ads_attribution_openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=investigate_attribution reward=0.09 done=false error=null
[END] success=true steps=3 score=0.900 rewards=0.09,0.76,0.67
...
```

This structured output is designed for easy monitoring, reproducible evaluation, and downstream parsing.

### Validate Submission
```bash
bash scripts/validate-submission.sh <your_space_url> .
```

---

## 📁 Project Structure

```
meta-ads-openenv/
├── openenv.yaml              # OpenEnv metadata
├── inference.py              # Required hackathon inference script
├── requirements.txt          # Dependencies
├── Dockerfile                # Container definition
│
├── meta_ads_env/            # Core environment
│   ├── env.py               # Main environment class
│   ├── models.py            # Pydantic models
│   ├── simulator.py         # Attribution simulator
│   ├── reward.py            # Reward function
│   ├── grader.py            # Task graders
│   └── tasks.py             # Task definitions
│
├── data/                    # Task scenarios
│   ├── easy_ads.json
│   ├── medium_ads.json
│   └── hard_ads.json
│
├── baseline/                # Baseline agent
│   ├── baseline_agent.py    # LLM-powered agent
│   └── run_baseline.py      # Evaluation script
│
├── evaluation/              # Evaluation tools
│   ├── llm_grader.py       # Optional LLM-as-judge
│   └── metrics.py          # Aggregate metrics
│
├── app/                     # Demo UI
│   └── app.py              # Gradio interface
│
└── scripts/                 # Utilities
    └── validate-submission.sh
```

---

## 🔬 Advanced Usage

### Validate OpenEnv Compliance
```bash
pip install openenv
openenv validate .
```

### Custom Training Loop
```python
for episode in range(100):
    obs = env.reset()
    done = False
    
    while not done:
        action = your_policy.select_action(obs)
        obs, reward, done, info = env.step(action)
        your_policy.update(obs, action, reward)
```

### LLM-as-Judge Evaluation
```python
from evaluation.llm_grader import LLMGrader

grader = LLMGrader(model="Qwen/Qwen2.5-72B-Instruct")
result = grader.grade_trajectory(
    task_id="hard_full_attribution_audit",
    history=env.state().history,
    initial_context=initial_obs.context,
    final_context=final_obs.context
)
```

---

## 📄 License

MIT License - a permissive license that allows use, modification, and distribution with attribution. See LICENSE for full terms.

---

## 🙏 Acknowledgments

Built to demonstrate how AI agents can solve high-impact marketing optimization problems in realistic attribution environments. Inspired by real Meta Ads attribution challenges faced by performance teams at scale.

**OpenEnv**: RL environment specification  
**Meta Ads Manager**: Real-world attribution dynamics and constraints  
**Digital Marketing Community**: Practical insights from attribution and measurement operations

---

**🚀 Making attribution-aware AI optimization practical and measurable**
