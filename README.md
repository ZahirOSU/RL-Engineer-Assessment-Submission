# RL Training Task: ML Experiment Log Analysis

A reinforcement learning training task that evaluates a model's ability to correctly filter and aggregate experimental results based on precise specifications.

## Overview

| Property | Value |
|----------|-------|
| **Task Type** | Data filtering & aggregation |
| **Skill Tested** | Attention to specification details |
| **Target Pass Rate** | 10-40% |
| **Expected Answer** | 54140 |
| **Dataset Size** | 15 experiments |

## Task Description

The model is given results from a hyperparameter sweep and must:

1. **Filter** experiments matching ALL conditions:
   - `status == "completed"`
   - `test_accuracy > 0.82` *(strictly greater)*
   - `gpu_hours < 48` *(strictly less)*
   - `epochs >= 50` *(greater or equal)*

2. **Calculate** for each: `experiment_id × epochs`

3. **Return** the sum of all calculated values

## Why This Task?

### Real-World Relevance
Analyzing ML experiment logs is a daily task for ML engineers:
- *"Which runs converged with accuracy above our threshold?"*
- *"What's the total compute cost for our successful experiments?"*

### Skill It Teaches
The task tests **precision with inequalities**:

| Condition | Type | Common Mistake |
|-----------|------|----------------|
| `> 0.82` | strict | Using `>=` |
| `< 48` | strict | Using `<=` |
| `>= 50` | non-strict | Using `>` |

Models must read specifications carefully and implement them exactly.

## Edge Cases

The dataset contains deliberate boundary cases:

| Exp ID | Value | Condition | Included? | Reason |
|--------|-------|-----------|-----------|--------|
| 102 | acc = **0.82** | `> 0.82` | ❌ | Exactly at boundary |
| 103 | gpu = **48** | `< 48` | ❌ | Exactly at boundary |
| 105 | epochs = **50** | `>= 50` | ✅ | Meets non-strict |
| 111 | acc = **0.8201** | `> 0.82` | ✅ | Just above boundary |
| 113 | gpu = **47** | `< 48` | ✅ | Just below boundary |
| 108 | status = **failed** | `== completed` | ❌ | Wrong status |

## Failure Modes

Each mistake produces a **unique wrong answer**:

| Error | Wrong Answer | Correct | Difference |
|-------|--------------|---------|------------|
| `>= 0.82` instead of `> 0.82` | 59,240 | 54,140 | +5,100 |
| `<= 48` instead of `< 48` | 61,865 | 54,140 | +7,725 |
| `> 50` instead of `>= 50` | 43,240 | 54,140 | -10,900 |
| Forgot status check | 70,340 | 54,140 | +16,200 |

This allows precise identification of which concept the model failed on.

## Repository Structure

```
.
├── main.py              # Task implementation (380 lines)
├── data.json            # Experiment dataset (15 records)
├── README.md            # This documentation
├── pyproject.toml       # Project configuration
├── .python-version      # Python version (3.13)
└── .gitignore           # Git ignore rules
```

## Installation & Usage

### Prerequisites
- Python 3.11+
- Anthropic API key

### Setup

```bash
# Clone the repository
git clone https://github.com/ZahirOSU/RL-Engineer-Assessment-Submission.git
cd RL-Engineer-Assessment-Submission

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run with uv (recommended)
uv run main.py

# Or with pip
pip install anthropic
python main.py
```

### Expected Output

```
Data verification:
  Loaded 15 experiments
  Computed answer: 54140
  Expected answer: 54140
  Match: ✓

=================================================================
  RL TRAINING TASK: ML Experiment Log Analysis
=================================================================
  Model:           claude-3-5-haiku-latest
  Test iterations: 10
  Expected answer: 54140
-----------------------------------------------------------------
  ✓ Test  1:      54140 (expected 54140)
  ✗ Test  2:      59240 (expected 54140)
  ✓ Test  3:      54140 (expected 54140)
  ✗ Test  4:      61865 (expected 54140)
  ...
-----------------------------------------------------------------
  RESULTS:
    Passed:    3/10
    Failed:    7/10
    Pass rate: 30.0%

  WRONG ANSWER DISTRIBUTION:
    59240: 3 occurrence(s)
    61865: 2 occurrence(s)
-----------------------------------------------------------------
  ✓ Pass rate is WITHIN target range (10-40%)
=================================================================
```

## Design Decisions

### Why 4 Conditions?
More conditions = more opportunities for mistakes = harder task. Four conditions with mixed inequality types achieves the 10-40% target.

### Why These Boundary Values?
- `0.82` and `48` test strict inequality understanding
- `50` tests non-strict inequality understanding
- Values like `0.8201` and `47` test "just barely" cases

### Why This Calculation?
`experiment_id × epochs` requires the model to:
1. Correctly identify which rows pass the filter
2. Extract two different columns
3. Perform multiplication and summation

This catches models that filter correctly but make calculation errors.

## Technical Details

### Grading
The grader accepts multiple formats:
- Integer: `54140`
- Float: `54140.0`
- String: `"54140"`

Tolerance: ±0.5 (handles floating point)

### Agent Loop
- Max steps: 10
- Tools: `python` (code execution), `submit` (answer submission)
- Model: `claude-3-5-haiku-latest`

## Author

**Zahir Alsulaimawi**  
 


