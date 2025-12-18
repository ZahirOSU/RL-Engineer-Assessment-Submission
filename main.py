#!/usr/bin/env python3
"""
RL Training Task: ML Experiment Log Analysis

This task evaluates a model's ability to correctly filter and aggregate
experimental results based on multiple conditions with boundary edge cases.

The task tests attention to specification details - specifically the difference
between strict inequalities (>, <) and non-strict inequalities (>=, <=).

Target pass rate: 10-40%
Expected answer: 54140

Author: Zahir Alsulaimawi
Repository: https://github.com/ZahirOSU/RL-Engineer-Assessment-Submission
"""

import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-3-5-haiku-latest"
MAX_AGENT_STEPS = 10
DEFAULT_NUM_RUNS = 10
DATA_FILE = Path(__file__).parent / "data.json"
EXPECTED_ANSWER = 54140


# =============================================================================
# DATA HANDLING
# =============================================================================

def load_experiment_data() -> list[dict]:
    """
    Load ML experiment results from JSON file.
    
    Returns:
        List of experiment dictionaries containing metrics and metadata.
    
    Raises:
        FileNotFoundError: If data.json is missing.
        json.JSONDecodeError: If data.json is malformed.
    """
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_expected_answer(experiments: list[dict]) -> int:
    """
    Compute the correct answer using the exact filtering logic.
    
    Filter conditions (ALL must be true):
        - status == "completed"
        - test_accuracy > 0.82  (strictly greater)
        - gpu_hours < 48        (strictly less)
        - epochs >= 50          (greater or equal)
    
    Calculation: sum of (experiment_id * epochs) for filtered experiments.
    
    Args:
        experiments: List of experiment dictionaries.
    
    Returns:
        The correct numerical answer.
    """
    filtered = [
        exp for exp in experiments
        if (
            exp["status"] == "completed"
            and exp["test_accuracy"] > 0.82
            and exp["gpu_hours"] < 48
            and exp["epochs"] >= 50
        )
    ]
    return sum(exp["experiment_id"] * exp["epochs"] for exp in filtered)


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_task_prompt(experiments: list[dict]) -> str:
    """
    Construct the task prompt with embedded experimental data.
    
    The prompt is carefully worded to specify:
        - Exact filter conditions with strict/non-strict inequalities
        - The calculation formula
        - Expected output format
    
    Args:
        experiments: List of experiment dictionaries to embed.
    
    Returns:
        Complete prompt string for the agent.
    """
    # Format data for readability
    data_str = json.dumps(experiments, indent=2)
    
    return f"""You are an ML engineer analyzing results from a hyperparameter sweep.

Below is the experimental data in JSON format:

```json
experiments = {data_str}
```

## Your Task

1. **Filter** the experiments where ALL of these conditions are TRUE:
   - status equals "completed"
   - test_accuracy is STRICTLY GREATER than 0.82 (i.e., > 0.82, not >= 0.82)
   - gpu_hours is STRICTLY LESS than 48 (i.e., < 48, not <= 48)  
   - epochs is GREATER THAN OR EQUAL to 50 (i.e., >= 50)

2. **Calculate** for each filtered experiment: experiment_id × epochs

3. **Return** the SUM of all calculated values

Use the python tool to write and execute code, then submit your final numerical answer."""


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS: list[ToolUnionParam] = [
    {
        "name": "python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Use print() to output results. Returns stdout as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit",
        "description": "Submit your final numerical answer to complete the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "number",
                    "description": "The final computed numerical answer"
                }
            },
            "required": ["answer"],
        },
    },
]


def execute_python_code(code: str) -> dict[str, Any]:
    """
    Execute Python code safely and capture stdout.
    
    Args:
        code: Python code string to execute.
    
    Returns:
        Dictionary with 'output' (stdout string) and 'error' (exception info or None).
    """
    try:
        stdout_buffer = StringIO()
        with redirect_stdout(stdout_buffer):
            exec(code, {"__builtins__": __builtins__}, {})
        return {
            "output": stdout_buffer.getvalue(),
            "error": None
        }
    except Exception as e:
        return {
            "output": None,
            "error": f"{type(e).__name__}: {str(e)}"
        }


# =============================================================================
# GRADING
# =============================================================================

def grade_answer(submitted: Any, expected: int) -> bool:
    """
    Evaluate if the submitted answer is correct.
    
    Handles multiple submission formats:
        - Integer: 54140
        - Float: 54140.0
        - String: "54140"
    
    Args:
        submitted: The agent's submitted answer.
        expected: The correct answer.
    
    Returns:
        True if answer is correct within tolerance, False otherwise.
    """
    if submitted is None:
        return False
    
    try:
        # Convert string to number if needed
        if isinstance(submitted, str):
            submitted = float(submitted)
        
        # Allow small tolerance for floating point
        return abs(float(submitted) - float(expected)) < 0.5
    except (ValueError, TypeError):
        return False


# =============================================================================
# AGENT EXECUTION
# =============================================================================

async def run_agent(prompt: str, verbose: bool = False) -> Any:
    """
    Execute the agent loop until an answer is submitted or max steps reached.
    
    Args:
        prompt: The task prompt.
        verbose: If True, print agent's actions.
    
    Returns:
        The submitted answer, or None if no answer was submitted.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(MAX_AGENT_STEPS):
        # Get model response
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        tool_results = []
        submitted_answer = None

        # Process response content
        for block in response.content:
            if block.type == "text" and verbose:
                preview = block.text[:100].replace("\n", " ")
                print(f"    Step {step + 1}: {preview}...")

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                # Handle python tool
                if tool_name == "python":
                    code = tool_input.get("code", "")
                    result = execute_python_code(code)
                    
                    if verbose:
                        if result["error"]:
                            print(f"    Step {step + 1}: Python error - {result['error'][:50]}")
                        else:
                            output_preview = result["output"][:50].replace("\n", " ")
                            print(f"    Step {step + 1}: Python output - {output_preview}")

                # Handle submit tool
                elif tool_name == "submit":
                    submitted_answer = tool_input.get("answer")
                    result = {"status": "submitted", "answer": submitted_answer}
                    
                    if verbose:
                        print(f"    Step {step + 1}: Submitted answer - {submitted_answer}")

                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # No tools used - agent stopped
        if not tool_results:
            if verbose:
                print(f"    Step {step + 1}: No tool use, ending.")
            break

        # Update conversation history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Answer submitted - we're done
        if submitted_answer is not None:
            return submitted_answer

    return None


# =============================================================================
# TEST EXECUTION
# =============================================================================

async def run_single_test(
    test_id: int,
    prompt: str,
    expected: int,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    """
    Run a single test iteration.
    
    Args:
        test_id: Test number for display.
        prompt: The task prompt.
        expected: Expected answer.
        verbose: If True, print detailed output.
    
    Returns:
        Tuple of (test_id, passed, submitted_answer).
    """
    if verbose:
        print(f"\n  Test {test_id}:")
    
    try:
        answer = await run_agent(prompt, verbose=verbose)
        passed = grade_answer(answer, expected)
    except Exception as e:
        print(f"  ✗ Test {test_id}: ERROR - {type(e).__name__}: {e}")
        return test_id, False, None
    
    # Format output
    symbol = "✓" if passed else "✗"
    answer_str = str(answer) if answer is not None else "None"
    print(f"  {symbol} Test {test_id:2d}: {answer_str:>10} (expected {expected})")
    
    return test_id, passed, answer


async def run_evaluation(
    num_runs: int = DEFAULT_NUM_RUNS,
    verbose: bool = False
) -> float:
    """
    Run the complete evaluation suite.
    
    Args:
        num_runs: Number of test iterations.
        verbose: If True, print detailed output.
    
    Returns:
        Pass rate as a percentage.
    """
    # Load data and verify
    experiments = load_experiment_data()
    computed = compute_expected_answer(experiments)
    
    if computed != EXPECTED_ANSWER:
        raise ValueError(
            f"Expected answer mismatch: computed {computed}, expected {EXPECTED_ANSWER}"
        )
    
    # Build prompt
    prompt = build_task_prompt(experiments)
    
    # Print header
    print("=" * 65)
    print("  RL TRAINING TASK: ML Experiment Log Analysis")
    print("=" * 65)
    print(f"  Model:           {MODEL}")
    print(f"  Test iterations: {num_runs}")
    print(f"  Expected answer: {EXPECTED_ANSWER}")
    print("-" * 65)
    
    # Run tests concurrently
    tasks = [
        run_single_test(i + 1, prompt, EXPECTED_ANSWER, verbose=verbose)
        for i in range(num_runs)
    ]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    passed_count = sum(1 for _, passed, _ in results if passed)
    failed_count = num_runs - passed_count
    pass_rate = (passed_count / num_runs) * 100
    
    # Analyze wrong answers
    wrong_answers: dict[Any, int] = {}
    for _, passed, answer in results:
        if not passed and answer is not None:
            key = int(answer) if isinstance(answer, (int, float)) else str(answer)
            wrong_answers[key] = wrong_answers.get(key, 0) + 1
    
    # Print results
    print("-" * 65)
    print("  RESULTS:")
    print(f"    Passed:    {passed_count}/{num_runs}")
    print(f"    Failed:    {failed_count}/{num_runs}")
    print(f"    Pass rate: {pass_rate:.1f}%")
    
    if wrong_answers:
        print("\n  WRONG ANSWER DISTRIBUTION:")
        for ans, count in sorted(wrong_answers.items(), key=lambda x: -x[1]):
            print(f"    {ans}: {count} occurrence(s)")
    
    print("-" * 65)
    
    # Check target range
    if 10 <= pass_rate <= 40:
        print("  ✓ Pass rate is WITHIN target range (10-40%)")
    elif pass_rate < 10:
        print("  ⚠ Pass rate is BELOW target range (10-40%)")
    else:
        print("  ⚠ Pass rate is ABOVE target range (10-40%)")
    
    print("=" * 65)
    
    return pass_rate


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for the task evaluation."""
    # Verify data integrity
    experiments = load_experiment_data()
    computed = compute_expected_answer(experiments)
    
    print(f"\nData verification:")
    print(f"  Loaded {len(experiments)} experiments")
    print(f"  Computed answer: {computed}")
    print(f"  Expected answer: {EXPECTED_ANSWER}")
    print(f"  Match: {'✓' if computed == EXPECTED_ANSWER else '✗'}")
    print()
    
    # Run evaluation
    await run_evaluation(num_runs=DEFAULT_NUM_RUNS, verbose=False)


if __name__ == "__main__":
    asyncio.run(main())
