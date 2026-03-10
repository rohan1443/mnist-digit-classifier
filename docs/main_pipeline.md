# `main_pipeline.m`

## What This Script Is
This is the master launcher. It is the "control room" of the whole project.

## What It Does
- Shows a menu to user.
- Lets user run:
  - full pipeline
  - one step only
  - results summary
- Tracks total run time for full execution.

## Approach
Instead of manually running every script and risking missed steps, this file gives a structured flow.

The intended step order is:
1. `load_data.m`
2. `preprocess_data.m`
3. `extract_features.m`
4. `train_models.m`
5. `evaluate_model.m`

## Process (Simple View)
- Read user choice.
- Switch-case dispatches to matching function.
- Each function calls target script with `run('src/...')`.
- For full mode, print completion checkpoints and timing.

## Why It Matters For Assignment
- Shows reproducible workflow.
- Makes demo smoother in front of examiner.
- Reduces human mistakes between stages.

## Inputs and Outputs
Input:
- User keyboard choice.

Output:
- Runs selected stage(s).
- Console summary and timing.

## Notes
The file is a wrapper/orchestrator, not a model itself.
