# Plivo LLM Speedrun — Experiment Run Log

## Run 00 — Official Starter Baseline

- **Checkpoint:** `baseline.pt`
- **Hypothesis:** Establish the untouched starter implementation as the reference point for all subsequent experiments.
- **Changes:** None. Used the provided model, byte tokenizer, Adam optimizer, constant learning rate of `3e-4`, batch size `8`, and 2,000 optimizer steps.
- **Parameters:** 1,339,840
- **Training time:** 58 seconds
- **Final reported training loss:** 1.7315
- **Dev BPB:** 2.3718
- **Conclusion:** The starter model trains stably, but its constant learning rate, untied embedding/output weights, plain Adam optimizer, and lack of gradient clipping or scheduling provide several controlled optimization opportunities.