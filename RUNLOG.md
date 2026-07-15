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

## Run 01 — Weight Tying

- **Checkpoint:** `exp01_weight_tying.pt`
- **Hypothesis:** Sharing the token embedding and output projection weights may improve parameter efficiency and regularization while freeing capacity for later architecture scaling.
- **Single change:** Set `tie_weights = True`; all other model and training settings remained identical to the official baseline.
- **Parameters:** 1,298,880
- **Training time:** 54 seconds
- **Final reported training loss:** 1.7651
- **Dev BPB before:** 2.3718
- **Dev BPB after:** 2.4122
- **Result:** Rejected as a standalone change.
- **Conclusion:** Weight tying reduced the parameter count by 40,960 but worsened dev BPB by 0.0404. The shared embedding/output representation likely constrained this byte-level model within the limited 2,000-step training regime. It may be reconsidered only if the saved parameters are reinvested into a stronger architecture.