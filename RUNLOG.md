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

## Run 02 — Increased Batch Size to 32

- **Checkpoint:** `exp02_batch32.pt`
- **Hypothesis:** Increasing the batch size from 8 to 32 will expose the model to four times as many token positions within the fixed 2,000-step budget and provide a more stable gradient estimate.
- **Single change:** Increased `--batch` from `8` to `32`; architecture, tokenizer, optimizer, learning rate, seed, sequence length, and step count remained unchanged.
- **Parameters:** 1,339,840
- **Training time:** 230 seconds
- **Final reported training loss:** 1.4430
- **Dev BPB before:** 2.3718
- **Dev BPB after:** 2.0943
- **Absolute improvement:** 0.2775 BPB
- **Relative improvement:** 11.70%
- **Result:** Accepted as the current best configuration.
- **Conclusion:** Increasing the batch size substantially improved generalization under the fixed optimizer-step budget. Batch size 32 processed approximately 8.19 million token positions, compared with approximately 2.05 million for the baseline. The result indicates that the starter configuration was primarily limited by training-token throughput rather than model capacity.

## Run 03 — Increased Batch Size to 64

- **Checkpoint:** `exp03_batch64.pt`
- **Hypothesis:** Increasing the batch size from 32 to 64 will process twice as many token positions within the fixed 2,000-step budget and may further improve gradient stability and generalization.
- **Single change:** Increased `--batch` from `32` to `64`; architecture, tokenizer, optimizer, learning rate, seed, sequence length, and step count remained unchanged.
- **Parameters:** 1,339,840
- **Training time:** 463 seconds
- **Final reported training loss:** 1.3442
- **Dev BPB before:** 2.0943
- **Dev BPB after:** 1.9969
- **Absolute improvement over previous best:** 0.0974 BPB
- **Relative improvement over previous best:** 4.65%
- **Absolute improvement over baseline:** 0.3749 BPB
- **Relative improvement over baseline:** 15.81%
- **Result:** Accepted as the current best configuration.
- **Conclusion:** Increasing the batch size from 32 to 64 produced another substantial improvement, confirming that additional training-token throughput remains valuable under the fixed optimizer-step limit. The smaller gain relative to the previous batch-size increase indicates diminishing returns, but batch size 64 remains clearly superior to all earlier configurations.

## Run 04 — Increased Batch Size to 128

- **Checkpoint:** `exp04_batch128.pt`
- **Hypothesis:** Increasing the batch size from 64 to 128 will process twice as many token positions within the fixed 2,000-step budget and may continue improving generalization, although diminishing returns are expected.
- **Single change:** Increased `--batch` from `64` to `128`; architecture, tokenizer, optimizer, learning rate, seed, sequence length, and step count remained unchanged.
- **Parameters:** 1,339,840
- **Training time:** 1,027 seconds
- **Final reported training loss:** 1.2767
- **Dev BPB before:** 1.9969
- **Dev BPB after:** 1.9380
- **Absolute improvement over previous best:** 0.0589 BPB
- **Relative improvement over previous best:** 2.95%
- **Absolute improvement over baseline:** 0.4338 BPB
- **Relative improvement over baseline:** 18.29%
- **Result:** Accepted as the current best configuration.
- **Conclusion:** Batch size 128 produced another valid improvement, confirming that increased token throughput remained beneficial. However, the gain was smaller than the batch 32 and batch 64 improvements, while training time increased to more than 17 minutes. This demonstrates diminishing returns from further batch scaling, so subsequent experiments will target tokenizer and optimization efficiency instead.

## Run 05 — Lossless Hybrid UTF-8 Tokenizer

- **Checkpoint:** `exp05_hybrid_tokenizer.pt`
- **Hypothesis:** Replacing frequently occurring non-ASCII UTF-8 byte sequences with single character tokens will reduce sequence length for Hindi and other multilingual text while retaining raw-byte fallback for arbitrary unseen UTF-8 input.
- **Single change:** Replaced the 256-token byte tokenizer with a lossless 320-token hybrid tokenizer containing all 256 byte tokens plus the 64 most frequent non-ASCII characters learned exclusively from `train_corpus.txt`; all model and training settings remained unchanged from Run 04.
- **Vocabulary size:** 320
- **Training corpus tokens before:** 7,318,592
- **Training corpus tokens after:** 5,716,436
- **Token reduction:** 21.80%
- **Parameters:** 1,360,320
- **Training time:** 829 seconds
- **Final reported training loss:** 1.5773
- **Dev BPB before:** 1.9380
- **Dev BPB after:** 1.9020
- **Absolute improvement over previous best:** 0.0360 BPB
- **Relative improvement over previous best:** 1.86%
- **Absolute improvement over baseline:** 0.4698 BPB
- **Relative improvement over baseline:** 19.81%
- **Result:** Accepted as the current best configuration.
- **Conclusion:** The hybrid tokenizer improved BPB while remaining exactly lossless and robust to arbitrary UTF-8 through byte fallback. Its 21.80% reduction in training-token count provides more linguistic context within the fixed 128-token window, particularly for Devanagari text, confirming that multilingual token efficiency was a meaningful limitation of the original byte tokenizer.