# Final Configuration Notes

1. The final model is a four-layer, four-head causal GPT with a 160-dimensional embedding, a 128-token context window, pre-layer normalization and GELU feed-forward blocks.
2. It contains 1,360,320 parameters, safely below the 2,000,000-parameter limit.
3. The tokenizer uses all 256 raw-byte tokens plus 64 frequent non-ASCII character tokens learned only from the provided training corpus.
4. Raw-byte fallback makes the tokenizer exactly lossless for arbitrary UTF-8 while reducing the training sequence length by 21.80%.
5. Batch size 128 increases useful token exposure within the fixed 2,000-optimizer-step budget.
6. The final training recipe uses AdamW, a peak learning rate of 6e-4, 100 warm-up steps, cosine decay to 6e-5, weight decay of 0.1 and gradient clipping at 1.0.
7. The final checkpoint records exactly 2,000 optimizer steps and achieves 1.8773 BPB on the provided development evaluation file.
8. This improves the official starter baseline of 2.3718 BPB by 0.4945 BPB, or 20.85%.
9. Weight tying was rejected because it worsened dev BPB to 2.4122 despite reducing the parameter count.
10. All training was performed locally on CPU using only the supplied training corpus, pure PyTorch and standard-library components.
