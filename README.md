# Plivo 2,000-Step LLM Speedrun

Final submission for the Plivo AI/ML Assignment Round, IIT Kharagpur.

## Final Result

- **Development BPB:** 1.8773
- **Parameters:** 1,360,320
- **Optimizer steps:** 2,000
- **Improvement over starter baseline:** 20.85%
- **Execution environment:** CPU only

## Final System

- Four-layer, four-head causal GPT
- 160-dimensional token embeddings
- Context length of 128 tokens
- Lossless hybrid UTF-8 tokenizer with raw-byte fallback
- Batch size 128
- AdamW optimizer
- Linear warm-up and cosine learning-rate decay
- Decoupled weight decay and gradient clipping

## Training

The training corpus must be supplied through `--data`.

```bash
python train.py \
  --data ../data/train_corpus.txt \
  --steps 2000 \
  --batch 128 \
  --lr 6e-4 \
  --out ckpt.pt
```

## Evaluation

```bash
python evaluate.py \
  --checkpoint ckpt.pt \
  --text_file ../data/dev_eval.txt
```

Expected development result:

```json
{"bpb": 1.8773, "n_params": 1360320, "steps": 2000, "tokens_in_eval": 113387, "tokens_scored": 113386}
```

## Tokenizer

The tokenizer contains all 256 raw-byte tokens and 64 frequent non-ASCII
character tokens learned only from the supplied training corpus. Raw-byte fallback
keeps encoding lossless for arbitrary UTF-8 text.

## Documentation

- `RUNLOG.md` contains the complete experiment history.
- `NOTES.md` describes the final configuration in ten sentences.
- `SUMMARY.html` provides the technical report and contribution disclosure.
