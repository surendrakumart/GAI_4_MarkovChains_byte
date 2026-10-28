# Generation report

- Method: word-level Markov chain with order 3 and context backoff.
- Corpus: `corpus/training_text.txt` (original synthetic corpus; CC0 1.0).
- Prompts: 20 starting-word seeds listed in `prompts.txt`.
- Saved results: 20 files in `generated_samples/`.
- Runtime dependencies: Python standard library only.
- Default reproducibility seed: 2026 (each prompt uses a deterministic offset).

## Quality observation

The generator preserves some local word sequences from the corpus, but the samples can be grammatically awkward or semantically inconsistent. These are expected limitations of a small statistical model trained on a short corpus. The detailed comparison is in `comparison_note.md`.
