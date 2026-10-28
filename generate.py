#!/usr/bin/env python3
"""Generate text samples using a small word-based Markov chain.

No third-party packages are required. The corpus and prompts are stored in this
project directory so the run is reproducible.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus" / "training_text.txt"
PROMPTS_PATH = ROOT / "prompts.txt"
OUTPUT_DIR = ROOT / "generated_samples"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?;,]")


def tokenize(text: str) -> list[str]:
    """Return lowercase word and punctuation tokens."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def read_seeds(path: Path) -> list[str]:
    """Read numbered seed words from prompts.txt, ignoring comments."""
    seeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"\s*\d+\.\s*([A-Za-z]+(?:'[A-Za-z]+)?)\s*$", line)
        if match:
            seeds.append(match.group(1).lower())
    return seeds


class MarkovTextGenerator:
    """A word-level Markov chain with context backoff."""

    def __init__(self, tokens: list[str], order: int = 3) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        if len(tokens) <= order:
            raise ValueError("Corpus is too small for the selected Markov order")
        self.order = order
        self.tokens = tokens
        self.transitions: dict[int, DefaultDict[tuple[str, ...], list[str]]] = {}
        for context_size in range(1, order):
            table: DefaultDict[tuple[str, ...], list[str]] = defaultdict(list)
            for index in range(len(tokens) - context_size):
                context = tuple(tokens[index:index + context_size])
                next_token = tokens[index + context_size]
                table[context].append(next_token)
            self.transitions[context_size] = table
        self.positions: dict[str, list[int]] = defaultdict(list)
        for index, token in enumerate(tokens):
            if token.isalpha() or "'" in token:
                self.positions[token].append(index)

    def generate(self, seed: str, token_limit: int = 55, rng: random.Random | None = None) -> str:
        """Generate a sample beginning with a seed word from the corpus."""
        if rng is None:
            rng = random.Random()
        seed = seed.lower().strip()
        if seed not in self.positions:
            raise ValueError(f"Seed '{seed}' does not appear in the corpus")
        if token_limit < 3:
            raise ValueError("token_limit must be at least 3")

        # Start with a corpus occurrence of the seed and enough following tokens
        # to initialize the highest-order context when possible.
        # Prefer sentence-start occurrences when available, so each sample
        # begins with a plausible capitalization and sentence boundary.
        sentence_starts = [
            i for i in self.positions[seed]
            if i == 0 or self.tokens[i - 1] in {".", "!", "?"}
        ]
        candidates = sentence_starts or self.positions[seed]
        possible_starts = [i for i in candidates if i + 2 < len(self.tokens)]
        start = rng.choice(possible_starts or candidates)
        output = self.tokens[start:min(start + 3, len(self.tokens))]

        while len(output) < token_limit:
            chosen: str | None = None
            max_context = min(self.order - 1, len(output))
            for size in range(max_context, 0, -1):
                context = tuple(output[-size:])
                candidates = self.transitions[size].get(context)
                if candidates:
                    chosen = rng.choice(candidates)
                    break
            if chosen is None:
                # This is a defensive fallback; corpus transitions normally
                # provide a continuation for every non-final token.
                chosen = rng.choice(self.tokens)
            output.append(chosen)

        # Stop at the first sentence-ending punctuation after a useful length,
        # while preserving the requested minimum amount of generated content.
        if len(output) > 12:
            for index in range(12, len(output)):
                if output[index] in {".", "!", "?"}:
                    output = output[:index + 1]
                    break
        return detokenize(output)


def detokenize(tokens: Iterable[str]) -> str:
    """Make simple readable text from word/punctuation tokens."""
    text = ""
    no_space_before = {".", ",", ";", "!", "?"}
    for token in tokens:
        if token in no_space_before:
            text = text.rstrip() + token + " "
        else:
            text += token + " "
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Capitalize the beginning of the sample and sentence beginnings.
    if text:
        text = text[0].upper() + text[1:]
        text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with a word-based Markov chain.")
    parser.add_argument("--order", type=int, default=3, help="Markov order (default: 3)")
    parser.add_argument("--tokens", type=int, default=55, help="Maximum generated tokens per sample")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for reproducible samples")
    args = parser.parse_args()

    if not CORPUS_PATH.exists() or not PROMPTS_PATH.exists():
        print("Error: expected corpus/training_text.txt and prompts.txt beside this script.")
        return 1

    tokens = tokenize(CORPUS_PATH.read_text(encoding="utf-8"))
    generator = MarkovTextGenerator(tokens, order=args.order)
    seeds = read_seeds(PROMPTS_PATH)
    if not seeds:
        print("Error: no valid seed words found in prompts.txt")
        return 1

    missing = [seed for seed in seeds if seed not in generator.positions]
    if missing:
        print("Error: these seed words are absent from the corpus: " + ", ".join(missing))
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove only previously generated sample_*.txt files to avoid stale outputs.
    for old_file in OUTPUT_DIR.glob("sample_*.txt"):
        old_file.unlink()

    for index, seed in enumerate(seeds, start=1):
        rng = random.Random(args.seed + index)
        sample = generator.generate(seed, token_limit=args.tokens, rng=rng)
        output_path = OUTPUT_DIR / f"sample_{index:02d}.txt"
        output_path.write_text(
            f"Sample {index:02d}\nSeed prompt: {seed}\nMarkov order: {args.order}\n\n{sample}\n",
            encoding="utf-8",
        )
        print(f"[{index:02d}/{len(seeds):02d}] {output_path.relative_to(ROOT)} — seed: {seed}")

    print(f"\nGenerated {len(seeds)} samples in {OUTPUT_DIR.relative_to(ROOT)}/")
    print("Run details: word-level Markov chain; no external Python packages required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
