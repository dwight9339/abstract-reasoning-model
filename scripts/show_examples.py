"""Print a few generated problems so you can eyeball what the model will see.

    python scripts/show_examples.py --count 3 --max-depth 4
"""

import argparse
import random

from arm.data import GraphConfig, Vocab, generate, to_text

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=3)
parser.add_argument("--min-depth", type=int, default=1)
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = GraphConfig(min_depth=args.min_depth, max_depth=args.max_depth)
vocab = Vocab(cfg.num_ids)
rng = random.Random(args.seed)

for i in range(args.count):
    p = generate(cfg, rng)
    print(f"--- example {i}: depth={p.depth} label={p.label} edges={len(p.edges)} ---")
    print(to_text(p))
    print("tokens:", vocab.encode(p))
    print()
