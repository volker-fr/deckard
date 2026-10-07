# Generates fp32 reference fixtures for `deckard verify` on Linux, mirroring
# the Core ML synthetic fixture shapes and feed format. The reference logits
# come from the pinned fp32 HuggingFace checkpoint itself, so both the Candle
# and ONNX Runtime native backends must reproduce them within the 0.002 score
# tolerance.

import argparse
import json
import math
from pathlib import Path

LENGTHS = (32, 128, 512)
STEPS = (11, 97)


def generate(path, lengths=LENGTHS, steps=STEPS):
    print(f"Generating fixtures from {path}")
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=config, torch_dtype=torch.float32
    )
    model.eval()

    cases = []
    with torch.inference_mode():
        for length in lengths:
            for step in steps:
                for padded in (False, True):
                    # Fully-wrapped CLS..SEP feeds as the native runtime expects.
                    # The "padded" variant zeroes an interior run (its masked
                    # positions never contribute) rather than appending after
                    # SEP, which both backends reject.
                    interior = [3 + index * step for index in range(length - 2)]
                    if padded:
                        interior[(length - 2) // 2] = 0
                    ids = [1] + interior + [2]
                    mask = [int(value != 0) for value in ids]
                    inputs = torch.tensor([ids], dtype=torch.int64)
                    attention = torch.tensor([mask], dtype=torch.int64)
                    logit = float(model(input_ids=inputs, attention_mask=attention).logits[0, 0])
                    cases.append({
                        "name": f"length{length}-values{step}-batch1" + ("-padded" if padded else ""),
                        "logit": logit,
                        "score": 1 / (1 + math.exp(-logit)),
                        "feed": {"input_ids": [ids], "attention_mask": [mask]},
                    })
                    print(f"  {cases[-1]['name']}: logit={logit:.6f}")
    return {"cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("fixtures.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True)
    result = generate(args.checkpoint)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()