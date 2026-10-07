# Exports the pinned fp32 Gradient checkpoint to a single dynamic-sequence
# model.onnx for the ONNX Runtime backend (deckard ... --onnx on Linux).
#
# The ONNX graph mirrors the Candle backend exactly: a
# DebertaV2ForSequenceClassification with one label, fp32 arithmetic, and the
# HF DebertaV2 masking/relative-position implementation that Candle replicates.

import argparse
import json
import tempfile
from pathlib import Path


def verify_checkpoint(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    if config.get("torch_dtype") != "float32":
        raise SystemExit("The ONNX export requires the pinned fp32 checkpoint.")
    if config.get("model_type") != "deberta-v2" or config.get("hidden_size") != 1024:
        raise SystemExit("Unexpected model configuration.")
    if len(config.get("id2label", {})) != 1:
        raise SystemExit("Unexpected label count for the Gradient ONNX export.")


def fold_relative_position_ifs(model):
    # Every DebertaV2 attention layer exports
    #   if key_len != query_len: relative_pos = build_relative_position(...)
    #   else: relative_pos = <shared relative_pos>
    # as an ONNX If. Self-attention always takes the else branch, but the two
    # branches yield rank 3 vs rank 4 tensors, so the If output has no static
    # rank and the OpenVINO GPU plugin refuses to compile the graph. Replace
    # each If with an Identity of its else-branch input and drop the dead
    # condition chains.
    from onnx import helper

    graph = model.graph
    nodes = []
    folded = 0
    for node in graph.node:
        if node.op_type != "If":
            nodes.append(node)
            continue
        else_branch = next(a.g for a in node.attribute if a.name == "else_branch")
        if len(else_branch.node) != 1 or else_branch.node[0].op_type != "Identity":
            raise SystemExit(f"Unexpected If node {node.name}: else branch is not an Identity.")
        nodes.append(helper.make_node(
            "Identity", [else_branch.node[0].input[0]], list(node.output), name=node.name))
        folded += 1
    if folded == 0:
        raise SystemExit("Expected relative-position If nodes in the DebertaV2 export.")

    live = {output.name for output in graph.output}
    kept = []
    for node in reversed(nodes):
        if any(name in live for name in node.output):
            kept.append(node)
            live.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    del graph.value_info[:]


def export_model(checkpoint, output):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float32
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    clean = [1, 2]
    ids = torch.tensor([clean], dtype=torch.int64)
    mask = torch.ones_like(ids)
    # torch's ONNX serializer writes large weights as external-data files next
    # to its output; keep them in a scratch directory that is always removed.
    spill = tempfile.TemporaryDirectory(dir=output.parent, prefix=".onnx-export-")
    tmp = Path(spill.name) / "model.onnx"

    class TraceModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return model(input_ids, attention_mask=attention_mask).logits

    trace_model = TraceModel().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(trace_model, (ids, mask), check_trace=False)

    torch.onnx.export(
        traced,
        (ids, mask),
        str(tmp),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Load the graph and re-save a self-contained model.onnx so the backend can
    # load it from a single file (and from memory) without external data.
    import onnx

    model = onnx.load(tmp, load_external_data=True)
    spill.cleanup()
    fold_relative_position_ifs(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output), save_as_external_data=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model.onnx"))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing model.onnx whose pin no longer matches.")
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint)
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        output.unlink()
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True)
    export_model(args.checkpoint, output)
    print(f"Exported {output}")


if __name__ == "__main__":
    main()