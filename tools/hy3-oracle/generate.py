#!/usr/bin/env python3
"""Generate independent reduced-Hy3 vectors with pinned Transformers.

This development tool is deliberately outside the Rust dependency graph. It
requires an exact checkout of the commit recorded below and never downloads a
model or tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRANSFORMERS_COMMIT = "3e80155a968c1080f11b2710e8b31741ac5ab0ed"
TOKEN_IDS = (3, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_source(source: Path) -> None:
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != TRANSFORMERS_COMMIT:
        raise RuntimeError(
            f"Transformers checkout is {actual}, expected {TRANSFORMERS_COMMIT}"
        )
    sys.path.insert(0, str(source / "src"))


def load_bundle(bundle: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["transformers_commit"] != TRANSFORMERS_COMMIT:
        raise RuntimeError("weight bundle targets a different Transformers commit")
    gguf = (bundle / manifest["gguf_file"]).read_bytes()
    if sha256(gguf) != manifest["gguf_sha256"]:
        raise RuntimeError("reduced GGUF hash does not match its manifest")

    raw = np.memmap(bundle / manifest["weights_file"], dtype="<f4", mode="r")
    tensors: dict[str, np.ndarray] = {}
    for entry in manifest["tensors"]:
        start = entry["offset_bytes"] // 4
        end = start + entry["element_count"]
        values = np.asarray(raw[start:end], dtype=np.float32).copy()
        if sha256(values.astype("<f4", copy=False).tobytes()) != entry["sha256_f32le"]:
            raise RuntimeError(f"dequantized tensor hash mismatch: {entry['name']}")
        # GGUF ne[0] is contiguous. Reversing the shape produces the usual
        # framework row-major view without transposing payload bytes.
        framework_shape = tuple(reversed(entry["shape"]))
        tensors[entry["name"]] = values.reshape(framework_shape)
    return tensors, manifest


def build_model(tensors: dict[str, np.ndarray], oracle_config: dict[str, Any]):
    from transformers import HYV3Config, HYV3ForCausalLM, __version__

    if __version__ != "5.6.0":
        raise RuntimeError(f"loaded Transformers {__version__}, expected 5.6.0")
    config = HYV3Config(
        vocab_size=oracle_config["vocab_size"],
        hidden_size=oracle_config["hidden_size"],
        intermediate_size=oracle_config["intermediate_size"],
        num_hidden_layers=oracle_config["num_hidden_layers"],
        num_attention_heads=oracle_config["num_attention_heads"],
        num_key_value_heads=oracle_config["num_key_value_heads"],
        head_dim=oracle_config["head_dim"],
        max_position_embeddings=oracle_config["max_position_embeddings"],
        initializer_range=0.006,
        rms_norm_eps=oracle_config["rms_norm_eps"],
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        num_experts=oracle_config["num_experts"],
        num_experts_per_tok=oracle_config["num_experts_per_tok"],
        num_shared_experts=oracle_config["num_shared_experts"],
        moe_intermediate_size=oracle_config["moe_intermediate_size"],
        router_scaling_factor=oracle_config["router_scaling_factor"],
        enable_moe_fp32_combine=True,
        mlp_layer_types=["dense", "sparse"],
        output_router_logits=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": oracle_config["rope_theta"],
            "factor": oracle_config["rope_factor"],
            "original_max_position_embeddings": oracle_config["rope_original_context"],
            "beta_fast": oracle_config["rope_beta_fast"],
            "beta_slow": oracle_config["rope_beta_slow"],
        },
    )
    config._attn_implementation = "eager"
    model = HYV3ForCausalLM(config).to(dtype=torch.float32, device="cpu").eval()
    state = make_state_dict(tensors)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def make_state_dict(tensors: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": tensor(tensors["token_embd.weight"]),
        "model.norm.weight": tensor(tensors["output_norm.weight"]),
        "lm_head.weight": tensor(tensors["output.weight"]),
    }
    for layer in range(2):
        gguf = f"blk.{layer}"
        model = f"model.layers.{layer}"
        for suffix, target in [
            ("attn_norm.weight", "input_layernorm.weight"),
            ("attn_q.weight", "self_attn.q_proj.weight"),
            ("attn_q_norm.weight", "self_attn.q_norm.weight"),
            ("attn_k.weight", "self_attn.k_proj.weight"),
            ("attn_k_norm.weight", "self_attn.k_norm.weight"),
            ("attn_v.weight", "self_attn.v_proj.weight"),
            ("attn_output.weight", "self_attn.o_proj.weight"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ]:
            state[f"{model}.{target}"] = tensor(tensors[f"{gguf}.{suffix}"])
    for suffix, target in [
        ("ffn_gate.weight", "gate_proj.weight"),
        ("ffn_up.weight", "up_proj.weight"),
        ("ffn_down.weight", "down_proj.weight"),
    ]:
        state[f"model.layers.0.mlp.{target}"] = tensor(tensors[f"blk.0.{suffix}"])

    prefix = "model.layers.1.mlp"
    state[f"{prefix}.gate.weight"] = tensor(tensors["blk.1.ffn_gate_inp.weight"])
    state[f"{prefix}.e_score_correction_bias"] = tensor(tensors["blk.1.exp_probs_b"])
    gate = tensors["blk.1.ffn_gate_exps.weight"]
    up = tensors["blk.1.ffn_up_exps.weight"]
    state[f"{prefix}.experts.gate_up_proj"] = tensor(
        np.concatenate((gate, up), axis=1)
    )
    state[f"{prefix}.experts.down_proj"] = tensor(
        tensors["blk.1.ffn_down_exps.weight"]
    )
    for suffix, target in [
        ("ffn_gate_shexp.weight", "gate_proj.weight"),
        ("ffn_up_shexp.weight", "up_proj.weight"),
        ("ffn_down_shexp.weight", "down_proj.weight"),
    ]:
        state[f"{prefix}.shared_experts.{target}"] = tensor(
            tensors[f"blk.1.{suffix}"]
        )
    return state


def array(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )


class Capture:
    def __init__(self, model):
        self.model = model
        self.values: dict[str, np.ndarray] = {}
        self.active_layer = 0
        self.router_ids: list[int] = []
        self.handles = []
        self._install_hooks()
        self._install_rope_capture()

    def _save(self, name: str, value: torch.Tensor) -> None:
        self.values[name] = array(value).reshape(-1).copy()

    def _install_hooks(self) -> None:
        for layer_id, layer in enumerate(self.model.model.layers):
            prefix = f"block{layer_id}"
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    lambda _module, _args, layer_id=layer_id: setattr(
                        self, "active_layer", layer_id
                    )
                )
            )
            self.handles.append(
                layer.input_layernorm.register_forward_hook(
                    lambda _m, _i, output, prefix=prefix: self._save(
                        f"{prefix}.attention_normalized", output
                    )
                )
            )
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _m, _i, output, prefix=prefix: self._save(
                        f"{prefix}.values", output
                    )
                )
            )
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, inputs, prefix=prefix: self._save(
                        f"{prefix}.attention_context", inputs[0]
                    )
                )
            )
            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    lambda _m, _i, output, prefix=prefix: self._save(
                        f"{prefix}.attention_delta", output
                    )
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda _m, inputs, prefix=prefix: self._save(
                        f"{prefix}.attention_residual", inputs[0]
                    )
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda _m, _i, output, prefix=prefix: self._save(
                        f"{prefix}.ffn_normalized", output
                    )
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    lambda _m, _i, output, prefix=prefix: self._save(
                        f"{prefix}.ffn_delta", output
                    )
                )
            )
        sparse = self.model.model.layers[1].mlp
        self.handles.append(
            sparse.gate.register_forward_hook(self._capture_router)
        )
        self.handles.append(
            self.model.model.norm.register_forward_pre_hook(
                lambda _m, inputs: self._save("final.hidden", inputs[0])
            )
        )
        self.handles.append(
            self.model.model.norm.register_forward_hook(
                lambda _m, _i, output: self._save("final.normalized", output)
            )
        )

    def _capture_router(self, _module, _inputs, output) -> None:
        routed = sorted(int(value) for value in output[2].reshape(-1))
        if routed:
            self.router_ids = routed

    def _install_rope_capture(self) -> None:
        import transformers.models.hy_v3.modeling_hy_v3 as modeling

        original = modeling.apply_rotary_pos_emb

        def wrapped(q, k, cos, sin, unsqueeze_dim=1):
            rotated_q, rotated_k = original(q, k, cos, sin, unsqueeze_dim)
            prefix = f"block{self.active_layer}"
            self._save(f"{prefix}.queries", rotated_q)
            self._save(f"{prefix}.keys", rotated_k)
            return rotated_q, rotated_k

        modeling.apply_rotary_pos_emb = wrapped
        self._restore_rope = lambda: setattr(
            modeling, "apply_rotary_pos_emb", original
        )

    def reset(self) -> None:
        self.values.clear()
        self.router_ids.clear()

    def close(self) -> None:
        self._restore_rope()
        for handle in self.handles:
            handle.remove()


def run(model) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    capture = Capture(model)
    all_arrays: dict[str, np.ndarray] = {}
    steps = []
    cache = None
    try:
        with torch.inference_mode():
            for step, token_id in enumerate(TOKEN_IDS):
                capture.reset()
                output = model(
                    input_ids=torch.tensor([[token_id]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                if cache is None or cache.get_seq_length() != step + 1:
                    raise RuntimeError("official graph did not advance the KV cache")
                normalized = torch.from_numpy(
                    capture.values["block1.ffn_normalized"]
                ).reshape(1, -1)
                router = model.model.layers[1].mlp.gate
                router_logits = torch.nn.functional.linear(
                    normalized.float(), router.weight.float()
                )
                _, router_ids = torch.topk(
                    torch.sigmoid(router_logits)
                    + model.model.layers[1].mlp.e_score_correction_bias,
                    router.top_k,
                    dim=-1,
                    sorted=False,
                )
                capture.router_ids = sorted(
                    int(value) for value in router_ids.reshape(-1)
                )
                logits = array(output.logits[0, -1]).reshape(-1).copy()
                probabilities = array(torch.softmax(output.logits[0, -1], dim=-1))
                capture.values["final.logits"] = logits
                capture.values["final.probabilities"] = probabilities
                step_prefix = f"step{step}"
                for name, values in capture.values.items():
                    all_arrays[f"{step_prefix}.{name}"] = values
                steps.append(
                    {
                        "token_id": token_id,
                        "selected_experts": list(capture.router_ids),
                        "greedy_id": int(torch.argmax(output.logits[0, -1])),
                        "hashes": {
                            name: sha256(values.tobytes())
                            for name, values in sorted(capture.values.items())
                        },
                    }
                )
    finally:
        capture.close()
    return all_arrays, steps


def main() -> None:
    args = parse_args()
    verify_source(args.transformers_source.resolve())
    tensors, manifest = load_bundle(args.bundle.resolve())
    model = build_model(tensors, manifest["config"])
    # Transformers may install an available kernel implementation on first
    # use. Warm that dispatch before hooks so both recorded steps traverse the
    # same settled graph.
    with torch.inference_mode():
        model(input_ids=torch.tensor([[0]], dtype=torch.long), use_cache=False)
    arrays, steps = run(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    report = {
        "format": "lightbridge-transformers-hy3-oracle-v1",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "transformers_version": "5.6.0",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "weight_manifest_sha256": sha256(
            (args.bundle / "manifest.json").read_bytes()
        ),
        "gguf_sha256": manifest["gguf_sha256"],
        "token_ids": list(TOKEN_IDS),
        "steps": steps,
        "npz_sha256": sha256(args.output.with_suffix(".npz").read_bytes()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
