"""Generate and verify deterministic Adaptive-SDF x4 parity fixtures."""

# ruff: noqa: E501, E701, E702

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from sdf_upscaler import adaptive_color_sdf as adaptive
from sdf_upscaler import main_adaptive_color_sdf_options, upscale_adaptive_sdf_x4
from sdf_upscaler.alpha_sdf import resize_alpha_main_x4
from sdf_upscaler.rgb_gate import _srgb_to_linear, resize_bounded_rgb_lanczos
from sdf_upscaler.sal_classic import (
    main_sal_classic_rgb_options,
    segment_guided_normalized_lanczos_rgb,
)
from sdf_upscaler.structure_fields import extract_structure_field
from sdf_upscaler.structure_fusion import edge_local_structure_fusion_x4

OUTPUT = Path("fixtures/parity/adaptive")
EXPECTED_SHA = "289a9a0f6444883306fee5d7599ce0af3d34244780515e25409bcac91fb38074"


def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def fixed() -> np.ndarray:
    h, w = 18, 22; yy, xx = np.mgrid[:h, :w]; rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = np.stack(((xx * 17 + yy * 3) % 256, (xx * 7 + yy * 19) % 256, (xx * 13 + yy * 11) % 256), axis=2).astype(np.uint8)
    mask = ((xx - 10.5) / 7.5) ** 2 + ((yy - 8.5) / 6.0) ** 2 <= 1.0; rgba[..., 3] = np.where(mask, 255, 0); rgba[3, 6:16, 3] = 128
    return rgba


def edge(hidden: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    value = np.zeros((7, 9, 4), np.uint8); value[..., :3] = (210, 55, 40); value[2:5, 2:7, 3] = 255; value[1, 2:7, 3] = 96; value[5, 2:7, 3] = 192; value[..., :3][value[..., 3] == 0] = hidden; return value


def opaque() -> np.ndarray: value = np.zeros((5, 7, 4), np.uint8); value[..., :3] = (20, 140, 220); value[..., 3] = 255; return value
def transparent() -> np.ndarray: value = opaque(); value[..., 3] = 0; return value
def line() -> np.ndarray: value = np.zeros((9, 11, 4), np.uint8); value[..., :3] = (20, 245, 110); value[4, 1:10, 3] = 255; return value
def odd() -> np.ndarray: value = edge(); return value[:5, :7]


def normalize(value: np.ndarray) -> tuple[np.ndarray, str]:
    array = np.asarray(value)
    if array.dtype.kind == "f": return np.ascontiguousarray(array.astype("<f4")), "little-endian float32"
    if array.dtype.kind in "bu": return np.ascontiguousarray(array.astype(np.uint8)), "uint8"
    return np.ascontiguousarray(array.astype("<i4")), "little-endian int32"


def fixed_arrays(rgba: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    final, diag = upscale_adaptive_sdf_x4(rgba)
    if sha(final.tobytes()) != EXPECTED_SHA: raise ValueError("adaptive fixed digest drift")
    alpha_lr = rgba[..., 3].astype(np.float32) / 255.0
    alpha_hr = resize_alpha_main_x4(alpha_lr)
    sal_base, _ = segment_guided_normalized_lanczos_rgb(
        rgba[..., :3],
        alpha_lr,
        alpha_hr,
        scale=4,
        **main_sal_classic_rgb_options(),
    )
    bounded = adaptive._bounded_visible_rgb(rgba, radius=16.0, feather=8.0)
    bounded_base = resize_bounded_rgb_lanczos(bounded)
    structure_field = extract_structure_field(bounded)
    structure_output, _ = edge_local_structure_fusion_x4(
        bounded,
        bounded_base,
        structure_field=structure_field,
        scale=4,
        max_residual=0.02,
        strength=1.0,
        reprojection_gate_enabled=False,
    )
    arrays = {
        "fixed/input_rgba_u8": rgba, "fixed/final_rgba_u8": final, "fixed/final_rgb_u8": final[..., :3], "fixed/canonical_alpha_hr_f32": alpha_hr,
        "fixed/sal_base_rgb_u8": sal_base, "fixed/sal_base_linear_f32": _srgb_to_linear(sal_base.astype(np.float32) / 255.0),
        "fixed/bounded_visible_rgb_u8": bounded, "fixed/bounded_lanczos_rgb_u8": bounded_base,
        "fixed/bounded_lanczos_linear_f32": _srgb_to_linear(bounded_base.astype(np.float32) / 255.0),
        "fixed/structure_features_hwc_f32": structure_field.features, "fixed/structure_edge_mask_u8": structure_field.edge_mask,
        "fixed/structure_contour_labels_i32": structure_field.contour_labels,
        "fixed/structure_output_rgb_u8": structure_output, "fixed/structure_output_linear_f32": _srgb_to_linear(structure_output.astype(np.float32) / 255.0),
        "fixed/sal_gate_hr_f32": diag.sal.gate_hr, "fixed/sal_condition_hr_f32": diag.sal.condition_hr, "fixed/sal_contour_support_hr_f32": diag.sal.contour_support_hr, "fixed/sal_residual_luma_hr_f32": diag.sal.residual_luma_hr, "fixed/sal_active_mask_hr_u8": diag.sal.active_mask_hr,
        "fixed/structure_candidate_hr_f32": diag.structure.structure_candidate_hr, "fixed/structure_residual_hr_f32": diag.structure.structure_residual_hr, "fixed/structure_gate_hr_f32": diag.structure.structure_gate_hr, "fixed/structure_reprojection_gain_lr_f32": diag.structure.reprojection_gain_lr, "fixed/structure_reprojection_gate_hr_f32": diag.structure.reprojection_gate_hr, "fixed/structure_applied_gate_hr_f32": diag.structure.applied_gate_hr, "fixed/structure_unconstrained_rgb_hr_u8": diag.structure.unconstrained_rgb_hr,
        "fixed/interior_gate_hr_f32": diag.structure_interior_gate_hr, "fixed/median_active_mask_hr_u8": diag.median_active_mask_hr, "fixed/median_residual_luma_hr_f32": diag.median_residual_luma_hr, "fixed/reprojection_gate_lr_f32": diag.reprojection_gate_lr, "fixed/reprojection_gate_hr_f32": diag.reprojection_gate_hr,
    }
    scalars = {"expected_final_rgba_sha256": EXPECTED_SHA, "lr_error_base": diag.lr_error_base, "lr_error_candidate_before": diag.lr_error_candidate_before, "lr_error_candidate_after": diag.lr_error_candidate_after, "lr_error_final": diag.lr_error_final, "global_reprojection_fallback": diag.global_reprojection_fallback, "final_not_worse_than_base": diag.lr_error_final <= diag.lr_error_base + 1e-9}
    return arrays, scalars


def build() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays, scalars = fixed_arrays(fixed()); cases = []
    variants = {"hidden_rgb_a": edge((0,0,0)), "hidden_rgb_b": edge((255,13,231)), "opaque": opaque(), "partial_alpha": edge(), "transparent": transparent(), "line_art": line(), "odd_dimensions": odd()}
    for name, rgba in variants.items():
        output, diag = upscale_adaptive_sdf_x4(rgba); arrays[f"cases/{name}/input_rgba_u8"] = rgba; arrays[f"cases/{name}/output_rgba_u8"] = output; cases.append({"id": name, "output_sha256": sha(output.tobytes()), "shape": list(output.shape), "fallback": diag.global_reprojection_fallback})
    supplied = np.full((20, 28), 0.37, np.float32); rgba = opaque(); output, _ = upscale_adaptive_sdf_x4(rgba, alpha_hr=supplied); arrays["cases/supplied_alpha/input_rgba_u8"] = rgba; arrays["cases/supplied_alpha/hr_alpha_f32"] = supplied; arrays["cases/supplied_alpha/output_rgba_u8"] = output; cases.append({"id":"supplied_alpha", "output_sha256":sha(output.tobytes()), "shape":list(output.shape), "alpha_preserved":bool(np.array_equal(output[...,3], np.rint(supplied*255).astype(np.uint8)) )})
    entries=[]; normalized={}
    for name, value in sorted(arrays.items()):
        array,dtype=normalize(value); normalized[name]=array; raw=array.tobytes(); entries.append({"id":name,"path":f"{name}.raw","shape":list(array.shape),"dtype":dtype,"byte_length":len(raw),"sha256":sha(raw)})
    manifest={"schema_version":1,"kind":"python_adaptive_sdf_x4_parity","pipeline_id":adaptive.ADAPTIVE_COLOR_SDF_PIPELINE_ID,"scale":4,"options":main_adaptive_color_sdf_options(),"versions":{x:importlib.metadata.version(x) for x in("numpy","pillow","scipy","scikit-image","sdf-upscaler")},"fixed_scalars":scalars,"absent_stages":{"backprojection_iteration_planes":"canonical routine retains these only as local loop state","premultiplied_reprojection_planes":"canonical routine does not return local base/candidate/guarded arrays"},"arrays":entries,"cases":cases}
    return manifest,normalized


def generate(output: Path) -> None:
    manifest,arrays=build(); output.mkdir(parents=True,exist_ok=True)
    for entry in manifest["arrays"]:
        path=output/entry["path"]; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(arrays[entry["id"]].tobytes())
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); verify(output)


def verify(output: Path) -> None:
    stored=json.loads((output/"manifest.json").read_text(encoding="utf-8")); regenerated,arrays=build()
    if stored != regenerated: raise ValueError("manifest differs from regenerated canonical output")
    for entry in stored["arrays"]:
        actual=(output/entry["path"]).read_bytes(); expected=arrays[entry["id"]].tobytes()
        if actual != expected or len(actual) != entry["byte_length"] or sha(actual) != entry["sha256"]: raise ValueError(f"fixture mismatch: {entry['id']}")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=OUTPUT); parser.add_argument("--verify",action="store_true"); args=parser.parse_args()
    if args.verify: verify(args.output)
    else: generate(args.output)
    return 0

if __name__=="__main__": raise SystemExit(main())
