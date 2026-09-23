"""
RunPod Serverless handler for TRELLIS.2 (microsoft/TRELLIS.2-4B).

Input (job["input"]):
    image_base64   : str, required. A base64-encoded image (no data: URI
                      prefix — strip that on the caller side if present).
    output_compression : str, optional, default "gzip". Pass "none" to get
                      raw base64 instead of gzip+base64 (matches the pattern
                      used by other RunPod 3D-generation workers, so a
                      caller already handling one is not surprised by the
                      other).
    simplify_target : int, optional. Passed to mesh.simplify(); the repo's
                      own example caps this at 16_777_216 (nvdiffrast's
                      internal limit) — we default to that ceiling and clamp
                      any caller-supplied value to it. This operates on the
                      O-Voxel structure itself, before export.
    decimation_target : int, optional, default 50_000. Passed to
                      o_voxel.postprocess.to_glb() — controls the vertex
                      count of the *final exported GLB mesh*. This is a
                      separate, much smaller number than simplify_target;
                      matches the repo's own official usage example.
    texture_size    : int, optional, default 2048. Texture resolution baked
                      into the exported GLB, also passed to to_glb().

Output:
    {"status": "success", "glb_gzip_base64": "..."}         (default)
    {"status": "success", "glb_base64": "..."}               (if output_compression="none")
    {"status": "error", "message": "..."}

Design notes:
    - The pipeline is loaded ONCE at module import time (outside handler()),
      not per-request. RunPod keeps a worker process alive between requests
      when min workers > 0, or for the lifetime of a cold-started worker
      while it services queued jobs — reloading an 8B-parameter pipeline on
      every single call would make every request pay the full model-load
      cost, which is the single biggest avoidable latency source in a
      serverless image-generation worker.
    - Output is gzip-compressed by default. A textured mesh easily exceeds a
      few MB, and RunPod (like most serverless platforms) charges/limits on
      response payload size; gzip typically cuts glTF/GLB binary payloads by
      30-50% since they contain a fair amount of repetitive structure.
    - We explicitly refuse to return a payload over a preventive size
      ceiling rather than letting a huge JSON blob potentially get truncated
      or rejected upstream with a less useful error.
    - pipeline.run(image)[0] returns a MeshWithVoxel object. It has
      .simplify() but NOT .export() — that was the bug that produced
      "'MeshWithVoxel' object has no attribute 'export'". Per the repo's
      own official example, exporting to GLB requires first converting via
      o_voxel.postprocess.to_glb(...), and it's THAT returned object which
      has .export(). See _export_glb() below.
"""

import base64
import gzip
import os
import tempfile
import traceback

import runpod

# --- Model load: happens once, at cold start, before the handler loop starts ---
print("[trellis2-worker] Loading TRELLIS.2-4B pipeline (cold start)...")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
_pipeline.cuda()

print("[trellis2-worker] Pipeline loaded and moved to GPU. Ready for jobs.")

# nvdiffrast's internal vertex-count ceiling; mirrors the value used in the
# model card's own minimal example (mesh.simplify(16_777_216)).
_MESH_SIMPLIFY_CEILING = 16_777_216

# Defaults for the GLB export step (o_voxel.postprocess.to_glb), matching
# the repo's own official usage example. decimation_target controls the
# *exported* mesh's vertex count — a much smaller number than
# simplify_target, which operates on the O-Voxel structure beforehand.
_DEFAULT_DECIMATION_TARGET = 50_000
_DEFAULT_TEXTURE_SIZE = 2048

# Same preventive payload ceiling pattern used by other RunPod 3D workers in
# this space: reject an oversized output explicitly rather than emitting a
# JSON blob the platform may truncate or reject with a less specific error.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024 - 64 * 1024  # 10 MiB - 64 KiB


def _decode_input_image(image_base64: str) -> Image.Image:
    """Decode a base64 image string into a PIL Image.

    Accepts a bare base64 string. If the caller forgot to strip a data URI
    prefix (e.g. "data:image/png;base64,..."), we strip it here rather than
    failing — that is the single most common integration mistake and costs
    nothing to handle defensively.
    """
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return Image.open(tmp_path).convert("RGBA")
    finally:
        os.unlink(tmp_path)


def _export_glb(
    mesh,
    simplify_target: int,
    decimation_target: int = _DEFAULT_DECIMATION_TARGET,
    texture_size: int = _DEFAULT_TEXTURE_SIZE,
) -> bytes:
    """Run mesh simplification, convert to a real exportable GLB, and
    return its bytes via a temp file.

    Two distinct steps, both required (this mirrors the repo's own official
    usage example — see the module docstring):
      1. mesh.simplify(simplify_target) — operates on the MeshWithVoxel /
         O-Voxel structure itself. MeshWithVoxel has this method but NOT
         .export().
      2. o_voxel.postprocess.to_glb(...) — converts that O-Voxel structure
         into an actual exportable mesh (cleaning, remeshing, UV unwrap,
         texture baking). The OBJECT THIS RETURNS is what has .export().
    """
    target = min(int(simplify_target), _MESH_SIMPLIFY_CEILING)
    mesh.simplify(target)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        glb.export(tmp_path, extension_webp=True)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def handler(job):
    """RunPod Serverless entry point. See module docstring for I/O contract."""
    try:
        job_input = job.get("input") or {}

        image_base64 = job_input.get("image_base64")
        if not image_base64:
            return {"status": "error", "message": "image_base64 is required in input"}

        compression = (job_input.get("output_compression") or "gzip").lower()
        simplify_target = job_input.get("simplify_target", _MESH_SIMPLIFY_CEILING)
        decimation_target = job_input.get("decimation_target", _DEFAULT_DECIMATION_TARGET)
        texture_size = job_input.get("texture_size", _DEFAULT_TEXTURE_SIZE)

        image = _decode_input_image(image_base64)

        with torch.inference_mode():
            outputs = _pipeline.run(image)

        if not outputs:
            return {"status": "error", "message": "Pipeline produced no output for this image"}

        mesh = outputs[0]
        glb_bytes = _export_glb(mesh, simplify_target, decimation_target, texture_size)

        if compression == "none":
            payload_bytes = base64.b64encode(glb_bytes)
            field_name = "glb_base64"
        else:
            payload_bytes = base64.b64encode(gzip.compress(glb_bytes))
            field_name = "glb_gzip_base64"

        if len(payload_bytes) > _MAX_OUTPUT_BYTES:
            return {
                "status": "error",
                "message": (
                    f"Generated GLB is too large to return "
                    f"({len(payload_bytes)} bytes, limit {_MAX_OUTPUT_BYTES}). "
                    "Try a lower decimation_target."
                ),
            }

        return {
            "status": "success",
            field_name: payload_bytes.decode("ascii"),
        }

    except Exception as e:  # noqa: BLE001 — a serverless handler must never
        # let an exception propagate past this point: RunPod expects a JSON
        # response either way, and a traceback string is far more actionable
        # to the caller (and to you, debugging a job from the dashboard)
        # than a bare "worker crashed" with no detail.
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})