# TRELLIS.2 — RunPod Serverless worker

Image-to-3D generation ([microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B))
packaged as a RunPod Serverless worker, called on demand from a chatbot.

## Why Serverless (not a Pod)

You pay only while a request is actively processing. With `Workers Min: 0`,
idle time between calls costs **$0** — the right shape for "call this only
when the chatbot needs it," as opposed to a Pod, which bills continuously
whether or not anyone is using it.

## 1. Build and publish the image

1. Push this repo to your own GitHub account.
2. GitHub Actions (`.github/workflows/build.yml`) builds and publishes
   automatically on every push to `main`, to:
   ```
   ghcr.io/<your-username>/<your-repo-name>:latest
   ghcr.io/<your-username>/<your-repo-name>:<commit-sha>
   ```
3. Make sure the resulting GHCR package is set to **public** (private
   packages need extra registry-auth configuration on the RunPod side).

The first build takes 20-40 minutes: it compiles native extensions
(flash-attn, nvdiffrast, cumesh, ...) from source and downloads the model
weights at build time, matching TRELLIS.2's own install instructions.

## 2. Deploy on RunPod

Serverless → New Endpoint:

| Setting | Value | Why |
|---|---|---|
| Container Image | `ghcr.io/<you>/<repo>:<commit-sha>` | Pin a specific SHA once you've confirmed a build works — not `:latest`, which can change under you |
| GPU | RTX 4090 (24GB) or better | TRELLIS.2 requires ≥24GB VRAM; validated by Microsoft on A100/H100 |
| Workers Min | **0** | Costs $0 at rest — this is the whole point |
| Workers Max | 2–3 | Caps concurrent cost if several users hit it at once |
| Idle Timeout | 60s | How long a warm worker waits for the next job before shutting down |
| Environment variables | none required | Model weights are baked into the image at build time |

## 3. Call it from your chatbot

```typescript
// lib/trellis2/client.ts
export async function generate3DModel(imageBase64: string): Promise<Buffer> {
  const response = await fetch(
    `https://api.runpod.ai/v2/${process.env.RUNPOD_TRELLIS2_ENDPOINT_ID}/runsync`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: {
          image_base64: imageBase64,
          output_compression: "gzip", // default; omit or set "none" for raw base64
        },
      }),
    }
  );

  const result = await response.json();

  if (result.output?.status !== "success") {
    throw new Error(result.output?.message ?? "TRELLIS.2 generation failed");
  }

  const gzipped = Buffer.from(result.output.glb_gzip_base64, "base64");
  const zlib = await import("node:zlib");
  return zlib.gunzipSync(gzipped); // raw .glb bytes, ready to save or forward
}
```

Wire this into an AI SDK `tool()` exactly like the pattern used for the
Daytona/Claude Code integration discussed earlier — call it only when the
user's request actually needs a 3D asset, not on every message.

### `/run` vs `/runsync`

- **`/runsync`** (used above): blocks until the job finishes, simplest for a
  chatbot tool call. Fine as long as your HTTP client's timeout tolerates a
  cold start plus generation time (expect low tens of seconds warm, more on
  a cold start).
- **`/run`**: returns immediately with a job ID; poll `/status/{id}`
  separately. Worth switching to if generation time makes `/runsync`
  impractical for your chatbot's request/response cycle.

## 4. Test locally before deploying (optional but recommended)

```bash
pip install runpod
python3 handler.py --test_input test_input.json
```

Replace the placeholder in `test_input.json` with a real base64-encoded PNG
first. This runs the actual handler function against a local GPU (or fails
fast and loudly if no GPU is available) — much faster to debug than
iterating via full RunPod deploys.

## Known limitations (inherited from TRELLIS.2 itself)

- **Small mesh holes possible.** TRELLIS.2 is a base model; occasional
  topological discontinuities in raw meshes are expected. For 3D-printing
  use, run a hole-filling pass afterward.
- **No aesthetic alignment.** The model is not RLHF-tuned to human style
  preference — output style follows the training distribution, not a
  particular "look."
- **Geographic licence restriction.** TRELLIS.2's community licence
  excludes the EU, UK, and South Korea. Confirm this is compatible with
  where you and your users are before shipping this in production.
