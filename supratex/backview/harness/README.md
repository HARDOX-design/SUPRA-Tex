# Back-view generation harness

This optional harness turns a validated back-camera normal pack and a front RGB
reference into an audited back RGBA observation. It runs **GPT-6 Astra** with
`reasoning_effort="ultra"` through Codex. The worker must have the built-in
`image_gen` tool. Astra orchestrates generation and visual review; the image
synthesis model behind the tool is not disclosed. A plain text/vision API is
insufficient. Missing capabilities fail explicitly, without model fallback.

The local image helpers use Pillow, NumPy, and SciPy, already included in the
project dependencies. Codex authentication and the required model/tool access
must be available in the execution environment.

## Prepare and run

The input is a validated `supratex.back_view_camera.v4` camera pack
containing `front_rgb.png`, `back_normal.png`, `back_mask.png`, and
`cameras.json`. The pack must retain its geometry and camera source bindings.
Camera metadata and images are copied byte for byte. Placeholder cameras and
masks inferred from normal colors are not used.

Regenerate camera packs made with earlier versions from the original inputs
using the current tools before running the harness.

```bash
python -m supratex.backview.harness prepare --pack-dir outputs/example/backview --run-root outputs/example/backview_generation
python -m supratex.backview.harness run --run-root outputs/example/backview_generation
python -m supratex.backview.harness run --run-root outputs/example/backview_generation --execute --workers 1
python -m supratex.backview.harness audit --run-root outputs/example/backview_generation
```

The first `run` command prints a plan and makes no model calls. `--execute`
starts real Codex workers, with at most three workers. Use `--case CASE_ID` to
select cases. Calling `prepare_run` repeatedly with different case IDs adds
cases to one run; existing inputs and prompt snapshots cannot be replaced.

Each worker sees only the back normal and front reference, in that order.
All object categories use the same saved English prompt. A fresh correction
uses the fixed base-plus-refinement prompt and the original inputs. Workers
preserve all attempts, actual tool returns, prompts, and visual reviews.
Terminal policy refusals and missing capabilities are recorded as failures.

Completed cases are reused on resume. A missing or stale review triggers
review-only recovery without a new generation. Visually rejected results
remain rejected; request a correction explicitly with `--retry-review`.
A failed worker cannot pass merely because an older image exists.

## Artifacts and acceptance

Each `results/CASE_ID/` contains immutable raw attempts and their prompt/tool
records, the selected `raw.png`, provenance, review, numbered worker invocation
and event logs, and worker completion records. Postprocessing adds:

- `back_rgba.png`: 1024-square observation with alpha exactly equal to the
  original renderer mask.
- `back_rgb.png`: the same observation composited once over white.
- `overlay.png`: generated geometry overlaid on the normal guide.
- `alignment.json`: hashes, framing transform, independent silhouette metrics,
  and brightness diagnostics.
- `validation.json`: postprocessing status and verification limits.

The report at `report/index.html` is offline and in English. It hides stale
processed images and links complete JSON audit records. `validate_result`
accepts a result only when all recorded inputs, prompts, attempts, outputs,
worker completion, and the current `acceptable` visual review agree. A
completed generation with `needs_review` or `needs_regeneration` remains
unaccepted for inference. Logs establish provenance and detect stale or changed
artifacts; local records are not cryptographic attestations of a remote service.

Postprocessing standardizes the canvas and, only when native alpha is
meaningful, estimates a uniform scale and translation. Scale is limited to
0.65–1.35 and translation to 15% of each canvas dimension. It does not rotate,
flip, deform individual parts, inpaint texture, or adjust brightness. The
original target mask is then applied. Matching final alpha proves the mask
constraint, **not** internal surface correspondence. The latter remains a
visual review limitation and is always recorded as unproven.

New generation creates new conditioning images and must be recorded as a new run.
