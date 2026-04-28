# Contributing to edge-infer

Thanks for considering a contribution! edge-infer is small enough that
the bar is low: file an issue, send a PR, or DM me — all welcome.

## Source of truth

`edge_infer.py` is the entire code generator. ~1,250 lines of Python.
Read it top-to-bottom; the structure is:

1. ONNX loader + `--check` op-classification scanner.
2. Per-tensor symmetric INT8 quantization helpers.
3. Weight-name sanitizer.
4. `ShapeTracker` (propagates tensor shapes through the graph).
5. `PredictEmitter` (walks the graph and emits Rust code).
6. CLI entry point.

The six `examples/*/generated/` crates are output of the generator,
not hand-written. **Don't edit them by hand.** Regenerate with:

```bash
uv run python edge_infer.py examples/<name>/<model>.onnx \
    --output examples/<name>/generated/ --quantize int8
```

## How to test a change

1. Run `--check` on every shipped example to confirm the compatibility
   scanner doesn't regress:

   ```bash
   for onnx in examples/*/[!_]*.onnx; do
       uv run python edge_infer.py --check "$onnx"
   done
   ```

2. Regenerate the canonical crate and cross-compile for Cortex-M4:

   ```bash
   uv run python edge_infer.py examples/mnist/mnist_cnn.onnx \
       --output /tmp/test_mnist/ --quantize int8
   cd /tmp/test_mnist
   cargo build --release --target thumbv7em-none-eabihf
   ```

3. (If you have QEMU): run the demo on the lm3s6965evb machine and
   check the predicted class against the test sample. See
   `examples/mnist/README.md` for the QEMU command.

4. Run the f32-vs-INT8 disagreement check on the full MNIST test set:

   ```bash
   uv run --extra train python scripts/eval_full_mnist.py
   ```

## Where to file what

- **New ONNX op needed** — open an issue from the
  [`op-request`](.github/ISSUE_TEMPLATE/op-request.yml) template. One
  issue per op so others can 👍 the ones they need.
- **My model didn't compile** — open an issue from the
  [`model-didnt-compile`](.github/ISSUE_TEMPLATE/model-didnt-compile.yml)
  template. The scanner output is enough; full repro is appreciated.
- **I tried it on real hardware** — open an issue from the
  [`hardware-test-report`](.github/ISSUE_TEMPLATE/hardware-test-report.yml)
  template. Even short reports help calibrate "what really fits."
- **Code change** — send a PR. Small focused PRs are easier to review
  than large ones; a single new op is a perfect first PR.

## Code style

- Python: `ruff` defaults. The codebase is plain stdlib + numpy + onnx;
  no async, no fancy abstractions, no decorators-as-frameworks. Keep it
  that way.
- Rust output: no `unsafe` unless there's a comment justifying it
  against [the array layout reference](https://doc.rust-lang.org/reference/type-layout.html#array-layout).
  Prefer safe `as_flattened_mut()` when possible.
- Commits: small, focused, present-tense. Bonus points for showing the
  generator output diff alongside the generator change.

## License

By contributing, you agree your contribution is licensed under both the
MIT and Apache-2.0 licenses (the project's dual-license).
