# AI-07 completed study and reproduction

The 8 October execution request superseded the preparation-only restriction. The bounded feasibility pilot, longer-final-training diagnostic and explicit modular reference are complete. The pilot and diagnostic fail the frozen competence gate. No causal-localization or confirmation stage is pending under this study.

Read the [final report](../01-Papers/AI-07-FEASIBILITY-REPORT.md) and [reproducibility guide](../REPRODUCIBILITY.md). `make check` verifies the preserved preparation inputs; `make integrity` verifies retained evidence. Fresh model execution uses the explicit commands in the reproduction guide and must write into a new directory.

All runs used the existing AI-05 CPU environment without modifying it. Keep one worker and one numerical thread, check available memory and disk before allocation, and preserve all failures. New output belongs on project storage. No GPU, model download, driver change, external scheduling or termination of other jobs is required.

A future scientific redesign would need competent composition before adaptation and a comparison that addresses transition difficulty. It should receive a separate protocol and evidence tree; the current results must remain intact. If a future extension uses the GPU, retain the portfolio's existing shared lock at `<coordination-folder>/2026-09-22/gpu.lock` relative to the project root and acquire it nonblocking.
