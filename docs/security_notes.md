# Security and Data Notes

- Local runs bind to `127.0.0.1`; Render runs bind to `0.0.0.0` behind Render's
  service URL.
- The training request schema accepts only SARSA (`S`), rejects extra fields,
  and enforces bounded numeric values before a subprocess can be started.
- All trainer and PyGame launcher commands use fixed executable/script paths,
  argument lists, and `shell=False`.
- Only one browser-started training process can run at once.
- Demonstration and artifact writes use the configured storage root, while
  generated episode filenames are sanitised.
- Demonstration JSONL is validated before warm-start training consumes it.
- SARSA checkpoints are NumPy arrays loaded with `allow_pickle=False`.
- Run IDs are backend generated; the frontend cannot choose an artifact path.

The hosted service is intended for a single trusted user. Before making it a
multi-user application, add authentication, CSRF/origin protections, rate
limiting, audit logging, and process-level authorization.
