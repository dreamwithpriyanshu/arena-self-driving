# Security and Data Notes

- The local server binds to `127.0.0.1`; it is not a network service.
- The training request schema accepts only SARSA (`S`), rejects extra fields,
  and enforces bounded numeric values before a subprocess can be started.
- All trainer and PyGame launcher commands use fixed executable/script paths,
  argument lists, and `shell=False`.
- Only one browser-started training process can run at once.
- Demonstration and artifact writes are constrained to project `data/` and
  `artifacts/` directories. Generated episode filenames are sanitised.
- Demonstration JSONL is validated before warm-start training consumes it.
- SARSA checkpoints are NumPy arrays loaded with `allow_pickle=False`.
- Run IDs are backend generated; the frontend cannot choose an artifact path.

The browser control surface is designed for one local user. If remote access is
required, add authentication, CSRF/origin protections, TLS, audit logging, and
process-level authorization before changing the bind address.
