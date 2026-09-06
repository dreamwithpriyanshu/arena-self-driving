# FastAPI Training Control Surface

The local FastAPI server starts the existing headless SARSA trainer as a child
process. It accepts only the `S` model label and bounded hyperparameters,
constructs a fixed argument list, and uses `shell=False`; client input never
becomes a shell command.

Run it from the project root after installing `requirements-native.txt`:

```powershell
python -m backend.main
```

It binds to `127.0.0.1` by default. Do not expose it to a network without
adding authentication, TLS, origin controls, and a review of the subprocess
authorization model.

The browser frontend is plain HTML, CSS, and JavaScript served by FastAPI.
This intentionally avoids a JavaScript build chain for a small local control
surface. Browser-started runs write their history below `artifacts/runs/`; the
`/runs` API also lists existing history files directly below `artifacts/`.

The **Drive and play** controls launch the existing local PyGame programs:
human driving records demonstrations, and agent play visualises the saved
SARSA checkpoint. These remain native windows because PyGame cannot be
embedded into a browser page. The API uses fixed script paths and `shell=False`
for those launches as well.
