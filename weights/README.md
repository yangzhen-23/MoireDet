# Checkpoint provenance

Each checkpoint must have a JSON sidecar manifest. By default, `model.pth`
uses the sidecar `model.pth.json`.

The example manifest's all-zero SHA-256 is deliberately invalid and cannot
validate a real checkpoint. Replace it with the exact lowercase SHA-256 of the
received checkpoint file and record complete provenance evidence.

Only these source types are accepted: `official_repository`, `author_direct`,
`author_team_direct`, `advisor_direct`, and `verified_mirror`.

`torch.load` is pickle-based. It can execute code during deserialization, so
the loader validates source provenance and checkpoint bytes before calling it;
only use checkpoints from these trusted sources.
