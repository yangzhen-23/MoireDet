# Checkpoint provenance

Each checkpoint must have a JSON sidecar manifest. By default, `model.pth`
uses the sidecar `model.pth.json`.

For the expected official file `PSENet_100_loss0.000000.pth`, the pinned
upstream `MoireDet/script/model_download.txt` records the author-provided
original link: https://drive.google.com/file/d/1QivNnHWaomJmUuBgueGzwtVooijsc_TH/view?usp=sharing
It is currently unavailable in this environment. This documentation records
the source only and does not claim a successful download.

The example manifest's all-zero SHA-256 is deliberately invalid and cannot
validate a real checkpoint. Replace it with the exact lowercase SHA-256 of the
received checkpoint file and record complete provenance evidence.

Only these source types are accepted: `official_repository`, `author_direct`,
`author_team_direct`, `advisor_direct`, and `verified_mirror`.

`torch.load` is pickle-based. It can execute code during deserialization, so
the loader validates source provenance and checkpoint bytes before calling it;
only use checkpoints from these trusted sources.
