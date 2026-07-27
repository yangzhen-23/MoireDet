# Checkpoint provenance

Each checkpoint must have a JSON sidecar manifest. By default, `model.pth`
uses the sidecar `model.pth.json`. The sidecar is acquisition evidence only:
it is ignored by Git and is not authorization to deserialize a checkpoint.

Loading requires two independent records to match exactly:

1. The local acquisition sidecar records the filename, source, retrieval date,
   provenance evidence, and SHA-256.
2. A reviewer must add the same complete record, including the authenticated
   SHA-256, to the committed `configs/trusted_checkpoints.json` repository
   trust store.

Copying or editing a sidecar is insufficient. The default trust store is
intentionally empty until an administrator/repository reviewer authenticates
an official checkpoint hash. Do not add a guessed, placeholder, or
self-computed hash to that store.

For the expected official file `PSENet_100_loss0.000000.pth`, the pinned
upstream `MoireDet/script/model_download.txt` records the author-provided
original link: https://drive.google.com/file/d/1QivNnHWaomJmUuBgueGzwtVooijsc_TH/view?usp=sharing
It is currently unavailable in this environment. This documentation records
the source only and does not claim a successful download.

The example manifest's all-zero SHA-256 is deliberately invalid and cannot
validate a real checkpoint. Replace it with the exact lowercase SHA-256 of the
received checkpoint file and record complete provenance evidence, then obtain
review for a matching committed trust-store entry.

Only these source types are accepted: `official_repository`, `author_direct`,
`author_team_direct`, `advisor_direct`, and `verified_mirror`.

PyTorch 1.10 `torch.load` is pickle-based and can execute code during
deserialization. The loader therefore checks the reviewed repository pin and
the immutable in-memory byte snapshot before calling it; only an authenticated
hash may be pinned.
