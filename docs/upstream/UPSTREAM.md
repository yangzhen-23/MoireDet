# Audited MoireDet upstream baseline

The existing `MoireDet/` directory is the official upstream source tree from
https://github.com/cong-yang/MoireDet at pinned commit
`afde899f3c3beee96160610ee450618136a38f7b`.

On 2026-07-27, `origin/main` exactly matched that commit.  No duplicate source
snapshot was created.  The runtime import roots are `MoireDet` and
`MoireDet/script`; the official sample is `MoireDet/script/00002423.png`.

The local runtime compatibility boundary consists of:

- `patches/0001-torchvision-load-state-dict-compat.patch`
- `patches/0002-disable-resnet-online-download.patch`

The first patch supports torchvision versions that no longer provide
`torchvision.models.utils.load_state_dict_from_url`.  The second prevents
`TripleBranchWithSpecificConv` from making an implicit ResNet-weight download
during construction.  These patches do not alter the target model's
computation after strict checkpoint loading, and leave the Performer layout
and batch-size behavior untouched.

The upstream repository showed no code license.  This record supplies
attribution and documents a local compatibility patch only; it does not grant
any license to the upstream code.

## Integrity records

Unpatched `MoireDet/lib/models/model.py` Git-object SHA-256:
`4a573e2112691b153af1b8a9ef904d6adfc75754e5dd48e6bd3b440086e3307c`.

Post-patch SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `MoireDet/lib/models/model.py` | `822CB5E091C3FC25F9C03B0A2D1D31194AEEE50A2CE18442AE01CD7F0300B120` |
| `MoireDet/lib/models/modules/resnet.py` | `35FBF70CA3AAEE0F8E5AA8794840D30206D377F45791856896D09EEFB211E332` |
| `MoireDet/lib/models/modules/resnet_dct.py` | `8C28FCEAE2D44BAB986F39E57663E2CF1268810CEF581037BBF1815D2F1F057A` |
