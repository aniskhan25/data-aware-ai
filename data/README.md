# data/

This directory holds **small manifests only**. Generated datasets never enter Git.

## Why the dataset is not in the repository

The tutorial dataset is deterministic: given a profile and a seed, generation
reproduces the same bytes and the same manifest, so committing tens of thousands of
images would add nothing but weight. It would also be a poor example — this
tutorial is about not putting large numbers of small files where they do not
belong.

Generate it instead:

```bash
python scripts/generate_dataset.py \
    --profile-config configs/datasets/balanced.yaml \
    --output "$TUTORIAL_ROOT/source" \
    --manifest "$TUTORIAL_ROOT/manifests/balanced.jsonl" \
    --workers 16
```

## manifests/

A manifest is JSON Lines, one record per sample:

```json
{"byte_size": 1192, "checksum": "b94605a6125bbb60", "class_id": 0,
 "estimated_decode_cost": 1024, "height": 32,
 "relative_path": "images/class_0000/s00000000.jpg",
 "sample_id": "s00000000", "width": 32}
```

Records are sorted by `sample_id` and written with sorted keys, so the file is
byte-identical for identical content and its hash is a reliable identity.

Every dataset representation reads the same manifest. That is what makes the layout
comparisons in Part III meaningful: loose files, a SquashFS image, and tar shards
are measured over identical records in an identical order. Comparison tools refuse
to compare runs whose `manifest_hash` differs, because those runs did not read the
same data.

`estimated_decode_cost` is a work proxy (pixel count), not measured seconds. Part VI
uses it to balance shards by estimated work rather than by sample count.

Manifests generated into this directory are gitignored. Commit one only if it is
small and a documented part of an example.
