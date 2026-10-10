# Local Model Layout

The training launcher never downloads model files. Put complete local Hugging
Face snapshots in this directory:

```text
model/
|-- qwen3_4B/
|   |-- config.json
|   |-- tokenizer_config.json
|   `-- model-*.safetensors
`-- all-MiniLM-L6-v2/
    |-- config.json
    |-- modules.json
    |-- tokenizer.json
    |-- model.safetensors
    `-- 1_Pooling/
        `-- config.json
```

For example, while network access is available:

```bash
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir model/qwen3_4B
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir model/all-MiniLM-L6-v2
```

Then validate that both configs exist:

```bash
test -f model/qwen3_4B/config.json
test -f model/all-MiniLM-L6-v2/config.json
```

`MODEL_ROOT`, `MODEL_PATH`, and `SENTENCE_TRANSFORMER_MODEL` may override these
locations. Model weights are ignored by Git and are not covered by this
repository's Apache License 2.0.
