# Training data

SenseSpec training recipes use text conversations by default. Prepare a local
JSONL file and set `TRAIN_DATA_PATH` to its absolute path when launching training.
See the [training guides](../README.md) for model setup and launch commands.

## Text conversations

Each line is a JSON record with a `conversations` array. Messages use `role` and
string-valued `content` fields:

```json
{"conversations":[{"role":"user","content":"Explain how a plant uses sunlight."},{"role":"assistant","content":"Plants use sunlight to turn water and carbon dioxide into sugars, storing energy that supports their growth and releasing oxygen in the process."}]}
```

The whole conversation provides context. The recipes supervise only the final
assistant answer and use the target checkpoint's chat template. Answers with
fewer supervised tokens than `dataset.min_loss_tokens` are filtered out.

Set `training.max_seq_length` and `inference.vllm.extra_args.max_model_len` to
the same context limit. Tokenized inputs must be shorter than that limit;
split or remove conversations that are too long. For large datasets,
`dataset.tokenization_batch_size` can reduce temporary preprocessing memory.

The bundled [text JSONL](sample_sensenova_loop_text.jsonl) illustrates the format.
Use your own conversations for training.

## Optional image input

Image input requires a dataset containing image references and explicitly
enabling the vision encoder. After setting up the environment, model path,
and GPUs as described in the [DFlash2 guide](../SenseNova-Loop-9B-Preview-dflah2/README.md):

```bash
export TRAIN_DATA_PATH="/absolute/path/to/image-conversations.jsonl"
export OUTPUT_DIR="$PWD/outputs/dflash2-images-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$OUTPUT_DIR/logs"
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh \
  inference.vllm.enable_multimodal=true \
  inference.vllm.extra_args.language_model_only=false
```

These options also apply to the other training recipes. Keep
`dataset.renderer=sensenova-loop` for the checkpoint's native conversation format.

For an image message, `content` is an ordered list of image and text items.
The assistant answer remains text:

```json
{"conversations":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///shared/images/example.png"}},{"type":"text","text":"Describe this image."}]},{"role":"assistant","content":"Write a complete reference answer describing the image here, including the details the draft model should learn to reproduce."}]}
```

Add more `image_url` items for multiple images. Supported image sources include
HTTP(S) URLs, `data:image/...;base64,...` URLs, `file:///absolute/path` URLs,
and file paths. Absolute paths or embedded data URLs are convenient for
multi-node training. Every inference node must be able to access each image;
relative paths are resolved from the inference process's working directory.

With image input enabled, the recipes allow up to four images per request and
set a maximum of 1048576 processed pixels per image. Video input is disabled.
Adjust these settings through:

- `inference.vllm.extra_args.limit_mm_per_prompt.image`
- `inference.vllm.extra_args.mm_processor_kwargs.max_pixels`

Expanded image tokens count toward the context limit. Resize images or split
conversations that exceed it. Only the assistant answer is supervised; image
and user tokens are excluded. The target model and vision encoder remain frozen.

The bundled [image JSONL](sample_sensenova_loop_multimodal.jsonl) illustrates
single-image and multiple-image messages with embedded image data.
