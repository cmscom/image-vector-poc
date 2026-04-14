/**
 * Generate Jina CLIP v2 image embeddings using Transformers.js (ONNX Runtime).
 *
 * Usage:
 *   node embed_images_jina_clip_v2.mjs --image-list <json> --output <json> [--dtype fp32|fp16|q4|q4f16|int8]
 *
 * Input JSON: [{"id": "...", "file_path": "..."}, ...]
 * Output JSON: {"model": "...", "dtype": "...", "count": N, "embeddings": [{"id": "...", "embedding": [...]}]}
 *
 * Note: Uses unified JinaCLIPModel ONNX (image+text in one graph). Passing only pixel_values
 * makes it skip the text branch (transformers.js handles dummy inputs internally).
 */

import { AutoModel, AutoProcessor, RawImage } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const DEFAULT_MODEL = 'jinaai/jina-clip-v2';

const { values: args } = parseArgs({
  options: {
    'image-list': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
    'model': { type: 'string', default: DEFAULT_MODEL },
  },
});

if (!args['image-list'] || !args['output']) {
  console.error('Usage: node embed_images_jina_clip_v2.mjs --image-list <json> --output <json> [--dtype fp32|fp16|q4|q4f16|int8]');
  process.exit(1);
}

function l2Normalize(arr) {
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  const c = Math.max(norm, 1e-8);
  return arr.map(v => v / c);
}

async function main() {
  const dtype = args['dtype'];
  const MODEL = args['model'];
  const imageList = JSON.parse(readFileSync(args['image-list'], 'utf-8'));

  console.error(`Model: ${MODEL}`);
  console.error(`dtype: ${dtype}`);
  console.error(`Images: ${imageList.length}`);

  const startLoad = performance.now();
  const proc = await AutoProcessor.from_pretrained(MODEL);
  const model = await AutoModel.from_pretrained(MODEL, { dtype });
  console.error(`Model loaded in ${((performance.now() - startLoad) / 1000).toFixed(1)}s`);

  let embeddingDim = null;
  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < imageList.length; i++) {
    const { id, file_path } = imageList[i];
    try {
      const image = await RawImage.read(file_path);
      const { pixel_values } = await proc.image_processor(image);
      const out = await model({ pixel_values });
      const raw = Array.from(out.l2norm_image_embeddings.data);
      if (embeddingDim === null) {
        embeddingDim = raw.length;
        console.error(`Embedding dim: ${embeddingDim}`);
      }
      results.push({ id, embedding: l2Normalize(raw) });
    } catch (err) {
      console.error(`Error ${i} (${file_path}): ${err.message}`);
      results.push({ id, embedding: new Array(embeddingDim || 1024).fill(0) });
    }
    if ((i + 1) % 10 === 0 || i === imageList.length - 1) {
      const elapsed = (performance.now() - startEmbed) / 1000;
      const speed = (i + 1) / elapsed;
      console.error(`  [${i + 1}/${imageList.length}] ${elapsed.toFixed(1)}s elapsed (${speed.toFixed(2)} img/s)`);
    }
  }

  const totalTime = (performance.now() - startEmbed) / 1000;
  writeFileSync(args['output'], JSON.stringify({
    model: MODEL,
    dtype,
    count: results.length,
    processing_time_seconds: totalTime,
    embeddings: results,
  }));
  console.error(`Done. ${results.length} embeddings → ${args['output']} (${totalTime.toFixed(1)}s)`);
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
