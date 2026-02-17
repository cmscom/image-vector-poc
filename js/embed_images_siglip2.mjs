/**
 * Generate SigLIP 2 image embeddings using Transformers.js (ONNX Runtime).
 *
 * Usage:
 *   node embed_images_siglip2.mjs --image-list <json_path> --output <json_path> [--dtype fp32] [--model <model_name>]
 *
 * Input JSON format: [{"id": "uuid", "file_path": "/absolute/path/to/image.jpg"}, ...]
 * Output JSON format: {"model": "...", "dtype": "...", "count": N, "embeddings": [{"id": "...", "embedding": [...]}]}
 */

import { AutoProcessor, SiglipVisionModel, RawImage } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const DEFAULT_MODEL = 'onnx-community/siglip2-base-patch16-224-ONNX';

// Parse CLI arguments
const { values: args } = parseArgs({
  options: {
    'image-list': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
    'model': { type: 'string', default: DEFAULT_MODEL },
  },
});

if (!args['image-list'] || !args['output']) {
  console.error('Usage: node embed_images_siglip2.mjs --image-list <json> --output <json> [--dtype fp32|fp16|q8|q4] [--model <model>]');
  process.exit(1);
}

const MODEL_NAME = args['model'];

/**
 * L2-normalize an embedding array (matches Python implementation).
 */
function l2Normalize(embedding) {
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  const clippedNorm = Math.max(norm, 1e-8);
  return embedding.map(val => val / clippedNorm);
}

async function main() {
  const dtype = args['dtype'];
  const imageList = JSON.parse(readFileSync(args['image-list'], 'utf-8'));

  console.error(`Model: ${MODEL_NAME}`);
  console.error(`dtype: ${dtype}`);
  console.error(`Images to process: ${imageList.length}`);

  // Load model and processor
  console.error('Loading model and processor...');
  const startLoad = performance.now();
  const processor = await AutoProcessor.from_pretrained(MODEL_NAME);
  const model = await SiglipVisionModel.from_pretrained(MODEL_NAME, { dtype });
  const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);
  console.error(`Model loaded in ${loadTime}s`);

  // Detect embedding dimension from first successful inference
  let embeddingDim = null;

  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < imageList.length; i++) {
    const { id, file_path } = imageList[i];

    try {
      // Load and preprocess image
      const image = await RawImage.read(file_path);
      const imageInputs = await processor(image);

      // Run vision model
      const { pooler_output } = await model(imageInputs);

      // Extract embedding and L2-normalize
      const rawEmbedding = Array.from(pooler_output.data);
      if (embeddingDim === null) {
        embeddingDim = rawEmbedding.length;
        console.error(`Embedding dimension: ${embeddingDim}`);
      }
      const embedding = l2Normalize(rawEmbedding);

      results.push({ id, embedding });
    } catch (err) {
      console.error(`Error processing image ${i} (${file_path}): ${err.message}`);
      // Push zeros as placeholder for failed images
      const dim = embeddingDim || 768;
      results.push({ id, embedding: new Array(dim).fill(0) });
    }

    // Progress every 10 images
    if ((i + 1) % 10 === 0 || i === imageList.length - 1) {
      const elapsed = ((performance.now() - startEmbed) / 1000).toFixed(1);
      const speed = ((i + 1) / parseFloat(elapsed)).toFixed(1);
      console.error(`  [${i + 1}/${imageList.length}] ${elapsed}s elapsed (${speed} img/s)`);
    }
  }

  const totalTime = ((performance.now() - startEmbed) / 1000).toFixed(1);

  // Write output
  const outputData = {
    model: MODEL_NAME,
    dtype,
    count: results.length,
    processing_time_seconds: parseFloat(totalTime),
    embeddings: results,
  };

  writeFileSync(args['output'], JSON.stringify(outputData));
  console.error(`Done. ${results.length} embeddings saved to ${args['output']} (${totalTime}s)`);
}

main().catch(err => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
