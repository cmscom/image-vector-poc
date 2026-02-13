/**
 * Generate SigLIP text embeddings using Transformers.js (ONNX Runtime).
 *
 * Usage:
 *   node embed_texts.mjs --texts <json_path> --output <json_path> [--dtype fp32]
 *
 * Input JSON format: ["text1", "text2", ...]
 * Output JSON format: {"model": "...", "dtype": "...", "count": N, "embeddings": [{"text": "...", "embedding": [...]}]}
 */

import { AutoTokenizer, SiglipTextModel } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const MODEL_NAME = 'Xenova/siglip-base-patch16-224';

const { values: args } = parseArgs({
  options: {
    'texts': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
  },
});

if (!args['texts'] || !args['output']) {
  console.error('Usage: node embed_texts.mjs --texts <json> --output <json> [--dtype fp32|fp16|q8|q4]');
  process.exit(1);
}

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
  const texts = JSON.parse(readFileSync(args['texts'], 'utf-8'));

  console.error(`Model: ${MODEL_NAME}`);
  console.error(`dtype: ${dtype}`);
  console.error(`Texts to process: ${texts.length}`);

  // Load model and tokenizer
  console.error('Loading model and tokenizer...');
  const startLoad = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
  const model = await SiglipTextModel.from_pretrained(MODEL_NAME, { dtype });
  const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);
  console.error(`Model loaded in ${loadTime}s`);

  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < texts.length; i++) {
    const text = texts[i];

    try {
      // Tokenize with max_length padding (matching Python's SigLIP requirement)
      const textInputs = await tokenizer(text, {
        padding: 'max_length',
        truncation: true,
      });

      // Run text model
      const { pooler_output } = await model(textInputs);

      // Extract embedding and L2-normalize
      const rawEmbedding = Array.from(pooler_output.data);
      const embedding = l2Normalize(rawEmbedding);

      results.push({ text, embedding });
    } catch (err) {
      console.error(`Error processing text ${i} ("${text.slice(0, 50)}"): ${err.message}`);
      results.push({ text, embedding: new Array(768).fill(0) });
    }

    // Progress every 10 texts
    if ((i + 1) % 10 === 0 || i === texts.length - 1) {
      const elapsed = ((performance.now() - startEmbed) / 1000).toFixed(1);
      console.error(`  [${i + 1}/${texts.length}] ${elapsed}s elapsed`);
    }
  }

  const totalTime = ((performance.now() - startEmbed) / 1000).toFixed(1);

  const output = {
    model: MODEL_NAME,
    dtype,
    count: results.length,
    processing_time_seconds: parseFloat(totalTime),
    embeddings: results,
  };

  writeFileSync(args['output'], JSON.stringify(output));
  console.error(`Done. ${results.length} text embeddings saved to ${args['output']} (${totalTime}s)`);
}

main().catch(err => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
