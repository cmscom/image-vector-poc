/**
 * Generate SigLIP text embeddings with optional preprocessing.
 *
 * Usage:
 *   node embed_texts_preprocessed.mjs --texts <json> --output <json> [--dtype fp32] [--preprocess none|lowercase|nfkc_lower|full_normalize]
 *
 * Preprocessing modes:
 *   none           - No preprocessing (default, same as embed_texts.mjs)
 *   lowercase      - text.toLowerCase()
 *   nfkc_lower     - NFKC normalize + lowercase
 *   full_normalize - lowercase + remove ASCII punctuation + whitespace normalize (Python canonicalize_text equivalent)
 *
 * Input JSON: ["text1", "text2", ...]
 * Output JSON: {"model": "...", "dtype": "...", "preprocess": "...", "count": N, "embeddings": [...]}
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
    'preprocess': { type: 'string', default: 'none' },
    'model-path': { type: 'string', default: '' },
  },
});

if (!args['texts'] || !args['output']) {
  console.error('Usage: node embed_texts_preprocessed.mjs --texts <json> --output <json> [--dtype fp32] [--preprocess none|lowercase|nfkc_lower|full_normalize] [--model-path <path>]');
  process.exit(1);
}

// ASCII punctuation characters (same as Python string.punctuation)
const ASCII_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';

/**
 * Preprocess text according to the specified mode.
 */
function preprocessText(text, mode) {
  switch (mode) {
    case 'none':
      return text;

    case 'lowercase':
      return text.toLowerCase();

    case 'nfkc_lower':
      return text.normalize('NFKC').toLowerCase();

    case 'full_normalize': {
      // Equivalent to Python SiglipTokenizer.canonicalize_text()
      let result = text.toLowerCase();
      // Remove ASCII punctuation
      result = result.split('').filter(c => !ASCII_PUNCTUATION.includes(c)).join('');
      // Normalize whitespace
      result = result.replace(/\s+/g, ' ').trim();
      return result;
    }

    default:
      console.error(`Unknown preprocess mode: ${mode}`);
      process.exit(1);
  }
}

/**
 * L2-normalize an embedding array.
 */
function l2Normalize(embedding) {
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  const clippedNorm = Math.max(norm, 1e-8);
  return embedding.map(val => val / clippedNorm);
}

async function main() {
  const dtype = args['dtype'];
  const preprocess = args['preprocess'];
  const modelPath = args['model-path'] || MODEL_NAME;
  const texts = JSON.parse(readFileSync(args['texts'], 'utf-8'));

  console.error(`Model: ${modelPath}`);
  console.error(`dtype: ${dtype}`);
  console.error(`Preprocess: ${preprocess}`);
  console.error(`Texts to process: ${texts.length}`);

  // Load model and tokenizer
  console.error('Loading model and tokenizer...');
  const startLoad = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(modelPath);
  const model = await SiglipTextModel.from_pretrained(modelPath, { dtype });
  const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);
  console.error(`Model loaded in ${loadTime}s`);

  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < texts.length; i++) {
    const originalText = texts[i];
    const processedText = preprocessText(originalText, preprocess);

    try {
      const textInputs = await tokenizer(processedText, {
        padding: 'max_length',
        truncation: true,
      });

      const { pooler_output } = await model(textInputs);

      const rawEmbedding = Array.from(pooler_output.data);
      const embedding = l2Normalize(rawEmbedding);

      results.push({ text: originalText, processed_text: processedText, embedding });
    } catch (err) {
      console.error(`Error processing text ${i} ("${originalText.slice(0, 50)}"): ${err.message}`);
      results.push({ text: originalText, processed_text: processedText, embedding: new Array(768).fill(0) });
    }

    if ((i + 1) % 10 === 0 || i === texts.length - 1) {
      const elapsed = ((performance.now() - startEmbed) / 1000).toFixed(1);
      console.error(`  [${i + 1}/${texts.length}] ${elapsed}s elapsed`);
    }
  }

  const totalTime = ((performance.now() - startEmbed) / 1000).toFixed(1);

  const output = {
    model: modelPath,
    dtype,
    preprocess,
    count: results.length,
    processing_time_seconds: parseFloat(totalTime),
    embeddings: results,
  };

  writeFileSync(args['output'], JSON.stringify(output));
  console.error(`Done. ${results.length} embeddings saved to ${args['output']} (${totalTime}s)`);
}

main().catch(err => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
