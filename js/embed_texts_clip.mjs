/**
 * Generate CLIP-L text embeddings using Transformers.js (ONNX Runtime).
 *
 * Usage:
 *   node embed_texts_clip.mjs --texts <json> --output <json> [--dtype fp32]
 *
 * Input JSON: ["text1", "text2", ...]
 * Output JSON: {"model": "...", "dtype": "...", "count": N, "embeddings": [...]}
 */

import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const MODEL_NAME = 'Xenova/clip-vit-large-patch14';

const { values: args } = parseArgs({
  options: {
    'texts': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
  },
});

if (!args['texts'] || !args['output']) {
  console.error('Usage: node embed_texts_clip.mjs --texts <json> --output <json> [--dtype fp32]');
  process.exit(1);
}

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

  console.error('Loading model and tokenizer...');
  const startLoad = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
  const model = await CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, { dtype });
  const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);
  console.error(`Model loaded in ${loadTime}s`);

  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < texts.length; i++) {
    const text = texts[i];

    try {
      const textInputs = await tokenizer(text, {
        padding: 'max_length',
        truncation: true,
      });

      const output = await model(textInputs);

      // CLIP outputs text_embeds (projected) or last_hidden_state
      let embeddingData;
      if (output.text_embeds) {
        embeddingData = output.text_embeds.data;
      } else {
        const keys = Object.keys(output);
        console.error(`  Available output keys: ${keys.join(', ')}`);
        for (const key of ['text_embeds', 'pooler_output', 'last_hidden_state']) {
          if (output[key]) {
            embeddingData = output[key].data;
            console.error(`  Using key: ${key}`);
            break;
          }
        }
        if (!embeddingData) {
          throw new Error(`No embedding found in output keys: ${keys.join(', ')}`);
        }
      }

      const rawEmbedding = Array.from(embeddingData);
      const embedding = l2Normalize(rawEmbedding);
      results.push({ text, embedding });
    } catch (err) {
      console.error(`Error processing text ${i} ("${text.slice(0, 50)}"): ${err.message}`);
      results.push({ text, embedding: new Array(768).fill(0) });
    }

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
  console.error(`Done. ${results.length} embeddings saved to ${args['output']} (${totalTime}s)`);
}

main().catch(err => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
