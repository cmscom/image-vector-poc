/**
 * Generate Jina CLIP v2 text embeddings using Transformers.js (ONNX Runtime).
 *
 * Usage:
 *   node embed_texts_jina_clip_v2.mjs --texts <json> --output <json> [--dtype fp32|fp16|q4|q4f16|int8]
 *
 * Input JSON: ["text1", "text2", ...]
 * Output JSON: {"model": "...", "dtype": "...", "count": N, "embeddings": [{"text": "...", "embedding": [...]}]}
 *
 * Tokenizer: XLMRobertaTokenizer (BPE) — handles CJK without <unk> fallback issues.
 * Recommended max_length: 77 (Jina CLIP convention; Gemma-style fix not needed for XLM-R).
 */

import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const DEFAULT_MODEL = 'jinaai/jina-clip-v2';
const MAX_LENGTH = 77;

const { values: args } = parseArgs({
  options: {
    'texts': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
    'model': { type: 'string', default: DEFAULT_MODEL },
  },
});

if (!args['texts'] || !args['output']) {
  console.error('Usage: node embed_texts_jina_clip_v2.mjs --texts <json> --output <json> [--dtype fp32|fp16|q4|q4f16|int8]');
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
  const texts = JSON.parse(readFileSync(args['texts'], 'utf-8'));

  console.error(`Model: ${MODEL}`);
  console.error(`dtype: ${dtype}`);
  console.error(`Texts: ${texts.length}`);

  const startLoad = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  const model = await AutoModel.from_pretrained(MODEL, { dtype });
  console.error(`Model loaded in ${((performance.now() - startLoad) / 1000).toFixed(1)}s`);
  console.error(`Tokenizer: ${tokenizer.constructor.name}`);

  const results = [];
  const startEmbed = performance.now();

  // バッチ推論: 全テキストを一度に
  try {
    const inp = await tokenizer(texts, { padding: true, truncation: true, max_length: MAX_LENGTH });
    const out = await model({ input_ids: inp.input_ids });
    const data = out.l2norm_text_embeddings.data;
    const dim = out.l2norm_text_embeddings.dims[1];
    console.error(`Embedding dim: ${dim}`);
    for (let i = 0; i < texts.length; i++) {
      const raw = Array.from(data).slice(i * dim, (i + 1) * dim);
      results.push({ text: texts[i], embedding: l2Normalize(raw) });
    }
  } catch (err) {
    console.error(`Batch failed: ${err.message}. Falling back to per-text.`);
    for (let i = 0; i < texts.length; i++) {
      try {
        const inp = await tokenizer([texts[i]], { padding: true, truncation: true, max_length: MAX_LENGTH });
        const out = await model({ input_ids: inp.input_ids });
        const raw = Array.from(out.l2norm_text_embeddings.data);
        results.push({ text: texts[i], embedding: l2Normalize(raw) });
      } catch (e) {
        console.error(`  text ${i} failed: ${e.message}`);
        results.push({ text: texts[i], embedding: new Array(1024).fill(0) });
      }
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
  console.error(`Done. ${results.length} text embeddings → ${args['output']} (${totalTime.toFixed(1)}s)`);
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
