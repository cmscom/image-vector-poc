/**
 * Debug SigLIP tokenizer: output token IDs and normalized text for comparison with Python.
 *
 * Usage:
 *   node debug_tokenizer.mjs --texts <json_path> --output <json_path>
 *
 * Input JSON: ["text1", "text2", ...]
 * Output JSON: [{ "text": "...", "token_ids": [...], "tokens": [...] }, ...]
 */

import { AutoTokenizer } from '@huggingface/transformers';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const MODEL_NAME = 'Xenova/siglip-base-patch16-224';

const { values: args } = parseArgs({
  options: {
    'texts': { type: 'string' },
    'output': { type: 'string' },
  },
});

if (!args['texts'] || !args['output']) {
  console.error('Usage: node debug_tokenizer.mjs --texts <json> --output <json>');
  process.exit(1);
}

async function main() {
  const texts = JSON.parse(readFileSync(args['texts'], 'utf-8'));

  console.error(`Model: ${MODEL_NAME}`);
  console.error(`Texts: ${texts.length}`);

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);

  const results = [];

  for (const text of texts) {
    // Tokenize with same settings as embed_texts.mjs
    const encoded = await tokenizer(text, {
      padding: 'max_length',
      truncation: true,
    });

    const inputIds = Array.from(encoded.input_ids.data).map(Number);

    // Also get tokens without padding for comparison
    const encodedNoPad = await tokenizer(text, {
      padding: false,
      truncation: true,
    });
    const tokenIdsNoPad = Array.from(encodedNoPad.input_ids.data).map(Number);

    // Decode individual tokens
    const tokens = [];
    for (const id of tokenIdsNoPad) {
      const decoded = tokenizer.decode([id], { skip_special_tokens: false });
      tokens.push(decoded);
    }

    results.push({
      text,
      token_ids_no_pad: tokenIdsNoPad,
      token_ids_padded: inputIds,
      tokens,
      num_tokens: tokenIdsNoPad.length,
    });
  }

  writeFileSync(args['output'], JSON.stringify(results, null, 2));
  console.error(`Done. ${results.length} results saved to ${args['output']}`);
}

main().catch(err => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
