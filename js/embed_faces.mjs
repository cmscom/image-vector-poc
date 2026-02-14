/**
 * Generate ArcFace face embeddings using ONNX Runtime (Node.js).
 *
 * Uses the same w600k_r50.onnx model as InsightFace (Python) to produce
 * identical 512-dimensional face embeddings from pre-aligned 112x112 face crops.
 *
 * Usage:
 *   node embed_faces.mjs --face-list <json_path> --output <json_path> --model-path <onnx_path>
 *
 * Input JSON: [{"id": "face_id", "file_path": "/path/to/face_crop_112x112.png"}, ...]
 * Output JSON: {"model": "arcface/w600k_r50", "count": N, "embeddings": [{"id": "...", "embedding": [...]}]}
 */

import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const { values: args } = parseArgs({
  options: {
    'face-list': { type: 'string' },
    'output': { type: 'string' },
    'model-path': { type: 'string' },
  },
});

if (!args['face-list'] || !args['output'] || !args['model-path']) {
  console.error('Usage: node embed_faces.mjs --face-list <json> --output <json> --model-path <onnx>');
  process.exit(1);
}

function l2Normalize(embedding) {
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  const clippedNorm = Math.max(norm, 1e-8);
  return embedding.map(val => val / clippedNorm);
}

async function preprocessFace(filePath) {
  // Read image as 112x112 RGB raw pixels
  const { data, info } = await sharp(filePath)
    .resize(112, 112)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;

  // Convert HWC RGB to CHW float32, normalize: (pixel - 127.5) / 127.5
  const tensor = new Float32Array(3 * 112 * 112);
  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * channels + c;
        const dstIdx = c * height * width + h * width + w;
        tensor[dstIdx] = (data[srcIdx] - 127.5) / 127.5;
      }
    }
  }

  return new ort.Tensor('float32', tensor, [1, 3, 112, 112]);
}

async function main() {
  const faceList = JSON.parse(readFileSync(args['face-list'], 'utf-8'));
  const modelPath = args['model-path'];

  console.error(`Model: ${modelPath}`);
  console.error(`Faces to process: ${faceList.length}`);

  // Load ONNX model
  console.error('Loading ArcFace ONNX model...');
  const startLoad = performance.now();
  const session = await ort.InferenceSession.create(modelPath);
  const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);
  console.error(`Model loaded in ${loadTime}s`);

  // Log model input/output info
  console.error(`Input names: ${session.inputNames.join(', ')}`);
  console.error(`Output names: ${session.outputNames.join(', ')}`);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const results = [];
  const startEmbed = performance.now();

  for (let i = 0; i < faceList.length; i++) {
    const { id, file_path } = faceList[i];

    try {
      const inputTensor = await preprocessFace(file_path);
      const feeds = { [inputName]: inputTensor };
      const output = await session.run(feeds);

      const rawEmbedding = Array.from(output[outputName].data);
      const embedding = l2Normalize(rawEmbedding);
      results.push({ id, embedding });
    } catch (err) {
      console.error(`Error processing face ${i} (${file_path}): ${err.message}`);
      results.push({ id, embedding: new Array(512).fill(0) });
    }

    if ((i + 1) % 10 === 0 || i === faceList.length - 1) {
      const elapsed = ((performance.now() - startEmbed) / 1000).toFixed(1);
      const speed = ((i + 1) / parseFloat(elapsed)).toFixed(1);
      console.error(`  [${i + 1}/${faceList.length}] ${elapsed}s elapsed (${speed} face/s)`);
    }
  }

  const totalTime = ((performance.now() - startEmbed) / 1000).toFixed(1);

  const outputData = {
    model: 'arcface/w600k_r50',
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
