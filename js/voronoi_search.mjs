/**
 * Voronoi partition search using pre-computed pivots (pure JS).
 *
 * Loads pivot centroids and face embeddings exported from Python (NB102),
 * performs pivot selection + cosine rerank, and outputs results as JSON.
 *
 * Embeddings are loaded from binary Float32 format (+ face_ids JSON) to
 * avoid Node.js string length limits with large JSON files.
 *
 * Usage:
 *   node voronoi_search.mjs \
 *     --pivots <voronoi_pivots.json> \
 *     --embeddings-bin <voronoi_embeddings.bin> \
 *     --face-ids <voronoi_face_ids.json> \
 *     --assignments <voronoi_assignments.json> \
 *     --verification <voronoi_verification.json> \
 *     --output <results.json> \
 *     --top-k 3
 */

import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

const { values: args } = parseArgs({
  options: {
    'pivots':         { type: 'string' },
    'embeddings-bin': { type: 'string' },
    'face-ids':       { type: 'string' },
    'assignments':    { type: 'string' },
    'verification':   { type: 'string' },
    'output':         { type: 'string' },
    'top-k':          { type: 'string', default: '3' },
  },
});

if (!args.pivots || !args['embeddings-bin'] || !args['face-ids'] || !args.assignments || !args.verification || !args.output) {
  console.error('Usage: node voronoi_search.mjs --pivots <json> --embeddings-bin <bin> --face-ids <json> --assignments <json> --verification <json> --output <json> [--top-k 3]');
  process.exit(1);
}

const topK = parseInt(args['top-k'], 10);

// --- Load data ---
console.log('Loading pivots...');
const pivotsData = JSON.parse(readFileSync(args.pivots, 'utf-8'));
const centroids = pivotsData.centroids; // Array of 256 x 512-D arrays
const DIM = pivotsData.dim;

console.log('Loading face IDs...');
const faceIds = JSON.parse(readFileSync(args['face-ids'], 'utf-8'));

console.log('Loading embeddings (binary Float32)...');
const embBuf = readFileSync(args['embeddings-bin']);
const embFloat32 = new Float32Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 4);
const nFaces = faceIds.length;

// Build face_id -> index map for fast lookup
const faceIdToIdx = new Map();
for (let i = 0; i < nFaces; i++) {
  faceIdToIdx.set(faceIds[i], i);
}

// Helper: get embedding by face index (returns a view into the Float32Array)
function getEmbedding(idx) {
  const offset = idx * DIM;
  return embFloat32.subarray(offset, offset + DIM);
}

console.log('Loading assignments...');
const assignmentsData = JSON.parse(readFileSync(args.assignments, 'utf-8'));
const assignments = assignmentsData.assignments;

console.log('Loading verification data...');
const verificationData = JSON.parse(readFileSync(args.verification, 'utf-8'));

console.log(`Loaded: ${pivotsData.n_pivots} pivots, ${nFaces} embeddings (${DIM}-D), ${verificationData.queries.length} queries`);
console.log(`Search config: top_k=${topK}`);

// --- Build pivot-to-faces index ---
const pivotToFaces = {};
for (const [faceId, pivotIds] of Object.entries(assignments)) {
  for (const pid of pivotIds) {
    if (!pivotToFaces[pid]) pivotToFaces[pid] = [];
    pivotToFaces[pid].push(faceId);
  }
}

// --- Utility functions ---
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function dotProductF32(f32arr, jsArr) {
  // Dot product between Float32Array subarray and JS number array
  let sum = 0;
  for (let i = 0; i < f32arr.length; i++) {
    sum += f32arr[i] * jsArr[i];
  }
  return sum;
}

function selectTopPivots(queryEmb, centroids, topK) {
  const sims = centroids.map((c, i) => ({ idx: i, sim: dotProduct(queryEmb, c) }));
  sims.sort((a, b) => b.sim - a.sim);
  return sims.slice(0, topK).map(s => s.idx);
}

// --- Run search for all queries ---
const results = [];
const t0 = Date.now();

for (const queryInfo of verificationData.queries) {
  const queryFaceId = queryInfo.query_face_id;
  const queryIdx = faceIdToIdx.get(queryFaceId);
  const queryEmbF32 = getEmbedding(queryIdx);
  // Convert to JS array for pivot selection (centroids are JS arrays)
  const queryEmb = Array.from(queryEmbF32);

  // Step 1: Select top-k pivots
  const topPivotIds = selectTopPivots(queryEmb, centroids, topK);

  // Step 2: Gather candidates from selected pivots
  const candidateSet = new Set();
  for (const pid of topPivotIds) {
    const faces = pivotToFaces[pid];
    if (faces) {
      for (const fid of faces) {
        candidateSet.add(fid);
      }
    }
  }
  // Remove self
  candidateSet.delete(queryFaceId);

  // Step 3: Cosine rerank using binary embeddings
  const candidates = Array.from(candidateSet);
  const scored = candidates.map(fid => {
    const idx = faceIdToIdx.get(fid);
    const emb = getEmbedding(idx);
    return {
      face_id: fid,
      score: dotProductF32(emb, queryEmb),
    };
  });
  scored.sort((a, b) => b.score - a.score);

  // Top-30 result
  const top30 = scored.slice(0, 30);

  results.push({
    query_face_id: queryFaceId,
    pivot_ids: topPivotIds,
    n_candidates: candidates.length,
    top30,
  });
}

const elapsed = Date.now() - t0;
console.log(`Search completed: ${results.length} queries in ${elapsed}ms (${(elapsed / results.length).toFixed(1)}ms/query)`);

// --- Compare with Python brute-force results ---
let top30ExactMatchCount = 0;
let top30OverlapSum = 0;
let scoreAbsDiffSum = 0;
let scoreAbsDiffCount = 0;

for (let qi = 0; qi < results.length; qi++) {
  const jsResult = results[qi];
  const pyBf = verificationData.bf_top30[qi];

  // Top-30 face ID overlap
  const jsTop30Ids = new Set(jsResult.top30.map(r => r.face_id));
  const pyTop30Ids = new Set(pyBf.top30.map(r => r.face_id));

  let overlap = 0;
  for (const fid of jsTop30Ids) {
    if (pyTop30Ids.has(fid)) overlap++;
  }
  top30OverlapSum += overlap;

  // Check if all 30 match exactly (same order)
  const jsIds = jsResult.top30.map(r => r.face_id);
  const pyIds = pyBf.top30.map(r => r.face_id);
  if (jsIds.length === pyIds.length && jsIds.every((id, i) => id === pyIds[i])) {
    top30ExactMatchCount++;
  }

  // Score difference for common items
  const pyScoreMap = {};
  for (const item of pyBf.top30) {
    pyScoreMap[item.face_id] = item.score;
  }
  for (const item of jsResult.top30) {
    if (pyScoreMap[item.face_id] !== undefined) {
      scoreAbsDiffSum += Math.abs(item.score - pyScoreMap[item.face_id]);
      scoreAbsDiffCount++;
    }
  }
}

const nQueries = results.length;
const avgOverlap = top30OverlapSum / nQueries;
const avgScoreDiff = scoreAbsDiffCount > 0 ? scoreAbsDiffSum / scoreAbsDiffCount : 0;

console.log('\n=== Python vs JS Comparison ===');
console.log(`Top-30 ID overlap (mean): ${avgOverlap.toFixed(1)} / 30 (${(avgOverlap / 30 * 100).toFixed(1)}%)`);
console.log(`Top-30 exact order match: ${top30ExactMatchCount} / ${nQueries} (${(top30ExactMatchCount / nQueries * 100).toFixed(1)}%)`);
console.log(`Score abs diff (mean): ${avgScoreDiff.toExponential(4)}`);

// --- Save results ---
const output = {
  config: {
    n_pivots: pivotsData.n_pivots,
    top_k: topK,
    n_queries: nQueries,
  },
  comparison: {
    top30_id_overlap_mean: avgOverlap,
    top30_id_overlap_pct: avgOverlap / 30 * 100,
    top30_exact_order_match: top30ExactMatchCount,
    top30_exact_order_match_pct: top30ExactMatchCount / nQueries * 100,
    score_abs_diff_mean: avgScoreDiff,
  },
  results,
};

writeFileSync(args.output, JSON.stringify(output, null, 2));
console.log(`\nResults saved to: ${args.output}`);
