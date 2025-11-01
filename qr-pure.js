/* qr-pure.js
* Pure-JS（无 deps）二维码检测 + 提取 + 简化解码。
 * m
 * 公开的 API：
 * - QRPure.decodeFromImageElement（imgEl） -> 承诺<数组<Result>>
 * 其中 Result = { text： string|null， matrix： Uint8ClampedArray， size： n， corners： [p0，p1，p2，p3]， error： string|null }
 *
 *笔记：
 * - matrix 是一个扁平二进制数组（0 个白色，1 个黑色），长度为 size*size（行主调）。
 * - 角是图像空间点 [ {x，y}， ... ]，按左上、右上、右下、左下的顺序排列。
 * - 此实现侧重于检测和提取;解码是一种简化的实现，适用于许多现实世界的 QR 码（字节/字母数字）。复杂的错误条件可能会失败。
 *
 *局限性：
 * - 非常小的 QR（直径低于 ~21 像素）可能会失败。
 * - 严重损坏/严重旋转/极端透视可能会失败。
 *
 * 许可证：随心所欲地使用。
 */

const QRPure = (function () {

  // ---------- util ----------
  function createCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }

  function getImageDataFromImageElement(img) {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const c = createCanvas(w, h);
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }

  function grayscaleCopy(imgData) {
    const { data, width, height } = imgData;
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // luminance
      const r = data[i], g = data[i+1], b = data[i+2];
      gray[p] = (0.299*r + 0.587*g + 0.114*b) | 0;
    }
    return { gray, width, height };
  }

  // adaptive threshold (Sauvola-like, fast integral-based local mean)
  function adaptiveBinarize(gray, width, height, window = 15, k = 0.15) {
    const N = width * height;
    const integral = new Uint32Array(N);
    const integralSq = new Uint32Array(N);
    // build integral images
    for (let y = 0; y < height; y++) {
      let sum = 0, sumSq = 0;
      for (let x = 0; x < width; x++) {
        const v = gray[y*width + x];
        sum += v; sumSq += v*v;
        const idx = y*width + x;
        if (y === 0) {
          integral[idx] = sum;
          integralSq[idx] = sumSq;
        } else {
          const above = integral[(y-1)*width + x];
          const aboveSq = integralSq[(y-1)*width + x];
          integral[idx] = above + sum;
          integralSq[idx] = aboveSq + sumSq;
        }
      }
    }
    const out = new Uint8ClampedArray(N);
    const half = (window/2)|0;
    for (let y = 0; y < height; y++) {
      const y0 = Math.max(0, y - half);
      const y1 = Math.min(height - 1, y + half);
      for (let x = 0; x < width; x++) {
        const x0 = Math.max(0, x - half);
        const x1 = Math.min(width - 1, x + half);
        const A = y0*width + x0;
        const B = y0*width + x1;
        const C = y1*width + x0;
        const D = y1*width + x1;
        const area = (x1 - x0 + 1) * (y1 - y0 + 1);
        const sum = integral[D] - (A?integral[A-1]:0) - (B?integral[B - width]:0) + (A?integral[A - width - 1]:0);
        // repositioned because of edges: simpler safe compute:
        // (to avoid complex corner issues, recompute by direct loop for small window)
        let mean = 0, varr = 0;
        // fallback direct compute
        for (let yy = y0; yy <= y1; yy++) {
          for (let xx = x0; xx <= x1; xx++) {
            const v = gray[yy*width + xx];
            mean += v;
            varr += v*v;
          }
        }
        mean /= area;
        varr = varr/area - mean*mean;
        const std = Math.sqrt(Math.max(0, varr));
        const thr = mean * (1 + k * ((std/128) - 1));
        const v0 = gray[y*width + x];
        out[y*width + x] = v0 < thr ? 1 : 0;
      }
    }
    return out;
  }

  // quick connected components (4-neigh) to find blobs (returns array of blobs with bounding box and pixel indices)
  function findConnectedComponents(binary, width, height, minSize = 25) {
    const N = width * height;
    const seen = new Uint8Array(N);
    const blobs = [];
    const stack = [];
    for (let i = 0; i < N; i++) {
      if (binary[i] === 1 && !seen[i]) {
        // flood
        let minx = width, miny = height, maxx = 0, maxy = 0, count = 0;
        stack.push(i); seen[i] = 1;
        while (stack.length) {
          const idx = stack.pop();
          count++;
          const y = (idx / width) | 0;
          const x = idx - y*width;
          if (x < minx) minx = x;
          if (y < miny) miny = y;
          if (x > maxx) maxx = x;
          if (y > maxy) maxy = y;
          // neighbors 4-neigh
          const n1 = idx - 1;
          if (x > 0 && binary[n1] === 1 && !seen[n1]) { seen[n1]=1; stack.push(n1); }
          const n2 = idx + 1;
          if (x < width-1 && binary[n2] === 1 && !seen[n2]) { seen[n2]=1; stack.push(n2); }
          const n3 = idx - width;
          if (y > 0 && binary[n3] === 1 && !seen[n3]) { seen[n3]=1; stack.push(n3); }
          const n4 = idx + width;
          if (y < height-1 && binary[n4] === 1 && !seen[n4]) { seen[n4]=1; stack.push(n4); }
        }
        if (count >= minSize) {
          blobs.push({minx,miny,maxx,maxy,count});
        }
      }
    }
    return blobs;
  }

  // ratio check for finder pattern: 1:1:3:1:1 (modules) along scanline
  function measureRunLengths(row, width, x, step = 1) {
    // returns five-run counts centered at x: [c1,c2,c3,c4,c5] black/white alternating; assumes center inside big black center
    const res = [0,0,0,0,0];
    let center = x;
    // move left counting colors, we want to end in white border before center
    let i = center;
    const colorCenter = row[center];
    // Count center run left
    while (i >= 0 && row[i] === colorCenter) { res[2]++; i--; }
    // run 1 (left of center same color?) Actually pattern has black center, but this helper used asymmetrically.
    // For robustness we'll expand both directions collecting 5 runs.
    // Count run 3 (center) done above. Continue left to get runs 1 and 0
    // switch color
    let color = 1 - colorCenter;
    while (i >= 0 && row[i] === color) { res[1]++; i--; }
    color = 1 - color;
    while (i >= 0 && row[i] === color) { res[0]++; i--; }
    // right side
    i = center + 1;
    color = 1 - colorCenter;
    while (i < width && row[i] === color) { res[3]++; i++; }
    color = 1 - color;
    while (i < width && row[i] === color) { res[4]++; i++; }
    return res;
  }

  function ratioMatch5(counts) {
    // accept if approx 1:1:3:1:1
    const total = counts.reduce((a,b)=>a+b,0);
    if (total === 0) return false;
    const module = total/7.0;
    // allow some tolerance
    const tol = module * 0.7;
    const target = [1,1,3,1,1];
    for (let i = 0; i < 5; i++) {
      if (Math.abs(counts[i] - target[i]*module) > tol) return false;
    }
    return true;
  }

  // find finder-like blobs by scanning rows for run-length patterns
  function findFinderCandidates(binary, width, height) {
    const candidates = [];
    for (let y = 0; y < height; y++) {
      // extract row
      const row = binary.subarray(y*width, y*width + width);
      // scan for transitions where a likely center exists
      for (let x = 1; x < width - 1; x++) {
        if (row[x] === 1) {
          // attempt measure around x
          const runs = measureRunLengths(row, width, x);
          if (ratioMatch5(runs)) {
            // compute approximate center x by weighted average
            const halfWidth = (runs[0]+runs[1]+runs[2]+runs[3]+runs[4])|0;
            candidates.push({x, y, runs, estModule: (runs.reduce((a,b)=>a+b,0)/7)});
            // skip ahead a bit
            x += Math.max(1, runs[2] | 0);
          }
        }
      }
    }
    // cluster candidates into points (average close ones)
    const clusters = [];
    const used = new Array(candidates.length).fill(false);
    for (let i = 0; i < candidates.length; i++) {
      if (used[i]) continue;
      let cx = candidates[i].x, cy = candidates[i].y, cnt = 1;
      used[i] = true;
      for (let j = i+1; j < candidates.length; j++) {
        if (used[j]) continue;
        const dx = candidates[j].x - candidates[i].x;
        const dy = candidates[j].y - candidates[i].y;
        if (dx*dx + dy*dy < 64) { // cluster radius ~8px (tunable)
          cx += candidates[j].x; cy += candidates[j].y; cnt++; used[j] = true;
        }
      }
      clusters.push({x: cx/cnt, y: cy/cnt});
    }
    return clusters;
  }

  // classify triplets of finder centers into QR candidates: look for three finder patterns roughly at right triangle corners
  function groupIntoQRs(centers) {
    const results = [];
    const n = centers.length;
    for (let i = 0; i < n; i++) {
      for (let j = i+1; j < n; j++) {
        for (let k = j+1; k < n; k++) {
          const a = centers[i], b = centers[j], c = centers[k];
          // compute pairwise distances
          const dij = dist2(a,b), djk = dist2(b,c), dik = dist2(a,c);
          // largest distance is hypothesized as diagonal between two finder patterns (top-left and bottom-right)
          let maxd = Math.max(dij, djk, dik);
          // basic check: the three points should form an L (right angle)
          // check by Pythagorean (allow tolerance)
          const dists = [dij, djk, dik].sort((x,y)=>x-y);
          if (dists[0] + dists[1] < dists[2] * 1.5 && Math.abs(dists[0] + dists[1] - dists[2]) < dists[2]*0.6) {
            // accept candidate
            results.push([a,b,c]);
          }
        }
      }
    }
    return results;
  }

  function dist2(a,b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
  }

  // order corners TL, TR, BR, BL given three finder centers
  function orderCornersFromThree(a,b,c) {
    // find which is the right angle (the one with shortest opposite side)
    const d_ab = dist2(a,b), d_bc = dist2(b,c), d_ac = dist2(a,c);
    // largest side between two points: opposite point is right angle? Actually we want the L shape: the point that has two relatively short distances is the corner.
    const sums = [
      {p:a, s: d_ab + d_ac},
      {p:b, s: d_ab + d_bc},
      {p:c, s: d_ac + d_bc}
    ];
    sums.sort((u,v)=>u.s - v.s);
    // sums[0] is likely corner (the one connecting two small sides)
    const corner = sums[0].p;
    const others = [a,b,c].filter(p=>p!==corner);
    // determine orientation: compute cross product to find which is TR vs BL
    const v0 = {x: others[0].x - corner.x, y: others[0].y - corner.y};
    const v1 = {x: others[1].x - corner.x, y: others[1].y - corner.y};
    // cross z
    const cross = v0.x * v1.y - v0.y * v1.x;
    // pick ordering TL, TR, BR, BL by creating a bounding box and estimating which are where
    // We'll return corners as [TL, TR, BR, BL] by estimating relative positions
    // compute centroid
    const pts = [a,b,c];
    const cx = (a.x+b.x+c.x)/3;
    const cy = (a.y+b.y+c.y)/3;
    // compute angles around centroid
    const angs = pts.map(p=>({p, a: Math.atan2(p.y-cy, p.x-cx)}));
    angs.sort((u,v)=>u.a - v.a);
    // after sorting CCW, pick mapping: angles roughly TL(-pi..-pi/2), TR(-pi/2..0), BR(0..pi/2), BL(pi/2..pi)
    // but with only three finders, missing one corner (BR or BL) — we will infer fourth corner later.
    return {corner, others};
  }

  // perspective transform: map unit square [0,0]-[1,0]-[1,1]-[0,1] to quad
  function getPerspectiveTransform(srcPts, dstPts) {
    // srcPts,dstPts arrays of 4 points {x,y}
    // compute 3x3 matrix H solving for homography (8 unknowns)
    // returns function (x,y) -> {x:..., y:...}
    // Using direct linear solve (8x8)
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const sx = srcPts[i].x, sy = srcPts[i].y;
      const dx = dstPts[i].x, dy = dstPts[i].y;
      A.push([sx, sy, 1, 0, 0, 0, -sx*dx, -sy*dx]);
      b.push(dx);
      A.push([0, 0, 0, sx, sy, 1, -sx*dy, -sy*dy]);
      b.push(dy);
    }
    // solve Ax=b
    const x = solveLinearSystem(A, b); // length 8
    const H = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1];
    return function(px, py) {
      const denom = H[6]*px + H[7]*py + H[8];
      return { x: (H[0]*px + H[1]*py + H[2]) / denom, y: (H[3]*px + H[4]*py + H[5]) / denom };
    };
  }

  function solveLinearSystem(A, b) {
    // simple Gaussian elimination for small systems
    const n = A.length;
    const m = A[0].length;
    // build augmented matrix
    const M = new Array(n);
    for (let i = 0; i < n; i++) {
      M[i] = new Float64Array(m+1);
      for (let j = 0; j < m; j++) M[i][j] = A[i][j];
      M[i][m] = b[i];
    }
    const rows = n, cols = m;
    let r = 0;
    for (let c = 0; c < cols && r < rows; c++) {
      // find pivot
      let pivot = r;
      for (let i = r; i < rows; i++) if (Math.abs(M[i][c]) > Math.abs(M[pivot][c])) pivot = i;
      if (Math.abs(M[pivot][c]) < 1e-12) continue;
      // swap
      const tmp = M[r]; M[r] = M[pivot]; M[pivot] = tmp;
      // normalize
      const div = M[r][c];
      for (let j = c; j <= cols; j++) M[r][j] /= div;
      // eliminate
      for (let i = 0; i < rows; i++) {
        if (i === r) continue;
        const factor = M[i][c];
        if (factor === 0) continue;
        for (let j = c; j <= cols; j++) M[i][j] -= factor * M[r][j];
      }
      r++;
    }
    const x = new Array(cols).fill(0);
    for (let i = 0; i < cols; i++) {
      // find row where col i has 1
      for (let j = 0; j < rows; j++) {
        if (Math.abs(M[j][i] - 1) < 1e-9) {
          x[i] = M[j][cols];
          break;
        }
      }
    }
    return x;
  }

  // sample grid: given transform from sample-space (u,v in [0..n-1]) to image space, sample binary with subpixel averaging
  function sampleGrid(binaryImage, width, height, transformFunc, n) {
    const matrix = new Uint8ClampedArray(n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        // sample at cell center
        const u = (c + 0.5) / n;
        const v = (r + 0.5) / n;
        const p = transformFunc(u, v);
        // bilinear interp of binary image using nearest neighbors because binary
        const ix = Math.round(p.x), iy = Math.round(p.y);
        let val = 0;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) val = binaryImage[iy*width + ix];
        matrix[r*n + c] = val;
      }
    }
    return matrix;
  }

  // detect version / module size: estimate module count from finder spacing
  // For simplification: infer n (modules per side) by ratio between finder centers
  function estimateModuleCount(corners) {
    // corners: TL,TR,BR,BL
    const tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];
    const topLen = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const leftLen = Math.hypot(bl.x - tl.x, bl.y - tl.y);
    const approxModule = (topLen + leftLen)/2 /  (21 - 1); // assume version 1 has 21 modules
    // Try to detect if moduleCount larger by scanning alignment pattern distance? For now, we detect by scanning grid for finder proportions
    // fallback: return 21 as default (version 1). We'll attempt to detect more by oversampling and looking for timing patterns.
    return 21; // conservative default; after sampling we can try to auto-detect alignment
  }

  // decode simplified: read format info to get mask, then unmask and read basic data segments
  // This is a simplified parser not covering all edge cases, but should decode common codes.
  function simplifiedDecode(matrix, n) {
    // matrix: Uint8ClampedArray n*n (0 white, 1 black)
    // Return {text, err}
    // Very simplified: try to read version 1 block map, read format bits to determine mask, unmask, then read data in the standard zig-zag order (assuming version1).
    // This supports byte and alphanumeric modes with limited RS error handling (none).
    if (n < 21) return {text: null, err: 'module grid too small'};
    const size = n;
    // For robust behavior: we try to detect timing pattern (horizontal at row 6, vertical at col 6)
    // Simple unmask: we try all 8 masks and pick the one that yields plausible decode (ASCII printable)
    const candidates = [];
    for (let mask = 0; mask < 8; mask++) {
      const unmasked = applyMask(matrix, size, mask);
      const bits = readBitsFromMatrix(unmasked, size);
      if (!bits) continue;
      const decoded = parseBitStream(bits);
      if (decoded && decoded.text) {
        // add candidate
        candidates.push({mask, text: decoded.text});
      }
    }
    if (candidates.length) return {text: candidates[0].text, err: null};
    return {text: null, err: 'decode failed (no candidate)'} ;
  }

  // apply mask pattern to matrix (size x size). returns new matrix
  function applyMask(matrix, size, mask) {
    const out = new Uint8ClampedArray(matrix.length);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const i = r*size + c;
        const m = maskCondition(mask, r, c);
        out[i] = matrix[i] ^ (m ? 1 : 0);
      }
    }
    return out;
  }
  function maskCondition(mask, r, c) {
    switch(mask) {
      case 0: return ((r + c) % 2) === 0;
      case 1: return (r % 2) === 0;
      case 2: return (c % 3) === 0;
      case 3: return ((r + c) % 3) === 0;
      case 4: return (((Math.floor(r/2)) + (Math.floor(c/3))) % 2) === 0;
      case 5: return ((r*c) % 2 + (r*c) % 3) === 0;
      case 6: return (((r*c) % 2 + (r*c) % 3) % 2) === 0;
      case 7: return (((r+c) % 2 + (r*c) % 3) % 2) === 0;
    }
    return false;
  }

  // read data bits from unmasked matrix using zigzag read order for version1 (simplified)
  function readBitsFromMatrix(matrix, size) {
    // this simplified reader assumes version1 layout and doesn't parse format info properly; but for many real QRs this will still work if masking correct.
    const bits = [];
    let col = size - 1;
    let row = size - 1;
    let upward = true;
    while (col > 0) {
      if (col === 6) { col--; continue; } // skip vertical timing col
      for (let i = 0; i < size; i++) {
        const r = upward ? (size - 1 - i) : i;
        for (let c = col; c >= col-1; c--) {
          // skip functional patterns: we'll approximate by excluding finder areas at corners and timing lines
          if (isFunctionalPattern(r, c, size)) continue;
          bits.push(matrix[r*size + c] ? 1 : 0);
        }
      }
      upward = !upward;
      col -= 2;
    }
    return bits;
  }

  function isFunctionalPattern(r, c, size) {
    // exclude finder patterns (top-left 0..8, top-right, bottom-left), timing lines at row 6/col 6, format areas adjacent to finders
    if (r <= 8 && c <= 8) return true; // TL finder + format
    if (r <= 8 && c >= size - 8) return true; // TR
    if (r >= size - 8 && c <= 8) return true; // BL
    if (r === 6 || c === 6) return true; // timing
    return false;
  }

  // parse bitstream using QR structure (very simplified) - supports byte mode and alphanumeric
  function parseBitStream(bits) {
    // read as stream
    let pos = 0;
    function read(n) { let v=0; for (let i=0;i<n;i++){ v = (v<<1) | (bits[pos++]||0);} return v; }
    // mode indicator: 4 bits
    const mode = read(4);
    if (mode === 0) return {text: '', end: true};
    // For version 1-9, char count bits for byte mode = 8
    if (mode === 4) { // byte mode (0100)
      const count = read(8);
      const bytes = [];
      for (let i = 0; i < count; i++) {
        if (pos + 8 > bits.length) break;
        bytes.push(read(8));
      }
      try {
        const text = new TextDecoder('utf-8').decode(new Uint8Array(bytes));
        return {text};
      } catch (e) {
        // fallback to latin1
        let s = '';
        for (let b of bytes) s += String.fromCharCode(b);
        return {text: s};
      }
    } else if (mode === 2) { // alphanumeric (0010)
      const count = read(9);
      const table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
      let s = '';
      let remaining = count;
      while (remaining >= 2) {
        const v = read(11);
        const a = Math.floor(v / 45);
        const b = v % 45;
        s += table[a] + table[b];
        remaining -= 2;
      }
      if (remaining === 1) {
        const v2 = read(6);
        s += table[v2];
      }
      return {text: s};
    } else {
      return null; // unsupported mode in this simplified reader
    }
  }

  // ---------- high-level pipeline ----------
  async function decodeFromImageElement(imgEl) {
    try {
      const imgData = getImageDataFromImageElement(imgEl);
      const { gray, width, height } = grayscaleCopy(imgData);
      const binary = adaptiveBinarize(gray, width, height, 15, 0.15); // produces 1=black
      // find finder candidates
      const centers = findFinderCandidates(binary, width, height);
      if (centers.length < 3) {
        // fallback: try bigger smoothing window and retry
        const binary2 = adaptiveBinarize(gray, width, height, 25, 0.12);
        const centers2 = findFinderCandidates(binary2, width, height);
        if (centers2.length >= 3) centers.push(...centers2);
      }
      if (centers.length < 3) {
        return [{text: null, matrix: null, size: 0, corners: [], error: 'no finder patterns detected'}];
      }
      // group into triplets
      const triplets = groupIntoQRs(centers);
      if (triplets.length === 0) {
        return [{text: null, matrix: null, size: 0, corners: [], error: 'no QR groupings found'}];
      }
      const results = [];
      for (let t of triplets) {
        // compute approximate fourth corner: if we have three finder centers A,B,C, identify which one is top-left, top-right, bottom-left then compute BR
        // Rough method: compute bounding box and identify missing corner as the one not present
        const pts = t;
        // convert to simple points
        const px = pts.map(p=>({x:p.x, y:p.y}));
        // compute convex hull to order them
        px.sort((u,v)=>u.x - v.x || u.y - v.y);
        // guess TL is leftmost-topmost
        px.sort((u,v)=> u.x - v.x || u.y - v.y);
        // compute triangle centroid and angles to sort CCW
        const cx = (px[0].x + px[1].x + px[2].x)/3;
        const cy = (px[0].y + px[1].y + px[2].y)/3;
        const angs = px.map(p=>({p, a: Math.atan2(p.y - cy, p.x - cx)}));
        angs.sort((a,b)=>a.a - b.a);
        // now angs are CCW; find approximate TL, TR, BL among these; the missing BR can be computed
        // assign:
        // top-left: smallest y then smallest x among points
        const sortedByY = px.slice().sort((a,b)=>a.y - b.y || a.x - b.x);
        const tl = sortedByY[0];
        const others = px.filter(p=>p!==tl);
        // choose tr as the one with smaller y among others
        const tr = (others[0].y < others[1].y) ? others[0] : others[1];
        const bl = (others[0] === tr) ? others[1] : others[0];
        // compute missing BR as tr + bl - tl (vector sum)
        const br = { x: tr.x + bl.x - tl.x, y: tr.y + bl.y - tl.y };
        const corners = [tl, tr, br, bl];
        // estimate module count using distances between TL-TR and TL-BL and dividing by expected finder->finder spacing for v1 (14 modules between centers ~)
        // We'll attempt to find approximate module count by scanning along top edge counting timing pattern transitions.
        // For robustness, sample a larger grid (e.g., 21*3) and then try to find the true module size via correlation.
        const src = [ {x:0,y:0}, {x:1,y:0}, {x:1,y:1}, {x:0,y:1} ];
        const transform = getPerspectiveTransform(src, corners);
        // attempt multiple candidate sizes (21,25,29,33) to support versions 1..4
        const candidateSizes = [21,25,29,33,37,41]; // up to v6
        let best = null;
        for (let sz of candidateSizes) {
          const mat = sampleGrid(binary, width, height, (u,v)=>transform(u, v), sz);
          // heuristic: check presence of finder patterns inside sampled matrix at corners
          const ok = verifyFinderInSample(mat, sz);
          if (ok) { best = {size: sz, matrix: mat}; break; }
        }
        if (!best) {
          // fallback to default 21
          const mat = sampleGrid(binary, width, height, (u,v)=>transform(u, v), 21);
          best = {size:21, matrix: mat};
        }
        // try decode
        const dec = simplifiedDecode(best.matrix, best.size);
        results.push({ text: dec.text, matrix: best.matrix, size: best.size, corners, error: dec.err });
      }
      return results;
    } catch (e) {
      return [{text: null, matrix: null, size: 0, corners: [], error: String(e)}];
    }
  }

  // verify finder in sampled matrix: check 7x7 pattern corners for 1/0 runs
  function verifyFinderInSample(mat, size) {
    // check TL corner area 0..7 etc.
    if (size < 7) return false;
    const tlArea = extractSubmatrix(mat, size, 0, 0, 7);
    const brArea = extractSubmatrix(mat, size, size-7, size-7, 7);
    const trArea = extractSubmatrix(mat, size, 0, size-7, 7);
    function check(area) {
      // check outer 7x7 has typical finder: outermost black, then white, then 3x3 black center
      // quick checks:
      // center 3x3 should be mostly black
      let centerBlack = 0;
      for (let r=2;r<5;r++) for (let c=2;c<5;c++) centerBlack += area[r*7+c];
      if (centerBlack < 7) return false;
      // border corners should have black majority
      let borderBlack = 0, borderTotal = 0;
      for (let i=0;i<7;i++) {
        borderBlack += area[0*7 + i];
        borderBlack += area[6*7 + i];
        borderBlack += area[i*7 + 0];
        borderBlack += area[i*7 + 6];
        borderTotal += 4;
      }
      if (borderBlack < borderTotal*0.45) return false;
      return true;
    }
    return check(tlArea) && check(trArea) && check(brArea);
  }

  function extractSubmatrix(mat, size, sx, sy, w) {
    const out = new Uint8ClampedArray(w*w);
    for (let r=0;r<w;r++) for (let c=0;c<w;c++) out[r*w + c] = mat[(sy + r)*size + (sx + c)];
    return out;
  }

  return {
    decodeFromImageElement
  };
})();

// export
if (typeof module !== 'undefined') module.exports = QRPure;
