// src/main.ts

export type ShapeType = 'triangle' | 'square' | 'rectangle' | 'pentagon' | 'circle' | 'unknown';

export interface DetectedShape {
  type: ShapeType;
  bbox: { x: number; y: number; w: number; h: number };
  area: number;                    // number of pixels in component
  centroid: { x: number; y: number };
  contour: Array<{ x: number; y: number }>; // full contour (pixel coords)
  approx: Array<{ x: number; y: number }>;  // polygon after RDP (vertices)
  confidence?: number;             // heuristic confidence 0..1
}

export class ShapeDetector {
  // Main entry: analyze ImageData and return detected shapes
  detectShapes(imageData: ImageData): DetectedShape[] {
    const { width, height, data } = imageData;

    // 1) Convert to grayscale and binary (simple thresholding with Otsu-ish fallback)
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      // luma
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const threshold = this._autoThreshold(gray);
    const binary = new Uint8ClampedArray(width * height);
    for (let i = 0; i < gray.length; i++) binary[i] = gray[i] < threshold ? 1 : 0; // foreground = 1

    // 2) Basic morphological opening to reduce noise (erosion then dilation)
    // const cleaned = this._morphOpen(binary, width, height, 1);
    const cleaned = binary;


    // 3) Connected components labeling (4-neighbor)
    const labels = new Int32Array(width * height).fill(0);
    let nextLabel = 1;
    const components: { label: number; pixels: number[] }[] = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (cleaned[idx] === 1 && labels[idx] === 0) {
          // flood-fill
          const stack = [idx];
          labels[idx] = nextLabel;
          const pixels = [idx];
          while (stack.length) {
            const cur = stack.pop()!;
            const cx = cur % width;
            const cy = Math.floor(cur / width);
            const neighbors = [
              [cx - 1, cy],
              [cx + 1, cy],
              [cx, cy - 1],
              [cx, cy + 1],
            ];
            for (const [nx, ny] of neighbors) {
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nidx = ny * width + nx;
                if (cleaned[nidx] === 1 && labels[nidx] === 0) {
                  labels[nidx] = nextLabel;
                  stack.push(nidx);
                  pixels.push(nidx);
                }
              }
            }
          }
          components.push({ label: nextLabel, pixels });
          nextLabel++;
        }
      }
    }

    // Filter out tiny components
    const minSize = Math.max(10, Math.floor((width * height) * 0.00005)); // tiny area threshold
    const largeComps = components.filter(c => c.pixels.length >= minSize);

    const results: DetectedShape[] = [];

    for (const comp of largeComps) {
      // compute bbox, centroid, area
      let minX = width, minY = height, maxX = 0, maxY = 0, sumX = 0, sumY = 0;
      for (const idx of comp.pixels) {
        const x = idx % width;
        const y = Math.floor(idx / width);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumX += x;
        sumY += y;
      }
      const area = comp.pixels.length;
      const centroid = { x: sumX / area, y: sumY / area };
      const bbox = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };

      // 4) Extract contour by scanning border pixels of component
      const mask = new Uint8ClampedArray(bbox.w * bbox.h);
      for (const idx of comp.pixels) {
        const x = idx % width;
        const y = Math.floor(idx / width);
        const mx = x - bbox.x;
        const my = y - bbox.y;
        mask[my * bbox.w + mx] = 1;
      }
      const contour = this._traceContour(mask, bbox.w, bbox.h, bbox.x, bbox.y);

      // 5) Simplify contour to polygon using RDP
      const epsilon = Math.max(0.5, 0.006 * Math.sqrt(area));
      const approx = this._rdp(contour, epsilon);

    //   const approx = this._rdp(contour, 3); // epsilon tuned; adjust if needed

      // 6) Compute area (mask area used) and perimeter (contour length)
      const perimeter = this._perimeter(contour);
      const circularity = (4 * Math.PI * area) / (perimeter * perimeter + 1e-9);
      
      // 7) Classify shape
      let type: ShapeType = 'unknown';
      let confidence = 0;

      const vcount = approx.length;
      let avgAngle = 0;
      if (approx.length >= 3) {
        for (let i = 0; i < approx.length; i++) {
            const a = approx[(i - 1 + approx.length) % approx.length];
            const b = approx[i];
            const c = approx[(i + 1) % approx.length];
            avgAngle += this._angle(a, b, c);
        }
        avgAngle /= approx.length;
}

// 8) Classification (improved version)
const avgAngleDeg = avgAngle * 180 / Math.PI;
const ar = bbox.w / bbox.h;
const arRatio = ar < 1 ? 1 / ar : ar;
const sideCount = vcount;
console.log(`Detected vertices: ${sideCount}, circularity: ${circularity.toFixed(3)}`);


// heuristic logic combining circularity + vertex count + aspect ratio
if (sideCount <= 3) {
  // use angle consistency: average angle ≈ 60° for equilateral triangle
  if (avgAngleDeg > 40 && avgAngleDeg < 80) {
    type = 'triangle';
    confidence = 0.95;
  } else {
    type = 'triangle';
    confidence = 0.85; // fallback for irregular ones
  }
}

if (sideCount === 3) {
  type = 'triangle';
  confidence = 0.95;
} else if (sideCount === 4) {
  // Check angle and aspect ratio for squares vs rectangles
  if (arRatio < 1.15 && Math.abs(avgAngleDeg - 90) < 12) {
    type = 'square';
    confidence = 0.95;
  } else {
    type = 'rectangle';
    confidence = 0.9;
  }
} else if (sideCount === 5) {
  type = 'pentagon';
  confidence = 0.9;
} else if (sideCount > 5) {
  if (circularity > 0.7) {
    type = 'circle';
    confidence = 0.95;
  } else if (sideCount <= 7) {
    type = 'pentagon';
    confidence = 0.75;
  } else {
    type = 'unknown';
    confidence = 0.6;
  }
} else {
  // handle edge cases when vertex count is noisy
  if (circularity > 0.8) {
    type = 'circle';
    confidence = 0.9;
  } else {
    type = 'unknown';
    confidence = 0.5;
  }
}

console.log(`Shape debug → sides: ${sideCount}, avgAngle: ${avgAngleDeg.toFixed(1)}, arRatio: ${arRatio.toFixed(2)}, circularity: ${circularity.toFixed(2)}, final: ${type}`);

results.push({
  type,
  bbox,
  area,
  centroid,
  contour,
  approx,
  confidence: Math.min(1, Math.max(0, confidence)),
});




// use both vertex count and circularity
if (circularity > 0.78 && vcount >= 6) {
  type = 'circle';
  confidence = circularity;
} else if (vcount === 3) {
  type = 'triangle';
  confidence = 0.9;
} else if (vcount === 4) {
  const ar = bbox.w / bbox.h;
  const arRatio = ar < 1 ? 1 / ar : ar;
  if (arRatio < 1.15 && Math.abs((avgAngle * 180 / Math.PI) - 90) < 10) {
    type = 'square';
    confidence = 0.9;
  } else {
    type = 'rectangle';
    confidence = 0.85;
  }
} else if (vcount === 5) {
  type = 'pentagon';
  confidence = 0.85;
} else if (vcount > 5 && circularity > 0.45) {
  type = 'circle';
  confidence = circularity;
} else {
  type = 'unknown';
  confidence = 0.4;
}

      // heuristic rules:
      if (circularity > 0.72 && vcount > 6) {
        type = 'circle';
        confidence = Math.min(1, (circularity - 0.6) * 2); // rough
      } else {
        if (vcount === 3) {
          type = 'triangle';
          confidence = 0.9;
        } else if (vcount === 4) {
          // check aspect ratio to decide square vs rectangle
          const ar = bbox.w / bbox.h;
          const arRatio = ar < 1 ? 1 / ar : ar;
          if (arRatio < 1.12) {
            type = 'square';
            confidence = 0.9;
          } else {
            type = 'rectangle';
            confidence = 0.85;
          }
        } else if (vcount === 5) {
          type = 'pentagon';
          confidence = 0.85;
        } else if (vcount > 5) {
          // maybe circle-like
          if (circularity > 0.45) {
            type = 'circle';
            confidence = Math.max(0.6, circularity);
          } else {
            type = 'unknown';
            confidence = 0.4;
          }
        } else {
          type = 'unknown';
          confidence = 0.3;
        }
      }

      results.push({
        type,
        bbox,
        area,
        centroid,
        contour,
        approx,
        confidence: Math.min(1, Math.max(0, confidence)),
      });
    }

    return results;
  }

  // -------------------------
  // Helper functions
  // -------------------------

  // Otsu-like threshold: compute histogram and pick threshold maximizing between-class variance
  private _autoThreshold(gray: Uint8ClampedArray): number {
    const hist = new Array(256).fill(0);
    for (let v of gray) hist[v]++;
    const total = gray.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let varMax = 0;
    let threshold = 127;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) * (mB - mF);
      if (varBetween > varMax) {
        varMax = varBetween;
        threshold = t;
      }
    }
    return threshold;
  }

  // Simple morphology open: erosion then dilation using 3x3 cross
  private _morphOpen(bin: Uint8ClampedArray, w: number, h: number, iterations = 1): Uint8ClampedArray {
    let tmp = bin.slice() as Uint8ClampedArray;
    for (let it = 0; it < iterations; it++) {
      tmp = this._erode(tmp, w, h);
    }
    for (let it = 0; it < iterations; it++) {
      tmp = this._dilate(tmp, w, h);
    }
    return tmp;
  }

  private _erode(bin: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    const out = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let keep = 1;
        // cross-shaped neighborhood
        const checks = [
          [0, 0],
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ];
        for (const [dx, dy] of checks) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || bin[ny * w + nx] === 0) {
            keep = 0; break;
          }
        }
        out[y * w + x] = keep;
      }
    }
    return out;
  }

  private _dilate(bin: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    const out = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let any = 0;
        const checks = [
          [0, 0],
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ];
        for (const [dx, dy] of checks) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && bin[ny * w + nx] === 1) {
            any = 1; break;
          }
        }
        out[y * w + x] = any;
      }
    }
    return out;
  }

  private _angle(a: any, b: any, c: any): number {
    const ab = { x: b.x - a.x, y: b.y - a.y };
    const cb = { x: b.x - c.x, y: b.y - c.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const magA = Math.hypot(ab.x, ab.y);
    const magC = Math.hypot(cb.x, cb.y);
    return Math.acos(dot / (magA * magC + 1e-9)); // radians
}


  // trace contour using a simple border-follow on mask (mask coord system)
  private _traceContour(mask: Uint8ClampedArray, mw: number, mh: number, offsetX = 0, offsetY = 0): Array<{ x: number; y: number }> {
    // find starting pixel (first border pixel)
    const idxOf = (x: number, y: number) => y * mw + x;
    let sx = -1, sy = -1;
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        const v = mask[idxOf(x, y)];
        if (v === 1) {
          // check if border (has background neighbor)
          const neighs = [
            [x - 1, y],
            [x + 1, y],
            [x, y - 1],
            [x, y + 1],
          ];
          let border = false;
          for (const [nx, ny] of neighs) {
            if (nx < 0 || nx >= mw || ny < 0 || ny >= mh || mask[idxOf(nx, ny)] === 0) {
              border = true; break;
            }
          }
          if (border) {
            sx = x; sy = y; break;
          }
        }
      }
      if (sx !== -1) break;
    }
    const contour: Array<{ x: number; y: number }> = [];
    if (sx === -1) return contour;

    // Moore-neighbor tracing (clockwise)
    let px = sx, py = sy;
    let dir = 0; // direction index around pixel
    const dirs = [
      [1, 0], [1, -1], [0, -1], [-1, -1],
      [-1, 0], [-1, 1], [0, 1], [1, 1],
    ];
    const visited = new Set<string>();
    let steps = 0;
    do {
      contour.push({ x: px + offsetX, y: py + offsetY });
      visited.add(`${px},${py}`);
      let found = false;
      // search neighbors starting from dir-1
      for (let k = 0; k < 8; k++) {
        const nd = (dir + 7 + k) % 8;
        const nx = px + dirs[nd][0];
        const ny = py + dirs[nd][1];
        if (nx >= 0 && nx < mw && ny >= 0 && ny < mh && mask[ny * mw + nx] === 1) {
          // move
          px = nx; py = ny;
          dir = nd;
          found = true;
          break;
        }
      }
      if (!found) break; // isolated point
      steps++;
      if (steps > 20000) break; // safety
    } while (!(px === sx && py === sy));
    return contour;
  }

  // Ramer-Douglas-Peucker polygon simplification
  private _rdp(points: Array<{ x: number; y: number }>, eps: number): Array<{ x: number; y: number }> {
    if (points.length < 3) return points.slice();
    const sqDistPointToLine = (p: any, a: any, b: any) => {
      const vx = b.x - a.x, vy = b.y - a.y;
      const wx = p.x - a.x, wy = p.y - a.y;
      const c1 = vx * wx + vy * wy;
      if (c1 <= 0) return (p.x - a.x) ** 2 + (p.y - a.y) ** 2;
      const c2 = vx * vx + vy * vy;
      if (c2 <= c1) return (p.x - b.x) ** 2 + (p.y - b.y) ** 2;
      const t = c1 / c2;
      const px = a.x + t * vx, py = a.y + t * vy;
      return (p.x - px) ** 2 + (p.y - py) ** 2;
    };
    const recurse = (pts: typeof points, start: number, end: number, out: boolean[]) => {
      let maxd = 0, idx = start;
      for (let i = start + 1; i < end; i++) {
        const d = sqDistPointToLine(pts[i], pts[start], pts[end]);
        if (d > maxd) { maxd = d; idx = i; }
      }
      if (Math.sqrt(maxd) > eps) {
        out[idx] = true;
        recurse(pts, start, idx, out);
        recurse(pts, idx, end, out);
      }
    };
    const keep = new Array(points.length).fill(false);
    keep[0] = keep[points.length - 1] = true;
    recurse(points, 0, points.length - 1, keep);
    const res: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < points.length; i++) if (keep[i]) res.push(points[i]);
    // If closed contour produced start==end repeated, ensure we return unique vertices
    if (res.length > 1 && res[0].x === res[res.length - 1].x && res[0].y === res[res.length - 1].y) {
      res.pop();
    }
    return res;
  }

  private _perimeter(contour: Array<{ x: number; y: number }>): number {
    if (!contour || contour.length < 2) return 0;
    let p = 0;
    for (let i = 0; i < contour.length; i++) {
      const a = contour[i];
      const b = contour[(i + 1) % contour.length];
      p += Math.hypot(a.x - b.x, a.y - b.y);
    }
    return p;
  }
}
