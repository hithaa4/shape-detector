// src/main.ts

export type ShapeType = 'triangle' | 'star' | 'rectangle' | 'pentagon' | 'circle' | 'unknown';

export interface DetectedShape {
  type: ShapeType;
  bbox: { x: number; y: number; w: number; h: number };
  area: number;
  centroid: { x: number; y: number };
  contour: Array<{ x: number; y: number }>;
  approx: Array<{ x: number; y: number }>;
  confidence?: number;
}

export class ShapeDetector {
  detectShapes(imageData: ImageData): DetectedShape[] {
    const { width, height, data } = imageData;

    // 1) Convert to grayscale and binary
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const threshold = this._autoThreshold(gray);
    const binary = new Uint8ClampedArray(width * height);
    const meanGray = gray.reduce((a, b) => a + b, 0) / gray.length;
    const invert = meanGray > 127;
    for (let i = 0; i < gray.length; i++) {
      binary[i] = invert ? (gray[i] < threshold ? 1 : 0) : (gray[i] > threshold ? 1 : 0);
    }

    const cleaned = binary;

    // 2) Connected components labeling
    const labels = new Int32Array(width * height).fill(0);
    let nextLabel = 1;
    const components: { label: number; pixels: number[] }[] = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (cleaned[idx] === 1 && labels[idx] === 0) {
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
    const minSize = Math.max(50, Math.floor((width * height) * 0.00001));
    const largeComps = components.filter(c => c.pixels.length >= minSize);

    const results: DetectedShape[] = [];

    for (const comp of largeComps) {
      // Compute bbox, centroid, area
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

      // Extract contour
      const mask = new Uint8ClampedArray(bbox.w * bbox.h);
      for (const idx of comp.pixels) {
        const x = idx % width;
        const y = Math.floor(idx / width);
        const mx = x - bbox.x;
        const my = y - bbox.y;
        mask[my * bbox.w + mx] = 1;
      }
      const contour = this._traceContour(mask, bbox.w, bbox.h, bbox.x, bbox.y);

      // **FIXED: Better epsilon for RDP - adaptive based on shape characteristics**
      const perimeter = this._perimeter(contour);
      let epsilon = Math.max(0.02 * perimeter, 2.5);
      let approx = this._rdp(contour, epsilon);
      
      // If we get too many vertices, try more aggressive simplification
      if (approx.length > 8) {
        epsilon = Math.max(0.035 * perimeter, 3.5);
        approx = this._rdp(contour, epsilon);
      }
      
      // Post-process: remove nearly-collinear vertices
      approx = this._removeCollinearPoints(approx, 10); // 10 degree tolerance

      // Compute circularity
      const circularity = (4 * Math.PI * area) / (perimeter * perimeter + 1e-9);
      
      let type: ShapeType = 'unknown';
      let confidence = 0;

      const vcount = approx.length;
      const ar = bbox.w / bbox.h;
      const arRatio = ar < 1 ? 1 / ar : ar;

      console.log(`Shape debug → sides: ${vcount}, circularity: ${circularity.toFixed(3)}, arRatio: ${arRatio.toFixed(2)}, area: ${area}`);

      // **FIXED: Better classification logic**
      if (circularity > 0.75) {
        // High circularity = circle (check this first)
        type = 'circle';
        confidence = Math.min(1, circularity);
      } else if (vcount === 3) {
        // Triangle detection
        type = 'triangle';
        const angles = this._getAllAngles(approx);
        const avgAngle = angles.reduce((a, b) => a + b, 0) / angles.length;
        const angleVariance = this._angleVariance(approx);
        
        // Higher confidence for more regular triangles
        confidence = 0.8;
        if (angleVariance < 200) confidence = 0.9; // More uniform angles
        if (Math.abs(avgAngle - 60) < 10) confidence = 0.95; // Close to equilateral
        
        console.log(`Triangle angles: ${angles.map(a => a.toFixed(1)).join(', ')}, avg: ${avgAngle.toFixed(1)}, variance: ${angleVariance.toFixed(1)}`);
      } else if (vcount === 4) {
        // Could be square, rectangle, OR a triangle with one extra vertex
        const angles = this._getAllAngles(approx);
        const avgAngle = angles.reduce((a, b) => a + b, 0) / angles.length;
        
        // Check if this is actually a triangle misidentified as quadrilateral
        // Look for one angle close to 180° (nearly collinear point)
        const maxAngle = Math.max(...angles);
        const minAngle = Math.min(...angles);
        
        console.log(`4-sided angles: ${angles.map(a => a.toFixed(1)).join(', ')}, max: ${maxAngle.toFixed(1)}, min: ${minAngle.toFixed(1)}`);
        
        if (maxAngle > 170 || minAngle < 10) {
          // One vertex is nearly collinear - this is likely a triangle
          type = 'triangle';
          confidence = 0.85;
          console.log(`Reclassified as triangle (collinear vertex detected)`);
        }else {
          type = 'rectangle';
          confidence = 0.9;
        }
      } else if (vcount === 5) {
        // Could be pentagon OR a square with one extra vertex
        const angles = this._getAllAngles(approx);
        const maxAngle = Math.max(...angles);
        const minAngle = Math.min(...angles);
        
        if (maxAngle > 170 || minAngle < 10) {
          // Likely a quadrilateral
          if (arRatio < 1.2){
            type = 'rectangle';
            confidence = 0.85;
          }
        } else {
          type = 'pentagon';
          confidence = 0.85;
        }
      } else if (vcount > 5 && vcount < 10) {
        // Polygon with many sides - could still be a circle that wasn't simplified enough
        if (circularity > 0.6) {
          type = 'circle';
          confidence = circularity;
        } else {
          type = 'unknown';
          confidence = 0.5;
        }
      } else {
        type = 'unknown';
        confidence = 0.3;
      }

      console.log(`Final: ${type}, confidence: ${confidence.toFixed(2)}`);
    
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

  // Helper to get all angles in a polygon
  private _getAllAngles(points: Array<{ x: number; y: number }>): number[] {
    if (points.length < 3) return [];
    const angles: number[] = [];
    for (let i = 0; i < points.length; i++) {
      const a = points[(i - 1 + points.length) % points.length];
      const b = points[i];
      const c = points[(i + 1) % points.length];
      const ang = this._angle(a, b, c) * 180 / Math.PI;
      angles.push(ang);
    }
    return angles;
  }

  // Remove nearly collinear points from polygon
  private _removeCollinearPoints(points: Array<{ x: number; y: number }>, toleranceDeg: number): Array<{ x: number; y: number }> {
    if (points.length <= 3) return points;
    
    const result: Array<{ x: number; y: number }> = [];
    const n = points.length;
    
    for (let i = 0; i < n; i++) {
      const prev = points[(i - 1 + n) % n];
      const curr = points[i];
      const next = points[(i + 1) % n];
      
      const angle = this._angle(prev, curr, next) * 180 / Math.PI;
      
      // Keep point if angle is significantly different from 180° (not collinear)
      if (angle < (180 - toleranceDeg) || points.length <= 3) {
        result.push(curr);
      }
    }
    
    // Ensure we have at least 3 points
    return result.length >= 3 ? result : points;
  }

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

  private _angle(a: any, b: any, c: any): number {
    const ab = { x: b.x - a.x, y: b.y - a.y };
    const cb = { x: b.x - c.x, y: b.y - c.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const magA = Math.hypot(ab.x, ab.y);
    const magC = Math.hypot(cb.x, cb.y);
    return Math.acos(Math.max(-1, Math.min(1, dot / (magA * magC + 1e-9))));
  }

  private _angleVariance(points: Array<{ x: number; y: number }>): number {
    const angles = this._getAllAngles(points);
    if (angles.length === 0) return 0;
    const mean = angles.reduce((a, b) => a + b, 0) / angles.length;
    const variance = angles.reduce((a, b) => a + (b - mean) ** 2, 0) / angles.length;
    return variance;
  }

  private _traceContour(mask: Uint8ClampedArray, mw: number, mh: number, offsetX = 0, offsetY = 0): Array<{ x: number; y: number }> {
    const idxOf = (x: number, y: number) => y * mw + x;
    let sx = -1, sy = -1;
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        const v = mask[idxOf(x, y)];
        if (v === 1) {
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

    let px = sx, py = sy;
    let dir = 0;
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
      for (let k = 0; k < 8; k++) {
        const nd = (dir + 7 + k) % 8;
        const nx = px + dirs[nd][0];
        const ny = py + dirs[nd][1];
        if (nx >= 0 && nx < mw && ny >= 0 && ny < mh && mask[ny * mw + nx] === 1) {
          px = nx; py = ny;
          dir = nd;
          found = true;
          break;
        }
      }
      if (!found) break;
      steps++;
      if (steps > 20000) break;
    } while (!(px === sx && py === sy));
    return contour;
  }

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