// k-nearest-neighbours classification on 2-D points.
//
// A query is labelled by majority vote among the k closest training points
// (squared Euclidean distance). `classify` returns the winning class together
// with the per-class vote counts. `looAccuracy` is the leave-one-out training
// accuracy: each point is classified by its k nearest *other* points.

// Training points are {x, y, c} with class c in {0, 1}.
export function classify(train, x, y, k) {
  const d = train.map((p) => ({
    d: (p.x - x) ** 2 + (p.y - y) ** 2,
    c: p.c,
  }));
  d.sort((a, b) => a.d - b.d);
  let v0 = 0,
    v1 = 0;
  for (let i = 0; i < Math.min(k, d.length); i++) {
    if (d[i].c === 0) v0++;
    else v1++;
  }
  return { cls: v1 > v0 ? 1 : 0, v0, v1 };
}

// Leave-one-out training accuracy: a point never votes for itself.
export function looAccuracy(train, k) {
  if (!train.length) return 0;
  let correct = 0;
  train.forEach((p) => {
    const d = train.map((q) => ({
      d: (q.x - p.x) ** 2 + (q.y - p.y) ** 2,
      c: q.c,
      self: q === p,
    }));
    d.sort((a, b) => a.d - b.d);
    let v0 = 0,
      v1 = 0,
      used = 0;
    for (let i = 0; i < d.length && used < k; i++) {
      if (d[i].self) continue;
      if (d[i].c === 0) v0++;
      else v1++;
      used++;
    }
    if ((v1 > v0 ? 1 : 0) === p.c) correct++;
  });
  return correct / train.length;
}
