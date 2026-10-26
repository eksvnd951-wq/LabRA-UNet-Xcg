#!/usr/bin/env python3
"""
Split a Roboflow export into train / valid / test by source photo (default 7:2:1).

Roboflow names every file <source>.rf.<hash>.<ext>; the augmented copies of one photo
share <source>. When several exports are merged into one folder, files whose names
occur in more than one export carry a "<dataset>-<version>_" prefix. This script
  * groups all files of src/{train,valid,test} by source photo,
  * drops photos whose label file is empty (not annotated),
  * assigns the photos at random (fixed seed) to train / valid / test,
  * copies every file of a training photo, but only the un-augmented copy of a
    validation or test photo (the copy whose bytes are identical in all merged
    exports, or the only copy),
and writes <out>/<split>/{images,labels,masks} and <out>/split.csv.

Usage: python make_split.py --src Bacteria-35 --out Bacteria-35-split
"""
import argparse, collections, csv, hashlib, os, random, re, shutil

NAME = re.compile(r"^(?:[A-Za-z]+-\d+_)?(.+?)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="merged export with train/valid/test (images, labels, masks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, nargs=3, default=(0.7, 0.2, 0.1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = collections.defaultdict(list)          # source photo -> [(split dir, file name)]
    for split in ("train", "valid", "test"):
        d = os.path.join(a.src, split, "images")
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            m = NAME.match(f)
            if not m: raise SystemExit(f"unexpected file name: {f}")
            files[m.group(1)].append((split, f))

    def label(split, f): return os.path.join(a.src, split, "labels", os.path.splitext(f)[0] + ".txt")
    def md5(split, f): return hashlib.md5(open(os.path.join(a.src, split, "images", f), "rb").read()).hexdigest()

    original, dropped = {}, []
    for src, fs in files.items():
        groups = collections.defaultdict(list)
        for s, f in fs: groups[md5(s, f)].append((s, f))
        orig = max(groups.values(), key=len)[0]     # identical in every export = un-augmented copy
        if os.path.getsize(label(*orig)) == 0 or not open(label(*orig)).read().strip():
            dropped.append(src); continue
        original[src] = orig

    sources = sorted(original); random.Random(a.seed).shuffle(sources)
    n_tr, n_va = round(a.ratio[0] * len(sources)), round(a.ratio[1] * len(sources))
    assign = {s: "train" for s in sources[:n_tr]}
    assign.update({s: "valid" for s in sources[n_tr:n_tr + n_va]})
    assign.update({s: "test" for s in sources[n_tr + n_va:]})

    rows = []
    for src, split in sorted(assign.items(), key=lambda kv: (kv[1], kv[0])):
        for s, f in (files[src] if split == "train" else [original[src]]):
            for sub, name in (("images", f), ("labels", os.path.splitext(f)[0] + ".txt"),
                              ("masks", os.path.splitext(f)[0] + ".png")):
                os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
                shutil.copy2(os.path.join(a.src, s, sub, name), os.path.join(a.out, split, sub, name))
            rows.append({"split": split, "source_photo": src, "file": f})
    with open(os.path.join(a.out, "split.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["split", "source_photo", "file"]); w.writeheader(); w.writerows(rows)

    count = collections.Counter(assign.values())
    print(f"photos: {len(sources)} annotated ({len(dropped)} without labels dropped: {', '.join(sorted(dropped))})")
    for split in ("train", "valid", "test"):
        print(f"  {split}: {count[split]} photos, {sum(r['split'] == split for r in rows)} files")


if __name__ == "__main__":
    main()
