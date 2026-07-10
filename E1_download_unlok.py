"""
E1_download_unlok.py  - FIXED FOR LOCAL COCO IMAGES
-----------------------------------------------------
Downloads UnLOK-VQA from HuggingFace and prepares forget/retain splits.
Uses your existing local COCO images from:
    D:\\vqav2\\COCO Images\\train2014\\  (COCO_train2014_XXXXXXXXXXXX.jpg)
    D:\\vqav2\\COCO Images\\val2014\\    (COCO_val2014_XXXXXXXXXXXX.jpg)

No internet download needed for images that exist locally.
Missing images are downloaded from COCO CDN as fallback.

Usage
-----
    py E1_download_unlok.py                  # uses default COCO paths
    py E1_download_unlok.py --verify_only    # just check what exists
    py E1_download_unlok.py --coco_root "D:\\vqav2\\COCO Images"  # custom root
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    UNLOK_DIR, UNLOK_JSON, UNLOK_IMAGES_DIR,
    UNLOK_FORGET_DIR, UNLOK_RETAIN_DIR,
)

# -- Default local COCO root (edit if different) -------------------------------
DEFAULT_COCO_ROOT = Path(r"D:\vqav2\COCO Images")

# COCO CDN fallback (val2017 - most UnLOK IDs are here)
COCO_CDN_VAL2017   = "http://images.cocodataset.org/val2017/{:012d}.jpg"
COCO_CDN_TRAIN2014 = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
COCO_CDN_VAL2014   = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


# ------------------------------------------------------------------------
# STEP 1: BUILD LOCAL IMAGE INDEX
# ------------------------------------------------------------------------

def build_local_index(coco_root: Path) -> dict:
    """
    Scan all COCO subdirectories and build a dict:
        image_id (int) -> local Path

    Handles these naming conventions:
        COCO_val2014_000000000042.jpg
        COCO_train2014_000000000042.jpg
        000000000042.jpg              (val2017 style)
    """
    print(f"[E1] Scanning local COCO images in: {coco_root}")
    index = {}

    subdirs = [
        coco_root / "val2014",
        coco_root / "train2014",
        coco_root / "val2017",
        coco_root / "train2017",
        coco_root,                   # images directly in root
    ]

    import re
    pattern = re.compile(r"(\d{6,12})\.jpg$", re.IGNORECASE)

    for d in subdirs:
        if not d.exists():
            continue
        count = 0
        for f in d.glob("*.jpg"):
            m = pattern.search(f.name)
            if m:
                img_id = int(m.group(1))
                if img_id not in index:        # prefer val2014 over train
                    index[img_id] = f
                    count += 1
        if count:
            print(f"  {d.name}: {count} images indexed")

    print(f"  Total unique image IDs found locally: {len(index)}")
    return index


# ------------------------------------------------------------------------
# STEP 2: DOWNLOAD UnLOK-VQA DATASET
# ------------------------------------------------------------------------

def download_unlok_dataset() -> list:
    """Download UnLOK-VQA JSON from HuggingFace."""
    print("[E1] Loading UnLOK-VQA from HuggingFace (vaidehi99/UnLOK-VQA)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets")
        sys.exit(1)

    ds = load_dataset("vaidehi99/UnLOK-VQA", split="train")
    print(f"  Downloaded {len(ds)} samples.")

    UNLOK_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for row in ds:
        records.append({
            "id":       int(row["id"]),
            "src":      str(row["src"]),
            "pred":     str(row["pred"]),
            "rephrase": list(row["rephrase"]) if row["rephrase"] else [],
            "loc":      list(row["loc"])      if row["loc"]      else [],
            "loc_ans":  list(row["loc_ans"])  if row["loc_ans"]  else [],
            "image_id": int(row["image_id"]),
        })

    with open(UNLOK_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved: {UNLOK_JSON}  ({len(records)} records)")
    return records


# ------------------------------------------------------------------------
# STEP 3: COPY / DOWNLOAD IMAGES
# ------------------------------------------------------------------------

def resolve_images(records: list, local_index: dict) -> dict:
    """
    For each unique image_id in records, find or download the image.
    Returns  image_id -> Path  for successfully resolved images.
    """
    UNLOK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    needed = sorted(set(r["image_id"] for r in records))
    print(f"\n[E1] Resolving {len(needed)} unique images...")

    resolved = {}
    missing  = []

    for img_id in needed:
        dst = UNLOK_IMAGES_DIR / f"{img_id:012d}.jpg"

        # Already in our working dir
        if dst.exists():
            resolved[img_id] = dst
            continue

        # Copy from local COCO
        if img_id in local_index:
            shutil.copy2(local_index[img_id], dst)
            resolved[img_id] = dst
            continue

        missing.append(img_id)

    print(f"  Resolved from local COCO: {len(resolved)}/{len(needed)}")

    if missing:
        print(f"  {len(missing)} images not found locally - downloading from CDN...")
        downloaded = 0
        for i, img_id in enumerate(missing):
            dst = UNLOK_IMAGES_DIR / f"{img_id:012d}.jpg"
            # Try val2014, then train2014, then val2017
            urls = [
                COCO_CDN_VAL2014.format(img_id),
                COCO_CDN_TRAIN2014.format(img_id),
                COCO_CDN_VAL2017.format(img_id),
            ]
            success = False
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, dst)
                    success = True
                    break
                except Exception:
                    continue
            if success:
                resolved[img_id] = dst
                downloaded += 1
            else:
                print(f"    [warn] Could not resolve image {img_id}")

            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(missing)}] downloaded {downloaded}")

        print(f"  Downloaded from CDN: {downloaded}/{len(missing)}")

    total = len(resolved)
    print(f"  TOTAL resolved: {total}/{len(needed)}")
    return resolved


# ------------------------------------------------------------------------
# STEP 4: PREPARE FORGET / RETAIN SPLITS
# ------------------------------------------------------------------------

def prepare_splits(records: list, resolved: dict) -> tuple:
    """
    Build forget and retain annotation JSON files.

    Forget set  : (image, src question, pred answer) with rephrase_questions
    Retain set  : (image, loc question, loc_ans)  - UnLOK's locality proxy
    """
    UNLOK_FORGET_DIR.mkdir(parents=True, exist_ok=True)
    UNLOK_RETAIN_DIR.mkdir(parents=True, exist_ok=True)

    valid = [r for r in records if r["image_id"] in resolved]
    print(f"\n[E1] Valid samples (image resolved): {len(valid)}/{len(records)}")

    forget_items = []
    retain_items = []

    for r in valid:
        img_path = str(resolved[r["image_id"]])
        answer   = str(r["pred"]).strip()

        # -- Forget item ------------------------------------------------------
        rephrase_qs = []
        for q in r.get("rephrase", [])[:5]:   # keep up to 5 rephrases
            q = str(q).replace("nq question: ", "").strip()
            if q:
                rephrase_qs.append(q)

        forget_items.append({
            "entity":             f"unlok_{r['id']}",
            "image":              img_path,
            "question":           str(r["src"]).strip(),
            "answer":             answer,
            "aliases":            [answer.lower(), answer],
            "rephrase_questions": rephrase_qs,
            "image_id":           r["image_id"],
            "sample_id":          r["id"],
        })

        # -- Retain / locality items ------------------------------------------
        loc_qs  = r.get("loc",     [])
        loc_ans = r.get("loc_ans", [])
        for q, a in zip(loc_qs, loc_ans):
            q = str(q).replace("nq question: ", "").strip()
            a = str(a).strip()
            if q and a:
                retain_items.append({
                    "entity":   f"loc_{r['id']}",
                    "image":    img_path,
                    "question": q,
                    "answer":   a,
                    "aliases":  [a.lower(), a],
                    "sample_id": r["id"],
                })

    # Save
    forget_ann = UNLOK_FORGET_DIR / "annotations.json"
    retain_ann = UNLOK_RETAIN_DIR / "annotations.json"

    with open(forget_ann, "w", encoding="utf-8") as f:
        json.dump(forget_items, f, indent=2)
    with open(retain_ann, "w", encoding="utf-8") as f:
        json.dump(retain_items, f, indent=2)

    print(f"  Forget split : {len(forget_items)} items  ->  {forget_ann}")
    print(f"  Retain split : {len(retain_items)} items  ->  {retain_ann}")
    return forget_items, retain_items


# ------------------------------------------------------------------------
# VERIFY
# ------------------------------------------------------------------------

def verify():
    print("\n[E1] VERIFICATION")
    ok = True

    if UNLOK_JSON.exists():
        with open(UNLOK_JSON, encoding="utf-8") as f:
            rec = json.load(f)
        print(f"  Dataset JSON : {len(rec)} records  OK")
    else:
        print(f"  Dataset JSON : MISSING  X")
        ok = False

    img_count = len(list(UNLOK_IMAGES_DIR.glob("*.jpg"))) if UNLOK_IMAGES_DIR.exists() else 0
    print(f"  Working images : {img_count}")

    forget_ann = UNLOK_FORGET_DIR / "annotations.json"
    if forget_ann.exists():
        with open(forget_ann, encoding="utf-8") as f:
            fgt = json.load(f)
        missing_img = sum(1 for x in fgt if not Path(x["image"]).exists())
        print(f"  Forget split : {len(fgt)} items, {missing_img} missing images")
        if fgt:
            sample = fgt[0]
            print(f"  Sample forget : Q='{sample['question'][:60]}' A='{sample['answer']}'")
    else:
        print(f"  Forget split : MISSING  X")
        ok = False

    retain_ann = UNLOK_RETAIN_DIR / "annotations.json"
    if retain_ann.exists():
        with open(retain_ann, encoding="utf-8") as f:
            ret = json.load(f)
        print(f"  Retain split : {len(ret)} items")
    else:
        print(f"  Retain split : MISSING  X")
        ok = False

    status = "READY - run E2, E4 next" if ok else "INCOMPLETE - check errors above"
    print(f"\n  STATUS: {status}")
    return ok


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root", default=str(DEFAULT_COCO_ROOT),
        help=r"Root of your COCO images, e.g. D:\vqav2\COCO Images"
    )
    parser.add_argument("--verify_only", action="store_true",
                        help="Just verify existing files, do not download")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip HuggingFace download (use existing JSON)")
    args = parser.parse_args()

    if args.verify_only:
        verify()
        return

    # -- Step 1: Load / download dataset JSON ----------------------------------
    if args.skip_dataset and UNLOK_JSON.exists():
        print(f"[E1] Using existing dataset JSON: {UNLOK_JSON}")
        with open(UNLOK_JSON, encoding="utf-8") as f:
            records = json.load(f)
        print(f"  {len(records)} records.")
    else:
        records = download_unlok_dataset()

    # -- Step 2: Build local image index ---------------------------------------
    coco_root  = Path(args.coco_root)
    local_idx  = build_local_index(coco_root)

    # Quick coverage check
    needed     = set(r["image_id"] for r in records)
    local_hit  = needed & set(local_idx.keys())
    print(f"\n[E1] Coverage check:")
    print(f"  UnLOK needs    : {len(needed)} unique image IDs")
    print(f"  Found locally  : {len(local_hit)} ({100*len(local_hit)/len(needed):.1f}%)")
    print(f"  Need to download: {len(needed) - len(local_hit)}")

    # -- Step 3: Copy / download images ----------------------------------------
    resolved = resolve_images(records, local_idx)

    # -- Step 4: Prepare splits ------------------------------------------------
    prepare_splits(records, resolved)

    # -- Step 5: Verify --------------------------------------------------------
    verify()

    print("\n[E1] DONE.")
    print("  Next steps:")
    print("    py E2_run_audit_per_entity.py --resume")
    print("    py E4_eval_unlok.py --resume")


if __name__ == "__main__":
    main()


