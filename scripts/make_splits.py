#!/usr/bin/env python3
from pathlib import Path
import re, random, json

SEED = 42
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.15, 0.15

ROOT = Path("data/raw/kth")
SPLITS_DIR = Path("splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Map folder names to label ids (edit if your folder names differ)
LABELS = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5,
}

person_re = re.compile(r"(person\d+)", re.IGNORECASE)

def person_id_from_name(name: str) -> str:
    m = person_re.search(name)
    if not m:
        raise ValueError(f"Could not find personXX in filename: {name}")
    return m.group(1).lower()

def main():
    random.seed(SEED)

    samples = []  # (relpath, label, person)
    for cls, y in LABELS.items():
        cls_dir = ROOT / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for vid in sorted(cls_dir.glob("**/*.avi")):
            rel = vid.as_posix()
            person = person_id_from_name(vid.name)
            samples.append((rel, y, person))

    # group by person
    by_person = {}
    for rel, y, p in samples:
        by_person.setdefault(p, []).append((rel, y))

    persons = sorted(by_person.keys())
    random.shuffle(persons)

    n = len(persons)
    n_train = int(n * TRAIN_PCT)
    n_val = int(n * VAL_PCT)
    train_people = set(persons[:n_train])
    val_people = set(persons[n_train:n_train + n_val])
    test_people = set(persons[n_train + n_val:])

    def write_split(fname, people_set):
        lines = []
        for p in sorted(people_set):
            for rel, y in by_person[p]:
                lines.append(f"{rel} {y}\n")
        (SPLITS_DIR / fname).write_text("".join(lines), encoding="utf-8")
        return len(lines)

    c_train = write_split("train.txt", train_people)
    c_val = write_split("val.txt", val_people)
    c_test = write_split("test.txt", test_people)

    # save label map for the report / reproducibility
    (SPLITS_DIR / "label_map.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")

    print(" Done")
    print(f"People: train={len(train_people)} val={len(val_people)} test={len(test_people)} (total={n})")
    print(f"Videos: train={c_train} val={c_val} test={c_test} (total={c_train+c_val+c_test})")
    print(f"Wrote: {SPLITS_DIR/'train.txt'}, {SPLITS_DIR/'val.txt'}, {SPLITS_DIR/'test.txt'}")

if __name__ == "__main__":
    main()
