import re

def people(path):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(person\d+)", line, re.I)
            if m:
                s.add(m.group(1).lower())
    return s

t = people("splits/train.txt")
v = people("splits/val.txt")
e = people("splits/test.txt")

print("overlap train/val", len(t & v))
print("overlap train/test", len(t & e))
print("overlap val/test", len(v & e))
