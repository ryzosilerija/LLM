"""Load experiment definitions and resolve each config into a full run config."""
import json
import os


def parse_ids(spec, all_ids):
    """'01-18,R01-R03,25' -> list of ids in file order."""
    if not spec:
        return list(all_ids)
    wanted = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            inside = False
            for i in all_ids:
                if i == a:
                    inside = True
                if inside:
                    wanted.add(i)
                if i == b:
                    break
        else:
            wanted.add(part)
    unknown = wanted - set(all_ids)
    if unknown:
        raise SystemExit(f"unknown config ids: {sorted(unknown)}")
    return [i for i in all_ids if i in wanted]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def load_lrs(path):
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def resolve(exp, entry, lrs, overrides=None):
    """Merge baseline + config entry + CLI overrides, and pick the LR."""
    cfg = dict(exp["baseline"])
    cfg.update({k: v for k, v in entry.items() if k not in ("id", "group", "seeds")})
    if "lr" not in entry:
        cfg["lr"] = lrs.get(f"config:{entry['id']}", lrs[cfg["optimizer"]])
    cfg.update(overrides or {})
    return cfg


def run_name(entry, seed):
    return f"{entry['id']}_{entry['optimizer']}_{entry['activation']}_{entry['norm']}_{entry['positional']}_s{seed}"


def parse_overrides(pairs):
    """['d_model=128', 'max_steps=500'] -> dict with numbers parsed."""
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, "summary.json"))
