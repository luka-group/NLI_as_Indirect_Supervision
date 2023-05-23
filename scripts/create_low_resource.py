import json, argparse, os
from collections import defaultdict
import random, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str, choices=["chemprot_hf_new", "DDI_hf"])
    parser.add_argument('mode', type=str, choices=["1perc", '5perc', '10perc', '8shot', '50shot'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(42)
    pwd = Path(__file__).parent.resolve()
    args = parse_args()
    root = pwd.parent / "data"
    out_dir = root / f"{args.ds}_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    for file in (root / args.ds).iterdir():
        if file.stem != "train":
            # only copy
            shutil.copy(file, out_dir / file.name)
            continue

        with open(file) as f:
            data = [json.loads(l) for l in f]
        if 'perc' in args.mode:
            num = int(args.mode.replace("perc", "")) / 100 # 10perc -> 0.1
            labels = [d['label'] for d in data]
            _, sample = train_test_split(data, test_size=num, stratify=labels)
        else: # shot
            assert 'shot' in args.mode
            num = int(args.mode.replace("shot", ""))

            label2instances = defaultdict(list)
            for d in data:
                label2instances[d['label']].append(d)
            for l in label2instances:
                label2instances[l] = random.sample(label2instances[l], k=num)
            sample = [d for v in label2instances.values() for d in v] 
        print(args.mode, f"data={len(data)}", f"sample={len(sample)}")
        with open(out_dir / file.name, 'w') as f: 
            for s in sample:
                f.write(json.dumps(s) + "\n")
        
    

    