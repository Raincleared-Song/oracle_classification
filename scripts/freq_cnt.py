from data.utils import get_supervised_data, get_oracle_labeled_data, get_handa_classes_data, get_handa_category_data, get_oracle_db_data, get_jiaguwenbian_data
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="handa")
parser.add_argument('--data_root', default = "/data/private/songchenyang/hanzi_filter/classes")

args = parser.parse_args()
ratios = (1.0, )
if args.dataset == "supervised":
    data, _, y2label = get_supervised_data(data_paths=["data/supervised_data/data_1", "data/supervised_data/data_2"],
                                                                 formats=["jpg", "png"], ratios=ratios)
elif args.dataset.startswith("oracle"):
    data, args.num_classes, y2label = get_oracle_labeled_data(ratios=ratios)
elif args.dataset == "handa":
    data, args.num_classes, y2label = get_handa_classes_data(ratios=ratios, data_root=args.data_root)
elif args.dataset == "handa_category":
    data, args.num_classes, y2label = get_handa_category_data(ratios=ratios, data_root=args.data_root)
elif args.dataset == "handa_oracle_db":
    data, args.num_classes, y2label = get_oracle_db_data(ratios=ratios)
elif args.dataset == "jgwb":
    data, args.num_classes, y2label = get_jiaguwenbian_data(ratios=ratios)

y2cnt = {}
# for folder in os.listdir("your_path"):
#     y2cnt[folder] = len(os.listdir(folder))
for img, y in data:
    if y2cnt.get(y) is None:
        y2cnt[y] = 0
    y2cnt[y] += 1

print(len(y2cnt))
json.dump(y2cnt, open(f"./data/{args.dataset}_freq_{len(y2cnt)}.json",'w'))

sorted_y = sorted(y2cnt.keys(), key=lambda y: y2cnt[y])
f = open(f"./data/{args.dataset}.tsv", 'w')
f.write("y\tcnt\n")
for y in sorted_y:
    f.write(f"{y}\t{y2cnt[y]}\n")
f.close()
print("freq\tclasses\tnumber")
tot = 0
n = 0
x = 0
for ind, y in enumerate(sorted_y):
    if n==0:
        if y2cnt[y] > x:
            print(f"{x}\t{ind}\t{tot}")
            x += 1
            if x > 5:
                n = y2cnt[y] // 10 + 1
    elif y2cnt[y] > n*10:
        print(f"{n * 10}\t{ind}\t{tot}")
        n = y2cnt[y] // 10 + 1
    tot += y2cnt[y]
print(f"{y2cnt[y]}\t{ind + 1}\t{tot}")
