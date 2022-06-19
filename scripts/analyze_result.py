import os

f = open("./log/classes_linear_64.txt", 'r')
pred2freq, label2freq = {}, {}
pair2path = {}
pair2freq = {}
cnt = 0
# data_root = f"/home/hx/zeyuan/SimCLR-release/classes/"
data_root = f"/data/private/songchenyang/hanzi_filter/classes"
def get_number_of_char(ch: int):
    try:
        return len(os.listdir(f"{data_root}/{ch}/0"))//10
    except:
        return 0
def get_ord_from_str(s: str):
    if '+' in s:
        s = s.split('+')[0]
    return int(s)
for line in f:
    cnt += 1
    path, pred, label = line.split()
    pred = get_ord_from_str(pred)
    label = get_ord_from_str(label)
    if pred != label:
        continue
    pair = f"{pred},{label}"
    if pair2path.get(pair) is None:
        pair2path[pair] = []
    pair2path[pair].append(path)
    pred2freq[pred] = pred2freq.get(pred, 0) + 1
    label2freq[label] = label2freq.get(label, 0) + 1
    pair2freq[pair] = pair2freq.get(pair, 0) + 1
print(cnt)
print(f"correct labels: {len(label2freq)}")

for c, freq in sorted(pred2freq.items(), key=lambda x:x[1], reverse=True)[:100]:
    print(f"{chr(c)}: {freq}/{get_number_of_char(c)}")
print(f"label2freq: {len(label2freq)}")
for c, freq in sorted(label2freq.items(), key=lambda x:x[1], reverse=True)[:100]:
    print(f"{chr(c)}: {freq}/{get_number_of_char(c)}")

print(f"pair2freq: {len(pair2freq)}")

f2 = open("handa_pair2paths.md", 'w')
for pair, freq in sorted(pair2freq.items(), key=lambda x:x[1], reverse=True)[:100]:
    c1, c2 = list(map(int, pair.split(",")))
    print(f"{chr(c1)}({c1})\t{chr(c2)}({c2})\t{freq}")
    f2.write(f"{chr(c1)}({c1}) {chr(c2)}({c2}) {freq}\n")
    for path in pair2path[pair]:
        f2.write(f"![]({path})\n")

f2.close()