import json

f1 = open("./feats/handa_paths.json", 'r')
paths = []
base1 = "handa_classes"
base2 = "handa_classes_white"

for path in json.load(f1):
    paths.append(f"{base2}{path[len(base1):]}")

json.dump(paths, open("./feats/handa_white_paths.json", 'w'))