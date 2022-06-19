import os
from tqdm import tqdm
from utils import load_json, save_json


def main():
    results = {}
    appear_chs = {}
    # wzb data
    char_base = '/var/lib/shared_volume/home/linbiyuan/corpus/wenbian/labels_页数+序号+著录号+字形_校对版_060616/char'
    png_list = sorted(os.listdir(char_base))
    for img in tqdm(png_list, desc='images'):
        assert img.startswith("甲骨文字編-") and img.endswith(".png")
        img_root = img[6:-4]
        _, page, ch_idx, char, sub_idx, book_age = img_root.split('_')
        if sub_idx[-1] == '.':
            sub_idx = sub_idx[:-1]
        page, ch_idx, sub_idx = int(page), int(ch_idx), int(sub_idx)
        appear_chs.setdefault(char, [])
        if ch_idx not in appear_chs[char]:
            appear_chs[char].append(ch_idx)
        if ch_idx not in results:
            results[ch_idx] = []
        results[ch_idx].append({
            'ch': char,
            'sub': sub_idx,
            'img': img,
            'src': 'wzb',
        })
    # chant data, from 5000
    chant_idx = 5000
    filter_chs = ['※', '■', '（', '〔', '）', '〕', '□', '…', '*']
    handa_parts = ['B', 'D', 'H', 'L', 'S', 'T', 'W', 'Y', 'HD']
    handa_prefix = '/data/private/songchenyang/hanzi_filter/handa'
    for part in handa_parts:
        meta = load_json(f'../hanzi_filter/handa/{part}/oracle_meta_{part}.json')
        for book in tqdm(meta, desc=part):
            if len(book['r_chars']) == 0:
                continue
            for ch in book['r_chars']:
                if ch['char'] in filter_chs:
                    continue
                assert os.path.exists(f'{handa_prefix}/{part}/characters/{ch["img"]}')
                char, img = ch['char'], f'{part}/characters/{ch["img"]}'
                if char not in appear_chs:
                    ch_idx = chant_idx
                    appear_chs[char] = [chant_idx]
                    chant_idx += 1
                else:
                    ch_idx = appear_chs[char][0]
                if ch_idx not in results:
                    results[ch_idx] = []
                results[ch_idx].append({
                    'ch': char,
                    'sub': -1,
                    'img': img,
                    'src': 'chant',
                })
    save_json(results, 'orcal/oracle_classification_combine.json')
    save_json(appear_chs, 'orcal/oracle_char_to_index.json')
    print(len(results), len(appear_chs), sum(len(val) for val in results.values()))  # 6996 5009 462544


if __name__ == '__main__':
    main()
