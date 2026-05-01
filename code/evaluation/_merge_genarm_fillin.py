import json

main_path = 'results_genarm_1000/GenARM_0.9help_0.1harm/generation.json'
fill_path = 'results_genarm_1000_fillin/GenARM_0.9help_0.1harm/generation.json'
m = json.load(open(main_path))
f = json.load(open(fill_path))
print(f'[merge] main={len(m)}  fillin={len(f)}')
m_by_uid = {x['uid']: x for x in m}
for x in f:
    m_by_uid[x['uid']] = x
merged = sorted(m_by_uid.values(), key=lambda x: int(x['uid'].replace('eval', '')))
print(f'[merge] merged total={len(merged)}')
json.dump(merged, open(main_path, 'w'), ensure_ascii=False, indent=4)
print(f'[merge] wrote {main_path}')
