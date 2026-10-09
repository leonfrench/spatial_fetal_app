from pathlib import Path
import json, hashlib, zipfile
root=Path('/content/fetal/GSM9629296')
m=json.loads((root/'manifest.json').read_text())
for entry in m['files']:
 p=root/entry['path']
 assert p.stat().st_size==entry['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
paths=sorted(p for p in root.rglob('*') if p.is_file())
for i in range(4):
 dest=root.parent/f'export_{i}.zip'
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED) as z:
  for p in paths[i::4]: z.write(p,p.relative_to(root))
 print(dest.name,dest.stat().st_size,hashlib.sha256(dest.read_bytes()).hexdigest())
print('files',len(paths),'max',max(p.stat().st_size for p in paths))
print('progressive',len(m['progressive']),'spatial',len(m['spatial']))
