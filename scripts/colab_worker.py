"""Runs only on the Colab VM: conversion, tests, validation and packaging."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    code=Path(__file__).resolve().parents[1]
    work=Path(os.environ['FETAL_WORK'])
    out=Path(os.environ['FETAL_OUTPUT'])
    work.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    stages=[]
    for script,args in [('scripts/prepare_ranks.py',[]),
                        ('tests/test_rank_io.py',[]),
                        ('scripts/validate_export.py',[str(out)])]:
        t=time.perf_counter()
        print(f'COLAB STAGE: {script}',flush=True)
        with subprocess.Popen([sys.executable,'-u',str(code/script),*args],cwd=code,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
            for line in process.stdout:
                print(line,end='',flush=True)
            if process.wait()!=0:
                raise RuntimeError(f'VM stage failed: {script}')
        stages.append(dict(script=script,seconds=round(time.perf_counter()-t,3),passed=True))
    report=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=platform.node(),platform=platform.platform(),python=sys.version,
        processing_location='Google Colab VM',raw_input=str(work/'source.h5ad'),
        output=str(out),physical_memory_bytes=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        peak_memory_note='Linux child-process peak RSS; excludes parent kernel and is not a prediction for larger inputs',
        disk_free_bytes=shutil.disk_usage(work).free,stages=stages,
        processing_and_validation_seconds=round(time.perf_counter()-started,3),
        source_code_sha256={str(p.relative_to(code)):digest(p) for p in code.rglob('*.py')})
    (out/'colab_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    # Stream small transport bundles, keeping transfer memory independent of
    # total dataset size. Every bundle is removed locally after extraction.
    transport=work/'transport'; transport.mkdir(exist_ok=True)
    files=[]; bundles=[]; batch=[]; batch_size=0
    def write_bundle(paths):
        name=f'part-{len(bundles):05d}.zip'
        target=transport/name
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for path in paths:
                z.write(path,path.relative_to(out))
        bundles.append(dict(path=name,bytes=target.stat().st_size,sha256=digest(target)))
    for path in sorted(p for p in out.rglob('*') if p.is_file()):
        size=path.stat().st_size
        if size>=95_000_000:
            raise RuntimeError(f'Artifact exceeds size limit: {path}')
        if batch and batch_size+size>16_000_000:
            write_bundle(batch); batch=[]; batch_size=0
        batch.append(path); batch_size+=size
        files.append(dict(path=str(path.relative_to(out)),bytes=size,sha256=digest(path)))
    if batch: write_bundle(batch)
    transfer=dict(version=1,remote_output=str(out),files=files,bundles=bundles,
                  total_processed_bytes=sum(p['bytes'] for p in files))
    (transport/'transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    print('COLAB COMPLETE:',json.dumps(dict(n_files=len(files),n_bundles=len(bundles),
          total_bytes=transfer['total_processed_bytes'],execution=report)),flush=True)


if __name__=='__main__': main()
