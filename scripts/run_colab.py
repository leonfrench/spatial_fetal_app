"""Local standard-library launcher. All scientific computation runs on Colab.

Example: python scripts/run_colab.py --dataset GSM9629296 --high-mem
Use --reuse-session NAME instead to reuse an existing runtime.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import uuid
import zipfile

DEFAULT_URL='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9629nnn/GSM9629296/suppl/GSM9629296_CS12-13_E2S3_HESTA.h5ad'
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_path(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe transfer path: {name}')
    path=(root/str(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Transfer path escapes output: {name}')
    return path


def download_output(cli,session,remote_transport,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fetal-transfer-') as temporary:
        temporary=Path(temporary)
        transfer_path=temporary/'transfer.json'
        subprocess.run([cli,'download','-s',session,remote_transport+'/transfer.json',str(transfer_path)],check=True)
        transfer=json.loads(transfer_path.read_text())
        expected={entry['path']:entry for entry in transfer['files']}
        extracted=set()
        for bundle in transfer['bundles']:
            local=temporary/'bundle.zip'
            remote=remote_transport+'/'+bundle['path']
            subprocess.run([cli,'download','-s',session,remote,str(local)],check=True)
            if local.stat().st_size!=bundle['bytes'] or digest(local)!=bundle['sha256']:
                raise ValueError(f'Bundle checksum failed: {remote}')
            with zipfile.ZipFile(local) as archive:
                for entry in archive.infolist():
                    if entry.filename not in expected or entry.filename in extracted:
                        raise ValueError(f'Unexpected/duplicate file: {entry.filename}')
                    safe_path(destination,entry.filename)
                    archive.extract(entry,destination)
                    extracted.add(entry.filename)
            local.unlink()
        if extracted!=set(expected):
            raise ValueError('Incomplete transfer')
        # Streaming byte verification only: no H5AD, NumPy or Parquet reads locally.
        for name,entry in expected.items():
            path=safe_path(destination,name)
            if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
                raise ValueError(f'Artifact checksum failed: {name}')
        (destination/'transfer_receipt.json').write_text(json.dumps(dict(
            verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            session=session,remote_transport=remote_transport,
            verification='streaming SHA-256 only; all scientific validation performed on Colab',
            **transfer),indent=2)+'\n')
        print(f'Download complete: {len(expected)} verified files; {transfer["total_processed_bytes"]:,} bytes -> {destination}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='GSM9629296')
    parser.add_argument('--url',default=DEFAULT_URL)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--colab',default=str(ROOT/'.conda-env/bin/colab'))
    runtime=parser.add_mutually_exclusive_group()
    runtime.add_argument('--high-mem',action='store_true',help='Create a new high-memory CPU runtime')
    runtime.add_argument('--reuse-session',help='Use an existing runtime (does not change its memory tier)')
    parser.add_argument('--session',help='Name of a NEW session; default is unique')
    parser.add_argument('--timeout',type=int,default=21600,help='Remote execution timeout in seconds (default 6 hours)')
    parser.add_argument('--min-cells',type=int,default=0)
    parser.add_argument('--shard-cells',type=int,default=512)
    parser.add_argument('--grid',type=int,default=8)
    parser.add_argument('--skip-spatial',action='store_true',help='Export only progressive batches')
    parser.add_argument('--sampling',choices=['random','spatial'],default='random')
    parser.add_argument('--download-only',metavar='REMOTE_TRANSPORT',help='Resume retrieval from a completed run; requires --reuse-session')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.dataset):
        parser.error('Dataset must be a simple identifier')
    if args.min_cells<0 or args.shard_cells<1 or args.grid<1 or args.timeout<1:
        parser.error('Invalid size, threshold or timeout')
    if args.session and args.reuse_session:
        parser.error('--session and --reuse-session cannot be combined')
    if args.download_only and not args.reuse_session:
        parser.error('--download-only requires --reuse-session')
    destination=(args.output or ROOT/'data'/args.dataset).resolve()
    if destination.exists() and any(destination.iterdir()) and not args.download_only:
        parser.error('Output already contains files; choose a new --output (or --download-only to resume)')
    cli=str(Path(args.colab).resolve())
    suffix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]
    session=args.reuse_session or args.session or 'fetal-'+suffix
    if args.download_only:
        download_output(cli,session,args.download_only.rstrip('/'),destination)
        return
    def colab(*arguments):
        subprocess.run([cli,*arguments],check=True)
    if not args.reuse_session:
        colab('new','-s',session,*(['--high-mem'] if args.high_mem else []))
    colab('status','-s',session)
    print(f'Processing session: {session}. Runtime is left available after completion.',flush=True)
    colab('install','-s',session,'anndata','numpy','pandas','pyarrow','scipy','h5py')
    remote_work='/content/fetal-runs/'+args.dataset+'-'+suffix
    remote_bundle='/content/fetal-code-'+suffix+'.zip'
    remote_transport=remote_work+'/transport'
    settings=dict(FETAL_WORK=remote_work,FETAL_OUTPUT=remote_work+'/output',
        FETAL_INPUT=remote_work+'/source.h5ad',FETAL_URL=args.url,FETAL_DATASET=args.dataset,
        FETAL_MIN_CELLS=str(args.min_cells),FETAL_SHARD_CELLS=str(args.shard_cells),
        FETAL_GRID=str(args.grid),FETAL_SAMPLING=args.sampling,FETAL_SKIP_SPATIAL=str(int(args.skip_spatial)),FETAL_SEED='20261008',FETAL_VALIDATION_REPORT='validation_colab.json')
    print(f'VM working directory: {remote_work}',flush=True)
    print(f'Resume downloads after completion with --reuse-session {session} --download-only {remote_transport}',flush=True)
    with tempfile.TemporaryDirectory(prefix='fetal-code-') as temporary:
        temporary=Path(temporary)
        bundle=temporary/'code.zip'
        with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in ['scripts/prepare_ranks.py','scripts/rank_io.py','scripts/spatial_order.py',
                             'scripts/validate_export.py','scripts/colab_worker.py','tests/test_rank_io.py']:
                archive.write(ROOT/relative,relative)
        colab('upload','-s',session,str(bundle),remote_bundle)
        bootstrap=temporary/'launch_remote.py'
        bootstrap.write_text(
            'import os, pathlib, subprocess, sys, zipfile\n'
            f'os.environ.update({settings!r})\n'
            f'code = pathlib.Path({remote_work!r}) / "code"\n'
            'code.mkdir(parents=True, exist_ok=True)\n'
            f'with zipfile.ZipFile({remote_bundle!r}) as archive: archive.extractall(code)\n'
            'with subprocess.Popen([sys.executable, "-u", str(code / "scripts/colab_worker.py")], '
            'stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:\n'
            '    for line in process.stdout: print(line, end="", flush=True)\n'
            '    if process.wait() != 0: raise RuntimeError("Colab pipeline failed; see VM log")\n')
        colab('exec','-s',session,'--timeout',str(args.timeout),'-f',str(bootstrap))
    download_output(cli,session,remote_transport,destination)


if __name__=='__main__': main()
