# Spatial fetal rank data

Prepared on a CPU Google Colab runtime using the configured `google-colab-cli`.
The initial source is GEO **GSM9629296 / CS12-13_E2S3_HESTA**.

## Local Streamlit explorer

```bash
.conda-env/bin/python -m pip install -r requirements-app.txt
.conda-env/bin/streamlit run app.py
```

Open http://localhost:8501. The app defaults to CS20 at `data/GSM9629349/`
(265,988 bins) and does not use Colab. Scoring starts automatically.
The slice selector also offers CS12–13 and CS23. The progressive files
use spatial refinement order, in batches of 32,768 bins. Occupied grid blocks
are sampled at widths 16, 8, 4, 2, and 1, retaining earlier representatives and
adding the bin nearest each newly unrepresented block's center. Additions
within a level are shuffled with a fixed seed so partial levels cover the tissue.
The first full batch includes several levels; it is not a 16-bin-resolution image.
The browser receives compact source-grid positions and integer tissue codes,
then indexed float64 score updates. Geometry is reused after acknowledgment;
a replaced iframe receives a fresh snapshot. At the initial zoom, a canvas
raster aggregates source bins into blocks sized for the display resolution.
Numeric blocks show the mean of scored values; tissue blocks show the majority
annotation (alphabetical tie break). Zooming restores individual source bins.
Unscored occupied blocks receive a display-only neighboring preview using a
multi-source grid flood fill, with Manhattan distance and deterministic ties.
Empty blocks stay transparent; subpixel gaps may be merged in an overview block.
Hover distinguishes block means, previews and individual-bin values. Statistics,
color limits and CSV exports always use original scores, never raster values.
The small slice uses the same spatial
refinement order in two batches of 8,192 and 5,065 bins.
Rebuild this format on Colab with `--sampling spatial --shard-cells 32768
--skip-spatial` using `scripts/run_colab.py`; no duplicate spatial files are needed.

- Choose Human, Mouse, or Rhesus Macaque for both input lists; enter symbols
  separated by whitespace, commas or semicolons. Human inputs use HGNC symbols.
  Non-human inputs use case-sensitive source symbols (e.g. mouse `Sox2`).
  Both target and background lists are converted to human orthologs before
  dataset matching and background reranking. One-to-many mappings retain all
  human orthologs, deduplicated; unmapped source symbols are reported separately
  from human symbols absent from this dataset. A nonempty background with no
  mapped genes produces an error rather than reverting to the full universe.
  The bundled `data/homologene_to_human.csv.gz` is the unchanged mapping snapshot
  from [SEA-AD multi-polygenic app](https://github.com/leonfrench/SEA_AD_multi_polygenic_app/blob/main/data/homologene_to_human.csv.gz),
  downloaded 2026-10-08; conversion follows that app's `analysis.py` rules.
  No network request is needed at app runtime. Changing species reruns scoring.
  Inputs are case-normalized and deduplicated. Unknown symbols are reported.
- Optional background means the **complete universe, including targets**.
  Targets outside it are excluded and explicitly reported. A blank background
  uses all 17,799 retained genes. Empty targets or a universe without any
  non-target genes are rejected.
- For a custom universe, positive-expression genes are sorted by their original
  rank within each observation; average tie ranks and the zero tie are
  recomputed inside that universe. These reconstructed ranks give the exact
  target-versus-complement AUROC without needing the original counts.
- `spatial_app/scoring.py` uses NumPy indexing, sorting and `bincount` over whole
  batches, with no per-cell scoring loop and no dense cells-by-genes matrix.
- Colour limits are the min/max across currently scored observations and update
  after each batch. A constant result uses a narrow padded scale. The range
  depends on completed batches, not the current zoom viewport.
- Toggle **Colour by cell type** on for all 19 source labels; the colour key occupies a fixed area in both modes, preserving the viewport.
  The score calculation is preserved.
- Drag to pan, scroll to zoom, double-click to reset, or save the current tissue
  raster as PNG. The sidebar color switch preserves the viewport. New inputs
  clear previous results and restart scoring when the gene field loses focus.
  Completed results can be downloaded as CSV.

The cell-type enrichment table is calculated once after all batches finish and
reused until the slice or input query changes. Before completion it shows a
waiting message. The table performs a second-level analysis:
it ranks scored bins by their gene-set AUROC, then calculates a one-versus-rest
AUROC for each annotation. Values above 0.5 indicate higher-ranking bins of
that type; values below 0.5 indicate lower-ranking bins. This is not the mean
or median of the original gene-set scores. Two-sided Mann–Whitney U p-values
use the asymptotic distribution with tie and continuity correction. BH adjusted
p-values correct across all annotated types for the current query (untestable
types count as p=1 for correction but display missing values). Only fully scored queries produce enrichment results. Spatial correlation means
bin-level independence is imperfect, so p-values are exploratory rather than
replicate-level evidence. The table and individual-bin scores can be downloaded
separately when scoring is complete.

Rank-file caching is bounded to one batch to limit server memory use. A Streamlit fragment processes one
batch per update, leaving the input controls usable. The scoring timer reports
computation only; disk reads and browser rendering are additional.

Original 512-observation-batch benchmark for all 13,257 observations: approximately 0.09 seconds of
full-universe scoring, or 1.19 seconds with a 10,000-gene background and explicit
re-ranking. Reading the 26 rank files took 0.48 seconds. These measurements are
machine/cache dependent and exclude the deliberately progressive UI updates.

The mathematical tests compare to direct positive/negative gene pairs and fresh
SciPy ranks, including ties, all-zero observations and restricted universes.
App tests cover progressive updates, changing backgrounds and invalid queries.

```bash
.conda-env/bin/python -m pip install pytest scipy
.conda-env/bin/python -m pytest tests/test_scoring.py tests/test_app.py -q
```

`FETAL_APP_DATA=/absolute/path/to/export` can select another prepared dataset;
this initial viewer is tested with the small slice and plots all metadata points.

## Run entirely on Colab, then download the processed files

`scripts/run_colab.py` is a standard-library launcher for the installed Colab CLI.
The H5AD download, HGNC filtering, ranking, scientific validation, edge-case tests
and packaging all run on the **Colab VM**. The local machine only uploads code,
downloads small processed bundles, extracts them and checks streaming SHA-256
hashes. It never downloads or opens the raw H5AD or loads expression arrays.
Local NumPy, Pandas, SciPy and AnnData installations are not needed for this workflow.

For the next slice, create a high-memory CPU runtime and process its URL:

```bash
.conda-env/bin/python scripts/run_colab.py \
  --dataset NEXT_SLICE \
  --url 'https://example.org/next_slice.h5ad' \
  --high-mem \
  --output data/NEXT_SLICE
```

Replace the dataset name and URL. `--high-mem` requests the Colab CLI's native
high-memory CPU option; availability depends on the account. To use a runtime
already created with the desired memory, use `--reuse-session SESSION_NAME`
instead. Reusing a session does not resize it. The launcher prints the runtime's
status and keeps it available after completion. Stop it when finished with
`.conda-env/bin/colab stop -s SESSION_NAME`.

The complete VM-only test uses a fresh source download and a fresh VM working
directory, with output under `data/GSM9629296-colab-tested/`. The earlier
`data/GSM9629296/` export remains available. The new directory includes:

- `colab_execution.json`: VM hostname, physical memory, peak child-process RAM,
  processing times and checksums of the code run on the VM.
- `validation_colab.json`: scientific validation results and timings measured
  **on Colab**.
- `transfer_receipt.json`: checksums and sizes of every file downloaded locally.

Verified on 2026-10-08 using the standard CPU session `fetal-prep` (13.6 GB
physical RAM): 50.9 seconds for download/conversion and 58.9 seconds including
all VM tests and validation; peak child-process RSS 571.8 MB. All 80 processed
files (107.4 MB) were retrieved and checksum-verified. The 77 data artifacts are
byte-for-byte identical to the earlier export. These measurements cover this
slice, not a future larger one; the high-memory runtime option was not provisioned
for this test.

The launcher uses a unique `/content/fetal-runs/` work directory for each run,
preventing reuse of another slice's cached input. Only small transport bundles
are downloaded, one at a time, and each is deleted locally after extraction.
The output directory must be empty for a new run. The default VM execution
timeout is six hours; override it with `--timeout SECONDS` if needed.

If transfer is interrupted after processing finishes, use the remote transport
path printed by the launcher to retry without recomputing:

```bash
.conda-env/bin/python scripts/run_colab.py \
  --reuse-session SESSION_NAME \
  --download-only /content/fetal-runs/RUN_DIRECTORY/transport \
  --output data/NEXT_SLICE
```

Resume mode re-downloads the processed bundles and verifies them. It can write
into a partially downloaded destination. Keep the Colab runtime alive until
retrieval completes; its storage is ephemeral.

## Output

Additional completed slice: **GSM9629349 / CS20_E1S7** is stored in
`data/GSM9629349/`. GEO identifies it as CS20, Embryo1, Section7:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9629349

| Dataset | Observations | Protein-coding genes | Cell types | Random / spatial batches | Total output | Largest file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM9629296 (CS12-13 E2S3) | 13,257 | 17,799 | 19 | 2 / 47 | 107.3 MB | 31.90 MB |
| GSM9629349 (CS20 E1S7) | 265,988 | 18,622 | 21 | 520 / 556 | 1.048 GB | 21.99 MB |

CS20 E1S7 was downloaded, processed and validated on the standard-RAM Colab
session `fetal-prep`: 480.7 seconds for those stages and 2.040 GB peak
child-process RSS. The 6.932 GB raw H5AD stayed on Colab. Its execution,
validation and transfer reports are included with the processed data. Each
dataset has its own gene order and universe; always load its own `genes.parquet`
and manifest rather than reusing another slice's indices.

`data/GSM9629296/` contains 13,257 spatial observations, 19 source cell-type
labels, and 17,799 HGNC protein-coding genes. Each file is below 33 MB;
both expression layouts plus metadata total approximately 107.3 MB (decimal).
Commit the extracted directory, not a combined archive or the 740 MB H5AD.

| File | Contents |
| --- | --- |
| `manifest.json` | Gene universe, provenance, source/reference SHA-256, ordered shard paths, bounds, sizes, checksums, validation |
| `cells.parquet` | Stable `cell_index`, source observation ID, all source `.obs` columns including `x`, `y`, `cellid`, `celltype`, `stage`; random sampling order and tile assignments |
| `genes.parquet` | Column order, approved symbols, HGNC/Ensembl IDs, recomputed detection counts and totals, source names |
| `gene_mapping.parquet` | Audit of every source gene, mapping/exclusion reason and original `.var` fields |
| `hgnc_protein_coding.parquet` | Snapshot of 19,299 approved HGNC protein-coding genes and previous symbols |
| `progressive/*.npz` | Two disjoint random batches: 8,192 + 5,065 observations |
| `spatial/*.npz` | 47 batches from occupied tiles in an 8 x 8 grid, up to 512 observations each |

The small slice was regrouped locally into 8,192-observation random batches
without changing ranks or random order. Spatial batches remain at their original
size. The large slice and the historical Colab test export are unchanged. To
regroup an existing export, use `scripts/rebatch_progressive.py DATASET_DIR
--batch-size 8192`; it validates a staging copy before replacing the live export
and keeps the previous version in an ignored backup directory.

The two expression layouts intentionally duplicate ranks to support efficient
loading in either mode. Each layout covers every observation exactly once.
Both share `cell_index` and the same gene order. The app can ship only the
progressive layout initially if spatial loading is unnecessary.

## Biological and ranking choices

- Use `layers/counts`, with finite nonnegative integer values, rather than
  scaled expression. Within-observation monotonic normalization preserves ranks.
- Match HGNC approved symbols first, then unambiguous previous symbols (or
  Ensembl IDs for a future input). Do not guess from aliases. Of 17,809 retained
  source columns, 17,245 match current symbols and 564 match previous symbols.
  Multiple columns mapping to the same approved gene are summed **before**
  ranking, yielding 17,799 distinct genes. Three ambiguous source names are
  excluded. Full details are in the mapping audit.
- Keep all mapped protein-coding genes by default (`min_cells=0`), including
  measured genes that are zero throughout this particular subset. Genes absent
  from the source are **not** introduced as zero-expression genes.
- Recompute gene detection counts from the actual observations. Source `.var`
  counts reach 23,376 despite this file containing only 13,257 observations and
  therefore cannot safely be used as current-subset detection counts.
- An optional minimum-cell filter is available during export. It changes the
  AUROC background universe; do not apply it independently to individual shards.
- Preserve source coordinate units and orientation. `obs[['x','y']]` exactly
  matches `obsm['spatial']`. The paper describes bin50 spatial analyses, and the
  coordinates are consistent with a grid. Treat rows as spatial observations;
  a cell-type annotation does not establish that a row is a segmented single cell.

HGNC reference: https://www.genenames.org/download/statistics-and-files/
The precise downloaded reference checksum is recorded in the manifest. The
included snapshot supports inspection without relying on later HGNC releases.
## Data credit and original viewers

The spatial transcriptomics data and tissue annotations used here were generated
by Pan and colleagues for the Human Embryogenesis Spatiotemporal Transcriptomic
Atlas (HESTA). Please credit the original study when using these data:

Pan, J. et al. **Spatiotemporal transcriptome atlas of human embryos after
gastrulation.** *Nature* (2026).
[doi:10.1038/s41586-026-10545-0](https://doi.org/10.1038/s41586-026-10545-0).

The authors also provide interactive data viewers through the
[HESTA portal](https://db.cngb.org/hesta/), including the
[spatial clustering and gene-expression viewer](https://db.cngb.org/hesta/spatial.clustering/)
and [3D reconstruction viewer](https://db.cngb.org/hesta/3d.reconstruction/).
The study's data are also available through the
[STOmicsDB project STT0000025](https://db.cngb.org/stomics/project/STT0000025/).

## Exact rank encoding

Each NPZ contains ordinary NumPy arrays, read with `allow_pickle=False`:

| Array | Type | Meaning |
| --- | --- | --- |
| `cell_index` | uint32 | Global observation IDs for this shard's row order |
| `indptr` | uint32 | CSR row pointers, length rows + 1 |
| `gene_index` | uint16 | Gene columns with nonzero counts |
| `rank2` | uint16 | **Twice the absolute ascending average rank**, for stored genes |
| `zero_rank2` | uint16 | Twice the shared zero-expression rank for each row; 0 when no zeros |

Ranks run from 1 to G, with larger values meaning higher expression. Average
ranks preserve all ties, including zero ties. Doubling stores half-integer ranks
exactly. If a row has Z zero-expression genes, their shared rank is `(Z+1)/2`.
The sparse arrays store positive-expression genes, **not** only highly ranked
genes. This is lossless for ranks, but does not retain expression magnitudes.
The encoding supports at most 32,767 genes; the exporter rejects larger universes.

For a gene set with k unique matched genes and G genes in the universe:

```
AUROC = (sum(set ranks) - k * (k + 1) / 2) / (k * (G - k))
```

This is a full set-versus-complement ROC AUC, not a truncated AUCell score.
0.5 is the tie/null reference, and larger values indicate greater expression
of the selected genes. Empty or full-universe sets return NaN. Missing genes
must be reported and excluded from k; duplicated input symbols count once.

## Reader example

From the project root, after installing `requirements-reader.txt`:

```python
from pathlib import Path
import pandas as pd
from scripts.rank_io import load_manifest, load_shard, resolve_symbols, auroc

root = Path('data/GSM9629296')
manifest = load_manifest(root)
cells = pd.read_parquet(root / 'cells.parquet').set_index('cell_index')
genes = pd.read_parquet(root / 'genes.parquet')
selected, missing = resolve_symbols(genes, ['SOX2', 'PAX6'])

# Iterate for progressively more observations, caching completed scores.
for path in manifest['progressive']:
    shard = load_shard(root / path)
    scores = auroc(shard, selected, manifest['n_genes'])
    view = cells.loc[shard['cell_index'], ['x', 'y', 'celltype']].copy()
    view['auroc'] = scores
    # Add view to the growing plot; no reranking is required.
```

The first random batch has 8,192 observations; more batches increase the number
of displayed observations without changing existing scores or spatial bin size.
The shuffle uses seed 20261008 and is uniform without replacement, not stratified
by cell type. Rare types can be absent from early batches.

For a zoom viewport, call `spatial_paths(manifest, [xmin,ymin,xmax,ymax])`, load
those files, then filter observations to the exact viewport bounds. Use
`cell_index` to deduplicate against previously loaded random/spatial observations.
Cache ranks by dataset/file and scores by dataset/gene-set definition. Do not
recompute the gene universe when zooming or loading another batch.

NPZ uses DEFLATE compression for transport and ordinary in-memory NumPy arrays
once loaded. It is not memory-mapped or column-selective: cache a shard after its
first load. Parquet keeps metadata compact. Remote download latency and hosting
bandwidth are additional to the filesystem timings in `validation_colab.json`
(or `validation_local.json` in the original export).

## Reproduce or process another H5AD

The exporter reads CSR count blocks of 256 observations, ranks one observation
at a time, and uses disk-backed sparse temporary arrays. It does not allocate a
dense observations x genes matrix. Input must provide CSR `layers/counts`, gene
symbols/Ensembl IDs as `.var.index`, and `.obs.x/.obs.y`; adapt explicitly for a
different schema. Detection filtering and duplicate aggregation precede ranking.

Use the launcher above for the complete workflow. It also accepts `--min-cells`,
`--shard-cells` and `--grid`. The following environment variables are lower-level
exporter settings; the launcher sets them on the VM, not on the local machine.

The current dataset was left on session `fetal-prep` under `/content/fetal/`.
Colab storage is ephemeral; the local exported files are the durable deliverable.

Use `colab exec --env KEY=VALUE` to configure:

- `FETAL_WORK`: working directory (default `/content/fetal`).
- `FETAL_URL`: source download URL; use a new work directory for each source.
- `FETAL_INPUT`: already downloaded H5AD, otherwise `WORK/source.h5ad`.
- `FETAL_OUTPUT`: an empty destination directory, otherwise `WORK/GSM9629296`.
- `FETAL_DATASET`: dataset identifier in the manifest (default `GSM9629296`).
- `FETAL_MIN_CELLS`: default 0; e.g. 3 requires detection in three observations.
- `FETAL_SHARD_CELLS`: default 512, controls load granularity.
- `FETAL_GRID`: default 8; increase for finer viewport selection in larger inputs.
- `FETAL_SEED`: default 20261008.

Reference downloads are cached in the work directory. Preserve the original
`hgnc_complete_set.tsv` for bit-for-bit reruns. New work directories download the
then-current reference and record its checksum. Raw H5AD and temporary files are
excluded from Git. For very large atlases, total repository size can become a
constraint despite small individual files; the manifest/shard design also works
with static object storage. Metadata is currently one Parquet file and the
exporter fails if any artifact reaches 95 MB; partition metadata for atlases
large enough to reach that limit.

## Validation

- 64 random source observations: decoded ranks equal SciPy average ranks exactly;
  rank-sum AUROC agrees with independently enumerated positive/negative gene pairs.
- Every generated shard: all arrays equal the reference sparse ranks, with each
  layout covering every observation once.
- On Colab: every rank sum checked, both layouts compared observation by
  observation, and sample reader AUROCs checked independently.
- Local download: streaming SHA-256 checks only; no scientific computation.
- `tests/test_rank_io.py`: all-zero rows, positive ties, zero ties, no-zero rows,
  duplicate genes, empty/full sets, invalid indices and viewport intersections.

The launcher runs both `tests/test_rank_io.py` and `scripts/validate_export.py`
inside the VM automatically and stops on a failure before packaging.

## Active local data


Only the three coarse-to-fine exports (`data/GSM9629296`, `data/GSM9629349`,
and `data/GSM9629365`)
and the species conversion file are retained. Obsolete exports, batch backups,
raw H5AD input, and unused per-export gene-mapping/HGNC snapshots were removed.
Active exports retain rank batches, cells/genes metadata, manifests, and historical
validation/provenance reports; those reports describe the original full export
before cleanup. The preparation scripts can regenerate the mapping snapshots.

## Development

This application was developed using Codex with GPT-6 Astra.

## CS23 E2S8

The slice selector includes CS23 · Embryo 2 · Section 8
([GSM9629365](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9629365)).
Its 825,115 bins, 18,646 retained protein-coding genes, and 22 tissue labels were
processed and validated on a standard-RAM Colab VM. The coarse-to-fine export
uses 26 progressive batches (25 × 32,768 bins, then 5,915), with no duplicate
spatial files. The largest file is 62.1 MB; the export is approximately 1.35 GB.
The 17.9 GB source H5AD stays on Colab and is not downloaded locally.
