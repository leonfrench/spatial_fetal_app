"""HomoloGene conversion using the SEA-AD app's supplied mapping snapshot."""
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd

MAPPING_PATH = Path(__file__).resolve().parents[1]/'data/homologene_to_human.csv.gz'
SPECIES = ('Human', 'Mouse', 'Rhesus Macaque')


@lru_cache(maxsize=1)
def load_ortholog_map():
    frame = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)
    if not {'species', 'source_symbol', 'human_symbol'}.issubset(frame.columns):
        raise ValueError('Ortholog mapping is missing required columns.')
    mapping = {}
    for species, source, human in frame[['species', 'source_symbol', 'human_symbol']].itertuples(index=False, name=None):
        mapping.setdefault(species, {}).setdefault(source, set()).add(human)
    return mapping


def map_input(text, species):
    """Return deduplicated human symbols and unmapped source symbols.

    Source symbol matching is case-sensitive, as in the reference app. Expand
    one-to-many orthologs before deduplication and dataset/background matching.
    """
    if species not in SPECIES:
        raise ValueError(f'Unsupported species: {species}')
    requested = sorted(set(filter(None, re.split(r'[\s,;]+', text.strip()))))
    if species == 'Human':
        return requested, []
    mapping = load_ortholog_map()[species]
    converted = sorted({human for gene in requested for human in mapping.get(gene, ())})
    return converted, [gene for gene in requested if gene not in mapping]
