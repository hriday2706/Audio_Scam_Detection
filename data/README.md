# VoxShield Dataset Documentation

## Primary Dataset: VISHGUARD (Vish-Guard v3.0)

- **Source DOI**: [10.5281/zenodo.20429763](https://doi.org/10.5281/zenodo.20429763)
- **Zenodo Record**: https://zenodo.org/records/20429763
- **Associated Publication**: Nature Scientific Data ([https://www.nature.com/articles/s41597-026-07724-z](https://www.nature.com/articles/s41597-026-07724-z))
- **Title**: yasserhmimou9/Data-Paper-Vish-Guard-Dataset: Initial Data Release:Vish-Guard v3.0
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Access Date**: 2026-10-30
- **Archive File**: `vishguard_dataset.zip`
- **File Size**: 6,454,015,665 bytes (~6.45 GB)
- **MD5 Checksum**: `b6bf4c29812159ddecf7e6042fe0958f`
- **Language Scope**: English (`en`), Modern Standard Arabic (`ar`), French (`fr`)
- **Primary Operational Focus**: English subset (1,065 balanced calls: 536 legitimate, 529 fraudulent)

### Directory Layout
The raw dataset archive contains:
- `vishguard_data.json`: Comprehensive sample metadata, duration, emotional tags, tone, persuasion patterns, transcript script, and audio references.
- `audio/`: 16 kHz 16-bit mono WAV files organized by sample ID `{id}.wav`.
- `transcripts/`: Plain text reference transcripts `{id}.txt`.
- `scripts/`: Scenario source scripts.

### Secondary / External Datasets (Cataloged)
- **ICFD-31k**: Hinglish / Indian English synthetic phone transcripts ([HF: rishia2220/icfd-31k](https://huggingface.co/datasets/rishia2220/icfd-31k)). Text-only external validation.
- **TeleAntiFraud**: Chinese multi-turn scam calls ([HF: JimmyMa99/TeleAntiFraud](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud)).
- **ASLC-448**: Mendeley Arabic scam conversations ([Mendeley Data](https://data.mendeley.com/datasets/p384bgyzz3/5)).

### Ethical and Leakage Protocols
- Raw audio files and full binary archives are kept strictly local in `data/raw/` and excluded from Git.
- Derived feature caches and manifests are isolated under `data/processed/` and `data/manifests/`.
- Cross-split contamination is strictly audited and prevented.
