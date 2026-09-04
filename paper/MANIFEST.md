# PegaFlow paper artifact manifest

- Repository: `vLLM-HUST/pegaflow-hust`.
- Status: advisor-authored storyline draft on the existing Draft PR; implementation and experiments remain student-owned.
- Source base: `2913b5dd8d0885b22818c42eae31dd5753e6dfbb` (`origin/main`).
- Immutable content commit: `2120fff`.
- Hash rule: SHA-256 over canonical Git blob bytes at that content commit; checkout line-ending conversion is excluded.
- TeX entrypoint: `paper/main.tex`.
- Bibliography: `paper/references.bib`; primary-source system references are cited in the paper.
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`.
- PDF: `paper/build/main.pdf`; 3 pages.
- Build transcript: `paper/build/tectonic.log`; Tectonic exit code 0.
- Build-log scan: no overfull box, undefined reference, missing-character, or TeX error.
- Visual inspection: all three pages rendered and inspected; no clipping, overlap, blank page, or unreadable element.
- `main.tex` SHA256: `3C47E1418AFC4718C539F4704211E3839254AE1C1296E069F28FBB52EAA1FC4E`.
- `references.bib` SHA256: `8A77FE148BC65A1027D25BE62EDF9E2198CAA48F40896C8BA701DFFDB66E4FD9`.
- `EVIDENCE_LEDGER.md` SHA256: `13C60D9931547A4260970ACDA03F4D037C2B3BC2855354AD46CD047B00B6DB32`.
- `main.pdf` SHA256: `9974B72827F938037C4F03FADA3D533D79AB31B1765ADD52CD98072EF173A66C`.
- `tectonic.log` SHA256: `64977711CAE9F9177B5CD353AE8E0D066B2FCAAFF43C02F0BA89D8E68F19F32A`.
- Validation: paper build and `git diff --check`; code behavior is unchanged.

## Evidence boundary

Existing traces motivate measurement but do not yet establish that pooled P/D host memory causes uncontrolled RDMA tails. The paper therefore gates the mechanism on a fresh matched necessity experiment. Its primary claim is tail predictability at fixed total CPU memory and non-inferior cache hit rate; a 50% increase in effective residence is secondary and cannot substitute for the p99 result.
