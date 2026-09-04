# PegaFlow paper artifact manifest

- Repository: `vLLM-HUST/pegaflow-hust`.
- Status: advisor-authored storyline draft on the existing Draft PR; implementation and experiments remain student-owned.
- Source base: `1c9f1fc48c1239a8f8813056aba6b6696ba41fe7` (`origin/main`).
- Immutable content commit: `71731f6`.
- Hash rule: SHA-256 over canonical Git blob bytes at that content commit; checkout line-ending conversion is excluded.
- TeX entrypoint: `paper/main.tex`.
- Bibliography: `paper/references.bib`; primary-source system references are cited in the paper.
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`.
- PDF: `paper/build/main.pdf`; 3 pages.
- Build transcript: `paper/build/tectonic.log`; Tectonic exit code 0.
- Build-log scan: no overfull box, undefined reference, missing-character, or TeX error.
- Visual inspection: all three pages rendered and inspected; no clipping, overlap, blank page, or unreadable element.
- `main.tex` SHA256: `C54BD1196E345A7EB0B843A96179B502A541AC7DE795B080F3ED067AE68EA58D`.
- `references.bib` SHA256: `8A77FE148BC65A1027D25BE62EDF9E2198CAA48F40896C8BA701DFFDB66E4FD9`.
- `EVIDENCE_LEDGER.md` SHA256: `13C60D9931547A4260970ACDA03F4D037C2B3BC2855354AD46CD047B00B6DB32`.
- `main.pdf` SHA256: `A7B00894B6D9A9A443BC3616B00B423328B85FF4126830BBA673D428AD1E6589`.
- `tectonic.log` SHA256: `DE3DF3FDF1D1E48606D63627D51C873A3F369F80F525C1885B690CBA2B591914`.
- Validation: paper build and `git diff --check`; code behavior is unchanged.

## Evidence boundary

Existing traces motivate measurement but do not yet establish that pooled P/D host memory causes uncontrolled RDMA tails. The paper therefore gates the mechanism on a fresh matched necessity experiment. Its primary claim is tail predictability at fixed total CPU memory and non-inferior cache hit rate; a 50% increase in effective residence is secondary and cannot substitute for the p99 result.
