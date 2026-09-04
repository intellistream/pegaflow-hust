# PegaFlow paper artifact manifest

- Repository: `vLLM-HUST/pegaflow-hust`.
- Status: advisor-authored storyline draft on the existing Draft PR; implementation and experiments remain student-owned.
- Source base: `1c9f1fc48c1239a8f8813056aba6b6696ba41fe7` (`origin/main`).
- TeX entrypoint: `paper/main.tex`.
- Bibliography: `paper/references.bib`; primary-source system references are cited in the paper.
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`.
- PDF: `paper/build/main.pdf`; 3 pages.
- Build transcript: `paper/build/tectonic.log`; Tectonic exit code 0.
- Build-log scan: no overfull box, undefined reference, missing-character, or TeX error.
- Visual inspection: all three pages rendered and inspected; no clipping, overlap, blank page, or unreadable element.
- `main.tex` SHA256: `065BA014547661089B3E0F19997FBBCDC9865E7A061E5BF02D3FF52280D0C86D`.
- `references.bib` SHA256: `001C75DC8A206BC5164E7D020BAF9CA1AC44A82C1A6FBA67024780425677C6A7`.
- `EVIDENCE_LEDGER.md` SHA256: `27B381A8D025E47D1B515D9AD02602453069E7B6FB81541AA8FA28456A82AEF6`.
- `main.pdf` SHA256: `7EABD2B1BE08CEBBE1699EDBA873903E64F65F144B16639ECCA831D5953E6447`.
- `tectonic.log` SHA256: `AD12D5B4A9E94833655E45AE7AC3FDC3D1FEC936E90B9D4A2FD00EEC90E009FA`.
- Validation: paper build and `git diff --check`; code behavior is unchanged.

## Evidence boundary

Existing traces motivate measurement but do not yet establish that pooled P/D host memory causes uncontrolled RDMA tails. The paper therefore gates the mechanism on a fresh matched necessity experiment. Its primary claim is tail predictability at fixed total CPU memory and non-inferior cache hit rate; a 50% increase in effective residence is secondary and cannot substitute for the p99 result.
