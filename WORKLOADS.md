# Workload source

PegaFlow pins the policy-neutral `intellistream/llm-serving-workloads` catalog at
`ceef92d52e0c49f26ba1efc6706edd1f6df5d913` under
`workloads/llm-serving-workloads`. The catalog supplies matched request families,
presets, and trace identities for storage-path comparisons; this index does not
select an experiment or claim a transfer improvement.

The pinned private catalog has no repository license file, so it is internal-only
and must not be redistributed as a licensed dataset. Each experiment must freeze
case IDs, seeds, and calibration/validation/test membership. Updates require a
reviewed gitlink change and revalidation; no floating branch is accepted.
