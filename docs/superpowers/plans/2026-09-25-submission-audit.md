# AsOfCast M2 submission revalidation

Goal: verify the existing M2 scope and prevent asynchronous responses from mixing replay cases.
Scope authority: ../specs/2026-09-18-active-sensor-acquisition-design.md
Base remote commit: e5374058c445369b1cc2cac0303c656cfe3ea1a8 (main 0ea6489 + evidence export workflow).

## Constraints
No model retraining claims from UI tests. Preserve frozen origin, target and retrospective separation.
Use a separate branch. Do not merge, deploy or submit an application as part of this pass.
Use actual FastAPI model responses for browser tests; deliberately delay delivery to reproduce races.
The local browser bridge is not proof of public HTTP availability.
Do not add a license grant or change model thresholds to improve displayed results.

## Tasks
1. Baseline: exact source checksum, existing pytest and historical ETTh1 bundle verification.
2. RED: browser tests for delayed acquisition after case change/reset; delayed timeline; overlapping pulls; edited case ID.
3. GREEN: one request-generation boundary across state, acquisition and timeline; serialize acquisition and invalidate edited controls.
4. Validate: repeat Python and browser suites; real ETTh1 browser flow on desktop/mobile; update stale browser-check script and cache version.
5. Evidence: preserve source and new test reports in CI, separately check public demo without changing its deployment.
6. Documents: update M1 application text for M2 with evidence limits; record exact JD-visible requirements and unverified items.
7. Final diff review; publish review branch/PR only, no main merge.

## Review focus
Cancellation losing a race must not allow obsolete responses to mutate state or status.
Changing case/scenario/wait while a pull is pending must abandon that old interaction.
Double activation must not issue parallel stateless pulls with the same acquired list.
A typed but unapplied case ID must never be sent with the previous case's sensor state.
Error responses must release the busy state without re-enabling stale-case acquisition.
