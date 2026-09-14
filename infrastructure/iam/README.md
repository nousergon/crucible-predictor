# IAM — not here

`alpha-engine-predictor-role`'s IAM policy is codified in the private
`nous-ergon-ops` repo, not in this repo:

```
nous-ergon-ops/infrastructure/iam/alpha-engine-predictor-role/alpha-engine-predictor-policy.json
```

It is applied to live AWS automatically on merge to that repo's `main`
(`iam-apply-on-merge.yml`), which re-runs the drift checker afterward to
prove live matches codified.

**Never apply IAM from this repo.** A public copy of this policy lived
here until `alpha-engine-config-I8143`: it was not wired to anything, drifted
from the live role, and was applied by hand to production AWS on 2026-09-12 —
overwriting the owned policy and holding `nous-ergon-ops` IAM drift red for
two days. `tests/test_no_iam_policy_documents_reappear.py` in this repo
guards against a policy document or an `apply.sh` reappearing under
`infrastructure/`.
