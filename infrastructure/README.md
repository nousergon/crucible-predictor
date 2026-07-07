## Relocated to nous-ergon-ops (private)

The following operational files were relocated to the private
`nousergon/nous-ergon-ops` repo (mirrored layout) in the Phase-2 scoped
ops migration (alpha-engine-config#636, 2026-06-11). Each was verified
consumer-free (no workflow/test/SF-literal/box-runtime path) before
removal. Operators: find them at `nous-ergon-ops/<this-repo>/<same-path>`.

- `add-training-cron.sh` (one-time cron registrar; the registered crontab invokes spot_train.sh, which stays)
- `create-legacy-read-metric.sh`, `setup-cloudwatch-alarms.sh` (one-shot CloudWatch setup)
