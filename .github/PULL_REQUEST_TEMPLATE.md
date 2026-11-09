<!-- Delete the section that does not apply. -->

## Loss-study result submission

**Loss:** `<name>`

| model | trained | tested | notes |
|---|---|---|---|
| basic_unet | ☐ | ☐ | |
| ipwg_net | ☐ | ☐ | |
| fully_connected | ☐ | ☐ | |
| encoder_decoder | ☐ | ☐ | |
| dense_encoder_decoder | ☐ | ☐ | |
| efficient_net_v2 | ☐ | ☐ | |
| convnext | ☐ | ☐ | |
| ResNet | ☐ | ☐ | |

Checklist
- [ ] Ran via `python loss_study/run_loss.py <loss>` with **no** `--subset` / `--max-epochs` / `--limit-batches` / `--batch-size` overrides (100 epochs, subset `m`)
- [ ] `loss_study/compute.toml`, `dataset_*.toml`, `models.toml` are unchanged
- [ ] `python loss_study/validate_results.py models/*/test_results/*_<loss>_*.nc` passes
- [ ] Only `test_results/*.nc`, `netcdf_metrics/*.nc` (and, if changed, `loss_study/losses/<loss>.toml`) are included — no `.pt`, `checkpoints/`, `lightning_logs/`
- [ ] If `[args]` in the loss TOML were changed, the reason is explained below

Notes (failures, hardware, anything unusual):

---

## Code change

- [ ] `pytest tests/` passes
- [ ] New losses are registered with `@register_loss` and have a `loss_study/losses/<name>.toml`
