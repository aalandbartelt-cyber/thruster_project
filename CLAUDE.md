# CLAUDE.md — Thruster Project Context

## Project Identity

航天器推进器数字孪生与健康监测系统 (Spacecraft Thruster Digital Twin & Health Monitoring), CRAIC 2026 参赛项目, 上海大学.

## Communication

- 用户 (Mary) 说中文, 回答用中文
- 风格: 克制、技术向, 不加 emoji
- 不需要结尾总结 (用户已明确反馈)

## Key Files

| File | Role |
|------|------|
| `app_v3.py` | Streamlit 主应用 (V3 UI + V4 scoring) |
| `inference_utils.py` | 推理封装 + MC Dropout + 健康评分 |
| `data_pipeline_v2.py` | 17维特征工程 + 双输出标签 |
| `model_train_v2.py` | Attention-LSTM 训练 |
| `model_test_v2.py` | 测试评估 + npy 导出 |
| `model_shap_v2.py` | SHAP 归因分析 |
| `model_finetune_v2.py` | 个体推进器微调 |
| `scripts/calibrate_baseline.py` | p50 baseline 标定 (无 torch 依赖) |
| `app_v2.py` | V2 回滚基线 (只读不改) |
| `NIGHT_LOG.md` | 通宵优化日志 + V4 迭代记录 |

## Scoring Architecture (V4)

```
per-dimension: stat_score × eng_score × anom_score → geometric mean → dim_score
overall: (T_dim · M_dim · I_dim)^(1/3) × (0.85 + 0.15·conf/100)
```

- p50 baseline (not RMSE) to avoid circular calibration
- Adaptive engineering tolerance: `max(eng_tol_spec, 3*baseline/|actual|)`
- Isp masked to MFR > 10 mg/s to avoid numerical explosion
- 2-of-3 voting for combined anomaly mask
- See `docs/APP_V3_GUIDE.md` for full rationale

## Environment

- Primary: WSL Ubuntu 22.04 at `\\wsl.localhost\Ubuntu-22.04\home\mary\thruster_project`
- Shell: bash (Unix paths)
- Python: `python3` (may need `wsl -d Ubuntu-22.04 -e bash -c` prefix if PATH issues)
- No torch in WSL Python — use calibration script's inlined approach when needed

## Git

- Repo at `\\wsl.localhost\Ubuntu-22.04\home\mary\thruster_project`
- Branch: `feature/v3-night-optimization`
- `app_v2.py` preserved as rollback baseline — never modify

## Design Conventions

- No emoji in UI labels (NASA-style restraint)
- Geometric mean for multi-dimensional scores (bucket effect — weak dimension pulls down total)
- CSS pre-injected once (not repeated per element)
- Streamlit session_state for history (last 10 reports)
- `render_mission_header()` encapsulated — called in both file-loaded / no-file branches
