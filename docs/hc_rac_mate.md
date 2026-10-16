# HC-RAC trên MATE: cách chạy và cách đọc chỉ số chẩn đoán

Preset: `--config corac_mate` (HC-RAC, `corac_segment_len: 16`) +
`--env-config mate_belief_entity_gru_4v8_0` (agent `mate_entity_gru`, twin
`mate_entity_gru_twin`, `kl_weight` 1.0, 500 bước/episode, obstacle-free 4v8-0).

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --env-config mate_belief_entity_gru_4v8_0 \
  --config corac_mate \
  --max-steps 500000 \
  with seed=0 use_cuda=True device=0 buffer_cpu_only=True \
  use_wandb=True wandb_project=rac-mate \
  wandb_run_name=HCRAC-L16-mate4v8_0-500k-seed0 \
  use_tensorboard=False
```

Hai đối chứng cần chạy cùng ngân sách bước và cùng seed:

```bash
# RAC (one-hot bin), cùng env preset nên cùng kiến trúc twin
python src/main.py --env-config mate_belief_entity_gru_4v8_0 --config dual_iql_ree ...
# IQL (không có twin)
python src/main.py --env-config mate_entity_gru --config iql ...
```

`corac_mate` kế thừa `corac`, nên tự có `twin_target_clamp: true` (xem
`paper_revision/hc_rac_audit.md` §3.2, §4.1).

## Chỉ số cần xem (đã được log tự động)

| Chỉ số | Ý nghĩa | Bình thường | Cờ đỏ |
|---|---|---|---|
| `twin_target_over_return_max` | `twin_target_mean / episode_returns_max` — twin có đang học giá trị vượt quá return thật không | ≤ ~1 | > 1.5 và tăng dần: twin đang "ratchet"; kiểm tra `twin_target_clamp` có bật không |
| `kl_constraint` (raw) và `kl_loss` (= `kl_weight × kl_constraint`) | áp lực teacher lên Q hành vi | `kl_loss` nhỏ hơn `td_loss` | `kl_loss` ~ `td_loss` hoặc lớn hơn, hoặc `inf`: teacher đang chi phối, Q hành vi sẽ không hội tụ (xem `td_loss` của RAC để so) |
| `twin_delta_relative_scale` | `‖Δ(c)‖/‖W_base‖` trên bank — hypernet đã bỏ xa đầu cơ sở bao xa | vài phần mười, tăng chậm | tăng theo cấp số nhân: mất ổn định kiểu `twin_hyper_hidden=256` |
| `corac_phi_saturated` | tỉ lệ `|cos| > 0.999` trong `φ(R)` | thấp (< ~0.2) | cao: `return_var` quá nhỏ → φ bão hoà, mất độ phân giải liên tục |
| `corac_return_std` / `corac_return_mean` | thống kê chạy dùng để chuẩn hoá return | bám theo thang return MATE | `std` ~ 0 trong khi `mean` lớn: bộ chuẩn hoá hỏng |
| `corac_embed_batch_std` | độ phân tán của `ū` giữa các episode trong batch | > 0 rõ rệt | ~ 0: nửa TCL của context là vô dụng (32/48 chiều chỉ đưa nhiễu vào `max_c`) |
| `twin_td_loss`, `td_loss` | độ hội tụ của twin và của Q hành vi | cùng bậc với RAC | `td_loss` cao gấp nhiều lần RAC: Q hành vi không hội tụ |
| `teacher_q_mean` | giá trị Q của teacher | xấp xỉ thang return | lớn hơn `episode_returns_max` nhiều lần |

`corac_nce_accuracy`, `corac_pos_beats_hard`, `corac_hard_valid`, `corac_nce_loss`
cho biết encoder TCL có học được gì không (chance = `1/(batch × n_agents)`).

## Lưu ý khi resume

Từ commit `d5a4c61`, `save_models`/`load_models` lưu cả labeler
(`context_labeler.th`: encoder TCL, thống kê return, context bank). Checkpoint cũ
không có file này vẫn load được nhưng sẽ cảnh báo và labeler khởi động lại từ đầu —
trước đây đây là nguyên nhân âm thầm làm HC-RAC tệ đi sau mỗi lần resume.
