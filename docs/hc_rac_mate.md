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
`paper_revision/hc_rac_audit.md` §3.2, §4.1) và `corac_min_std_frac: 0.05`.

## Checklist cho run mới (so với run HCRAC-L16 cũ)

**Bắt buộc — nếu thiếu thì kết quả không so sánh được:**

1. `--config corac_mate`: L=16 + `twin_target_clamp` + `corac_min_std_frac` 0.05.
   Ba thứ này nhắm đúng ba lỗi đã đo: ratchet target twin, mất labeler khi resume,
   và φ bão hoà khi return co cụm.
2. **San bằng protocol với RAC/IQL**: cùng `kl_weight`, `buffer_size`, `slot_number`,
   `batch_size`, `training_iters`, `lr`, `epsilon_anneal_time`, `t_max`, `seed`.
   Run RAC trong chart dùng `kl_weight 0.5` + `buffer 500` trong khi HCRAC chạy
   `1.0` + `1000` — ratchet rất nhạy với cả hai, nên hoặc chạy lại RAC ở 1.0/1000,
   hoặc chạy HCRAC ở 0.5/500, đừng so lệch.
3. Giữ `twin_hyper_hidden: 64`. Đặt 256 làm twin nổ (KL = inf, behavior Q sụp).
4. Kiểm tra log: không còn `random_ expects` (crash bộ lấy cặp TCL ở L≥2 đã sửa), và
   nếu có resume thì checkpoint phải chứa `context_labeler.th`.

**Kỳ vọng định lượng sau khi sửa** (xem §3.6 của audit):

| Chỉ số | Run cũ (HCRAC-L16) | Kỳ vọng run mới | RAC tham chiếu |
|---|---|---|---|
| `twin_target_mean` | ~400–500 (phình 5–6×) | ~80–150 | ~100 |
| `twin_td_error_abs` | ~5–10 và tăng | phẳng, cùng bậc RAC | ~0.5–1 |
| `twin_td_loss` | ~300–500, tăng đơn điệu | phẳng | ~30–50 |
| `twin_target_over_return_max` | (chưa có) | ≤ ~1 | — |
| `corac_phi_saturated` | (chưa có) | thấp | — |

Nếu `twin_td_error_abs` vẫn tăng đơn điệu sau khi bật clamp thì giả thuyết ratchet sai và
cần xem lại `twin_delta_relative_scale` + `kl_constraint`.

## Chỉ số cần xem (đã được log tự động)

| Chỉ số | Ý nghĩa | Bình thường | Cờ đỏ |
|---|---|---|---|
| `twin_target_over_return_max` | `twin_target_mean / episode_returns_max` — twin có đang học giá trị vượt quá return thật không | ≤ ~1 | > 1.5 và tăng dần: twin đang "ratchet"; kiểm tra `twin_target_clamp` có bật không |
| `kl_constraint` (raw) và `kl_loss` (= `kl_weight × kl_constraint`) | áp lực teacher lên Q hành vi | `kl_loss` nhỏ hơn `td_loss` | `kl_loss` ~ `td_loss` hoặc lớn hơn, hoặc `inf`: teacher đang chi phối, Q hành vi sẽ không hội tụ (xem `td_loss` của RAC để so) |
| `twin_delta_relative_scale` | `‖Δ(c)‖/‖W_base‖` trên bank — hypernet đã bỏ xa đầu cơ sở bao xa | vài phần mười, tăng chậm | tăng theo cấp số nhân: mất ổn định kiểu `twin_hyper_hidden=256` |
| `corac_phi_saturated` | tỉ lệ `|cos| > 0.999` trong `φ(R)` | thấp (< ~0.2) | cao: `return_var` quá nhỏ → φ bão hoà, mất độ phân giải liên tục |
| `corac_return_std` / `corac_return_mean` | thống kê chạy dùng để chuẩn hoá return (**std đã áp sàn tương đối** `0.05·|mean|`) | `std` bám theo thang return MATE | `std` ~ 0 trong khi `mean` lớn: bộ chuẩn hoá hỏng |
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
