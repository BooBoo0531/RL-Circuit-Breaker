# 📚 ÔN TẬP ĐỒ ÁN RL CIRCUIT BREAKER — 2 TIẾNG (TỪ ZERO → SẴN SÀNG VẤN ĐÁP)

> **Mục tiêu**: Sáng mai báo cáo trước thầy. Đi từ "không biết gì về RL" → trả lời được mọi câu hỏi từ cơ bản đến nâng cao.
>
> **Cách dùng**: Đọc tuần tự, đừng nhảy phase. Mỗi phase có "Self-check" — trả lời được mới qua phase tiếp theo.

---

## ⏱ TIMELINE 2 TIẾNG (120 phút)

| Phase | Thời gian | Nội dung |
|---|---|---|
| **Phase 1** | 0–25 phút | Hiểu Circuit Breaker là gì + bài toán đang giải quyết |
| **Phase 2** | 25–55 phút | RL cơ bản: MDP, state, action, reward, episode |
| **Phase 3** | 55–80 phút | PPO và DQN: cách hoạt động, khác biệt, công thức |
| **Phase 4** | 80–95 phút | Đồ án: 7 feature, 5 action, reward function, 5 scenario |
| **Phase 5** | 95–110 phút | Kết quả + cách phân tích |
| **Phase 6** | 110–120 phút | Q&A — 30+ câu hỏi vấn đáp với đáp án ngắn gọn |

---

# 🟢 PHASE 1: HIỂU BÀI TOÁN (25 phút)

## 1.1. Microservices và vấn đề nó tạo ra (5 phút)

**Microservices**: Thay vì viết 1 ứng dụng to (monolith), người ta chia thành nhiều dịch vụ nhỏ độc lập, gọi nhau qua HTTP/gRPC.

**Ví dụ Bookinfo (đề tài dùng)**:
```
Browser → productpage → details
                     → reviews → ratings
```
4 dịch vụ này được triển khai trên Kubernetes (K8s), giao tiếp qua **Istio** (service mesh).

**Vấn đề**: Khi `ratings` chậm → `reviews` chờ → `productpage` chờ → toàn bộ hệ thống treo. Đây gọi là **cascade failure** (lỗi dây chuyền).

## 1.2. Circuit Breaker là gì? (8 phút)

**Ý tưởng**: Giống cầu dao điện trong nhà — khi điện chập, cầu dao tự ngắt để bảo vệ hệ thống.

**3 trạng thái**:
- `CLOSED` (đóng): Bình thường — request đi qua thoải mái
- `OPEN` (mở): Đã ngắt — từ chối tất cả request (fail-fast, không gọi service đang lỗi)
- `HALF-OPEN`: Thử nghiệm — cho 1 ít request đi qua xem service đã hồi phục chưa

**Cấu hình quan trọng** (trong Istio DestinationRule):
- `consecutiveErrors`: bao nhiêu lỗi liên tiếp thì OPEN (vd: 3, 5, 10)
- `interval`: khoảng thời gian theo dõi
- `baseEjectionTime`: thời gian giữ OPEN trước khi thử HALF-OPEN

**Retry** (trong Istio VirtualService):
- `retryAttempts`: nếu request lỗi, thử lại bao nhiêu lần (vd: 1, 2, 3)
- `perTryTimeout`: timeout cho mỗi lần thử

## 1.3. Vấn đề của cấu hình tĩnh — đó là lý do dùng RL (7 phút)

**Cấu hình tĩnh = đặt một lần, không đổi**. Vấn đề:

| Quá nghiêm ngặt (CB mở sớm) | Quá lỏng lẻo (CB không kịp mở) |
|---|---|
| `consecutiveErrors=2` | `consecutiveErrors=20` |
| Mới lỗi vài request đã cắt | Lỗi nhiều mới mở → cascade |
| **False positive cao** | **Bảo vệ kém** |
| Throughput giảm | Khi lỗi tăng đột biến không kịp phản ứng |

→ **Không có cấu hình nào tốt cho mọi tình huống**. Khi traffic bình thường nên lỏng, khi lỗi tăng nên thắt chặt.

**Giải pháp của đồ án**: Dùng **Reinforcement Learning** để agent tự học cách điều chỉnh cấu hình theo trạng thái runtime của hệ thống.

## 1.4. Bài toán cụ thể (5 phút)

**Đầu vào**: Vector 7 chiều (telemetry runtime)
- `rps` (requests per second): tải hiện tại
- `p50, p90, p99`: latency ở percentile 50, 90, 99 (ms)
- `err_rate`: tỉ lệ request lỗi [0,1]
- `cb_state`: trạng thái CB (0=CLOSED, 1=OPEN)
- `retry_rate`: tỉ lệ request đã retry [0,1]

**Đầu ra**: 1 trong 5 mức cấu hình
| Action | Tên | consecutiveErrors | retryAttempts | Đặc điểm |
|---|---|---|---|---|
| 0 | Emergency | 1 | 0 | Rất nghiêm ngặt |
| 1 | Strict | 3 | 1 | Nghiêm ngặt |
| 2 | Moderate | 5 | 2 | Cân bằng (baseline) |
| 3 | Relaxed | 10 | 3 | Linh hoạt |
| 4 | Liberal | 20 | 5 | Rất linh hoạt |

**Mục tiêu**: Agent học cách CHỌN action phù hợp với state để tối đa hoá throughput, giảm lỗi, tránh false positive.

### ✅ Self-check Phase 1
1. Cascade failure là gì? Vì sao Circuit Breaker giải quyết được?
2. Vì sao cấu hình tĩnh không đủ?
3. Đề tài có 7 feature gì? 5 action gì? Action nào là baseline?

---

# 🟡 PHASE 2: RL CƠ BẢN — MDP (30 phút)

## 2.1. Reinforcement Learning là gì? (5 phút)

**Định nghĩa**: Agent học **bằng cách thử và sai**. Không có ai dạy "action này đúng/sai" — chỉ có **reward** (điểm thưởng) sau mỗi action.

**Vòng lặp**:
```
Agent quan sát State → Chọn Action → Môi trường trả Reward + State mới
                          ↑                          ↓
                          └──── Học từ reward ←──────┘
```

**3 nhân vật**:
- **Agent** (tác tử): người ra quyết định (cũng chính là model RL)
- **Environment** (môi trường): nơi action được áp dụng và reward được tính
- **Policy** (chính sách): hàm π(s) → a, nói cho agent biết với state s thì chọn action gì

**Ví dụ đời thường**: Học chơi game Mario.
- State = vị trí Mario, vị trí kẻ địch trên màn hình
- Action = nhảy / chạy / bắn
- Reward = +1 khi qua màn, -1 khi chết

## 2.2. MDP — Markov Decision Process (10 phút)

Đây là **cách formal** để diễn tả bài toán RL. Một MDP gồm 5 thành phần `(S, A, P, R, γ)`:

| Ký hiệu | Tên | Trong đồ án |
|---|---|---|
| `S` | Tập trạng thái (state space) | Vector 7 chiều, mỗi chiều ∈ [0,1] |
| `A` | Tập hành động (action space) | 5 action rời rạc {0,1,2,3,4} |
| `P(s'\|s,a)` | Hàm chuyển trạng thái | Xác định bởi dataset offline (thay vì xác suất tường minh) |
| `R(s,a)` | Hàm reward | 5 thành phần: throughput + latency + error + FP + bonus |
| `γ` | Discount factor | 0.99 — đề tài dùng |

**Tính chất Markov**: State hiện tại đủ để quyết định tương lai — không cần lịch sử quá khứ.

**Discount factor γ** (rất hay được hỏi):
- γ = 0: chỉ quan tâm reward ngay (myopic - thiển cận)
- γ = 1: quan tâm reward tương lai bằng hiện tại (có thể không hội tụ)
- γ = 0.99: cân bằng — tương lai quan trọng nhưng giảm dần

**Công thức return** (tổng reward chiết khấu):
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
    = Σ γ^k · r_{t+k}
```

## 2.3. Episode — đơn vị huấn luyện (5 phút)

Một **episode** = chuỗi (state, action, reward) từ lúc bắt đầu đến khi kết thúc.

**Trong đồ án**:
- Mỗi episode = **100 timesteps**
- Mỗi episode bắt đầu bằng `env.reset()` — chọn ngẫu nhiên 1 trong 5 scenario
- Kết thúc khi đủ 100 step (terminated=True)

**Tổng số timesteps huấn luyện**:
- PPO: 500.000 (≈ 5.000 episode)
- DQN: 1.000.000 (≈ 10.000 episode)

## 2.4. Mục tiêu của RL — tối đa hoá expected return (5 phút)

**Mục tiêu cuối cùng**:
```
π* = argmax_π E[Σ γ^t · r_t]
```
**Dịch ra tiếng Việt**: Tìm chính sách π sao cho **tổng reward chiết khấu kỳ vọng** đạt giá trị lớn nhất.

**Hai trường phái thuật toán**:
1. **Value-based** (DQN): Học Q(s,a) — giá trị của cặp (state, action). Sau đó chọn action có Q lớn nhất: π(s) = argmax_a Q(s,a).
2. **Policy-based** (PPO): Học π(a|s) trực tiếp — phân phối xác suất trên các action.

## 2.5. Bellman Equation — viên gạch nền tảng (5 phút)

**Q-value** (giá trị state-action): "Nếu ở state s tôi chọn action a, rồi sau đó chơi tối ưu, tổng reward kỳ vọng là bao nhiêu?"

**Bellman Equation**:
```
Q(s,a) = r + γ · max_{a'} Q(s', a')
         ↑       ↑
       reward  giá trị tương lai tối đa
       hiện tại  ở state mới
```

**Trực giác**: Giá trị hiện tại = reward ngay + γ × giá trị tốt nhất ở bước tiếp theo.

DQN học cách **xấp xỉ** Q(s,a) bằng neural network, dùng phương trình này làm "target" để tối ưu.

### ✅ Self-check Phase 2
1. MDP gồm những thành phần nào? γ trong đề tài bằng bao nhiêu? Nghĩa là gì?
2. Value-based và Policy-based khác nhau thế nào?
3. Bellman equation viết ra được không?
4. Một episode trong đề tài dài bao nhiêu step?

---

# 🟠 PHASE 3: PPO VÀ DQN (25 phút)

## 3.1. DQN — Deep Q-Network (10 phút)

### Ý tưởng
- Học hàm Q(s,a) bằng neural network (MLP)
- Output mạng = vector 5 chiều, mỗi chiều = Q-value của 1 action
- Chọn action: a = argmax Q(s,a)

### Kiến trúc
```
State 7D → MLP (Hidden layers) → 5 Q-values (Q(s,a₀)...Q(s,a₄))
                                       ↓
                              argmax → Action ∈ {0..4}
```

### 3 thành phần đặc trưng của DQN

**1. Replay Buffer** (200.000 transitions trong đề tài)
- Lưu các transition `(s, a, r, s', done)` đã trải qua
- Khi train, lấy ngẫu nhiên minibatch 64 từ buffer → phá vỡ tương quan thời gian
- Lý do quan trọng: nếu học liên tục từ chuỗi tuần tự, mạng dễ overfit vào pattern gần nhất

**2. Target Network** (sync mỗi 2.000 step)
- Có 2 mạng: **online network** Q (cập nhật mỗi step) và **target network** Q' (cố định)
- Loss = MSE(Q_online(s,a),  r + γ · max_a' Q_target(s',a'))
- Lý do: nếu dùng cùng 1 mạng để tính cả Q và target → vòng lặp tự đuổi đuôi (chasing its own tail) → phân kỳ
- Soft update tau=0.005 thay vì hard copy

**3. Epsilon-greedy exploration**
- Với xác suất ε: chọn action ngẫu nhiên (explore)
- Với xác suất 1-ε: chọn action argmax Q (exploit)
- Trong đề tài: ε giảm tuyến tính từ **1.0 → 0.05** trong 40% bước đầu (≈ 400K steps)
- Sau đó giữ ε = 0.05 cho phần còn lại

### Hyperparameter quan trọng (đề tài)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `learning_rate` | 5e-5 | Nhỏ để tránh phân kỳ Q-value |
| `buffer_size` | 200.000 | Bộ nhớ kinh nghiệm |
| `learning_starts` | 10.000 | Warmup, chỉ thu thập, không học |
| `batch_size` | 64 | Minibatch cho gradient descent |
| `train_freq` | 4 | Mỗi 4 step train 1 lần |
| `target_update_interval` | 2.000 | Sync target network |
| `gamma` | 0.99 | Discount factor |
| `total_timesteps` | 1.000.000 | Tổng số bước (gấp đôi PPO) |

### Vì sao DQN cần 1M timesteps (gấp đôi PPO)?
- DQN là **off-policy**: học từ transition đơn lẻ trong buffer
- Mỗi gradient step chỉ khai thác 1 minibatch nhỏ
- Cần thời gian để buffer đầy + bao phủ không gian state đa dạng

## 3.2. PPO — Proximal Policy Optimization (10 phút)

### Ý tưởng
- Học **trực tiếp policy π(a|s)** — phân phối xác suất trên các action
- Output mạng = 5 logits → softmax → 5 xác suất action
- Chọn action: lấy mẫu từ phân phối (training) hoặc argmax (evaluation)

### Kiến trúc Actor-Critic
```
              ┌→ Actor head  → π(a|s) (5 xác suất action)
State 7D → MLP┤
              └→ Critic head → V(s) (giá trị state)
```

- **Actor**: ra quyết định
- **Critic**: ước lượng V(s) = giá trị kỳ vọng nếu ở state s

### Khái niệm cốt lõi: Advantage
```
A(s,a) = Q(s,a) - V(s)
```
**Trực giác**: "Action a tốt hơn bao nhiêu so với mức trung bình của state s?"
- A > 0: action này đáng làm
- A < 0: action này tệ hơn trung bình

PPO dùng **GAE** (Generalized Advantage Estimation) với `λ=0.95` để tính advantage ổn định.

### Clip Surrogate Objective (đặc trưng quan trọng nhất của PPO)

**Vấn đề**: Nếu cập nhật policy quá mạnh, agent có thể "quên" những gì đã học → sụp đổ.

**Giải pháp PPO**: Giới hạn mức độ thay đổi policy mỗi update.

```
Loss = E[ min(  r(θ)·A,  clip(r(θ), 1-ε, 1+ε)·A  )]

với r(θ) = π_new(a|s) / π_old(a|s)  ← tỉ lệ policy mới/cũ
    ε = clip_range = 0.2 trong đề tài
```

**Trực giác**: Nếu r(θ) > 1.2 hoặc < 0.8 (đổi quá mạnh), cắt nó lại — không cho gradient lớn xảy ra.

### Hyperparameter quan trọng (đề tài)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `learning_rate` | 3e-4 | Standard PPO lr |
| `n_steps` | 1024 | Rollout dài 1024 step trước mỗi update |
| `n_epochs` | 10 | Tái sử dụng rollout 10 lần |
| `batch_size` | 64 | Minibatch trong mỗi epoch |
| `gamma` | 0.99 | Discount |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.2 | Giới hạn r(θ) |
| `ent_coef` | 0.01 | Hệ số entropy bonus (khuyến khích exploration) |
| `max_grad_norm` | 0.5 | Clip gradient để chống nổ |
| `total_timesteps` | 500.000 | Tổng số bước |

### On-policy vs Off-policy (rất hay hỏi)
- **PPO (on-policy)**: Chỉ học từ data thu thập bằng policy HIỆN TẠI. Sau khi update, bỏ rollout cũ.
- **DQN (off-policy)**: Có thể học từ data cũ trong buffer (do policy cũ thu thập).

→ DQN tận dụng data tốt hơn nhưng nhiễu hơn. PPO ổn định hơn nhưng "tốn data".

## 3.3. Bảng so sánh PPO vs DQN (5 phút)

| Tiêu chí | PPO | DQN |
|---|---|---|
| Trường phái | Policy-based | Value-based |
| Học cái gì | π(a\|s) trực tiếp | Q(s,a) |
| Kiến trúc | Actor-Critic | Q-Network + Target Network |
| Buffer | Rollout 1024 (xoá sau update) | Replay Buffer 200K (giữ lâu) |
| On/Off-policy | On-policy | Off-policy |
| Exploration | Stochastic policy + entropy | Epsilon-greedy |
| Inference | Sampling hoặc argmax | argmax Q |
| Total timesteps | 500K | 1M (gấp đôi) |
| Stability | Cao (clip) | Phải có target network |

### ✅ Self-check Phase 3
1. Vì sao DQN cần Replay Buffer? Target Network để làm gì?
2. PPO clip ratio nghĩa là gì? Vì sao cần?
3. On-policy vs off-policy khác nhau thế nào?
4. Advantage là gì? Tại sao PPO dùng nó?

---

# 🔵 PHASE 4: ÁP DỤNG VÀO ĐỒ ÁN (15 phút)

## 4.1. State Vector chi tiết — 7 chiều

| # | Feature | Ý nghĩa | Range gốc | Chuẩn hoá |
|---|---|---|---|---|
| 1 | rps | Requests per second | [0, ~1000] | rps / 1000 |
| 2 | p50 | Latency 50 percentile | [0, ~1000] ms | p50 / 1000 |
| 3 | p90 | Latency 90 percentile | [0, ~1000] ms | p90 / 1000 |
| 4 | p99 | Latency 99 percentile | [0, ~1000] ms | p99 / 1000 |
| 5 | err_rate | Tỉ lệ lỗi | [0, 1] | giữ nguyên |
| 6 | cb_state | Trạng thái CB | {0, 1} | giữ nguyên |
| 7 | retry_rate | Tỉ lệ retry | [0, 1] | giữ nguyên |

**Vì sao chuẩn hoá về [0,1]?**
- Gradient ổn định (feature có cùng scale)
- Tránh exploding gradient khi rps và p99 có giá trị lớn
- Mạng học nhanh hơn

## 4.2. Reward Function — 5 thành phần

```python
reward = throughput + latency + error + false_positive + action_bonus
```

| Thành phần | Công thức | Range | Ý nghĩa |
|---|---|---|---|
| `throughput` | min(rps/20, 5.0) | 0 → +5 | Khuyến khích xử lý nhiều request |
| `latency` | -5 nếu p99 > 500 else 0 | {0, -5} | Phạt vi phạm SLO |
| `error` | -10 × err_rate | 0 → -10 | Phạt theo tỉ lệ lỗi |
| `false_positive` | -3 nếu (cb=1 AND err<0.05) | {0, -3} | Phạt CB mở khi không cần |
| `action_bonus` | +2 nếu action phù hợp với err_rate | {0, +2} | Hỗ trợ học |

**Range tổng**: -18 → +7 mỗi step

**Action bonus** (rất hay bị hỏi):
- err > 0.3 + chọn Emergency/Strict (action 0,1) → +2
- err < 0.05 + chọn Relaxed/Liberal (action 3,4) → +2
- Khác → 0

→ Đây **không phải rule-based** vì agent vẫn có thể chọn khác. Chỉ là tín hiệu định hướng nhỏ trong tổng reward.

## 4.3. Năm Scenario

| Scenario | Tên | Đặc điểm | Hành vi mong đợi |
|---|---|---|---|
| S1 | Normal | Tải bình thường, lỗi thấp | Liberal/Relaxed |
| S2 | Delay 200ms | Độ trễ vừa | Moderate |
| S3 | Delay 800ms | Độ trễ cao, gần SLO | Strict |
| S4 | Abort 20% | Lỗi trung bình | Strict |
| S5 | Abort 40% | Lỗi cao | Emergency/Strict |

**Random scenario mỗi episode** → agent học chính sách tổng quát, không overfit 1 tình huống.

## 4.4. Quy trình huấn luyện — đặc điểm quan trọng

### Best Checkpoint (phải nói được)
- **EvalCallback**: cứ 10.000 steps, đánh giá 5 episode (deterministic)
- Nếu mean reward > best hiện tại → ghi đè `best_model.zip`
- **Lý do**: trong RL, mô hình có thể bị **policy degradation** — học tốt rồi bị xấu đi cuối quá trình
- Theo gợi ý của thầy: **dùng best, không dùng final**

### Offline Training
- Dùng `behavioral_dataset.csv` (thu thập từ K3s + Bookinfo + Fortio + Istio fault injection)
- An toàn (không hỏng cluster), tái lập được, tiết kiệm

### Online Inference (Bảng 15 trong báo cáo)
4 bước:
1. Thu telemetry từ Istio + Prometheus
2. Predict action với deterministic=True
3. Áp dụng vào DestinationRule + VirtualService
4. Chờ control interval, lặp lại

### ✅ Self-check Phase 4
1. Vì sao chuẩn hoá state về [0,1]?
2. Reward có 5 thành phần — kể được không?
3. Best checkpoint vs Final model: chọn cái nào? Vì sao?

---

# 🟣 PHASE 5: KẾT QUẢ VÀ PHÂN TÍCH (15 phút)

## 5.1. Kết quả chính (từ Bảng 21 trong báo cáo)

### Comprehensive Eval Report (50 episodes/policy)

| Policy | Mean Reward | Std | SLO Compliance | Avg Err Rate | FP Rate |
|---|---|---|---|---|---|
| Random | -20.02 | 243.63 | 72.0% | 0.245 | 38.5% |
| Rule-based | 36.07 | 230.08 | 74.0% | 0.219 | 0.0% |
| **PPO** | **80.07** | 234.42 | **92.0%** | 0.212 | 0.0% |
| **DQN** | 18.04 | 231.71 | 80.0% | **0.165** | 0.0% |

**Ý nghĩa**:
- **PPO thắng tổng** (mean reward cao nhất, SLO compliance cao nhất)
- **DQN thắng err_rate** (thấp nhất 0.165)
- **Cả PPO và DQN đều thắng baseline Rule-based**

## 5.2. Vì sao Std cao (>200)?

**Đây là câu hỏi RẤT hay được hỏi**.

**Trả lời**: Std cao **không phải do model học kém**, mà do:
1. **Môi trường stochastic**: mỗi episode random 1 trong 5 scenario
2. Reward range của scenario rất khác nhau:
   - S1 (Normal): reward ~ +6 mỗi step → +600 cả episode
   - S5 (Abort 40%): reward ~ -3 mỗi step → -300 cả episode
3. → Mean ± Std rộng là điều **bản chất**

**Bằng chứng**: Khi đánh giá balanced (mỗi scenario chạy đều) trên 1000 episode, PPO mean = 34.60, DQN = 34.62 → **chênh lệch chỉ 0.01**, hai model thực sự tương đương khi cân bằng input.

## 5.3. Reward Curve và hội tụ

**Cách đọc reward curve**:
- **Đường mờ** (raw): reward từng episode — dao động bình thường
- **Đường đậm** (moving average window=50): xu hướng — đây mới là thứ cần nhìn

**Tốt**: Moving average tăng rồi bằng phẳng.
**Cần dùng best checkpoint**: Moving average tăng rồi giảm (degradation).

## 5.4. Action Distribution (theo scenario)

Khi đánh giá per-scenario:
- S1 (Normal): cả PPO và DQN dominant chọn **Relaxed/Liberal** ✓
- S5 (Abort 40%): cả 2 dominant chọn **Emergency/Strict** ✓
- → Agent học được mapping state → action đúng

## 5.5. Action Appropriateness

Tỉ lệ % action match với domain logic:
- High err (>0.3) → chọn 0 hoặc 1
- Low err (<0.05) → chọn 3 hoặc 4
- Mid → chọn 1, 2, 3

**Kết quả đề tài**: PPO, DQN, Rule-based đều đạt **100%** appropriateness.

## 5.6. Statistical Significance

Welch's t-test:
- PPO vs Random: p < 0.05 → khác biệt có ý nghĩa thống kê
- DQN vs Random: p < 0.05 → khác biệt có ý nghĩa
- PPO vs DQN: p > 0.05 → **không khác biệt có ý nghĩa** (do scenario chi phối)

### ✅ Self-check Phase 5
1. PPO mean reward bao nhiêu? Cao hơn baseline bao nhiêu %?
2. Vì sao Std cao? Có phải model kém không?
3. Action appropriateness là gì? Đạt bao nhiêu %?

---

# 🔴 PHASE 6: 30+ Q&A VẤN ĐÁP THƯỜNG GẶP (10 phút cuối)

> Đọc lướt — câu nào còn bí thì quay lại phase tương ứng.

## A. Câu hỏi cơ bản về RL (10 câu)

**Q1. RL là gì? Khác gì với supervised learning?**
A: RL học bằng reward (không có nhãn đúng/sai). Supervised có dataset (X, y); RL chỉ có (state, action, reward) tự sinh ra qua tương tác.

**Q2. MDP gồm gì?**
A: 5 thành phần: S (state), A (action), P (transition), R (reward), γ (discount).

**Q3. γ là gì? Đề tài bằng bao nhiêu?**
A: Discount factor — γ ∈ [0,1]. Đề tài 0.99 nghĩa: reward sau 100 step có trọng số 0.99^100 ≈ 0.37.

**Q4. Vì sao RL phù hợp cho bài toán này?**
A: Cấu hình tĩnh không thích nghi được; điều kiện vận hành thay đổi liên tục; RL học được mapping state → action tối ưu thông qua reward.

**Q5. Episode dài bao lâu?**
A: 100 timesteps. Tổng PPO 500K timesteps ≈ 5000 episode. DQN 1M ≈ 10000 episode.

**Q6. Chính sách (policy) là gì?**
A: Hàm π(s) → a, hoặc π(a|s) là phân phối xác suất. Là output cuối cùng của quá trình huấn luyện.

**Q7. State space của đề tài có chiều nào?**
A: 7 chiều, đã chuẩn hoá [0,1]: rps, p50, p90, p99, err_rate, cb_state, retry_rate.

**Q8. Action space rời rạc hay liên tục?**
A: Rời rạc, 5 action: Emergency, Strict, Moderate, Relaxed, Liberal.

**Q9. Vì sao chuẩn hoá state về [0,1]?**
A: Gradient ổn định, neural network học nhanh hơn, tránh exploding gradient.

**Q10. Reward range là gì?**
A: -18 đến +7 mỗi step. Episode 100 step → -1800 đến +700.

## B. Câu hỏi về PPO/DQN (10 câu)

**Q11. PPO và DQN khác nhau ở đâu?**
A: PPO policy-based (học π trực tiếp, on-policy); DQN value-based (học Q(s,a), off-policy + replay buffer).

**Q12. Replay Buffer là gì? Vì sao DQN cần?**
A: Bộ nhớ lưu transition (s,a,r,s') cũ. Lấy mẫu ngẫu nhiên giúp phá vỡ tương quan thời gian, tăng hiệu quả sử dụng dữ liệu.

**Q13. Target Network để làm gì?**
A: Tránh phân kỳ Q-value. Dùng mạng cố định (sync 2K steps) để tính target r + γ·max Q(s',a'), tránh "chasing its own tail".

**Q14. Epsilon-greedy là gì?**
A: Cơ chế exploration của DQN. Với xác suất ε chọn random, 1-ε chọn argmax. ε giảm từ 1.0 → 0.05 trong 40% bước đầu.

**Q15. PPO clip ratio là gì?**
A: r(θ) = π_new/π_old. Nếu r vượt [1-0.2, 1+0.2], cắt lại. Mục đích: ngăn policy thay đổi quá mạnh trong 1 update.

**Q16. Advantage là gì?**
A: A(s,a) = Q(s,a) - V(s). Action a tốt hơn trung bình bao nhiêu. PPO dùng để hướng update đúng hướng.

**Q17. Actor-Critic là gì?**
A: Kiến trúc PPO. Actor π(a|s) ra quyết định; Critic V(s) đánh giá state để tính advantage.

**Q18. n_steps=1024, n_epochs=10 nghĩa là gì?**
A: PPO thu rollout 1024 step, sau đó tái sử dụng rollout đó 10 epoch (≈ 160 update) trước khi thu rollout mới.

**Q19. Vì sao DQN train 1M timesteps trong khi PPO chỉ 500K?**
A: DQN học từ transition đơn lẻ (bootstrap target network nhiễu hơn). Cần thời gian lấp đầy buffer + bao phủ state space.

**Q20. Vì sao chọn cả PPO và DQN?**
A: 2 trường phái đại diện (policy-based vs value-based). So sánh có ý nghĩa học thuật. Action space rời rạc nhỏ phù hợp cả 2.

## C. Câu hỏi về kết quả + đồ án (10 câu)

**Q21. PPO đạt mean reward bao nhiêu? Hơn baseline bao nhiêu %?**
A: PPO 80.07, Rule-based 36.07. Cải thiện ≈ 122% (= (80.07-36.07)/36.07 × 100).

**Q22. Vì sao Std cao như vậy (>200)?**
A: Môi trường stochastic — mỗi episode random 1 scenario có reward range rất khác nhau (S1 +600, S5 -300). Std cao là bản chất, không phải model kém.

**Q23. Best checkpoint là gì? Khác final model thế nào?**
A: Best = mô hình có mean reward cao nhất trong các lần đánh giá. Final = mô hình tại bước cuối. Dùng best vì RL có thể bị policy degradation.

**Q24. Vì sao train offline thay vì online?**
A: An toàn (không hỏng prod), tái lập (dataset cố định), tiết kiệm (không cần duy trì cluster lâu).

**Q25. Action appropriateness đạt bao nhiêu %?**
A: PPO, DQN, Rule-based đều 100%. Có nghĩa agent luôn chọn action match với domain logic.

**Q26. Khi inference online làm sao agent hoạt động?**
A: 4 bước: (1) thu telemetry từ Istio + Prometheus, (2) predict action deterministic, (3) cập nhật DestinationRule + VirtualService qua K8s API, (4) chờ control interval rồi lặp.

**Q27. Cấu hình tĩnh có thể thay RL được không?**
A: Không, vì điều kiện vận hành thay đổi. Đã chứng minh: PPO SLO 92% > Rule-based 74%, error rate cũng thấp hơn.

**Q28. Vì sao reward có 5 thành phần?**
A: Để cân bằng đa mục tiêu: throughput cao, latency thấp, error thấp, FP thấp, học nhanh. Nếu chỉ 1 mục tiêu, agent sẽ overfit.

**Q29. Reward action_bonus có làm RL trở thành rule-based không?**
A: Không. Bonus chỉ +2 trong tổng reward range -18 đến +7. Agent vẫn có thể chọn khác. Đây là tín hiệu định hướng, không ép buộc.

**Q30. Hạn chế của đồ án là gì?**
A: (1) cb_state chỉ {0,1} không có Half-Open, (2) reward function thiết kế thủ công, (3) chưa test trên hệ thống production thật, (4) 5 scenario có thể chưa bao quát hết tình huống thực tế.

## D. Câu hỏi bẫy (5 câu)

**Q31. Tại sao không train trực tiếp trên K8s?**
A: 4 lý do: (1) ảnh hưởng prod, (2) chậm, (3) không tái lập, (4) tốn cluster.

**Q32. Reward âm nhiều có vấn đề không?**
A: Không. Quan trọng là reward TƯƠNG ĐỐI giữa các action ở cùng state. Reward âm chỉ là scaling.

**Q33. Nếu thầy hỏi "Sao std cao thế?" — kết luận model có hội tụ?**
A: Có, vì khi balanced eval (1000 episode đều scenario), PPO mean = 34.60, DQN = 34.62 — chênh lệch 0.01, rất ổn định. Std cao là do scenario phân phối, không phải do model.

**Q34. PPO hay DQN tốt hơn?**
A: Phụ thuộc tiêu chí. PPO mean reward và SLO compliance cao hơn. DQN error rate thấp hơn. Khi balanced, hai model tương đương (chênh 0.01).

**Q35. Sao không dùng SAC hay A3C?**
A: SAC cho continuous action — đề tài này discrete. A3C phức tạp hơn, không cần thiết. PPO + DQN đủ đại diện 2 trường phái.

---

# 📋 CHEAT SHEET CUỐI CÙNG (in ra mang đi)

## Số liệu bắt buộc thuộc

| | PPO | DQN |
|---|---|---|
| LR | 3e-4 | 5e-5 |
| Total timesteps | 500K | 1M |
| Buffer | rollout 1024 | replay 200K |
| Batch | 64 | 64 |
| γ | 0.99 | 0.99 |
| Special | clip=0.2, GAE λ=0.95, n_epochs=10 | tau=0.005, target_update=2K, ε 1.0→0.05 |

## State 7D
`[rps, p50, p90, p99, err_rate, cb_state, retry_rate]` — chuẩn hoá [0,1]

## Action 5
`Emergency(0) | Strict(1) | Moderate(2-baseline) | Relaxed(3) | Liberal(4)`

## Reward 5 phần
`throughput(0..+5) + latency(0/-5) + error(0..-10) + FP(0/-3) + bonus(0/+2)`

## Kết quả chính
- PPO: 80.07 (mean), SLO 92%
- DQN: 18.04, err 0.165
- Rule-based: 36.07, SLO 74%

## 3 keyword "thầy thích nghe"
1. **Best checkpoint** chống policy degradation
2. **Stochastic environment** giải thích std cao
3. **Action appropriateness 100%** — agent học đúng

---

# 🎯 LỜI KHUYÊN CUỐI

1. **Đừng học thuộc lòng** — hiểu logic. Thầy hỏi xoáy là lộ ngay.
2. **Khi không biết**: nói "Theo nhóm em hiểu/quan sát thấy..." thay vì im lặng.
3. **Khi bí**: quay về số liệu cụ thể (mean reward, SLO%, scenarios). Số nói thay lý thuyết.
4. **Nhấn mạnh**: best checkpoint, scenario stochastic giải thích std, action appropriateness 100%, RL > Rule-based.
5. **Ngủ đủ giấc** > đọc thêm 1 tiếng. Não cần consolidate.

**CHÚC BẠN BẢO VỆ THÀNH CÔNG! 🚀**
