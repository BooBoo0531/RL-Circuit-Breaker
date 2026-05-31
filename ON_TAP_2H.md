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
| **Phase 6** | 110–120 phút | Q&A — 65+ câu hỏi vấn đáp với đáp án chi tiết |

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

### 🔥 Ví dụ minh hoạ cascade failure (rất hay được hỏi)

Giả sử bình thường mỗi service trả về trong 50ms. Đột nhiên `ratings` bị nghẽn (DB overload) → mất 5 giây mới trả lời.

**Không có Circuit Breaker**:
```
T=0s:    100 user vào trang productpage
T=0.05s: productpage → reviews (đợi)
T=0.05s: reviews → ratings (đợi 5s)
T=0.5s:  thêm 100 user vào → 100 thread reviews mới đợi ratings
T=2s:    400 user trong hệ thống, tất cả thread đều block trên ratings
T=5s:    thread pool reviews cạn kiệt → reviews từ chối connection mới
T=5s:    productpage cũng bị block → toàn bộ web sập
```
→ 1 service chậm → KÉO theo cả hệ thống chết. Đây là **cascade failure**.

**Có Circuit Breaker** (consecutiveErrors=3):
```
T=0s:    ratings bắt đầu chậm
T=0.5s:  3 request liên tiếp tới ratings vượt timeout → CB mở
T=0.5s:  reviews KHÔNG gọi ratings nữa, return ngay với fallback
         (vd: trả "no rating available")
T=0.5s:  productpage vẫn render được, chỉ thiếu phần rating
T=10s:   CB chuyển HALF-OPEN, thử 1 request → ratings đã hồi phục
T=10s:   CB đóng lại, hệ thống normal
```
→ CB như "fuse" — hỏng 1 phần, các phần khác vẫn sống.

## 1.2. Circuit Breaker là gì? (8 phút)

**Ý tưởng**: Giống cầu dao điện trong nhà — khi điện chập, cầu dao tự ngắt để bảo vệ hệ thống.

### Analogy cầu dao điện vs Circuit Breaker phần mềm

| Cầu dao điện trong nhà | Circuit Breaker phần mềm |
|---|---|
| Khi điện chập (quá dòng) → cầu dao nhảy | Khi service lỗi nhiều → CB mở |
| Cắt điện vào nhà → bảo vệ thiết bị khác | Cắt request → bảo vệ caller, không kéo theo |
| Sau 1 lúc bật lại thử | Sau `baseEjectionTime` chuyển HALF-OPEN |
| Nếu vẫn chập → nhảy lại | Nếu vẫn lỗi → quay về OPEN |

**3 trạng thái**:
- `CLOSED` (đóng): Bình thường — request đi qua thoải mái
- `OPEN` (mở): Đã ngắt — từ chối tất cả request (fail-fast, không gọi service đang lỗi)
- `HALF-OPEN`: Thử nghiệm — cho 1 ít request đi qua xem service đã hồi phục chưa

**Sơ đồ chuyển trạng thái**:
```
        đủ N lỗi liên tiếp
CLOSED ─────────────────→  OPEN
   ↑                         │
   │ thành công               │ chờ baseEjectionTime
   │                         ↓
   └──────────────────── HALF-OPEN
        thử 1 request, lỗi → quay lại OPEN
```

**Cấu hình quan trọng** (trong Istio DestinationRule):
- `consecutiveErrors`: bao nhiêu lỗi liên tiếp thì OPEN (vd: 3, 5, 10)
- `interval`: khoảng thời gian theo dõi
- `baseEjectionTime`: thời gian giữ OPEN trước khi thử HALF-OPEN

**Ví dụ YAML cấu hình Istio thực tế** (đề tài dùng):
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ratings-cb
spec:
  host: ratings
  trafficPolicy:
    outlierDetection:
      consecutiveErrors: 5      # ← tham số agent điều chỉnh
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 100
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ratings-vs
spec:
  hosts: [ratings]
  http:
  - route:
    - destination: { host: ratings }
    retries:
      attempts: 2               # ← tham số agent điều chỉnh
      perTryTimeout: 2s
```

**Retry** (trong Istio VirtualService):
- `retryAttempts`: nếu request lỗi, thử lại bao nhiêu lần (vd: 1, 2, 3)
- `perTryTimeout`: timeout cho mỗi lần thử

**Lưu ý quan trọng**: agent không tự "viết" YAML, mà chọn 1 trong 5 preset (Emergency...Liberal). Mỗi preset là 1 cặp `(consecutiveErrors, retryAttempts)` cố định.

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

### 🎮 Ví dụ Mario chi tiết (để hiểu trial-and-error)

Vòng 1: Mario gặp con Goomba
- Action ngẫu nhiên: chạy thẳng → đụng Goomba → chết → reward = -1
- Agent ghi nhớ: "ở state này (Goomba phía trước), action 'chạy thẳng' → -1"

Vòng 2: Cùng state đó
- Action ngẫu nhiên (vì chưa biết): nhảy → qua được Goomba → reward = +0.5
- Agent ghi nhớ: "action 'nhảy' tốt hơn 'chạy thẳng' ở state này"

Vòng 100: Cùng state đó
- Agent đã học: chọn 'nhảy' với xác suất 95% (exploit), 5% chọn khác (explore)

**Bài học cho đồ án CB**:
- State = "err_rate cao 0.4, p99 cao 800ms"
- Vòng đầu agent random → có khi chọn Liberal → reward âm to
- Sau nhiều lần học → ở state này luôn chọn Emergency/Strict

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

### 🧮 Ví dụ tính Bellman bằng số (rất tốt cho vấn đáp)

Giả sử agent đang ở state s, có 2 action {a₀, a₁}, γ=0.9.

**Bước hiện tại**:
- Chọn a₀ → nhận reward r=+1 → đến state s'

**Tại s' agent có 2 lựa chọn** (giả sử Q-table đã học):
- Q(s', a₀) = 5
- Q(s', a₁) = 8 ← max

**Áp dụng Bellman**:
```
Q(s, a₀) = r + γ · max_{a'} Q(s', a')
        = 1 + 0.9 × 8
        = 1 + 7.2
        = 8.2
```

**Nếu γ=0.99** (đề tài): Q(s,a₀) = 1 + 0.99×8 = 8.92 → tương lai có trọng số CAO hơn.
**Nếu γ=0.5** (myopic): Q(s,a₀) = 1 + 0.5×8 = 5 → quan tâm tương lai ÍT hơn.

→ Đây chính là tín hiệu mà DQN dùng để cập nhật mạng neural.

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

### 🔄 Luồng DQN train 1 step (rất hay được hỏi)

```
1. ENV trả state s_t
2. Với xác suất ε: a_t = random; với 1-ε: a_t = argmax Q_online(s_t)
3. ENV step: nhận r_t, s_{t+1}, done
4. Lưu (s_t, a_t, r_t, s_{t+1}, done) vào replay buffer
5. Nếu step ≥ learning_starts (10K) VÀ step % train_freq == 0 (mỗi 4 step):
   a. Sample minibatch 64 transitions ngẫu nhiên từ buffer
   b. Tính target:
      y = r + γ · max_{a'} Q_target(s', a')   (nếu không done)
      y = r                                    (nếu done)
   c. Loss = MSE(Q_online(s,a), y)
   d. Backprop, cập nhật Q_online
6. Mỗi 2000 step: soft-sync Q_target ← τ·Q_online + (1-τ)·Q_target  (τ=0.005)
7. Cập nhật ε theo lịch tuyến tính (1.0 → 0.05 trong 40% bước đầu)
```

**Ví dụ số cụ thể**:
- step=15000 (đã warmup xong, bắt đầu train)
- ε hiện tại: 1.0 - (15000/400000)·0.95 = 1.0 - 0.0356 = 0.964 → vẫn explore mạnh
- step=400000: ε = 0.05 → chủ yếu exploit
- step=1000000: ε = 0.05 (đã đông cứng)

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

### 🧮 Ví dụ PPO clip bằng số (hay được hỏi)

Giả sử ở state s, action a có:
- π_old(a|s) = 0.5 (policy cũ chọn a với xác suất 50%)
- Sau khi học vài bước, π_new(a|s) = 0.8

**Tỉ lệ**: r(θ) = 0.8 / 0.5 = **1.6** → vượt giới hạn 1.2

**Trường hợp 1**: Advantage A > 0 (action tốt)
- Gốc: r·A = 1.6·A
- Clip: clip(1.6, 0.8, 1.2)·A = 1.2·A ← bị cắt
- min(1.6·A, 1.2·A) = 1.2·A → KHÔNG cho push policy mạnh hơn nữa

**Trường hợp 2**: Advantage A < 0 (action xấu)
- Gốc: r·A = 1.6·A (rất âm)
- Clip: 1.2·A (âm vừa)
- min(1.6·A, 1.2·A) = 1.6·A ← KHÔNG bị cắt
- → vẫn cho phép push policy GIẢM action xấu mạnh

**Bài học**: Clip giới hạn **upside** (tránh hưng phấn quá), không giới hạn **downside** (luôn cho phép sửa lỗi). Đây là asymmetric clip — đặc trưng của PPO.

### 🔄 Luồng PPO train 1 update cycle

```
1. ROLLOUT (2048 step):
   for t in 0..2047:
     - state s_t → π_old(a|s_t) → sample a_t
     - critic V(s_t)
     - env step → r_t, s_{t+1}
     - lưu (s_t, a_t, r_t, V(s_t), log_prob_old(a_t))
2. Tính Advantage cho từng step bằng GAE (λ=0.95):
   δ_t = r_t + γV(s_{t+1}) - V(s_t)
   A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
3. UPDATE (n_epochs=10):
   for epoch in 1..10:
     shuffle 2048 transitions
     for batch in 32 minibatch (mỗi 64):
       - forward π_new, V_new
       - r(θ) = exp(log_prob_new - log_prob_old)
       - L_clip = -min(r·A, clip(r, 0.8, 1.2)·A)
       - L_value = MSE(V_new, return_target)
       - L_ent = -ent_coef·H(π)  (khuyến khích exploration)
       - Loss = L_clip + 0.5·L_value + L_ent
       - backprop, clip grad ở 0.5
4. XOÁ rollout (on-policy), thu rollout mới
```

**Số liệu cụ thể**: 500K timesteps / 2048 = ~244 update cycles. Mỗi cycle có 10·32 = 320 gradient updates → tổng ~78K gradient updates.

### Hyperparameter quan trọng (đề tài)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `learning_rate` | 3e-4 | Standard PPO lr |
| `n_steps` | 2048 | Rollout dài 2048 step trước mỗi update |
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

### 🎬 Walkthrough 1 episode đầy đủ (rất quan trọng cho vấn đáp)

Giả sử `env.reset()` → random scenario S5 (Abort 40%).

**Step 0**:
- Row dataset: rps=18, p50=120, p90=400, p99=850, err_rate=0.42, cb=0, rt_rate=0.15
- State chuẩn hoá: [0.18, 0.12, 0.40, 0.85, 0.42, 0, 0.15]
- Agent (PPO đã train) predict: action=1 (Strict)
- Reward:
  - throughput = 0.18·5 = 0.9
  - latency = -5 (p99=850 > 500)
  - error = -10·0.42 = -4.2
  - fp = 0 (action=1 nhưng err=0.42 ≥ 0.05, không phải FP)
  - bonus = +2 (err>0.3 và action≤1) ✓
  - **total = 0.9 - 5 - 4.2 + 0 + 2 = -6.3**

**Step 1**:
- Row khác (cùng S5): rps=15, p99=920, err=0.45
- Agent: action=0 (Emergency)
- Reward = 0.75 - 5 - 4.5 + 0 + 2 = **-6.75**

**...lặp lại 100 step...**

**Cuối episode**: Tổng reward ≈ 100·(-6) ≈ -600 (vì S5 toàn khó).
- Đây là lý do **std cao**: nếu reset() trả S1 (Normal), tổng có thể +500. Chênh lệch 1000+ giữa các episode.

**So sánh nếu agent chọn sai action** (vd action=4 Liberal khi err=0.42):
- bonus = 0 (không match)
- Các thành phần khác giữ nguyên → reward = -6.3 - 2 = -8.3
- Cumulative qua 100 step: kém hơn 200 điểm → agent học được "không nên Liberal khi err cao"

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

# 🎛 PHASE 4B: PHÂN TÍCH SENSITIVITY THAM SỐ (bonus 5 phút)

> Thầy hay hỏi "Sao chọn giá trị này? Tăng/giảm thì sao?". Bảng dưới là answer.

## PPO — sensitivity từng tham số

| Tham số | Value | Vai trò | Nếu TĂNG | Nếu GIẢM |
|---|---|---|---|---|
| learning_rate | 3e-4 | Tốc độ cập nhật weight | Học nhanh nhưng dễ phân kỳ; loss dao động | Học chậm; có thể không hội tụ trong 500K |
| n_steps | 2048 | Độ dài rollout/update | Advantage estimate ổn định hơn nhưng tốn RAM | Update sớm hơn nhưng nhiễu, variance cao |
| n_epochs | 10 | Tái sử dụng rollout | Vắt kiệt rollout, có thể overfit batch → policy lệch | Lãng phí dữ liệu, học chậm |
| batch_size | 64 | Minibatch trong epoch | Gradient mượt hơn, chậm hơn | Gradient nhiễu, có thể giúp escape local minima |
| gamma | 0.99 | Discount factor | Quan tâm tương lai xa → khó học (variance ↑) | Myopic → bỏ qua lợi ích dài hạn |
| gae_lambda | 0.95 | GAE smoothing | Bias ↓, variance ↑ | Bias ↑, variance ↓ |
| clip_range | 0.2 | Giới hạn r(θ) | Cho update mạnh → có thể sụp đổ policy | Update yếu → học chậm, an toàn |
| ent_coef | 0.01 | Entropy bonus | Exploration mạnh → policy không hội tụ | Exploitation sớm → kẹt local optimum |
| max_grad_norm | 0.5 | Clip gradient | Cho gradient lớn → nguy cơ nổ | Quá an toàn, học chậm |

## DQN — sensitivity từng tham số

| Tham số | Value | Vai trò | Nếu TĂNG | Nếu GIẢM |
|---|---|---|---|---|
| learning_rate | 5e-5 | Tốc độ cập nhật Q | Q-value phân kỳ (lý do dùng nhỏ hơn PPO 6 lần) | Học rất chậm |
| buffer_size | 200000 | Bộ nhớ kinh nghiệm | Đa dạng cao, RAM tốn | Quên kinh nghiệm cũ, dễ overfit gần đây |
| learning_starts | 10000 | Warmup | Buffer đủ đa dạng trước khi train | Train sớm với data nghèo → bias |
| batch_size | 64 | Minibatch | Gradient mượt | Gradient nhiễu |
| tau | 0.005 | Soft update target | Target đuổi online nhanh → mất ổn định | Target quá lag, bias cao |
| gamma | 0.99 | Discount | Tương lai xa | Myopic |
| train_freq | 4 | Tần suất train | Train ít → exploration nhiều | Train nhiều → overfit batch hiện tại |
| target_update_interval | 2000 | Sync target hard | Target quá lag (bias) | Mất tính ổn định của target network |
| exploration_fraction | 0.4 | % training để decay ε | Explore lâu → học muộn nhưng đầy đủ | Exploit sớm → kẹt local |
| eps_final | 0.05 | ε cuối | Vẫn explore nhiều → policy không stable eval | Greedy hoàn toàn → có thể bỏ lỡ best action |

## Câu hỏi điển hình thầy có thể hỏi

**"Sao DQN learning_rate (5e-5) nhỏ hơn PPO (3e-4)?"**
→ DQN có bootstrap (Q target = r + γ·max Q'). Nếu lr lớn, error trong target nhân lên qua các update → phân kỳ Q. PPO không bootstrap policy gradient nên chịu lr lớn hơn được.

**"Sao tau=0.005 (rất nhỏ)?"**
→ Target network cần "lag" để stable. Tau=0.005 nghĩa mỗi update target chỉ kéo 0.5% về online. Nếu tau=1.0 → target = online luôn → mất ý nghĩa.

**"Sao exploration_fraction=0.4 mà không 0.1?"**
→ State space 7D liên tục, 5 scenario khác nhau. Cần 40% × 1M = 400K step để bao phủ đa dạng trước khi exploit. Nếu giảm 0.1 → exploit sớm khi buffer còn nghèo → suboptimal.

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

# 🔴 PHASE 6: 65+ Q&A VẤN ĐÁP THƯỜNG GẶP (10 phút cuối + tham khảo)

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

**Q18. n_steps=2048, n_epochs=10 nghĩa là gì?**
A: PPO thu rollout 2048 step, sau đó tái sử dụng rollout đó 10 epoch (≈ 320 update với batch_size=64) trước khi thu rollout mới.

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

## E. Câu hỏi NÂNG CAO — kỹ thuật chuyên sâu (15 câu)

**Q36. Vì sao reward action_bonus không làm thuật toán biến thành rule-based?**
A: Rule-based = mapping cố định state → action, không học. Còn ở đây, agent vẫn phải HỌC mapping qua gradient descent. Bonus +2 chỉ là 1 trong 5 thành phần reward (range tổng -18 → +7), agent có thể "bỏ qua" bonus nếu thành phần khác trade-off tốt hơn. Bằng chứng: action distribution của PPO không giống hệt rule (có biến thể tuỳ state lân cận).

**Q37. GAE λ=0.95 nghĩa là gì? Tại sao không dùng λ=1?**
A: GAE (Generalized Advantage Estimation) là weighted average của n-step advantages. λ=1 → Monte Carlo (variance cao, bias thấp). λ=0 → 1-step TD (variance thấp, bias cao). λ=0.95 cân bằng — kế thừa độ chính xác của MC nhưng giảm variance đáng kể, đặc biệt khi episode dài 100 step.

**Q38. Tại sao PPO dùng cả Actor và Critic? Bỏ Critic được không?**
A: Không. Critic V(s) cung cấp baseline cho Advantage A=Q-V. Nếu không có Critic → không tính được Advantage → policy gradient có variance cực cao. Critic giảm variance mà không thay đổi expected gradient (đây là tính chất của baseline).

**Q39. Replay buffer 200K lấp đầy mất bao lâu?**
A: 200K transitions / (1 transition mỗi step) = 200K step. Với learning_starts=10K, từ step 10K agent bắt đầu train với buffer chỉ 10K. Phải đến step 200K buffer mới đầy. Sau đó FIFO — transition cũ nhất bị đẩy ra. Đây là lý do DQN cần 1M timesteps.

**Q40. Cb_state trong state vector chỉ {0,1} có hạn chế gì?**
A: Có. Istio thực tế có HALF-OPEN nhưng dataset offline chỉ thu được CB ở trạng thái CLOSED (cb=0). Hạn chế: agent không học được hành vi khi CB đang HALF-OPEN. Đây là 1 trong 4 limitations đã ghi trong báo cáo.

**Q41. Khi inference online, control interval là bao lâu?**
A: Trong báo cáo đề xuất ~30s (tương đương `interval` của Istio outlierDetection). Mỗi 30s thu telemetry → predict action → cập nhật DestinationRule. Quá nhanh → cluster không kịp ổn định. Quá chậm → không phản ứng kịp khi traffic thay đổi.

**Q42. Vì sao chọn Stable-Baselines3 thay vì self-implement?**
A: SB3 là implementation chuẩn của PPO/DQN, được benchmark kỹ. Self-implement dễ sai bug subtle (vd: GAE off-by-one, target sync sai). Mục tiêu đề tài là so sánh PPO vs DQN, không phải re-implement RL → dùng SB3 hợp lý.

**Q43. Why deterministic=True khi evaluate?**
A: Khi train dùng stochastic policy (sample) để explore. Khi eval dùng argmax để có kết quả tái lập, không nhiễu bởi sampling. Điều này đảm bảo so sánh PPO vs DQN công bằng (đều argmax).

**Q44. Vì sao Welch's t-test mà không phải Student?**
A: Welch không giả định variance bằng nhau giữa 2 nhóm. Trong kết quả, std PPO=234 và DQN=231 không quá khác nhau, nhưng Welch an toàn hơn — không sai khi std bằng, chính xác khi std lệch.

**Q45. p > 0.05 giữa PPO và DQN nghĩa là gì?**
A: Không bác bỏ được null hypothesis "PPO = DQN". KHÔNG đồng nghĩa "PPO = DQN", chỉ là **không đủ bằng chứng** để khẳng định khác. Khi balanced eval, mean chênh 0.01 → thực sự tương đương.

**Q46. Có overfitting không? Làm sao biết?**
A: Có theo dõi qua eval callback (đánh giá định kỳ trên cùng env nhưng với deterministic policy). Nếu train reward tăng nhưng eval reward giảm → overfit. Đề tài không thấy gap rõ → best checkpoint capture được mô hình eval tốt nhất.

**Q47. Vì sao không dùng frame stacking (lịch sử)?**
A: State đã có đủ thông tin runtime tức thời (rps, latency, err, cb, retry). Bài toán có tính Markov xấp xỉ tốt — cấu hình CB phụ thuộc trạng thái hiện tại của metrics, không cần lịch sử dài. Frame stacking sẽ tăng input dim, tăng compute mà không lợi rõ.

**Q48. Reward bonus bị "leak" thông tin từ ground-truth không?**
A: Không leak. Bonus dựa trên `err_rate` mà state vector cũng chứa err_rate → agent đã thấy thông tin này. Bonus chỉ là tín hiệu định hướng, không cho thông tin "tương lai" mà state không có.

**Q49. Cận biên (boundary) action_bonus tại err=0.05 và err=0.3 có gây discontinuous không?**
A: Có discontinuous nhỏ (jump +2). Nhưng vì là 1/5 thành phần và state vẫn liên tục, neural network học smooth được. Có thể smooth bằng sigmoid nhưng đề tài giữ binary cho đơn giản, dễ explain.

**Q50. Nếu thay max_rps=1000 thành max_rps=100 (calibrate đúng dataset), kết quả thay đổi sao?**
A: Throughput component sẽ scale lớn hơn (rps thực ~84 → throughput sẽ gần 4 thay vì 0.4 trước normalize). Reward signal mạnh hơn → có thể học nhanh hơn. Đây là 1 trong các todo improvement đã ghi note trong code.

## F. Câu hỏi VẬN HÀNH thực tế (10 câu)

**Q51. Khi deploy production, agent chạy như nào?**
A: Đóng gói model `.zip` vào container Python. Chạy như sidecar/daemon: (1) gọi Prometheus API thu metrics mỗi 30s, (2) load model, predict action, (3) gọi Kubernetes API patch DestinationRule với preset tương ứng, (4) lặp lại.

**Q52. Nếu agent predict ra action gây sự cố, làm sao rollback?**
A: 3 cơ chế bảo vệ: (1) Safety guardrail — nếu err_rate đột nhiên tăng sau action mới → revert. (2) Human override — admin có thể disable agent, dùng default Moderate. (3) Action rate limiting — không được đổi action quá thường xuyên (vd ≥5 phút giữa các change).

**Q53. Làm sao monitor agent đang chạy?**
A: Log mỗi quyết định (state, action, predicted Q hoặc π) vào Prometheus. Tạo dashboard Grafana hiển thị action distribution theo thời gian, reward implicit (có thể tính từ telemetry). Alert nếu agent stuck 1 action quá lâu.

**Q54. Khi nào cần retrain agent?**
A: (1) Topology dịch vụ thay đổi (thêm service mới). (2) SLO thay đổi (vd p99 < 300ms thay vì 500ms). (3) Pattern traffic thay đổi (vd peak khác). (4) Drift detect — eval offline trên dataset gần đây thấy reward giảm.

**Q55. Có thể dùng cho service mesh khác Istio không?**
A: Có. Action 5 preset là abstract (Emergency...Liberal). Chỉ cần map sang config của mesh đó: Linkerd có `failure_accrual`, Consul có `passing` threshold. State vector cũng gen từ metrics chung (Prometheus chuẩn).

**Q56. Một cluster lớn có 100 service, mỗi service 1 agent — có scale được không?**
A: Có. Mỗi agent độc lập, state local. Có thể chạy 1 process Python load 100 model. Hoặc shared model nếu service homogenous. Bottleneck thực tế là latency Prometheus query, không phải inference (predict <1ms).

**Q57. Sao không dùng Multi-agent RL?**
A: Đề tài đơn service (per-service CB). MARL phức tạp hơn (joint state, credit assignment), benefit không rõ khi mỗi service có CB độc lập. Future work có thể coordinate giữa các CB liên quan.

**Q58. Cost computational của training?**
A: PPO 500K step ~ 1-2 giờ trên CPU thường (không cần GPU vì state nhỏ, MLP nhỏ). DQN 1M step ~ 3-4 giờ. Inference: <1ms/predict. Memory: replay buffer 200K × 7 float = ~5.6MB.

**Q59. Bookinfo là demo nhỏ, kết quả có generalize?**
A: Bookinfo nhỏ nhưng có 4 service và inter-dependency (productpage→reviews→ratings) — đủ để có cascade. Logic CB không phụ thuộc số service. Chính sách `state→action` học được áp dụng cho service nào có metrics tương tự (rps, latency, err).

**Q60. Nếu thầy yêu cầu chứng minh "RL thực sự cần thiết", nói gì?**
A: 3 luận điểm: (1) **Số liệu**: PPO mean reward 80.07 vs Rule-based 36.07 (cải thiện 122%); SLO 92% vs 74%. (2) **Adaptive**: Rule-based mỗi khi traffic pattern thay đổi phải tune lại thủ công, RL tự thích nghi qua reward. (3) **Generalization**: 5 scenario rất khác nhau, RL học mapping toàn cục, rule-based phải viết riêng cho từng case.

## G. Câu hỏi BẪY/PHẢN BIỆN (5 câu)

**Q61. "Em chỉ chứng minh trên 5 scenario, ngoài đời có vô số tình huống"**
A: Đúng, đó là limitation đã thừa nhận. Tuy nhiên, 5 scenario được chọn cover 2 trục chính: latency (delay 200/800ms) và error (abort 20%/40%). Chính sách học được dựa trên METRICS (state) — bất kể scenario nào tạo ra metrics tương tự, agent đáp ứng được. Future work: thêm scenario phức tạp hơn (correlated failure, tăng đột biến RPS).

**Q62. "Reward function chính em viết — vậy có khác gì rule không?"**
A: Khác về **cách dùng**. Rule-based: viết IF/THEN trực tiếp ra action. Reward function: viết tín hiệu chỉ dẫn (loss/reward), agent học MAPPING qua tối ưu hoá. Reward function có thể có nhiều mục tiêu mâu thuẫn (throughput vs error) — neural network học cân bằng tốt hơn human-tuned IF/THEN.

**Q63. "Tại sao em không thử continuous action (slider 0-1 cho consecutiveErrors)?"**
A: 2 lý do: (1) Istio config thực tế là số nguyên (consecutiveErrors là int). Discretize sẵn 5 preset là natural. (2) Discrete dễ explain với DevOps — preset Emergency/Strict/... có ngữ nghĩa, không phải số float khó hiểu.

**Q64. "Std cao chứng tỏ model unreliable"**
A: Sai logic. Std cao là do **environment stochastic** (5 scenario reward range khác nhau), không phải model nhiễu. Bằng chứng: balanced eval 1000 episode, PPO=34.60, DQN=34.62 — chênh 0.01 → cực kỳ stable. Std cao trong report là phản ánh tính phong phú của input, không phải bất ổn của output.

**Q65. "DQN error rate (0.165) tốt hơn PPO (0.212), sao chọn PPO?"**
A: Mục tiêu chính là **tối đa hoá reward tổng** (multi-objective). PPO mean reward 80.07 cao hơn rõ DQN 18.04. PPO trade-off chấp nhận err nhỉnh hơn để đạt throughput cao và FP thấp. Tuỳ ưu tiên: nếu đặt err lên đầu thì DQN, nếu cân bằng thì PPO. Đề tài đề xuất PPO làm default vì compliance SLO 92% vs DQN 80%.

---

# 📋 CHEAT SHEET CUỐI CÙNG (in ra mang đi)

## Số liệu bắt buộc thuộc

| | PPO | DQN |
|---|---|---|
| LR | 3e-4 | 5e-5 |
| Total timesteps | 500K | 1M |
| Buffer | rollout 2048 | replay 200K |
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
