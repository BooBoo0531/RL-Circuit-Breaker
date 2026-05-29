# ÔN TẬP BÁO CÁO ĐỒ ÁN
## Reinforcement Learning for Adaptive Circuit Breaking and Retry Strategy in Kubernetes Microservices
### Nhóm 12 – NT549.Q21

---

## PHẦN 1 – CÁC KHÁI NIỆM NỀN TẢNG

---

### 1.1. Circuit Breaker là gì?

**Khái niệm:**
Circuit Breaker (CB) là một design pattern trong microservices, hoạt động giống như cầu dao điện trong nhà.
Khi một service bị lỗi quá nhiều, CB sẽ "ngắt mạch" — tức là từ chối nhận request mới để service đó có thời gian phục hồi, thay vì tiếp tục nhận request và làm tình trạng tệ hơn.

**3 trạng thái của Circuit Breaker:**
```
CLOSED   → Bình thường, request đi qua thoải mái
OPEN     → Đang ngắt, từ chối tất cả request (fail-fast)
HALF-OPEN→ Thử nghiệm, cho một ít request đi qua để kiểm tra
```

**Vấn đề của cấu hình tĩnh:**
- Cấu hình cố định không phù hợp với mọi tình huống
- Quá chặt → CB mở nhầm khi hệ thống đang ổn (false positive) → giảm throughput
- Quá lỏng → CB không kịp phản ứng khi lỗi tăng đột biến → cascade failure

**Giải pháp của đồ án:**
Dùng RL để tự động điều chỉnh cấu hình CB theo trạng thái thực tế của hệ thống.

---

### 1.2. Reinforcement Learning là gì?

**Khái niệm:**
RL (Học tăng cường) là phương pháp học máy trong đó một Agent học cách hành động thông qua thử và sai. Agent không được dạy trực tiếp mà tự học từ phần thưởng (reward) nhận được sau mỗi hành động.

**Vòng lặp cơ bản:**
```
Agent quan sát State → chọn Action → nhận Reward + State mới → học → lặp lại
```

**Áp dụng vào đồ án:**
```
State   = [RPS, p50, p90, p99, err_rate, cb_state, retry_rate]  (7 features)
Action  = 5 cấu hình CB: Emergency / Strict / Moderate / Relaxed / Liberal
Reward  = đánh giá chất lượng hệ thống sau khi áp dụng cấu hình đó
```

---

### 1.3. PPO là gì?

**Khái niệm:**
PPO (Proximal Policy Optimization) là thuật toán RL thuộc nhóm Policy Gradient — học trực tiếp policy (chiến lược hành động). PPO đặc biệt ở chỗ giới hạn mức độ thay đổi policy sau mỗi lần cập nhật để tránh học sai hướng.

**Ưu điểm:** Ổn định, phù hợp với bài toán liên tục, ít bị dao động trong quá trình train.

**Config đang dùng:**
```
learning_rate = 3e-4
n_steps       = 2048
batch_size    = 64
total_timesteps = 1,000,000
```

---

### 1.4. DQN là gì?

**Khái niệm:**
DQN (Deep Q-Network) là thuật toán RL thuộc nhóm Value-based — học giá trị Q(s,a) của từng cặp (trạng thái, hành động), rồi chọn action có Q cao nhất. DQN phù hợp với không gian action rời rạc (discrete) như bài toán này (5 action).

**Ưu điểm:** Hiệu quả với action space nhỏ, có replay buffer giúp học từ kinh nghiệm cũ.

**Config đang dùng:**
```
learning_rate        = 5e-5
buffer_size          = 200,000
exploration_fraction = 0.4   (40% đầu dùng để khám phá)
total_timesteps      = 1,000,000
```

---

## PHẦN 2 – CÁC CÂU HỎI VỀ TRAINING VÀ KẾT QUẢ

---

### 2.1. Model đã hội tụ chưa?

**Khái niệm hội tụ:**
Model hội tụ khi reward trung bình không còn tăng nhiều nữa và ổn định quanh một giá trị — giống như học sinh đã đạt điểm tối đa, học thêm cũng không tăng điểm.

**Dấu hiệu chưa hội tụ (kết quả hiện tại):**
```
PPO: Mean = 80.07,  Std = 234.42  → Std/Mean = 293%  (quá cao)
DQN: Mean = 18.04,  Std = 231.71  → Std/Mean = 1284% (rất cao)
```
Std lớn hơn Mean nhiều lần → reward dao động mạnh → chưa ổn định.

**Nguyên nhân:**
1. Số timesteps chưa đủ (500k → cần 1M+)
2. Môi trường stochastic: mỗi episode random 1 trong 5 scenario
3. Reward range quá rộng (-253 đến +323) do scenario chi phối

**Kết luận:** Cần train thêm với 1M timesteps và xem reward curve trước khi đánh giá.

---

### 2.2. Checkpoint là gì? Tại sao cần dùng best checkpoint?

**Khái niệm checkpoint:**
Checkpoint = bản lưu trọng số (weights) của model tại một thời điểm trong quá trình train.
Giống như chụp ảnh màn hình trò chơi — bạn có thể quay lại bất kỳ điểm nào.

**Vấn đề policy degradation:**
Trong RL, model không cải thiện đều đặn. Reward có thể tăng rồi lại giảm:
```
Reward
  │        ●●●
  │      ●    ●●
  │    ●         ●●●  ← đang giảm (policy degradation)
  │●●
  └────────────────→ Timestep
          ↑
    Best checkpoint — nên lấy model ở đây
```

**Hướng dẫn của thầy:**
> "Có xu hướng giảm thì lấy policy vào lúc đẹp nhất"
→ Không lấy model lúc train xong, mà lấy model lúc reward cao nhất.

**Trong code:**
```
EvalCallback → tự động đánh giá mỗi 10k steps
             → lưu best_model.zip nếu tốt hơn lần trước
             → đây chính là best checkpoint
```

**File lưu:**
```
logs/ppo_best_model/best_model.zip  ← PPO best checkpoint
logs/dqn_best_model/best_model.zip  ← DQN best checkpoint
ppo_circuit_breaker.zip             ← PPO final model (lúc kết thúc train)
dqn_circuit_breaker.zip             ← DQN final model
```

---

### 2.3. Số liệu "không ảo" nghĩa là gì?

**Khái niệm:**
Số liệu ảo = kết quả trông đẹp nhưng không phản ánh thực tế, thường do:

| Lỗi phổ biến | Ví dụ | Cách tránh |
|---|---|---|
| Đánh giá trên data đã train | Train và test cùng dataset | Dùng data/scenario chưa thấy |
| Cherry-pick | Chạy 10 lần, báo lần tốt nhất | Báo mean ± std của tất cả |
| Quá ít episodes | Đánh giá 5 episodes | Dùng 50+ episodes |
| Chỉ dùng deterministic | Eval luôn chọn action tốt nhất | Test cả stochastic |

**Áp dụng vào đồ án:**
- Dùng 50 episodes để đánh giá (đủ lớn)
- Báo cả Mean và Std (không giấu độ dao động)
- So sánh với baseline (Random, Rule-based) để có context

---

## PHẦN 3 – CÁC TIÊU CHÍ TRƯỚC KHI CHẠY THỰC NGHIỆM

---

### 3.1. System Metrics là gì?

**Khái niệm:**
System metrics = các chỉ số đo chất lượng hệ thống microservice thực tế khi agent đang điều khiển — không phải điểm reward của model.

| Metric | Ý nghĩa | Tốt hơn khi |
|--------|---------|-------------|
| **SLO Compliance** | % thời gian p99 latency < 500ms | Càng cao càng tốt |
| **Avg Error Rate** | Tỉ lệ request bị lỗi trung bình | Càng thấp càng tốt |
| **Throughput** | Lượng request xử lý được (RPS) | Càng cao càng tốt |
| **FP Rate** | % lần CB mở nhầm khi hệ thống ổn | Càng thấp càng tốt, lý tưởng = 0% |
| **Reaction Time** | Số bước phản ứng khi lỗi tăng đột biến | Càng nhanh càng tốt |

**So sánh RL vs Rule-based — thế nào là tốt hơn:**
RL không cần thắng tất cả metrics. Chỉ cần thắng ít nhất 2/5 mà không thua nặng các metrics còn lại.

Ví dụ kết quả máy Windows (đã đạt):
```
DQN SLO: 82% > Rule-based 78%       ✓ tốt hơn
DQN Error Rate: 0.179 < 0.202       ✓ tốt hơn
DQN FP Rate: 0% = Rule-based 0%     ✓ ngang bằng
→ DQN đạt tiêu chí system metrics
```

---

### 3.2. Action Appropriateness là gì? Tại sao phải ≥ 90%?

**Khái niệm:**
Action Appropriateness = tỉ lệ % số lần agent chọn đúng action phù hợp với tình trạng hệ thống, theo logic domain knowledge.

**Quy tắc đánh giá:**
```
Tình huống                  Action đúng              Action sai
────────────────────────────────────────────────────────────────
err_rate > 0.3 (lỗi cao)  → Emergency/Strict (0,1)  Relaxed/Liberal
err_rate < 0.05 (ổn định) → Relaxed/Liberal  (3,4)  Emergency/Strict
err_rate 0.05–0.3 (trung) → Strict/Moderate  (1,2,3) Emergency/Liberal
```

**Ví dụ trực quan:**
Hệ thống error rate = 0.5 (50% request lỗi) mà agent chọn Liberal (mở hết CB)
→ Request tiếp tục vào service đang quá tải → cascade failure thật trên K3s/Azure.

**Tại sao phải ≥ 90%:**
- Kết quả hiện tại đã đạt 100% (cả PPO, DQN, Rule-based)
- 90% là ngưỡng tối thiểu chấp nhận được — hơn 1/10 quyết định sai là không đáng tin
- Trên môi trường thực, action sai = hệ thống bị ảnh hưởng thật

---

### 3.3. Bảng độ khó và mức độ ưu tiên các tiêu chí

| # | Tiêu chí | Độ khó | Ưu tiên | Lý do |
|---|----------|:------:|:-------:|-------|
| 4 | Action Appropriateness ≥ 90% | 3/10 | Cao nhất | Dễ đạt, đã có 100%, mất đi = model hỏng |
| 2 | RL tốt hơn Rule-based ≥ 1 metric | 5/10 | Cao | Bắt buộc cho kết luận đồ án |
| 1 | Reward curve tăng và bằng phẳng | 6/10 | Cao | Cần train đủ steps |
| 5 | Best checkpoint tốt hơn final | 4/10 | Trung bình | Quan trọng về phương pháp |
| 3 | Std ≤ 150 | 8/10 | Thấp nhất | Khó nhất do môi trường stochastic |

**Cần đạt 4/5 tiêu chí. Tiêu chí 2 và 4 là bắt buộc.**

---

## PHẦN 4 – QUY TRÌNH THỰC NGHIỆM

---

### 4.1. Thứ tự chạy sau khi train

```bash
# Bước 1: Train PPO (khoảng 20-40 phút)
python src/train_ppo.py

# Bước 2: Train DQN (khoảng 40-60 phút)
python src/train_dqn.py

# Bước 3: Kiểm tra reward curve
# Xem file ppo_reward_curve.png và dqn_reward_curve.png
# Moving average phải tăng và bằng phẳng ở cuối

# Bước 4: So sánh best checkpoint vs final model
python compare_checkpoints.py

# Bước 5: Đánh giá đầy đủ 11 metrics
python src/evaluate_comprehensive.py

# Bước 6: Chạy thực nghiệm trên K3s/Azure
```

---

### 4.2. Đọc kết quả compare_checkpoints.py như thế nào?

```
Best checkpoint : 95.20 +/- 180.30
Final model     : 60.10 +/- 220.50
Verdict         : BEST wins → policy degradation occurred, use best checkpoint
```

| Verdict | Ý nghĩa | Hành động |
|---------|---------|-----------|
| BEST wins | Policy bị degradation cuối train — thầy đúng | Dùng best checkpoint |
| SIMILAR | Không bị degradation | Dùng model nào cũng được |
| FINAL wins | EvalCallback có vấn đề hoặc eval_freq quá thưa | Kiểm tra lại callback |

---

### 4.3. Đọc kết quả reward curve như thế nào?

```
Tốt (hội tụ):              Chưa đủ:               Bị degradation:
    ___________              /\/\/\/\/\                /\
   /                        /                        /  \___
  /                        /                        /
 /                        /
```

- **Đường raw** (mờ): reward từng episode — dao động là bình thường
- **Moving average** (đậm): xu hướng chung — đây mới là thứ cần nhìn
- Nếu moving average tăng rồi bằng phẳng = hội tụ tốt
- Nếu moving average tăng rồi giảm = cần dùng best checkpoint

---

## PHẦN 5 – CÂU HỎI VẤN ĐÁP THƯỜNG GẶP

---

**Q: Tại sao dùng RL thay vì cấu hình tĩnh?**
A: Cấu hình tĩnh không thích nghi được với điều kiện thay đổi. Khi traffic tăng đột biến hay service bị lỗi, cấu hình tĩnh phản ứng không kịp hoặc phản ứng sai. RL quan sát trạng thái thực tế và điều chỉnh liên tục — giống người vận hành có kinh nghiệm thay vì quy trình cứng nhắc.

---

**Q: Tại sao chọn PPO và DQN, không dùng thuật toán khác?**
A: DQN phù hợp vì action space rời rạc nhỏ (5 action). PPO ổn định và phổ biến cho bài toán điều khiển liên tục. Hai thuật toán đại diện cho 2 trường phái khác nhau (value-based vs policy-based) nên so sánh có ý nghĩa học thuật.

---

**Q: Std cao như vậy có vấn đề không?**
A: Std cao một phần do môi trường stochastic — mỗi episode random 1 trong 5 scenario khác nhau, nên reward tự nhiên dao động. Quan trọng hơn là nhìn vào system metrics (SLO, error rate) để đánh giá chất lượng thực sự, thay vì chỉ nhìn mean reward.

---

**Q: RL thua Rule-based về mean reward thì kết luận thế nào?**
A: Mean reward không phải tiêu chí duy nhất. DQN đạt SLO Compliance 82% vs Rule-based 78%, và Error Rate thấp hơn. Rule-based luôn hành động 100% đúng về action appropriateness nhưng không tối ưu hóa được toàn bộ hệ thống. RL học được trade-off phức tạp hơn giữa các metrics.

---

**Q: Best checkpoint được lưu như thế nào?**
A: Dùng EvalCallback trong stable-baselines3. Cứ mỗi 10,000 steps, callback tự động chạy đánh giá 10 episodes. Nếu mean reward cao hơn lần trước thì lưu đè lên best_model.zip. Kết thúc training, best_model.zip chứa trọng số tại thời điểm model hoạt động tốt nhất.

---

**Q: Tại sao không train thêm cho đến khi hội tụ hoàn toàn?**
A: Đồ án có giới hạn thời gian (01/05 – 01/06/2026). Quan trọng hơn, môi trường stochastic với 5 scenario ngẫu nhiên sẽ luôn có Std cao nhất định — không thể loại bỏ hoàn toàn. Mục tiêu thực tế là đạt được xu hướng tăng rõ ràng và system metrics tốt hơn baseline, không nhất thiết phải hội tụ tuyệt đối.

---

**Q: Thực nghiệm trên K3s/Azure khác gì so với offline simulation?**
A: Offline simulation dùng dataset CSV để mô phỏng — nhanh, có thể lặp lại, nhưng không phản ánh đầy đủ hành vi hệ thống thực. K3s/Azure dùng Bookinfo thật với Istio thật, traffic do Fortio tạo ra, lỗi được inject thật — kết quả phản ánh đúng môi trường production hơn.

---

*File này được tổng hợp từ quá trình làm đồ án. Cập nhật lần cuối: 2026-05-30*
