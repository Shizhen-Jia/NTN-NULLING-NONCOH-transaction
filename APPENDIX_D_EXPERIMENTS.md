# Appendix D：E4–E8 可执行实验

## 1. λ 怎么替换，TN 约束怎么固定

**主实验固定 TN 要求，只扫描 NTN INR 门限 Γ；每个 Γ 都重新求整个联合策略。**

新波束使用附录 D 的硬约束，不再通过 λ 权衡泄漏：
$$
\max_v \operatorname{Re}(h^Hv),\quad
\operatorname{Im}(h^Hv)=0,\quad \|v\|_2\le1,\quad
C_r\|A_r^Hv\|_2+\rho_r\|v\|_2\le\sqrt{\Gamma},\ \forall r.
$$
这里 `v=√p w`，`p=||v||²` 是功率比例，实际发射功率为 `Pmax p`。
先用 SOCP 求各可观测状态/信息年龄下的服务值，再用完整历史占用测度 LP 联合选监听、服务/静默。
λ 仅在 E5 留作旧方法对照，不参与新联合优化。

主 Γ 扫描固定下列输入：

| 参数 | 含义 | 默认有限模型值 |
|---|---|---|
| θDL、θUL | TN 各截止窗口的需求 / 固定参考服务 | 0.40、0.50 |
| δTN | 每方向、每窗口允许未达标的概率 | 0.10 |
| δNTN | 每接收机允许 INR 超标的 DL 活动时间比例 | 0.08 |
| ΓdB | 被扫描的 INR 幅度门限 | −15、−10、−5、0 dB |
| 截止时间、调度、带宽、外部干扰、功率上限、初始信息 | 公共条件 | 各 Γ 相同 |

需求 `L=θ B_ref` 中的 `B_ref` 是**无监听空隙、无 NTN 约束的同一固定 TN 信道/调度的服务**，
它不随 Γ、待比较策略变化。默认有一个 TN 用户的 DL/UL 两类要求，分别在 tick 8 和12 检验：
$$\Pr\{B^{d}(0,t)<L^{d}(0,t)\}\le\delta^{TN},\quad d\in\{DL,UL\}.$$
不是把每一时刻 TN SINR 都强制不低于 −6 dB，也不是只要求平均总比特达标。
静态缓存补充实验没有时间轨迹，因此另外固定 `tn_snr_floor_db=-6` 做静态可行性标记，
不能与 E7 的窗口 QoS 混为一谈。

**Γ 降低是收紧瞬时保护幅度，δNTN 降低是收紧允许失败比例，两者不等价。**
主图只扫 Γ，δNTN 固定；额外提供 θDL×Γ 可行域热图。不可行会如实记录，
不会自动降低 θ 或提高 δTN。所有数值是研究配置，不声称是标准规定。

## 2. 保留与新增

- 原 `Nulling_CDF_SectorDrop.ipynb` 完全保留，包括已有输出。
- 新入口：`Nulling_CDF_SectorDrop_AppendixD.ipynb`，由原 notebook 深复制元数据后重构实验单元。
- 命令行入口：`run_appendix_d.py`。
- 实现：`appendix_d_experiments/`。
- 依赖：`requirements-appendix-d.txt`。
- 输出：`result/appendix_d_<时间>/`，不会替换论文现有 CDF、radiomap 或 main.pdf。

这里有两种数据来源，必须在论文图注中区分：

1. **声明的合成有限模型**：E4–E8 可直接运行，验证集合、SOCP、占用 LP、延迟因果性及联合策略。
2. **现有 Sionna 静态缓存**：直接利用原 7 GHz 信道、UL 推断的 DL 方向生成新的 Γ-CDF，
   按完整 macro 划分训练/校准/测试；只做静态经验验证。
   当前缓存没有真实监听失败/处理释放/跨时活动轨迹，不能据此宣称已完成物理 E7/E8。

E4 的经验边际覆盖不能直接塞入 E7 当逐状态风险认证。E7 目前使用自己的已声明有限信道/观测
模型计算精确条件失覆盖概率；接入物理数据后，应先取得符合 Appendix D 条件的风险表。

## 3. 运行

建议独立环境，避免升级现有 Sionna 环境中的 NumPy：

```bash
python -m venv --system-site-packages .venv-appendix-d
.venv-appendix-d/bin/python -m pip install -r requirements-appendix-d.txt
.venv-appendix-d/bin/python run_appendix_d.py --profile quick
```

如果使用 notebook，在安装这些依赖的 Python kernel 下运行。也可另装 `ipykernel` 注册独立 kernel；
默认 notebook 沿用原 `sionna20` 元数据，首个执行单元会检查当前 kernel 的依赖并给出说明。

```bash
# 正式规模：E4 增大完整场景样本，E7/E8 每配置10000条独立轨迹
.venv-appendix-d/bin/python run_appendix_d.py --profile full

# 只重跑联合策略；TN需求和两个失败预算固定
.venv-appendix-d/bin/python run_appendix_d.py --experiments E7 \
  --gamma-db -15 -10 -5 0 --theta-dl 0.40 --theta-ul 0.50 \
  --delta-tn 0.10 --delta-ntn 0.08

# 同时生成真实静态缓存的Gamma-CDF；省略max-sectors则使用全部测试扇区
.venv-appendix-d/bin/python run_appendix_d.py --experiments E5 \
  --cache-dir result/20260916_080614_095092 --cache-max-sectors 4
```

`quick` 用于功能与图表检查，不是精确估计1%尾部风险的正式统计规模。
`full` 不会把合成模型变为物理模型，也不会自动扩大完整历史图的决策数量。
完整历史规模指数增长，不应直接把 `epochs` 改成数十或数百。

输出目录若已有 `manifest.json` 会拒绝覆盖。每次记录 seed、依赖版本、配置、源代码快照、
源文件 SHA256，以及原 notebook 运行前后 SHA256。

## 4. 每组实验最终产物

| 实验 | 验证内容 | 可放论文的结果 |
|---|---|---|
| E4 | 完整相干信道集合、年龄/质量/监听长度、漏检及 UL 静默背景；完整场景分割校准 | 覆盖率与区间图、捕获/接受图、C/ρ及样本表、背景消融 |
| E5 | 支持函数可达性、SOCP、相位、功率回退、零预算零空间、秩/夹角 | Γ/ρ—功率/服务图、几何敏感性图、旧λ/无保护/名义零陷/robust/oracle对照表 |
| E6 | 共同风险条件下波束删减、完整策略混合枚举 vs 占用LP、随机化必要性、不可行性 | 目标/残差表、预算扫描图、随机化动作图、波束删减表 |
| E7 | F/T/L/J；同TN约束下扫描Γ并重求整体策略 | INR CDF、TN SNR CDF、净服务/保护折中、时间轴、θ×Γ可行域、策略成本表 |
| E8 | 延迟/失败、UL静默背景、背景缺失消融、增益界失配、burst相关性、新到达、禁止刷新 | 服务/超标比较图、逐轨迹原始数据、风险有效性标签、汇总表 |

每个实验同时输出矢量 `PDF`、预览 `PNG`、原始 `CSV`、`LaTeX .tex` 表，
部分 E4/E5 还保存 `NPZ`。E4/E5/E6/缓存表采用 `booktabs`；正文使用时需 `\usepackage{booktabs}`。
图的字号/配色可以调整后重画，CSV保留原始零功率和不可行状态。
E4/E5/E6/缓存的 .tex 是 tabular 片段；双栏论文中请包在 table* 内，或按栏宽缩放。
E7/E8 的 .tex 已包含 table* 环境，可以直接 input。

E7 重点文件：

- `gamma_inr_cdf.pdf`：前景接收机 INR，两幅分别是所有 DL 活动 tick 与 BS 发射条件下 CDF。
- `silent_ntn_and_tn_cdf.pdf`：UL 静默接收机 INR，以及全部 TN DL 时隙的 SNR。
- `service_protection_tradeoff.pdf`：F/T/L/J 净服务、RF空隙比例。
- `tn_demand_feasibility.pdf`：单独扫描 θDL 的可行域，不能混入“固定TN要求”主图。
- `timeline.pdf`、`timeline_J_gamma_*.csv`：gap、processing、可用信息年龄、功率、累计服务。
- `strategy_summary.csv`、`strategy_table.tex`：理论预测、Monte Carlo、逐窗口失败及成本。
- `policy_*_gamma_*.json`：真正执行的随机化策略、模型预测与数值残差。
- `inr_tn_samples_J_gamma_*.csv`：接收机、DL活动、BS是否发射、零INR、TN SNR 的逐样本记录。

全时域 CDF 包含 gap、mute、TN UL 时隙的零 BS 干扰，图上把零值放在 −60 dB 位置；
同时给出“BS正在发射”条件 CDF，避免静默造成的改善难以识别。原始CSV不把零值伪装成正功率。
CDF 是受控单 BS 干扰，不是全网络所有 BS 干扰求和。

## 5. 有限联合模型的具体含义

默认 M=4，12 ticks，三个可决策时点0/4/8，每个block固定3个TN DL时隙和1个UL时隙。
一个前景 NTN 接收机有两种有限物理信道状态，按已知 Markov 核演化；
另一个接收机始终 UL 静默但 DL 活动，由背景球保护。所有物理轨迹在选动作前生成，
控制器不能访问真实 NTN 状态。

动作包括不监听/监听1tick/监听2ticks，和服务/静默。`g_in=1, g_out=0, t_p=1`，
`t_ref=s+1`，`t_use=s+1+tau+t_p`。接受时输出有限状态方向；拒绝/无检测保留旧记录。
非阻塞处理期间使用旧记录，未释放结果不可见。RF占用同时影响相应DL/UL资源；
DL静默仍可服务原定TN UL。功率仅由当前可用信息的SOCP决定。

C=2及背景范数界是这个合成模型的**已知归一化边界**，不是 BS 从被动UL知道了真实NTN天线增益。
物理应用仍需接收增益、路径耦合/跨频变化和噪声下界的有效界或校准；不得用测试DL真值在线选波束。

有限模型按完整可观测历史展开，保留服务计数；LP 的 TN 成本是截止时未达标事件。
NTN成本由当前历史下真实信道不在集合的条件概率上界产生：
服务时 `c=eta*d`，mute时c=0，但DL活动时间仍进入d。
无beam切换/跨时能量成本；外部干扰固定；未来物理状态与连续波束无关。
这正是波束消去成立所需的范围。

F/T/L/J 都允许服务/静默并用同一个robust SOCP：

- F：在有限模型中离线选优固定实际监听日程、固定长度。
- T：固定长度（离线选1或2），监听时机自适应。
- L：实际监听起点日程固定（离线选优），每个起点可自适应选1或2，不能跳过。
- J：时机/长度/服务联合自适应。当前只有一个空间监听模板，不声称额外模板收益。

F的日程族是声明网格上的全部子集，并非只选一个未调优的周期。
各类都在相同已知模型中离线优化；Monte Carlo测试轨迹不参与调参。
J可能与T或L重合，不能预设完整方案在所有配置都严格领先。

E8 中 `bursty_same_marginals` 保持每次观察成功的边际概率，却引入时间相关；
冻结原观察核得到的策略，明确标为核失配压力测试。
`gain_bound_exceeded`、`missing_background_ablation` 也没有模型内保证。
`larger_silent_background` 可能使固定TN要求不可行；表中使用N/A，不能用零超标冒充有效解。
`modeled_new_arrival` 是已声明时间进入的背景接收机；活动分母从进入时刻开始计数。
`refresh_unavailable` 使用空合法监听日历，只允许降功率、服务/静默。

## 6. 统计与解释边界

- NTN指标取 `sum(exceeding active time)/sum(active time)`，包括漏检、UL静默和新到达接收机；
  不取episode比例的无权平均。本有限模型分母固定，数值虽相同，代码仍按总时长计算。
- TN每方向/窗口分别报失败概率与Clopper–Pearson区间。
- NTN按完整独立episode bootstrap，额外给保守Hoeffding上界；零观测不等于零风险。
- E4经验校准只有已声明的场景边际覆盖含义；测试覆盖失败会照实输出。
- E6是声明有限模型中的数学/数值核对；E7不是任意真实连续环境的全局最优证明。
- E7的默认服务单位是 `bits/Hz`（每tick为一个归一化时间单位）。设带宽B、tick秒数Δt后，
  比特为 `reported_value × B × Δt`，比较时B与Δt必须固定。

以后替换正文CDF时，优先用静态缓存输出检查Γ参数和物理量范围；真正用E7曲线支撑
“联合监听与服务优化”主张前，应将观测转移核、年龄集合、风险成本接入经过独立校准的物理轨迹。
当前实现与图注保留了这一区分。
