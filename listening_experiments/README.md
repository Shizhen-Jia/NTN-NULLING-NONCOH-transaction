# Fig. 10、11 独立实验程序

主入口：[Nulling_Listening_Fig10_Fig11.ipynb](../Nulling_Listening_Fig10_Fig11.ipynb)。与原 `Nulling_CDF_SectorDrop.ipynb` 平行，新增 Python 功能和测试全部在本目录；复用原有场景、MUSIC、功率拟合和波束函数。

## 运行

在项目根目录打开 notebook，选择已有 **sionna20** kernel，重启内核后从上往下执行。当前配置为 `MODE="run"`、`PROFILE="pilot"`、`CHANNEL_SOURCE="cached"`，直接复用首次运行保存的三个 RT 场景：

1. 配置：3 个独立场景，每场景 5 次活动/噪声重复，`p=(0.1,0.3,1)`、`L=(1,4,16,64)`，每窗 128 快拍。
2. 信道：读取 `CACHED_CHANNELS` 指向的冻结信道档案，核对 RT 物理参数；此模式不会调用 RT。需要新场景时改用 `native`。
3. 试验：有限快拍、MUSIC（rank/MDL 源数估计可选）、最终波束、完整受保护用户 INR 与逐服务链路 SINR，每个场景/重复单独保存。
4. 后处理：计算四个子图指标、95% 区间和工作点 CSV，导出 PDF/PNG。

`PROFILE="paper"` 切换为 20 个场景 × 20 次重复，`L=(1,2,4,8,16,24,32,48,57,64)`，MUSIC 搜索步长由 2° 改为 0.5°。应同时指定包含足够场景的缓存，或改用 `native + auto` 生成。当前冻结缓存只有三个场景。这是正式统计的起步配置，需根据场景聚类区间决定是否增加独立场景。

源数估计当前选择 `rank`，改为 `mdl` 即可切换。两种方法均使用实际有限快拍的样本协方差。当前 notebook 参数为：

```python
MUSIC = MusicConfig(
    source_estimation="rank",
    rank_relative_threshold=1e-5,
    rank_noise_margin=0.45,
    rank_noise_reference_snapshots=128,
    spatial_smoothing=True, subarray_rows=6, subarray_cols=6,
    forward_backward=True,
)
```

rank 的实际噪声裕量为 `0.45 * sqrt(128/T)`，因此在 `L=1,4,16,64`、每窗 128 快拍时分别为 **0.45、0.225、0.1125、0.05625**。从扣除已知噪声底的特征值中，保留大于 `max(1e-5 * 最大信号特征值, 实际裕量 * 噪声功率)` 的项。这是一条固定的快拍数规则，不使用实际活动数、真实方向或最终 INR 调阈值。

原先从 analytic 程序沿用的 `rank_noise_margin=0.001` 对样本噪声起伏过于敏感。新裕量只根据纯噪声训练选择，再用独立噪声验证，未按这三个场景的 INR 优化。它适用于当前 **8×8 阵列、6×6 前后向平滑、已知白噪声功率、128≤T≤8192**。改变阵列、平滑或噪声模型后应重新标定。低于参考快拍数时程序报错，避免外推未经验证的规则。`rank_noise_reference_snapshots=None` 表示常数裕量；加载缺少此字段的旧配置时保持原来的常数语义。MDL 不使用该缩放规则。

标定使用 2,000 次训练，并在正式网格的 10 个预算各做 1,000 次独立验证，均未观察到源数误报。每个预算的 95% Wilson 上限仍约为 0.38%，不能把未观察到误报解释为真实风险为零，或当作全网络虚警保证。[标定记录](../result/fig10_fig11/rank_calibration/ebb5d0a92c67bc5c/calibration.json)保存训练/验证种子、噪声样本统计、阈值及适用条件。可复现命令：

```bash
OPENBLAS_NUM_THREADS=1 python -m listening_experiments.calibrate_rank --cache result/fig10_fig11/channel_cache/324aed40805adac11ba8
```

所选方法、参考快拍数及每个预算的实际裕量都随结果归档；新参数建立独立结果目录，保留旧结果。

新参数已在同一批三个 RT 场景、每场景五次活动重复上完整复跑，输出目录为 [`runs/024a46c618b9d78e3fe1`](../result/fig10_fig11/runs/024a46c618b9d78e3fe1/analysis/1fdee43febe48c81/analysis.json)。原先三个全静默网络试验中的 36 个基站检查现均返回零源并退化为 MRT；信道、活动计数和 MRT 基准逐项保持相同。详见 [对照诊断](../result/fig10_fig11/runs/024a46c618b9d78e3fe1/rank_validation_comparison.json)。这次复跑用于参数适用性检查，不构成独立测试集上的性能结论。

生成新信道需要当前项目的 Sionna RT/Mitsuba 环境；复用信道及后处理仅需 NumPy、SciPy、Matplotlib。notebook 还使用 IPython。若小矩阵多线程开销较大，可在启动 Jupyter 前设置 `OPENBLAS_NUM_THREADS=1`。

## 信道复用与继续运行

| 设置或操作 | 行为 |
|---|---|
| `CHANNEL_SOURCE="cached"` | 显式读取 `CACHED_CHANNELS`，保留原始 RT 版本记录和信道文件身份，并核对当前物理配置。更新 MUSIC 代码不会使其重新追踪；缺失场景直接报错。 |
| `native` + `CACHE_POLICY="auto"` | 按 RT 参数、场景资产和相关代码的指纹复用已有信道，仅生成缺失场景。 |
| `native` + `CACHE_POLICY="reuse_only"` | 不允许触发 RT；缺失场景时报错。 |
| `CHANNEL_SOURCE="legacy"` | 设置 `LEGACY_CHANNELS` 和 `LEGACY_UL_SUBDIR`，读取已有 SectorDrop 的 `channels_XXXX.npz` / `sensing_XXXX.npz`。严格检查实际频率和物理阵列对应关系，不改写旧档案。 |
| 相同配置重新执行活动试验 | 跳过已经保存的 episode；中断的 episode 按相同种子重算。 |
| 改活动概率、监听预算、快拍数、λ、MUSIC 或感知功率 | 建立新的试验档案；`cached` 继续读取同一批信道。 |
| 改场景、部署、阵列、频率或 RT 相关代码 | 使用新的缓存指纹，生成相应信道。 |
| 增加 `NUM_MACROS` | 已有 RT 场景前缀可复用；新的整体试验档案重算活动试验。 |

旧缓存例子：`result/20260916_080614_095092/channels` 的 `ul_001` 是 6.3 GHz；实际可用场景数以目录内文件为准。原归档中的感知噪声不复用，噪声按新配置的 UL 功率重算。旧信道的部署参数以原运行记录为准，修改当前 `RTConfig` 不会改变导入的旧数据。

缓存保存的是静态窄带复信道：各用户各链路内部多径相干求和，保留各阵元系数，不是只保存信道功率。时间变化信道、不同延迟分辨率或另一频段不应通过修改后处理假装得到。

## 只改时间参数或重画

notebook 设置 `MODE="replot"`，将已打印的 `runs/<run_id>` 路径填到 `RUN_TO_REPLOT`，调整 `TIMING`、`ANALYSIS_THRESHOLD_DB` 或区间配置后执行。此时跳过所有信道读取、RT 和 MUSIC，实验参数从归档读取。

也可以在 Python 中直接调用：

```python
from listening_experiments import TimingConfig, analyze_run, plot_figures

analysis = analyze_run(
    "result/fig10_fig11/runs/<run_id>",
    TimingConfig(horizon=256, listen=1, switch=0.1, gap=1, processing=1),
    threshold_db=-3,
)
fig10, fig11 = plot_figures(analysis)
# 单纯重画已有汇总，不重新统计：
# fig10, fig11 = plot_figures(".../analysis/<analysis_id>/analysis.json")
```

`switch` 是每窗两次切频的**合计**；所有时间以一个监听窗口 τ₀ 为单位，属于规定的协议参数，不是测得的硬件时延。程序要求每个预算 `t_ready < H`。

## 统计口径

- 固定每窗快拍数，改变监听窗口数时总快拍数也改变。这不是固定总快拍数的连续/分开感知性能比较，也不包含式 (40) 的数值验证。
- 每用户/每窗独立 Bernoulli 活动；同一用户的活动、波形对所有基站共享，基站噪声独立。各参数共用随机数，各监听预算使用前缀；所有静默窗口保留噪声。
- 归一化观测为 `Y / sqrt(P_UL)`，噪声方差为 `N_BS/P_UL`。估计增益包含记录内活动占比，不用真实 `J_i` 或 `p` 反推活动时功率。真实方向、源数、活动计数不输入检测器。
- 受保护集合在活动前由 DL 耦合固定；未发射和未检测到的用户仍参加 INR 评价。每场景固定一轮 TN 配对，空服务扇区报错，不静默丢弃。
- 场景等权、场景内重复等权、试验内用户/链路等权。先逐链路计算 `log2(1+SINR)` 再平均。`R_eff` 是平均每链路的周期有效频谱效率，非网络总和。
- Fig. 10(a) 单独做廉价活动 MC，使用实际受保护人数，给出 Wilson 区间、解析概率和 union bound。其他三个面板对整个场景做配对 bootstrap；一个场景时不报告场景置信区间。零观测超限不能证明真实风险为零。
- MRT 无感知参考与同监听日程 MRT 参考在 Fig. 11(a) 分开；在 Fig. 11(b) 的数据时间超限比例上两者相同，合为一条线。

## 文件结构

```text
result/fig10_fig11/
  channel_cache/<rt_id>/
    rt_config.json
    macro_0000.npz                   # 静态复信道、几何、实际 RT 软件版本
    tn_drop/                         # 新 RT 的 TN 配对诊断
  runs/<run_id>/
    ac_config.json                  # 活动、物理功率、检测、代码和信道来源
    status.json
    episodes/macro_0000_episode_0000.npz
    analysis/<analysis_id>/
      analysis.json                 # 时间参数、阈值、区间配置与完整汇总
      ac_summary.csv
      ac_working_points.csv         # 当前网格包含指定工作点时生成
      coverage_model.csv
      scene_metrics.npz
      fig10_listening_protection.pdf / .png
      fig11_listening_service_cost.pdf / .png
```

episode 文件不展平：`inr_final_linear, J` 形状为 `[p,L,protected_user]`，`sinr_final_linear` 为 `[p,L,served_link]`；`ue_ids, tn_user_ids, tx_ids, macro_id, episode_id, round_id` 保留对应关系。MRT 原始值、估计源数、接受峰数、噪声估计、拟合残差及匿名方向/增益也一并保存。可据此追加 CDF 和按 `J=0` 分组的诊断；目前没有实现基于真实路径离线匹配的漏检/虚警统计。

`rank_noise_margins_by_budget` 记录实际传给检测器的裕量。修改前的已执行 notebook 备份在 `result/fig10_fig11/notebook_history/`；主 notebook 清除旧参数输出，避免将旧图误当作新配置结果。

配置改变产生不同目录；逐 episode 原子写入，不把未完成试验当作完整数据。时间参数与原始物理结果分开存档，避免只改服务协议时重算信道与检测。并发运行相同配置写同一目录不在当前支持范围内。

模块职责：`cache.py` / `rt_backend.py` 管信道；`observations.py` / `detection.py` / `evaluation.py` 管物理试验；`runner.py` 管断点；`analysis.py` / `plotting.py` 管统计与图片；`config.py` / `storage.py` 管配置与存储。

## 检查

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest discover -s listening_experiments/tests -v
```

覆盖分组累计与整段协方差一致性、活动前缀及跨基站波形、独立噪声与功率归一化、MDL/rank 两种源数规则和 rank 阈值、已知方向的 MUSIC/UL→DL 转移、空检测与 λ=0 的 MRT 退化、缓存扩展、断点恢复、不等用户数的场景等权统计以及重核时间参数。测试场景为控制输入，不用作论文性能结果。
