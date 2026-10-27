# MUSIC snapshot 实验：50 个 NTN-UE，两张单列图

主程序：[MUSIC_Snapshot_Experiment.ipynb](MUSIC_Snapshot_Experiment.ipynb)。
实现与英文说明：[snapshot_experiments](snapshot_experiments/README.md)。

## 正文图与统计口径

| 图 | 内容 | 尺寸 |
|---|---|---|
| 图 1(a) | 至少一条方向匹配成功的 NTN-UE 数量，跨 sector 去重 | 与 (b) 左右合排，3.5 × 1.9 in |
| 图 1(b) | 路径功率加权召回率、高 INR UE 检出率 | 图内 legend、95% 场景 bootstrap 置信带 |
| 图 2 | T=64、1024、8192 的 NTN INR CDF，加单条 no-nulling | 3.5 × 2.05 in |

Times New Roman；PDF 矢量图及 600 dpi PNG；图内不加标题。

**路径功率加权召回率** = 一对一匹配成功路径的功率之和 / 全部有效路径的功率之和。
每个 sector 内使用固定 5° 球面角距匹配，随后汇总所有场景、sector 和重复的分子分母。
逐路径功率之和不包含同一 UE 多径的相干交叉项，不能解释成实际干扰消除比例。

**高 INR UE 检出率** = 检测到的高 INR UE 数 / 全部高 INR UE 数。
高 INR UE 固定定义为 no-nulling 网络 INR > −3 dB，集合不随 T 改变。
在任意 sector 匹配到一条路径即算检出；实际保护效果仍由 nulling 后 INR 验证。

原始路径 precision/recall、TP/FP/FN 和已匹配角度误差保留在统计表与结果说明。
不删除真实信道中的弱路径，不为改善指标筛掉 INR CDF 中的未检测 UE。
所有 T 使用相同的 DL 有效链路掩码；nulling 后 INR 为零的样本仍保留。

## 固定实验设置

| 项目 | 设置 |
|---|---|
| 场景 | 20 个新随机 drop，每个 50 个 NTN-UE、12 个 TN-UE |
| 缓存 | 复用新的 [50-UE CDF 实验](result/CDF/20261026_220019_877592/) 同频信道 |
| UL/DL | 8 GHz，同频；200 MHz 噪声带宽 |
| 阵列与传播 | 8×8 阵列、6×6 子阵前后向平滑、多径深度 3 |
| T | 64、128、256、512、1024、2048、4096、8192 |
| 重复 | 每个场景 5 组波形/噪声，共 20×5×8=800 次网络检测 |
| nulling | snapshot 实验固定 λ=10¹³ |
| 观测 | UE 连续活动，同一 UE 的波形跨 sector 共享；sector 噪声独立 |

每个场景/重复使用同一最长记录的前缀；64 个快拍为一个流式计算块。
T 表示样本数量，程序没有定义硬件采样间隔或实际监听时间。
场景内保留全部重复与 T，再按场景做 2,000 次 bootstrap。
no-nulling 样本每个场景只计一次，不按重复、T 或 λ 重复拼接。

## T=64 标定与验证

先用 2,000 次独立纯噪声训练 T=64 的源数门限，再在八档 T 各用 1,000 次独立噪声验证。
冻结规则为 `margin(T)=0.70*sqrt(64/T)`；验证要求为每个 sector 源数误报率的 Wilson 95% 上界不超过 1%。
正式 notebook 使用第三组噪声再次验证实际缓存阵列。该标定不承诺有信号情况下或全网络的误报率。
门限随 T 按预定规则变化，因此结论描述整套检测协议随观测预算的表现。

先执行 3 场景 × 2 重复 × 3 档 T 的 smoke run，再执行完整扫描；核对共享 T 的结果一致。
输出位于 `result/MUSIC_SNAPSHOT/<timestamp>/`，与 `result/map`、`result/CDF` 平行。
新结果已完成 800 次网络检测并通过独立复算；旧的 `20261026_211714_127905` 版本已删除。
完整结果：[20261026_220701_036881](result/MUSIC_SNAPSHOT/20261026_220701_036881/RESULTS.md)。

## 频偏 CDF 与论文替换

独立的频偏 CDF 主程序保留理论协方差设置，改为每个 drop 50 个 NTN-UE。
λ=10¹¹、10¹³ 共用同一组 20 个新随机 drop；保留 −20%、−10%、0%、+10%、+20% 五档。
黑色 no-nulling 统一去重，灰色曲线使用 8 GHz 同频下未检测 UE 的 nulling 前 INR。
两面板合图保持 (a) NTN-UE INR、(b) TN-UE SINR，并更新 Overleaf 正文、图注及样本数。

频偏 CDF 已更新 Overleaf 正文 Figure 8、图注及性能数据，重新编译并同步，提交 `1998ea4`。
