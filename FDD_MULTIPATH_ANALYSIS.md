# FDD Nulling: Multipath Processing and Frequency-Transfer Analysis

This document accompanies `Nulling_CDF_SectorDrop.ipynb`. Sector placement and association rules are documented in [TN_SECTOR_DROP.md](TN_SECTOR_DROP.md).

## Multipath 处理过程的数学模型

本节从信号模型推导当前实验的处理流程：

$$
\text{逐径 CIR}\longrightarrow\text{相干窄带信道}
\longrightarrow R_{\mathrm U}\longrightarrow\text{空间平滑与 MUSIC}
\longrightarrow\widehat Q_{\mathrm U}\longrightarrow\widehat B_{\mathrm D}
\longrightarrow v_\lambda\longrightarrow\text{真实 DL INR/SINR}.
$$

这里必须区分三个数学对象：**物理信道中的相干路径和**、**估计时允许相关的路径协方差**、**波束设计中的非相干方向惩罚**。实验名称中的 NONCOH 指第三项，不能据此把真实多径信道改成路径功率之和。

### 1. 记号、阵列流形与适用范围

先固定一个 BS sector，省略 sector 下标。令 $q=(k,r)$ 表示 NTN UE $k$ 的第 $r$ 个接收天线分支，$\ell\in\mathcal P_q$ 表示该链路的一条有效路径。当前快照模型将不同 $q$ 视为独立信号源；同一个 $q$ 的所有路径共享一个波形。单天线 UE 时，$q$ 就是 UE 编号。

| 符号 | 含义 |
| --- | --- |
| $M=M_yM_z$ | BS 全阵列天线数，当前为 $8\times8=64$ |
| $M_s=m_ym_z$ | 平滑子阵天线数，当前为 $6\times6=36$ |
| $f_{\mathrm U},f_{\mathrm D}$ | UL、DL 载频；$\Delta f=f_{\mathrm D}-f_{\mathrm U}$ |
| $\mathbf r_m$ | 第 $m$ 个阵元相对于阵列参考点的物理位置，单位 m |
| $\Omega=(\phi,\theta)$ | BS 侧路径方向：方位角和天顶角，仰角为 $\pi/2-\theta$ |
| $\tau_{q\ell},\beta_{q\ell}(f)$ | 路径时延和包含载波传播相位的复系数 |
| $p_q,\sigma^2$ | 归一化 UL 源功率和每个 BS 阵元的噪声方差 |
| $L=\sum_q|\mathcal P_q|$、$K$ | 真实路径数、实际接受的候选角度数；均不必等于 UE 数 |

令方向单位向量和单位范数阵列流形为

$$
\mathbf d(\phi,\theta)=
\begin{bmatrix}\sin\theta\cos\phi&\sin\theta\sin\phi&\cos\theta\end{bmatrix}^{T},
\qquad
[\mathbf u(f,\Omega)]_m=
\frac{1}{\sqrt M}\exp\!\left(j\frac{2\pi f}{c}\mathbf r_m^{T}\mathbf d(\Omega)\right).
\tag{1}
$$

$\mathbf r_m$ 已包含 sector 的姿态旋转；使用局部坐标时应先做 local-to-global 旋转。正号与本项目保存的 BS 发射侧通道向量一致。下文把通道看作列向量，波束响应写成 $v^Hh$。内部 `channel_mode="conj"` 会同时共轭接收协方差与扫描流形，随后恢复到此约定进行全阵列拟合，不改变以下功率关系。

式 (1) 使用远场平面波和相同阵元响应假设，方向相关的公共阵元增益、极化与传播损耗吸收到 $\beta$ 中。真实 CIR 仍由射线追踪给出；若其阵列响应偏离这个流形，会表现为估计模型失配。

### 2. 从逐径 CIR 到相干窄带信道

令 $\alpha_{q\ell}(f)$ 表示尚未包含参考点载波时延相位的复传播系数，则

$$
\beta_{q\ell}(f)=\alpha_{q\ell}(f)e^{-j2\pi f\tau_{q\ell}},
\qquad
\mathbf a^{\mathrm b}_{q\ell}(f,t)=
\beta_{q\ell}(f)e^{j2\pi\nu_{q\ell}t}\mathbf u(f,\Omega_{q\ell}),
\tag{2}
$$

其中 $\nu_{q\ell}$ 为逐径 Doppler，$\mathbf a^{\mathrm b}_{q\ell}$ 是基带路径向量。因为 $\|\mathbf u\|=1$，静态时的全阵列路径功率为 $\|\mathbf a^{\mathrm b}_{q\ell}\|^2=|\beta_{q\ell}|^2$，不是单阵元功率。

**Sionna `paths.cir()` 的系数已经包含载波时延相位**，合成时不能再次乘 $e^{-j2\pi f\tau}$。基带系数及 Doppler 的定义见 [Sionna Paths 文档](https://nvlabs.github.io/sionna/rt/api/paths.html)。当前实验固定使用 $t=0$，没有根据快照编号推进 Doppler。

以 $\xi$ 表示相对于该载频的基带频率偏移，冲激响应和频响为

$$
\mathbf c_q(\zeta;t,f)=
\sum_{\ell\in\mathcal P_q}\mathbf a^{\mathrm b}_{q\ell}(f,t)\delta(\zeta-\tau_{q\ell}),
\qquad
\mathbf h_q(f,\xi,t)=
\sum_{\ell\in\mathcal P_q}\mathbf a^{\mathrm b}_{q\ell}(f,t)e^{-j2\pi\xi\tau_{q\ell}}.
\tag{3}
$$

式 (3) 的偏移频响保持阵列流形和材料响应为载频处的值。实际 FDD sweep 会在 UL 载频重新追踪，不是用式 (3) 把一个 DL CIR 简单平移到 UL。当前中心频率、静态窄带合成为

$$
\boxed{\mathbf h_q(f)=\sum_{\ell\in\mathcal P_q}
\beta_{q\ell}(f)\mathbf u(f,\Omega_{q\ell})}.
\tag{4}
$$

因此，对任意波束 $v$，接收功率包含路径交叉项：

$$
|v^H\mathbf h_q|^2=
\sum_\ell|\beta_{q\ell}|^2|v^H\mathbf u_{q\ell}|^2
+2\operatorname{Re}\!\sum_{\ell<m}\beta_{q\ell}\beta_{qm}^*
(v^H\mathbf u_{q\ell})(v^H\mathbf u_{qm})^*.
\tag{5}
$$

同向、等幅、反相的两条路径在式 (4) 中可以完全抵消，而逐径功率和仍为正。这个区别决定了最后的 INR 必须用完整复信道计算。

将宽带延迟波形近似为一个共同窄带波形，需要信号带宽与时延扩展满足相应窄带条件，例如 $B_{\mathrm{sig}}\Delta\tau\ll1$；阵列流形在带内不变还需要忽略波束斜视。当前实验直接研究中心频率切片，参数中的 200 MHz 只用于噪声与频带分离，不能据此声称已经验证 200 MHz 宽带下的这些条件。

传播控制只改变 $\mathcal P_q$：`max_depth` 限制每条路径允许的交互次数，不等于路径数；NTN 的 `nlos_only` 删除零交互直达径，保留间接径。路径集合为空时 $h_q=0$，链路记为 no-path。TN 同样按式 (4) 相干合成，其直达径不被 NTN 开关删除。

### 3. UL 观测模型：用户间独立，同一用户的路径相干

令 $s_q[n]$ 为第 $q$ 个源的单位功率符号，满足 $\mathbb E[s_q[n]s_{q'}[n]^*]=\delta_{qq'}$。采用上述通道约定的 UL 观测模型为

$$
\mathbf x[n]=\sum_q\sqrt{p_q}\mathbf h_q(f_{\mathrm U})s_q[n]+\mathbf w[n],
\qquad\mathbb E[\mathbf w\mathbf w^H]=\sigma^2I_M,
\tag{6}
$$

$$
R_{\mathrm U}=\mathbb E[\mathbf x\mathbf x^H]
=\sum_qp_q\mathbf h_q\mathbf h_q^H+\sigma^2I_M.
\tag{7}
$$

`analytic` 使用式 (7)；`sample` 根据式 (6) 生成 $N$ 个静态信道快照，再使用

$$
\widehat R_{\mathrm U}=\frac1N\sum_{n=1}^{N}\mathbf x[n]\mathbf x[n]^H.
\tag{8}
$$

**不是为每条 ray 独立生成 $s_{q\ell}[n]$。** 那样会人为消除路径相关项，得到另一个非相干传播模型。配置的 `music_noise_var` 就是这里的 $\sigma^2$；默认 $p_q=1$ 时，后续拟合的路径功率无需再除以源功率。

将所有路径展开，定义

$$
A_{\mathrm U}=[\mathbf u_1,\ldots,\mathbf u_L],\qquad
C_{\ell q}=\begin{cases}
\sqrt{p_q}\beta_{q\ell}(f_{\mathrm U}),&\ell\text{ 属于源 }q,\\
0,&\text{其他},
\end{cases}
\qquad Q_{\mathrm U}=CC^H.
\tag{9}
$$

则 $R_{\mathrm U}=A_{\mathrm U}Q_{\mathrm U}A_{\mathrm U}^H+\sigma^2I_M$。式 (9) 的 $\ell$ 使用全局路径编号。按用户排列路径时，$Q_{\mathrm U}$ 的用户内块为 $p_q\boldsymbol\beta_q\boldsymbol\beta_q^H$；不同用户间的块为零。这一块通常有非零的非对角元，却只有秩 1。

例如一个 UE 有两条路径：

$$
Q_{\mathrm U}=p\begin{bmatrix}
|\beta_1|^2&\beta_1\beta_2^*\\
\beta_2\beta_1^*&|\beta_2|^2
\end{bmatrix},\qquad
\operatorname{rank}(R_{\mathrm U}-\sigma^2I_M)=1\quad(h\ne0).
\tag{10}
$$

原始协方差的一个信号特征向量只张成 $h=\beta_1u_1+\beta_2u_2$ 的方向，一般不能张成 $u_1,u_2$ 两个方向。因此，无论有多少理想快照，都不能直接把原始信号秩当作 ray 数；需要下面的空间平滑。

### 4. 二维空间平滑如何恢复可分辨的方向

从完整矩形阵列中抽取所有平移的 $m_y\times m_z$ 子阵。子阵数和平均协方差为

$$
J=(M_y-m_y+1)(M_z-m_z+1),\qquad
R_{\mathrm{SS}}=\frac1J\sum_{j=1}^{J}S_jR_{\mathrm U}S_j^H,
\tag{11}
$$

其中 $S_j\in\{0,1\}^{M_s\times M}$ 选择第 $j$ 个子阵，各子阵按相同的局部几何顺序排列。当前 $8\times8\to6\times6$ 给出 $J=9$。实际有采样误差时，用 $\widehat R_{\mathrm U}$ 替代式 (11) 的 $R_{\mathrm U}$。

设 $A_s$ 的列是**单位范数子阵**流形，$\mathbf t_j$ 是该子阵中心相对于全阵列参考点的位移。平移只引入逐方向相位：

$$
S_jA_{\mathrm U}=\sqrt{\frac{M_s}{M}}A_sD_j,\qquad
D_j=\operatorname{diag}\!\left\{
 e^{j(2\pi f_{\mathrm U}/c)\mathbf t_j^T\mathbf d(\Omega_\ell)}
\right\}_{\ell=1}^{L}.
\tag{12}
$$

代入后得到

$$
R_{\mathrm{SS}}=\frac{M_s}{M}A_s\overline Q A_s^H+\sigma^2I_{M_s},
\qquad\overline Q=\frac1J\sum_jD_jQ_{\mathrm U}D_j^H.
\tag{13}
$$

$M_s/M$ 来自全阵列与子阵各自的单位范数归一化，不应漏掉。令

$$
\Gamma_{\ell m}=\frac1J\sum_j
 e^{j(2\pi f_{\mathrm U}/c)\mathbf t_j^T[\mathbf d(\Omega_\ell)-\mathbf d(\Omega_m)]},
\qquad\overline Q_{\ell m}=Q_{\mathrm U,\ell m}\Gamma_{\ell m}.
\tag{14}
$$

有 $\Gamma_{\ell\ell}=1$，而方向不同且平移相位有足够变化时，$|\Gamma_{\ell m}|<1$。两径例子因此满足

$$
\det(\overline Q)=p^2|\beta_1|^2|\beta_2|^2
\left(1-|\Gamma_{12}|^2\right)>0
\quad(\beta_1\beta_2\ne0,\ |\Gamma_{12}|<1).
\tag{15}
$$

即使每个子阵的单用户信号协方差都只有秩 1，多个子阵的平均也能产生秩 2，使两个路径方向进入信号子空间。这里是利用空间平移恢复秩，并没有让真实反射路径变成独立发射机。标准平移原理可参见 [MathWorks MUSIC 文档](https://www.mathworks.com/help/phased/ug/music-super-resolution-doa-estimation.html)；式 (12)–(15) 按本项目二维阵列及归一化约定展开。

对单个完全相干的 $L_q$ 径组，仅前向平滑有 $\operatorname{rank}(\overline Q_q)\le\min(L_q,J)$。恢复全部方向还要求相移向量足够独立、子阵流形满列秩，以及总有效方向数小于 $M_s$，为 MUSIC 留出噪声子空间。$L_q\le J$ 只是必要的计数条件之一，不保证可分辨。重合方向、平面阵列前后歧义或栅瓣等价方向可能仍使 $|\Gamma_{\ell m}|=1$。

开启 forward/backward 时，令 $\Pi$ 为子阵阵元中心反演的置换矩阵，实际扫描使用

$$
R_{\mathrm{FB}}=\frac12\left(R_{\mathrm{SS}}+\Pi R_{\mathrm{SS}}^*\Pi^T\right).
\tag{16}
$$

它要求子阵中心对称，可进一步改善相关信号的子空间结构，但不保证恢复所有路径。关闭 forward/backward 时扫描 $R_{\mathrm{SS}}$；关闭空间平滑时直接扫描原始全阵列协方差。

### 5. MUSIC 估角，以及回到全阵列估功率

将用于扫描的协方差作特征分解，选取估计信号维数 $d_s<M_s$，记其余特征向量组成 $E_n$。MUSIC 谱为

$$
P_{\mathrm{MUSIC}}(\Omega)=
\frac{1}{\mathbf u_s(f_{\mathrm U},\Omega)^HE_nE_n^H\mathbf u_s(f_{\mathrm U},\Omega)}.
\tag{17}
$$

搜索局部峰并细化得到 $\widehat\Omega_1,\ldots,\widehat\Omega_K$。在平滑后充分恢复信号秩且流形正确的理想条件下，真实路径流形与噪声子空间正交。实际 $d_s$ 由秩阈值等方法估计，峰筛选后可能有 $K<d_s$。这些是匿名空间方向，不携带 UE 身份，也不直接给出路径时延。重叠子阵不构成额外的独立时间快照，sample 模式下直接套用 MDL 只是启发式。无平滑时，上述 $M_s$ 和子阵流形分别替换为 $M$ 和全阵列流形。

接下来重建 $M$ 维全阵列流形

$$
\widehat A_{\mathrm U}=
[\mathbf u(f_{\mathrm U},\widehat\Omega_1),\ldots,
\mathbf u(f_{\mathrm U},\widehat\Omega_K)]\in\mathbb C^{M\times K}.
\tag{18}
$$

**角度来自平滑协方差，功率拟合则回到原始全阵列协方差** $R_{\mathrm U}^{\mathrm{obs}}$，其中 analytic 模式取式 (7)，sample 模式取式 (8)。具体估计为

$$
R_{\mathrm{sig}}=\frac12\left(R_{\mathrm U}^{\mathrm{obs}}+(R_{\mathrm U}^{\mathrm{obs}})^H\right)
-\widehat\sigma^2I_M,\qquad
Q_0=\widehat A_{\mathrm U}^{\dagger}R_{\mathrm{sig}}
(\widehat A_{\mathrm U}^{\dagger})^H,
\tag{19}
$$

$$
\frac{Q_0+Q_0^H}{2}=V\operatorname{diag}(\eta_i)V^H,\qquad
\widehat Q_{\mathrm U}=V\operatorname{diag}(\max(\eta_i,0))V^H,\qquad
\widehat g_{i,\mathrm U}=[\widehat Q_{\mathrm U}]_{ii}.
\tag{20}
$$

$\dagger$ 表示伪逆；当前截断阈值为最大奇异值的 $10^{-8}$。analytic 模式使用已知噪声方差，sample 模式从检测结果估计噪声。式 (20) 是在 $Q$ 空间进行半正定投影；一般不能称为带 PSD 约束的 $\|R_{\mathrm{sig}}-AQA^H\|_F^2$ 问题的精确全局解，因为 $A$ 的列不一定正交。

在方向准确、无失配且 $A_{\mathrm U}$ 满列秩时，$A_{\mathrm U}^{\dagger}A_{\mathrm U}=I$，即使真实 $Q_{\mathrm U}$ 因相干而奇异，也能从理想全阵列协方差恢复它。单用户两径时，两个对角元分别为 $p|\beta_1|^2,p|\beta_2|^2$。强相关角度使 $\widehat A_{\mathrm U}$ 病态，误差会被伪逆放大；重合方向则无法唯一拆分各径功率。

拟合残差 $\|R_{\mathrm{sig}}-\widehat A_{\mathrm U}\widehat Q_{\mathrm U}\widehat A_{\mathrm U}^H\|_F/\|R_{\mathrm{sig}}\|_F$ 和 $\operatorname{cond}(\widehat A_{\mathrm U})$ 用于诊断。拟合允许完整相关矩阵，不假设每个峰独立，也不需要事先知道哪些峰属于同一 UE。

### 6. 路径选择与 FDD 迁移的数学含义

将正的 $\widehat g_{i,\mathrm U}$ 从大到小排序为 $g_{(1)}\ge\cdots\ge g_{(K_+)}>0$。能量比例 $\rho$ 选择最小 $k_\rho$ 使

$$
\sum_{i=1}^{k_\rho}g_{(i)}\ge\rho\sum_{i=1}^{K_+}g_{(i)}.
\tag{21}
$$

再由 top-K 限制截断为最终集合 $\mathcal S$；额外的 beamformer cap 还可能继续减少项数。式 (21) 的“能量”是估计的逐径权重和，不是式 (5) 的相干接收功率，也不保证覆盖同一比例的真实 DL 干扰。所有漏检或舍弃的路径仍留在真实评估信道中。没有正权重时取空集合。

`angle` 模式使用同一物理阵列和估计方向，按 DL 载频重新构造

$$
\widehat{\mathbf u}_{i,\mathrm D}=\mathbf u(f_{\mathrm D},\widehat\Omega_i),\qquad
\widehat g_{i,\mathrm D}=s_f\widehat g_{i,\mathrm U},\quad
s_f=\begin{cases}
1,&\text{关闭功率修正},\\
(f_{\mathrm U}/f_{\mathrm D})^2,&\text{开启功率修正}.
\end{cases}
\tag{22}
$$

`raw` 模式直接取 $\widehat{\mathbf u}_{i,\mathrm D}=\widehat{\mathbf u}_{i,\mathrm U}$。两者都不传递 $\widehat Q_{\mathrm U}$ 的非对角相位。对于同一几何路径，在时延不变且 UL 系数非零时，实际跨频关系为

$$
\beta_{q\ell}(f_{\mathrm D})=\beta_{q\ell}(f_{\mathrm U})
\frac{\alpha_{q\ell}(f_{\mathrm D})}{\alpha_{q\ell}(f_{\mathrm U})}
 e^{-j2\pi\Delta f\tau_{q\ell}}.
\tag{23}
$$

路径对 $\ell,m$ 的相关项会额外包含 $e^{-j2\pi\Delta f(\tau_{q\ell}-\tau_{qm})}$ 和材料/天线响应变化。当前 MUSIC 流程没有估计时延或材料频率响应，无法由 UL 相关相位可靠重建这些 DL 交叉项。式 (22) 的功率修正仅近似自由空间频率缩放，不补偿式 (23) 的相位，也不保证 UL/DL 可见路径集合完全相同。

固定间距 $d=c/(2f_{\mathrm D})$ 时，UL 电间距为 $df_{\mathrm U}/c=f_{\mathrm U}/(2f_{\mathrm D})$。按 DL 重建流形可以校正已知的频率尺度，但不能修复 UL 阶段已经出现的角度误判、栅瓣混淆或漏检。

### 7. 非相干零陷目标与闭式特征向量解

使用最终选择的方向构造

$$
\boxed{\widehat B_{\mathrm D}=\sum_{i\in\mathcal S}\widehat g_{i,\mathrm D}
\widehat{\mathbf u}_{i,\mathrm D}\widehat{\mathbf u}_{i,\mathrm D}^H},\qquad
v^H\widehat B_{\mathrm D}v=
\sum_{i\in\mathcal S}\widehat g_{i,\mathrm D}|v^H\widehat{\mathbf u}_{i,\mathrm D}|^2.
\tag{24}
$$

该二次型惩罚每个方向上的泄漏，不使用不同方向之间的相位抵消。若另外假设路径相对相位独立、均匀随机，逐径功率和也可解释为相位平均后的泄漏；**当前静态仿真没有施加该随机相位平均假设**，这里将其作为设计准则。

对服务 TN 用户，令 $H_0\in\mathbb C^{M\times N_r}$ 为完整相干 DL 信道矩阵，$w_r$ 为基线 SVD 给出的接收合并器，$h_0=H_0w_r$，$\widetilde h_0=h_0/(\|h_0\|+\epsilon)$。固定 $w_r$ 后，优化为

$$
\max_{\|v\|=1}\left\{|v^H\widetilde h_0|^2-\lambda v^H\widehat B_{\mathrm D}v\right\}
=\max_{\|v\|=1}v^H\left(\widetilde h_0\widetilde h_0^H-\lambda\widehat B_{\mathrm D}\right)v.
\tag{25}
$$

由 Hermitian 矩阵的 Rayleigh 商性质，最优 $v_\lambda$ 是 $\widetilde h_0\widetilde h_0^H-\lambda\widehat B_{\mathrm D}$ 最大特征值对应的单位特征向量。它是信号增益与方向泄漏的软约束折中，有限 $\lambda$ 不保证每个方向上都有严格零点。当前接收合并器不会针对每个 $\lambda$ 联合重新优化。

$\lambda=0$ 时，$v$ 沿有效 TN 信道；无检测方向时惩罚也为零。若存在非空零陷子空间，且 TN 信道在其中有非零投影，则 $\lambda\to\infty$ 的方向趋向

$$
 v_\infty=\frac{P_\perp h_0}{\|P_\perp h_0\|},\qquad
 P_\perp=I_M-\widehat A_{\mathcal S}\widehat A_{\mathcal S}^{\dagger},
\tag{26}
$$

其中 $\widehat A_{\mathcal S}$ 收集正权重的所选 DL 流形。若它张满全空间，则不存在非零严格零陷波束。统一放大所有权重 $s_f$ 等价于把式 (25) 的 $\lambda$ 放大 $s_f$，所以功率修正也改变信号/干扰折中强度。

### 8. 用完整相干 DL 信道评估，而不是用惩罚值替代 INR

恢复 sector 下标 $b$，设各 BS 发射流独立、每流功率为 $P_b$。对 NTN UE $k$，当前接收天线功率求和的干扰模型为

$$
I_k=\sum_bP_b\sum_r\left|v_b^H\mathbf h_{bkr}(f_{\mathrm D})\right|^2
=\sum_bP_b\sum_r\left|\sum_{\ell\in\mathcal P_{bkr}}
\beta_{bkr\ell}(f_{\mathrm D})v_b^H\mathbf u_{bkr\ell}(f_{\mathrm D})\right|^2,
\qquad\mathrm{INR}_k=\frac{I_k}{N_{\mathrm{NTN}}}.
\tag{27}
$$

同一发射流的路径先相干相加，不同 BS 流之间再加功率。$N_{\mathrm{NTN}}$ 是实验配置的 INR 噪声功率，代码对多接收天线使用上述功率求和而不另做 NTN 合并器优化。当前 $P_b$ 使用共同的 `tx_power`。

对由 sector $b$ 服务的 TN 用户 $j$，使用其固定接收合并器 $w_j$：

$$
\mathrm{SINR}_j=\frac{P_b|v_b^HH_{bj}w_j|^2}
{N_{\mathrm{TN}}+\sum_{b'\ne b}P_{b'}|v_{b'}^HH_{b'j}w_j|^2},\qquad
\mathrm{SNR}_j=\frac{P_b|v_b^HH_{bj}w_j|^2}{N_{\mathrm{TN}}}.
\tag{28}
$$

所有 $H$ 和 $h$ 均含全部有效 DL 路径，保持原始增益，不使用式 (25) 中归一化后的期望信道代替实际接收功率。各指标转为 $10\log_{10}(\cdot)$ 后，对共同的有效 DL 评估样本形成经验 CDF。

为区分 oracle 与估计目标，在一个 sector 上定义两个真实矩阵：

$$
B_{\mathrm{path,D}}=\sum_{q,\ell}\mathbf a^{\mathrm b}_{q\ell}
\left(\mathbf a^{\mathrm b}_{q\ell}\right)^H,\qquad
R_{\mathrm{coh,D}}=\sum_q\mathbf h_q\mathbf h_q^H.
\tag{29}
$$

二者之差正是同一源不同路径的交叉项，一般不为零。multipath 的 `music_real_*` oracle 在式 (25) 中使用真实 $B_{\mathrm{path,D}}$，`B_nrmse` 也以它为参照；它们不以 $R_{\mathrm{coh,D}}$ 为目标。即使拥有完美路径信息，也不能保证该方向惩罚 oracle 在每个 UE 上都给出最低的瞬时 INR。

若 $v$ 与所有真实 DL 路径向量正交，则必然有 $v^Hh_q=0$，与各径相位无关。若存在角度误差、遗漏方向或有限 $\lambda$ 下的剩余响应，则式 (5) 的 DL 交叉项决定最终泄漏。因此，“跨频相位改变”可以改变残余干扰，却不能单独证明覆盖全部方向的严格零陷会失效。

### 9. 数学量与现有实现的对应

| 数学阶段 | 对应实现 | 该阶段使用的信息 |
| --- | --- | --- |
| 式 (3)–(4)：逐径相干合成 | `multipath_support.collapse_cir_to_narrowband` | 当前载频全部有效 CIR；不按强弱截断真信道 |
| 式 (6)–(8)：UL 协方差 | `ntn_music_detection` 中的静态/快照协方差函数 | 相干 UL 有效信道、源功率、噪声 |
| 式 (11)–(17)：平滑和估角 | `planar_subarrays`、`spatially_smooth_covariance`、`run_multipath_music_pipeline` | UL 协方差与已知阵列几何；不输入真路径角度 |
| 式 (18)–(20)：相关功率拟合 | `fit_correlated_path_covariance` | 原始全阵列 UL 协方差、估计角度 |
| 式 (21)–(24)：选择及 DL 惩罚 | `select_estimated_paths`、`transfer_music_peaks_to_dl`、功率修正 | 估计角度/权重、DL 阵列几何；不输入真实 DL 增益 |
| 式 (25)：求波束 | `BeamformingCalc.nulling_bf_music_noncoh` | 服务 TN 信道、固定接收合并器、所选方向惩罚 |
| 式 (27)–(29)：真实评估与 oracle | `nulling_cdf_utils.run_small_round` | 完整 DL 信道；真路径信息只供诊断及明确标注的 oracle |

## Fixed DL array and UL frequency sweep

The parameter cell of `Nulling_CDF_SectorDrop.ipynb` now includes:

```python
f_dl = 7e9
ul_frequency_percentages = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
num_macro_sims = 20
ul_to_dl_mode = "angle"
ul_dl_power_correction = True  # False restores uncorrected UL power weights
lambda_ranges_music_est = [1e10, 1e11, 1e12]
plot_oracle = False
```

Percentages are percentage points: `+5` gives 7.35 GHz and `-5` gives 6.65 GHz. The 8 × 8 BS has a fixed physical spacing of `c/(2*f_dl)`, about 21.414 mm. At UL, Sionna's wavelength-normalized spacing is `0.5*f_ul/f_dl`; its element positions in meters remain unchanged. Positive offsets can therefore introduce spatial aliasing; the code does not shrink the physical array to avoid it.

Each macro draws positions and traces DL TN/NTN channels once. Every percentage uses those same positions, satellite direction, sector orientations, DL TN association and scheduling. Nonzero offsets retrace the NTN channel at UL using the retained endpoints and fixed array. UL tracing uses reciprocal BS-to-NTN paths in the existing channel-vector convention, and restores DL scene state on success or failure. Zero offset reuses the original DL sensing channel; `ul_frequency_percentages=[0]` is the regression control.

In `angle` mode, blind MUSIC uses the actual UL electrical array geometry. Its anonymous estimated angles rebuild DL steering vectors; the estimated UL power weights are retained when power correction is off, or multiplied by `(f_UL/f_DL)**2` when it is on. No true DL channels, paired-user angles or DL gains are supplied to the transfer. This tests angular transfer and UL-derived weights, not instantaneous FDD channel reciprocity. Errors in angular estimation and UL aliasing propagate into the DL nulls. In `raw` mode, the UL vectors themselves are reused, providing a direct cross-frequency mismatch comparison. Both modes calculate INR/SNR/SINR against the same DL channels and DL evaluation mask.

The current settings produce $3 \times 7 = 21$ estimated curves plus one no-nulling baseline per CDF. The color legend lists the three lambdas and the black no-nulling baseline. A separate line-style legend lists each UL percentage and frequency; readers combine the two keys. Set `plot_oracle=True` for one additional true-DL oracle per lambda, or `plot_snr=True` for the optional SNR plot. The oracle does not depend on UL frequency and is only drawn once per lambda.

Runs are saved under `result/<YYYYMMDD_HHMMSS_microseconds>/`. Access one case as:

```python
case = nulling_cdf_results["by_percentage"][5.0]
values = case["est_inr_db"][1e11]
```

`run_config.json` records DL/UL frequencies, fixed spacing, bandwidths and the transfer mode. Common DL caches remain in `channels/channels_<macro>.npz`; UL sensing and MUSIC diagnostics live under `channels/ul_<index>/`. Diagnostics save both UL peak vectors and `peak_u_used_for_dl`. The combined metrics archive uses `ul_<index>/...` keys; separate per-percentage archives retain the existing metric names. Use `*_db_index_<lambda_index>` plus `lambda_ranges_*` for arbitrary lambdas; these indexed arrays avoid collisions from legacy rounded lambda names.

Nonzero offsets are checked for band separation using `abs(f_ul-f_dl) > (B+B_ul)/2`. With both bandwidths at 200 MHz, +/-2% at 7 GHz has overlapping bands; reduce both bandwidths below 140 MHz to treat it as a disjoint FDD case. Zero is exempt as a mathematical control. The optional `enforce_disjoint_bands=False` permits mathematical overlapping-band studies. This remains a narrowband array/channel experiment; bandwidth sets noise and checks separation, without modeling within-band beam squint.

The original element-pattern models and normalized sensing SNR are retained: `music_noise_var = N0_bs / Tx_power`. For a 23 dBm NTN-UE sensing-power experiment, explicitly use `Tx_power_handheld` in that denominator. No RF retuning transient, training duty-cycle overhead, cross-band calibration or material-specific DL gain prediction is introduced here. The optional power correction only applies a free-space frequency scaling. `music_covariance_mode="analytic"` still uses ideal covariance; choose `"sample"` to introduce the existing finite-snapshot estimator.

Validation: zero-offset regression and paired DL baseline/oracle invariance; positive/negative UL angle transfer, vector conventions, weights and state restoration; Cartesian-product plotting and archive round trips. A real Sionna LoS check verified fixed physical positions and frequency-dependent phase/gain. A small end-to-end Denver run used one macro, three frequencies, two lambdas, 12 TNs, eight NTN UEs and the 8 × 8 array, including saving and both CDF plots.

## Coherent multipath and NLOS controls

`Nulling_CDF_SectorDrop.ipynb` now imports the shared `multipath_support.collapse_cir_to_narrowband` helper. The existing `ntn_music_detection.collapse_cir_to_narrowband` import remains compatible. The notebook currently uses **`max_depth = 3`**. Setting `max_depth = 0` selects the legacy direct-path estimator. Example controls:

```python
max_depth = 3  # Sionna maximum path interaction depth
ntn_los_mode = "natural"  # or "nlos_only"
music_spatial_smoothing = True
music_smoothing_rows = 6
music_smoothing_cols = 6
music_forward_backward = True
multipath_top_k = None
multipath_energy_fraction = 1.0
```

Rerun the parameter cell and all subsequent cells. After loading updated Python modules, restart the kernel and run all cells so an old scene instance is not retained. `max_depth` is the only propagation-depth control and is passed to Sionna unchanged for TN DL, NTN DL and NTN UL. It specifies the maximum number of interactions along a path, not how many paths must exist. There is no separate `multipath` enable switch. Zero depth uses the original estimator; positive depth automatically selects coherent-path estimation and replaces diagonal covariance refinement with the correlated fit. Logs, plot titles and result archives record `max_depth` so runs with depths 1, 2, etc. remain distinguishable. If omitted from `run_nulling_cdf_experiment`'s path arguments, the experiment explicitly uses 0. Other Sionna propagation settings remain available. At zero depth, disabling NTN LOS leaves no NTN path; that case is reported without fabricating indirect paths.

`natural` includes all valid direct and indirect paths that the environment and selected propagation mechanisms support. A link is classified as NLOS if it has at least one valid nonzero indirect path and no valid nonzero direct path. `nlos_only` suppresses direct paths **only on NTN links**, at both UL and DL; this is an artificial direct-component removal test. TN links retain natural LOS/NLOS and gain multipath as well, keeping their serving-sector admission rule. A link with no valid path is reported separately; it is not called NLOS.

Depth, reflection/refraction/scattering/diffraction switches, tracing budgets, and solver seed are shared between NTN UL and DL. Direct-path mode preserves the original solver defaults. For positive depths, the notebook defaults to specular reflection and refraction, 100,000 launched samples and a 100,000 path cap per source. Choose the depth directly with `max_depth`; the notebook currently uses 3. Diffuse scattering and diffraction are configurable and disabled by default; scattering also requires suitable nonzero material scattering coefficients. Path finding at finite sampling budgets is approximate. Raising the depth or sampling budget can increase runtime and the number of discoverable paths.

### CIR synthesis and time axes

Sionna's `paths.cir()` returns baseband path coefficients `a` and delays `tau`. Use those coefficients, not the passband `paths.a` attribute. Their carrier propagation phase is already included.

```python
# a: [RX, RX_ANT, TX, TX_ANT, PATH, TIME]
h = collapse_cir_to_narrowband(a)                 # sum PATH, choose TIME=0
h_t = collapse_cir_to_narrowband(a, time_index=3)  # another time sample
h_all_times = collapse_cir_to_narrowband(a, time_index=None)
h_offset = collapse_cir_to_narrowband(
    a, tau=tau, frequency_offset_hz=1e6, valid_mask=paths.valid,
)
```

Summation is complex and coherent: two equal opposite-phase paths cancel. The helper never sums time samples or discards weak paths. Five-dimensional single-time CIRs and four-dimensional already-collapsed channels are accepted. For singleton TIME, default synthesis is numerically identical to the old sum. `time_index=None` returns `[RX, RX_ANT, TX, TX_ANT, TIME]`.

A nonzero frequency offset uses `exp(-j*2*pi*offset*tau)` and requires delays; it does not reapply the carrier phase. This helper's within-band offset holds the traced antenna response fixed and does not model beam squint. The notebook currently evaluates TIME=0 at each carrier, with bandwidth used for noise and band separation. It does not simulate OFDM symbols, ISI, CP violation, or Doppler time evolution. These require a separate wideband/time-domain experiment.

### Coherent-path estimation and full-array nulling

1. Form each UE's channel by coherently adding **all** its paths. In sample mode,
   one waveform per UE/RX antenna drives the combined channel; no independent
   random waveform is fabricated for each ray.
2. Form the full UL covariance. Extract translated 2D rectangular subarrays from
   the actual wavelength-normalized Sionna coordinates and average their
   covariances. Optional forward/backward averaging uses subarray centrosymmetry.
3. Estimate the number of spatial modes and MUSIC peaks on the smoothed
   covariance. This can resolve coherent paths under the usual array aperture,
   angular separation, SNR and subarray-rank conditions. It does not guarantee
   recovery of every ray, and does not remove spatial aliasing at higher UL
   frequencies. Set `music_spatial_smoothing=False` for the unsmoothed comparison.
   `detect_num_sources`, if set, means spatial modes here, not the number of UEs.
4. Rebuild each peak's **full-array** UL steering vector and fit
   `R_UL = A_UL Q_UL A_UL^H + sigma^2 I`, allowing a full correlated PSD `Q_UL`.
   A pseudoinverse followed by PSD projection estimates `Q_UL`; `diag(Q_UL)`
   provides full-array power weights. Conditioning and residuals are saved.
   This replaces diagonal/noncoherent power fitting; it does not perform the old
   diagonal-model joint angle refinement. MUSIC peak refinement remains active.
5. Select estimated directions per BS sector by power, optionally applying
   `multipath_top_k` and `multipath_energy_fraction`. When both are set, top-K can
   prevent reaching the requested energy fraction. The existing
   `max_detected_b_terms` can impose a further beamformer-side cap.
6. Transfer angles onto the full DL array for `ul_to_dl_mode="angle"`. Retain
   estimated UL weights when correction is off; otherwise scale them by `(f_UL/f_DL)**2`. The UL correlation phases are **not** copied into DL.
   `raw` mode retains the existing raw-vector mismatch control. Evaluate beams
   on the complete coherent DL channel, including every omitted/missed path.

The diagonal sum of selected path outer products is a direction-leakage penalty, not the instantaneous coherent DL covariance. UL weights also need not equal DL weights. In multipath mode, the optional `music_real_*` oracle uses all **true DL path vectors and powers** for the same directional penalty, with full coherent DL evaluation. It is a perfect-path-information comparison, not a guaranteed pointwise performance bound. Disabled mode retains the original effective-channel oracle. Subarray smoothing reduces estimation aperture; transmission still uses all 64 physical elements. Source counts must fit the subarray's noise-subspace requirements, and sample-mode MDL on overlapping smoothed data is a heuristic.

### Path diagnostics and reproducibility

Common DL caches now additionally include `channels/paths_dl_<macro>.npz` when `max_depth > 0`. They contain full TN/NTN CIRs and delays, valid-path masks, BS departure angles, powers, direct-path indicators and interaction types. Per-frequency `paths_ul_<macro>.npz` files retain the corresponding UL data.

MUSIC archives additionally save raw/smoothed UL covariances, raw signal rank, subarray shape/count, all candidate angles/powers before selection, and padded correlated-source covariance blocks. `candidate_counts` and `candidate_t_idx` define each block's active dimensions and ordering. Selected peaks and the vectors actually used for DL are saved separately.

`macro_stats[*].path_metrics_ul` and `.path_metrics_dl` report all valid paths, LOS/NLOS/no-path link counts, one-to-one angular matches and matched path-power fractions. The archive stores these dictionaries as JSON strings, avoiding pickle. All path truth is used only for evaluation or explicitly labeled oracle beams, never for the blind estimator. Matching includes every valid ray rather than only the strongest path per UE. Angular matching uses a 5-degree spherical angular gate. Coincident directions, front/back ambiguity and spatial aliasing can limit individual-ray matching even when a null suppresses the channel. The existing vector-correlation coverage diagnostics provide a complementary view. In multipath mode `noncoh_metrics` compare against the sum of true DL path outer products, not the coherent per-UE covariance.

### Tests in this repository

| File                                | Purpose                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/test_tn_sector_drop.py`    | Sector geometry, indoor/outdoor sampling, serving-power margin, retries and final path/UE alignment.                                       |
| `tests/test_music_accuracy.py`    | MUSIC angles and source counts, gain fitting, array conventions, sample noise, numerical scaling and output caching.                       |
| `tests/test_fdd_nulling.py`       | Fixed physical geometry across frequencies, UL-to-DL transfer, zero-offset regression, Cartesian-product CDFs and archives.                |
| `tests/test_multipath_support.py` | Coherent CIR/time handling, subarray smoothing, correlated powers, NLOS classification, path selection and full FDD multipath integration. |

Run all tests in the Sionna environment:

```bash
/home/shizhen/miniconda3/envs/sionna20/bin/python -m unittest discover -s tests -v
```

Multipath validation includes a two-path single-UE rank-1 covariance becoming rank 2 after smoothing, recovery of both directions and full-array powers, signed FDD transfer, and unchanged full-channel evaluation after top-K selection. Pre-change CDF arrays were also saved and compared exactly at `max_depth=0`. Real Sionna tests used a direct-plus-reflected link and its direct-path-removed variant at three frequencies. A small Denver run exercised 12 TNs, eight NTN UEs, an 8 × 8 array, three frequencies and two lambdas through saving and CDF plotting.

## Interpreting the frequency sweep

Here "frequency offset" means UL/DL carrier separation, not residual oscillator CFO or Doppler. DL stays at 7 GHz, and UL spans 5.6, 6.3, 6.65, 7, 7.35, 7.7, and 8.4 GHz. The estimator uses an analytic covariance by default; 800 configured snapshots do not introduce finite-snapshot randomness in that mode.

### What the saved 10-macro experiment actually shows

The following values are computed from `result/sector_drop_fdd_20260914_133236_039356/nulling_cdf_metrics.npz`, with 20 NTN devices, 12 TN devices, depth 3 and $\lambda = 10^{12}$. They describe the previous five-frequency, 10-macro run, not the new seven-frequency run. Lower INR means better NTN interference suppression.

| UL offset | UL frequency (GHz) | Median INR (dB) | 90th-percentile INR (dB) |
| --------- | -----------------: | --------------: | -----------------------: |
| -10%      |               6.30 |         -19.941 |                   -9.620 |
| -5%       |               6.65 |         -19.661 |                   -9.175 |
| 0%        |               7.00 |         -19.775 |                   -8.882 |
| +5%       |               7.35 |         -19.320 |                   -5.719 |
| +10%      |               7.70 |         -18.918 |                   -5.995 |

These data do not establish monotonic degradation with absolute frequency separation. Positive offsets generally perform worse for this lambda, whereas negative offsets can outperform zero. Other lambdas and CDF quantiles can cross. All five cases have exactly the same no-nulling DL INR samples. With only ten independent geometry draws, the curves are descriptive rather than a proof of an asymptotic trend; UE samples within a macro are not independent macro trials.

### Angle transfer already compensates the known array frequency dependence

`transfer_music_peaks_to_dl()` rebuilds each estimated direction using DL wavelength-normalized positions. Therefore an explanation based solely on reusing UL steering vectors at DL applies to `raw` mode, not to the current `angle` experiment. Transfer cannot repair a wrong angle, a missed path or an ambiguous UL spatial peak.

For the fixed physical spacing $d = c/(2f_{\mathrm{DL}})$, $d/\lambda_{\mathrm{UL}}$ equals 0.4, 0.45, 0.475, 0.5, 0.525, 0.55 and 0.6 for the seven offsets. Lower UL frequencies reduce electrical aperture; higher frequencies can introduce spatial aliasing for some directions once the spacing exceeds half a wavelength. Neither effect implies universal monotonic angle error. Spatial smoothing further reduces the estimation aperture to 6 × 6 while nulling still uses the full 8 × 8 array.

In the previous run, the macro-averaged `coverage95` was 0.9821 at zero offset, 0.9615 at +5%, and 0.9621 at +10%. This diagnostic measures DL path power with a steering-vector match of at least 0.95, not a 95% confidence interval. It is not the same as the one-to-one, 5-degree angular power coverage. These changes support an estimation/coverage contribution but do not isolate its causal share.

### Retained UL powers change the effective nulling strength

With power correction disabled, the implemented beam maximizes, for a unit-norm $v$,

$$
|\widetilde{h}_0^H v|^2 - \lambda v^H \widehat B_{DL}v,
\qquad
\widehat B_{DL}=\sum_\ell \widehat g_{\ell,UL}
\widehat u_{\ell,DL}\widehat u_{\ell,DL}^H.
$$

The desired effective channel is normalized. UL power weights are not normalized to a common trace or predicted at DL, and lambda is held fixed across frequencies. Thus identical lambda values do not imply identical effective penalty strengths. If every power weight is multiplied by $s$, the objective is exactly equivalent to replacing $\lambda$ by $s\lambda$, with directions held fixed.

For a free-space path with fixed antenna gains, a first-order approximation is

$$
g_{UL}/g_{DL}\approx(f_{DL}/f_{UL})^2.
$$

At +10%, the ratio is about 0.826; at -10%, it is about 1.235. At +20% and -20%, the corresponding ratios are about 0.694 and 1.563. Higher UL frequency can then underweight the DL interference penalty, while lower UL frequency can overweight it. This is a plausible explanation of the observed sign asymmetry, not a measured decomposition of the result. Frequency-dependent material responses and imperfect power fitting prevent applying this free-space approximation to every reflected or refracted path as an exact correction.

### Coherent propagation and the noncoherent directional penalty

A narrowband channel can be written as

$$
h(f)=\sum_\ell \alpha_\ell(f)e^{-j2\pi f\tau_\ell}u_\ell(f),
\qquad
\Delta\phi_{\ell m}=-2\pi\Delta f(\tau_\ell-\tau_m).
$$

Changing carrier frequency changes relative path phases and potentially path amplitudes. UL and DL coherent covariance matrices therefore need not coincide, even for shared geometric directions. Sionna's baseband CIR already contains the carrier propagation phase; it must not be applied a second time. See the [Sionna Paths documentation](https://nvlabs.github.io/sionna/rt/api/paths.html).

The estimator fits the full correlated $Q_{\mathrm{UL}}$ to obtain path powers, but the beamformer uses only $\operatorname{diag}(Q_{\mathrm{UL}})$. The off-diagonal UL correlation phases are saved as diagnostics, not transferred into DL. The directional penalty consequently omits coherent cross terms, while the evaluated DL INR retains them.

A change of relative phase alone is not sufficient to prove nulling degradation: if $v$ is orthogonal to every true DL path vector, their coherent sum is also zero for arbitrary path phases. Residual leakage arises when directions are missing or inaccurate, or finite lambda leaves a nonzero response along them. Their DL coherent combination then determines the actual interference. Stronger penalties cannot reconstruct unobserved directions.

### Diagnostics and controlled comparisons

`B_nrmse` in multipath mode compares the estimated directional penalty with the sum of true DL path outer products. It is not error against the coherent per-UE covariance. It need not vary monotonically with frequency offset or track every INR quantile. The unweighted relative gain error can be dominated by tiny matched true powers; do not interpret its very large values as a uniform gain error.

To distinguish mechanisms in future experiments:

1. Apply the approximate power correction
   `g_DL_hat = g_UL_hat * (f_UL/f_DL)**2`, holding estimated directions fixed,
   and compare with uncorrected results. This isolates a simple frequency-scale
   effect; it does not compensate material-specific gains.
2. Use a common trace for the penalty matrices, or sweep lambda separately per
   frequency, to examine direction/subspace quality at comparable penalty scale.
3. Enable `plot_oracle=True` to compare against true DL path directions/powers.
   This path oracle is not a guaranteed pointwise INR bound for the coherent
   channel. Keep it distinct from an instantaneous coherent-channel oracle.
4. Compare direct-path and multipath cases with controlled geometry, and inspect
   accepted peaks, coverage, fit conditioning, and path power diagnostics.

The initial seven-frequency, 20-macro run (`result/20260916_000256_575894/`) retained the original UL weights. The notebook now explicitly enables the power-correction switch described below for a new 20-macro run. The proposed angular and phase compensation methods remain separate future experiments.

## Optional UL-to-DL power correction

The notebook parameter `ul_dl_power_correction` is an on/off switch. `True`
uses `g_DL_hat = g_UL_hat * (f_UL/f_DL)**2`; `False` preserves the original
weights. The shared `run_nulling_cdf_experiment()` API defaults to `False`
for compatibility. The current notebook explicitly sets it to `True`.

For offsets `[-20, -10, -5, 0, 5, 10, 20]%`, the factors are
`[0.64, 0.81, 0.9025, 1, 1.1025, 1.21, 1.44]`.
The factor multiplies **power**, not channel amplitude, and is applied after
MUSIC/peak selection but before assembling the DL nulling penalty. It works
with either `angle` or `raw` vector transfer. It does not alter raw UL covariance,
fitted `Q_UL`, peak angles, selection scores, physical channels, or oracle powers.
Zero offset is a numerical no-op. This is a free-space approximation; it does not
predict material-specific frequency responses or coherent multipath phase.

The run config, metric archives and summary CSV record `ul_dl_power_correction`;
each frequency case also records `ul_dl_power_scale`. MUSIC archives retain
`peak_g_hat` as the original UL estimate and add `peak_g_used_for_dl` as the
weight actually passed toward the beamformer. The logs record the switch and
factor, and CDF titles state ON or OFF.

At fixed estimated directions, a common power factor is equivalent to scaling
lambda. Compare NTN INR together with TN SNR/SINR: a lower INR alone does not
establish an improved signal/interference tradeoff. This correction does not
recover missed paths or resolve angular aliases.

## Run archive and plotting

Every execution of the experiment cell creates a fresh local-time directory `result/YYYYMMDD_HHMMSS_microseconds/` before tracing begins. Older runs remain intact. The run records timezone-aware start/end timestamps and its status.

- `run_config.json`: actual frequencies, lambda values, random seeds, geometry,
  propagation, MUSIC, noise/power and plotting parameters.
- `run_status.json`: running, metrics_saved, complete or failed, with timestamps.
- `environment.json`: Python/platform and relevant installed package versions.
- `sources/` and `source_manifest.json`: notebook, helper modules and documentation
  snapshots with SHA-256 hashes. The notebook snapshot is the saved file on disk;
  `run_config.json` describes the parameters actually used by the running kernel.
- `run.log`: experiment stdout/stderr, including per-macro MUSIC/drop diagnostics.
- `nulling_cdf_metrics.npz` and `ul_*/nulling_cdf_metrics.npz`: combined and separate
  raw metric arrays, oracle arrays and diagnostic statistics, without pickle.
- `summary.csv`: sample counts and 10th/50th/90th percentiles for INR, SNR and SINR,
  including the no-nulling and oracle comparisons. Quantiles use finite samples;
  total and finite counts are reported separately.
- `channels/` and `tn_drop/`: shared DL channels, per-frequency UL channels,
  path/MUSIC diagnostics and TN admission reports.
- `nulling_inr_cdf.{png,pdf}` and `nulling_tn_sinr_cdf.{png,pdf}`: result figures;
  `plot_snr=True` additionally saves `nulling_tn_snr_cdf.{png,pdf}`.
- `cdf_style_map.json`: the color/lambda and dash/frequency mapping.
- `artifacts.json`: file names and sizes at successful completion.

The color legend lists lambda values and the black no-nulling reference once. The line-style legend lists UL offsets and frequencies once. Optional DL oracle curves use circle markers to distinguish them from the frequency styles.

Run all notebook cells with the `sionna20` kernel, or execute:

```bash
/home/shizhen/miniconda3/envs/sionna20/bin/python run_sector_drop.py
```

The command-line runner also saves `Nulling_CDF_SectorDrop.executed.ipynb` and `notebook_execution.log` in the run directory, and refreshes the working notebook outputs after successful completion. On execution failure it preserves the partial notebook and diagnostic log without replacing the working notebook.
