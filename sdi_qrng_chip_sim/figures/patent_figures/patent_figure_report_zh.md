# SDI QRNG 芯片专利图表仿真说明

本报告对应“一种具有片上自校准和动态熵提取控制的半设备无关量子随机数芯片及其随机数生成方法”的实施例仿真图。

## 图表列表

### 图 1: drift_fixed_vs_dynamic

- 文件：`/Users/sichen/Library/Mobile Documents/com~apple~CloudDocs/Business/Quantum_Device_Independent_server/QRNG/sdi_qrng_chip_sim/figures/patent_figures/patent_fig_1_drift_fixed_vs_dynamic.png`
- 说明：在光源功率漂移条件下，固定压缩率持续按预设速率输出，而动态压缩率随认证熵逐块调整输出长度。
- 关键结果：
  - `baseline_fixed_no_cal`: certified_rate=0.006389, stop_rate=0.000000, unsafe_block_rate=0.744444, unsafe_emitted_rate=0.018611, calibration_events=0
  - `dynamic_no_cal`: certified_rate=0.020485, stop_rate=0.000000, unsafe_block_rate=0.000000, unsafe_emitted_rate=0.000000, calibration_events=0
- 多 seed 统计：
  - `baseline_fixed_no_cal`: certified_rate_mean=0.004819 ± 0.001573, unsafe_block_rate_mean=0.807222 ± 0.062910
  - `dynamic_no_cal`: certified_rate_mean=0.018176 ± 0.003268, unsafe_block_rate_mean=0.000000 ± 0.000000

### 图 2: anomaly_event_calibration

- 文件：`/Users/sichen/Library/Mobile Documents/com~apple~CloudDocs/Business/Quantum_Device_Independent_server/QRNG/sdi_qrng_chip_sim/figures/patent_figures/patent_fig_2_anomaly_event_calibration.png`
- 说明：当片上监测能量超过安全包络时，事件触发自校准降低 omega_high，并恢复认证随机数输出。
- 关键结果：
  - `dynamic_no_cal`: certified_rate=0.000356, stop_rate=0.862500, unsafe_block_rate=0.000000, unsafe_emitted_rate=0.000000, calibration_events=0
  - `proposed_dynamic_event_cal`: certified_rate=0.030928, stop_rate=0.050000, unsafe_block_rate=0.000000, unsafe_emitted_rate=0.000000, calibration_events=26
- 多 seed 统计：
  - `dynamic_no_cal`: certified_rate_mean=0.002703 ± 0.002633, unsafe_block_rate_mean=0.000000 ± 0.000000
  - `proposed_dynamic_event_cal`: certified_rate_mean=0.021517 ± 0.003963, unsafe_block_rate_mean=0.000000 ± 0.000000

### 图 3: unsafe_block_safety

- 文件：`/Users/sichen/Library/Mobile Documents/com~apple~CloudDocs/Business/Quantum_Device_Independent_server/QRNG/sdi_qrng_chip_sim/figures/patent_figures/patent_fig_3_unsafe_block_safety.png`
- 说明：固定压缩率可能输出超过熵下界可认证的比特数；本方案通过动态输出控制避免 unsafe block。
- 关键结果：
  - `baseline_fixed_no_cal`: certified_rate=0.000333, stop_rate=0.088889, unsafe_block_rate=0.988889, unsafe_emitted_rate=0.029667, calibration_events=0
  - `proposed_dynamic_event_cal`: certified_rate=0.016964, stop_rate=0.005556, unsafe_block_rate=0.000000, unsafe_emitted_rate=0.000000, calibration_events=19
- 多 seed 统计：
  - `baseline_fixed_no_cal`: certified_rate_mean=0.000933 ± 0.000359, unsafe_block_rate_mean=0.968889 ± 0.011950
  - `proposed_dynamic_event_cal`: certified_rate_mean=0.018693 ± 0.001529, unsafe_block_rate_mean=0.000000 ± 0.000000

## 专利说明书可用结论

1. 在漂移环境下，动态熵提取根据每个 block 的认证熵改变输出长度，避免固定压缩率过度输出。
2. 在能量异常环境下，事件触发自校准把片上监测得到的能量上界拉回安全包络内，并恢复认证输出。
3. 固定输出方案会出现 unsafe block；本方案在同等输入数据下通过动态输出长度控制避免 unsafe block。
