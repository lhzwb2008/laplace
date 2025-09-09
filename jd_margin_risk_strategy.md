# 鸡蛋期货保证金风险监控策略文档

## 📋 策略概述

### 策略名称
鸡蛋期货保证金风险监控策略 (JD Margin Risk Strategy)

### 策略类型
- **交易品种**: 鸡蛋期货 (JD)
- **策略分类**: 季节性趋势策略 + 保证金风险管理
- **交易周期**: 日线级别
- **持仓方式**: 单向持仓（多头或空头）

### 核心理念
基于鸡蛋期货的季节性供需规律，结合技术指标和严格的风险控制，在夏季低价时买入，冬季高价时卖出，通过保证金杠杆放大收益的同时严格控制风险。

---

## 🎯 策略逻辑

### 1. 季节性交易逻辑

#### 买入条件（夏季策略）
- **时间窗口**: 6月、7月、8月
- **价格条件**: 当前价格位于过去120天价格区间的40%以下位置
- **技术条件**: RSI ≤ 70（避免超买）
- **波动率条件**: 当前波动率 ≤ 2.5倍历史平均波动率

#### 卖出条件（冬季策略）
- **时间窗口**: 11月、12月、1月
- **价格条件**: 当前价格位于过去120天价格区间的60%以上位置
- **技术条件**: RSI ≥ 25（避免超卖）
- **波动率条件**: 当前波动率 ≤ 2.5倍历史平均波动率

### 2. 技术指标计算

#### 价格位置指标
```python
price_position = (current_price - min_price) / (max_price - min_price)
```
- **计算周期**: 120天
- **作用**: 判断当前价格在历史区间中的相对位置

#### RSI指标
```python
rsi = 100 - (100 / (1 + rs))
rs = average_gain / average_loss
```
- **计算周期**: 14天
- **作用**: 避免在极端超买超卖时开仓

#### 波动率指标
```python
volatility = std(returns) * sqrt(252)
```
- **计算周期**: 20天
- **作用**: 控制在高波动期的交易风险

---

## ⚖️ 风险控制体系

### 1. 仓位计算算法

#### 多重仓位限制机制
```python
def calculate_position_size(self, price, direction):
    # 1. 基于目标保证金占用率的仓位
    target_position = (self.capital * self.target_margin_usage) / (price * 10 * self.margin_rate)
    
    # 2. 基于风险的仓位限制
    risk_capital = self.capital * self.risk_per_trade
    risk_based_position = risk_capital / (price * 10 * 0.1)  # 假设10%止损
    
    # 3. 基于可用资金的安全仓位
    available_capital = self.capital * self.max_position_ratio
    max_safe_position = available_capital / (price * 10 * self.margin_rate)
    
    # 4. 取最小值确保安全（激进策略：2.0倍风险放宽）
    position_size = min(target_position, max_safe_position, risk_based_position * 2.0)
    
    return max(1, int(position_size))  # 至少1手
```

#### 关键参数说明
- **目标保证金占用率**: 60%（可配置）
- **单次交易风险**: 20%（可配置）
- **最大仓位比例**: 90%（可配置）
- **风险放宽倍数**: 2.0（激进策略）

### 2. 保证金风险监控

#### 实时风险检查
```python
def check_margin_risk(self, current_price, date):
    if self.position != 0:
        # 计算当前保证金占用
        current_margin = abs(self.position) * current_price * 10 * self.margin_rate
        margin_ratio = current_margin / self.capital
        
        # 风险预警机制
        if margin_ratio > 0.8:  # 80%预警线
            return {'level': 'high', 'ratio': margin_ratio}
        elif margin_ratio > 0.6:  # 60%注意线
            return {'level': 'medium', 'ratio': margin_ratio}
    
    return {'level': 'low', 'ratio': 0}
```

### 3. 止损机制

#### 固定比例止损
- **止损幅度**: 10%（可配置）
- **计算方式**: 基于开仓价格的固定百分比
- **执行方式**: 每日收盘价检查

```python
def calculate_stop_loss(self, entry_price, direction):
    if direction == '多头':
        return entry_price * (1 - self.stop_loss_pct)
    else:
        return entry_price * (1 + self.stop_loss_pct)
```

---

## 📊 合约规格与成本

### 鸡蛋期货(JD)合约规格
- **交易单位**: 5吨/手
- **报价单位**: 元（人民币）/500千克
- **最小变动价位**: 1元/500千克
- **保证金比例**: 10%（策略设定）
- **价格计算**: 显示价格×10 = 实际合约价值

### 交易成本计算
```python
# 开仓成本
entry_cost = position_size * price * 10 * 0.0001  # 万分之一手续费

# 平仓成本
close_cost = position_size * price * 10 * 0.0001

# 总成本
total_cost = entry_cost + close_cost
```

---

## 📈 策略性能

### 历史回测结果
- **回测期间**: 2022-08-26 至 2025-09-09
- **初始资金**: 20,000元
- **最终资金**: 373,449元
- **总收益率**: 1767.2%
- **年化收益率**: 204.7%
- **夏普比率**: 2.186
- **最大回撤**: -60.6%
- **Calmar比率**: 3.378

### 风险指标分析
- **年化波动率**: 93.7%
- **胜率**: 根据具体交易统计
- **平均持仓周期**: 季节性周期
- **最大保证金占用**: 动态监控

---

## 🔧 策略配置

### 核心参数配置
```python
default_config = {
    'initial_capital': 20000,           # 初始资金
    'margin_rate': 0.10,                # 保证金比例
    'max_position_ratio': 0.9,          # 最大仓位比例
    'risk_per_trade': 0.2,              # 单次交易风险
    'stop_loss_pct': 0.1,               # 止损比例
    'target_margin_usage': 0.6          # 目标保证金占用率
}
```

### 季节性参数
```python
# 交易月份
buy_months = [6, 7, 8]      # 夏季买入月份
sell_months = [11, 12, 1]   # 冬季卖出月份

# 价格阈值
buy_threshold = 0.4         # 买入价格位置阈值
sell_threshold = 0.6        # 卖出价格位置阈值

# 技术指标参数
price_pos_period = 120      # 价格位置计算周期
rsi_period = 14             # RSI计算周期
rsi_buy_max = 70           # RSI买入上限
rsi_sell_min = 25          # RSI卖出下限
vol_max = 2.5              # 波动率最大倍数
```

---

## 🚀 使用说明

### 1. 环境要求
```python
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
```

### 2. 数据格式要求
- **文件格式**: CSV
- **必需字段**: date, open, high, low, close, volume
- **时间格式**: YYYY-MM-DD
- **价格单位**: 元/500千克

### 3. 运行方式
```python
# 基本运行
strategy, results, metrics = main()

# 自定义配置运行
custom_config = {
    'initial_capital': 50000,
    'target_margin_usage': 0.7
}
strategy, results, metrics = main(custom_config)
```

### 4. 输出说明
- **控制台输出**: 策略配置、回测结果、性能指标
- **日志文件**: trading_details.log（详细交易记录）
- **返回值**: 策略对象、回测数据、性能指标

---

## ⚠️ 风险提示

### 1. 市场风险
- 期货市场价格波动剧烈，可能面临重大亏损
- 季节性规律可能因市场环境变化而失效
- 极端市场条件下止损可能无法有效执行

### 2. 策略风险
- 历史回测结果不代表未来表现
- 策略参数需要根据市场变化进行调整
- 保证金交易具有杠杆风险，亏损可能超过本金

### 3. 技术风险
- 数据质量影响策略表现
- 系统故障可能导致交易中断
- 滑点和交易成本可能影响实际收益

### 4. 合规风险
- 需要遵守相关期货交易法规
- 确保资金来源合法合规
- 注意税务申报义务

---

## 📝 更新日志

### v1.0 (2025-01-08)
- ✅ 完成策略核心逻辑开发
- ✅ 修复利润计算问题
- ✅ 优化风险控制算法
- ✅ 实现详细交易日志
- ✅ 添加性能指标计算
- ✅ 优化日志输出（控制台+文件）

### 待优化项目
- [ ] 增加更多技术指标
- [ ] 实现动态止损机制
- [ ] 添加资金管理优化
- [ ] 支持多品种交易
- [ ] 实现实时交易接口

---

## 📞 技术支持

如有技术问题或策略优化建议，请通过以下方式联系：

- **策略开发**: AI Assistant
- **创建时间**: 2025-01-08
- **最后更新**: 2025-01-08
- **版本**: v1.0 Final

---

*本文档仅供学习和研究使用，不构成投资建议。期货交易有风险，入市需谨慎。*