# 复现与验证说明

## 环境

已验证Windows与Python 3.11.14。根目录pyproject.toml固定直接依赖：NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.10.6、pymoo 0.6.1.5。未提供全部传递依赖的跨平台锁文件，其他平台需运行同一测试集。

根目录环境服务于合成演示；完整研究依赖另见Paper/SourceCode/pyproject.toml。

## 运行

```powershell
python -m tn_dec_demo --quick --output demo_outputs/quick_new
python -m tn_dec_demo --seed 17 --rows 480 --evaluations 256 --output demo_outputs/full_new
python -m unittest discover -s tests -v
```

完整运行使用3个合成背景、3个优化种子，每种方法每次256次评价。quick模式使用1个背景、1个种子与64次评价。输出必须为新目录。

协议保存生成设定、种子、拆分和软件版本；选择锁在留出评价前写入；manifest记录产物SHA-256。哈希用于一致性核验，不是数字签名。

## 统计对象

预测指标来自20%的独立留出部分，选择只使用训练/验证数据。基础预处理在训练部分或堆叠内折拟合。解释输出为开发模型的验证集置换重要性。

搜索是在固定代理上的数值比较。归一化采用开发记录代理输出的第5与95百分位，共享参考点(1.1,1.1)。HV先按优化种子对情景平均，再汇总种子；不同数据生成种子的代理和尺度不同，不能把跨数据种子的HV当作同一外部基准。

搜索耗时包含代理与算法计算；流程耗时不含绘图和解释器冷启动。时间来自本机单进程实测，没有线上并发或生产吞吐测试。

## 重复运行

```powershell
python tools/benchmark_demo.py --output demo_outputs/benchmark_new
python tools/summarize_demo.py --input demo_outputs/benchmark_new --output docs/DEMO_RESULTS.md
```

不应根据同一留出集反复改动模型或生成机制后，继续称其为未使用测试。开发新方案时需要登记新的选择与评价协议。

## 受保护的研究材料

完整研究入口核验字段、时间范围和输入哈希。缺少授权材料时拒绝运行是预期行为；不要关闭检查或把合成数据放入原始数据路径。公开演示是无需这些材料即可验证流程的入口。
