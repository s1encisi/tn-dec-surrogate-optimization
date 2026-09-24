# TN / DEC 代理建模与多目标优化

分别预测出水总氮与全厂日总电耗，在固定工况和历史支持范围内比较候选方案。包含训练期预处理、留出集隔离、代理选择与同预算多目标搜索。

## 运行

使用 Python 3.11 或 3.12：

```bash
python -m pip install -e .
python -m tn_dec_demo --quick --output demo_outputs/quick_01
python -m unittest discover -s tests -v
```

输出目录必须尚不存在。公开演示独立生成合成数据，不读取工厂数据或论文材料；结果不等同于已实现的节能收益。

`tn_dec_demo/` 为演示实现，`Paper/SourceCode/` 与 `Paper/revision_20260905/code/` 保留研究源码。受保护数据、模型和论文不随仓库分发。

许可条款见 [LICENSE](LICENSE)。
