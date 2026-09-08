# TN–DEC surrogate modeling and multi-objective optimization

本私密仓库仅保存经过审查的计算代码、算法配置和测试。原始运行数据及研究交付材料保留在本地。

## 代码入口

- `Paper/SourceCode/src/taici/`：预测、最终代理、解释分析及历史强化学习模块。
- `Paper/SourceCode/scripts/` 和 `configs/`：科学计算入口及配置。
- `Paper/revision_20260905/code/reproduce_moo.py`：条件多目标优化的准备、调优、评价与分析。
- `Paper/revision_20260905/code/context_moo.py`：情景支持范围、坐标映射与多目标算法。
- `Paper/SourceCode/tests/` 与新版优化测试：实现验证。

## 本地环境

Python版本要求见 `Paper/SourceCode/pyproject.toml`。从仓库根目录，在独立环境中安装代码依赖：

```powershell
python -m pip install -e "./Paper/SourceCode[dev]"
git config --local core.hooksPath .githooks
python -B tools/check_repository_safety.py --self-test
```

完整实验还需要另行取得已授权的本地数据、冻结代理及证据文件，并恢复到代码要求的相对位置。仓库没有这些文件，不能独立复现论文数值；依赖真实数据的测试也需要对应本地输入。不要将这些输入加入版本控制。

## 上传范围

`.gitignore` 使用逐文件白名单。新增文件默认保持本地，只有审查内容并加入准确路径后才会纳入版本控制。

以下材料不在上传范围内：

- Excel、CSV及其他数据表、数据库、数组、训练数据和原始工艺资料。
- 训练模型、checkpoint、实验结果、运行日志和数据来源记录。
- 论文正文、SI、投稿材料、文献、图表、绘图数据及文档构建材料。
- 含厂区流程信息的绘图脚本、个人路径、凭据、环境文件和缓存。
- 历史实验、归档副本及本地虚拟环境。

本地原有根目录README保留，但不上传；GitHub展示本说明。

提交前检查暂存文件，推送前检查所推分支的完整提交历史：

```powershell
python -B tools/check_repository_safety.py --staged
python -B tools/check_repository_safety.py --history
```

检查会拒绝白名单外路径、数据和二进制类型、超大文件，以及常见凭据和个人信息模式。自动检查无法判断所有保密内容；加入新文件前仍需审阅。新克隆的仓库需要执行上面的 `core.hooksPath` 配置才能启用Git钩子。

不随仓库分发数据，不授予数据或模型的再分发权限。
