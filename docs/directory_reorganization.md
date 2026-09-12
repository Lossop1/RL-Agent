# 目录重组执行方案

**目标**：优化仓库目录结构，提升可维护性和清晰度

**执行日期**：2026-09-11

---

## 当前问题

1. **产物与源码混杂**：`output/`、`strategy_backups/` 与源码在同一级别
2. **文档分散**：架构文档、API 文档、归档文档分散在多个位置
3. **工具脚本分散**：部分工具在 `tools/`，部分在根目录
4. **测试镜像不完整**：`tests/` 目录未完全镜像 `autotuner/` 和 `products/` 结构
5. **配置文件分散**：SSH 配置、产品合同、环境变量分散

---

## 重组方案

### 阶段 1：产物目录统一（无需代码改动）

```bash
# 将所有运行产物移动到 output/ 下的子目录
mkdir -p output/strategies output/artifacts output/runs
mv strategy_backups/* output/strategies/  # 如果存在
# output/ 结构：
#   output/assets/        # 资产目录（已存在）
#   output/strategies/    # 历史策略备份
#   output/artifacts/     # 临时构建产物
#   output/runs/          # 运行记录
```

**影响**：无代码依赖，仅需更新 `.gitignore` 和文档引用

---

### 阶段 2：文档目录重组（需更新引用）

```bash
# 将文档按类型分类
mkdir -p docs/architecture docs/api docs/operations docs/archive

# 移动架构文档
mv docs/repository_architecture.md docs/architecture/
mv docs/traditional_control/ docs/architecture/traditional_control/

# 移动操作文档
mv docs/locomotion_console_startup.md docs/operations/
mv docs/traditional_control_demo.md docs/operations/

# 保持 docs/README.md 和 docs/taili_spec.md 在顶层
```

**影响**：需更新 `README.md` 和代码中的文档链接

---

### 阶段 3：配置文件统一（需更新代码）

```bash
# 将所有配置文件移动到 config/ 下
# 已有：config/products/taili.yaml, config/ssh.json.example
# 补充：
#   config/repository_structure.toml  # 从根目录移入
#   config/.env.example               # 环境变量模板
```

**影响**：需更新 `tools/check_repository_structure.py` 读取配置的路径

---

### 阶段 4：测试目录完善（需补充测试）

```bash
# 确保 tests/ 完整镜像源码结构
# 当前缺失的测试模块：
mkdir -p tests/autotuner/artifacts
mkdir -p tests/autotuner/infrastructure
mkdir -p tests/products/taili/core
mkdir -p tests/products/taili/traditional_control

# 为每个缺失模块添加占位测试文件
# tests/autotuner/artifacts/test_asset_catalog.py
# tests/autotuner/infrastructure/test_remote.py
# tests/products/taili/core/test_geometry.py
```

**影响**：提升测试覆盖率，需编写测试用例

---

### 阶段 5：工具脚本整理（可选）

```bash
# 将根目录的零散脚本移动到 tools/
# 保留：
#   pyproject.toml, requirements.txt, docker-compose.yml
# 移动到 tools/：
#   任何 *.sh, *.py 工具脚本（如果有）
```

---

## 执行顺序

1. **先执行阶段 1**（产物目录）：无风险，立即可做
2. **再执行阶段 2**（文档目录）：需要批量替换文档引用
3. **提交 Git**：确保中间状态可回退
4. **执行阶段 3**（配置文件）：需要修改代码路径
5. **测试验证**：运行 `pytest` 和 `check_repository_structure.py`
6. **执行阶段 4**（测试补充）：逐步补充测试用例
7. **最后执行阶段 5**（工具整理）：可选，低优先级

---

## 具体步骤

### 步骤 1：备份当前状态

```bash
git add -A
git commit -m "chore: 目录重组前的备份快照"
git tag pre-reorganization
```

### 步骤 2：执行阶段 1（产物目录）

```bash
# 如果 strategy_backups/ 存在
if [ -d "strategy_backups" ]; then
    mkdir -p output/strategies
    mv strategy_backups/* output/strategies/
    rmdir strategy_backups
fi

# 更新 .gitignore
echo "output/strategies/" >> .gitignore
echo "output/artifacts/" >> .gitignore
echo "output/runs/" >> .gitignore

git add -A
git commit -m "chore(目录结构): 统一运行产物到 output/ 子目录"
```

### 步骤 3：执行阶段 2（文档目录）

```bash
mkdir -p docs/architecture docs/api docs/operations

mv docs/repository_architecture.md docs/architecture/
mv docs/traditional_control/ docs/architecture/traditional_control/
mv docs/locomotion_console_startup.md docs/operations/
mv docs/traditional_control_demo.md docs/operations/

# 批量替换文档引用（需手工验证）
grep -r "docs/repository_architecture.md" . --include="*.md" --include="*.py"
# 手工修改每个引用为 docs/architecture/repository_architecture.md

git add -A
git commit -m "chore(目录结构): 重组文档目录，按类型分类"
```

### 步骤 4：执行阶段 3（配置文件）

```bash
# repository_structure.toml 已在 config/ 中，无需移动

# 更新代码中的配置路径（如果有硬编码）
# 运行检查确保一切正常
python tools/check_repository_structure.py

git add -A
git commit -m "chore(目录结构): 统一配置文件到 config/ 目录"
```

### 步骤 5：验证

```bash
# 运行所有检查
python tools/check_repository_structure.py
python -m pytest
python tools/check_doc_redundancy.py
python tools/check_comment_quality.py --path autotuner/

# 启动控制台测试
python -m autotuner.locomotion_console
```

---

## 回滚方案

如果重组后出现问题：

```bash
git reset --hard pre-reorganization
git tag -d pre-reorganization
```

---

## 预期收益

1. **清晰的源码/产物分离**：开发者不会误提交运行产物
2. **文档易于查找**：架构、API、操作文档分类明确
3. **配置集中管理**：所有配置在 `config/` 目录
4. **测试覆盖完整**：`tests/` 镜像源码结构
5. **符合工程规范**：目录结构清晰，职责明确

---

## 注意事项

1. 每个阶段执行后立即提交 Git
2. 文档引用需要手工验证，避免死链接
3. 配置路径修改后需要运行完整测试
4. 如果使用了绝对路径，需要全局搜索并替换
5. 重组期间暂停其他开发工作，避免合并冲突
