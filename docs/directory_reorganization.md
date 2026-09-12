# 目录重组方案

## 当前问题

1. **产物与源码混合**：`output/`、`frontend/dist/` 等生成目录与源码在同一层级
2. **历史文件位置不当**：`strategy_backups/` 归档内容未明确隔离
3. **工具分散**：部分工具脚本位置不统一
4. **配置文件分散**：SSH 配置、产品合同、结构定义分布在不同位置

## 目标结构

```
locomotion-workspace/
├── autotuner/              # 系统源码（产品无关）
│   ├── mechanisms/         # 第 6 层：奖励、指标、门控机制
│   ├── research/           # 第 3 层：研究协调、证据裁决
│   ├── artifacts/          # 资产目录与复用审批
│   ├── infrastructure/     # 第 2 层：SSH、进程、远程执行
│   ├── execution/          # 第 2 层：runtime、payload、部署
│   ├── control/            # 传统控制核心（产品无关）
│   ├── locomotion_console/ # 第 1 层：控制台后端
│   └── adapter/            # 机器人适配层
│
├── products/               # 产品实现（Taili 等）
│   └── taili/
│       ├── core/           # 第 5-6 层：观测、奖励、模型
│       ├── blind_locomotion/  # 任务适配
│       ├── traditional_control/  # 传统控制产品适配
│       ├── payload/        # Payload 清单与构建
│       └── ops/            # 运维插件
│
├── config/                 # 所有配置文件集中
│   ├── products/           # 产品合同
│   │   └── taili.yaml
│   ├── repository_structure.toml  # 结构定义
│   ├── ssh.json.example    # SSH 配置模板
│   └── ssh.json            # SSH 配置（gitignore）
│
├── tools/                  # 所有工具脚本
│   ├── check_repository_structure.py
│   ├── decouple_layers.py
│   ├── check_doc_redundancy.py
│   ├── check_comment_quality.py
│   ├── traditional_control/
│   │   └── build_bundle.py
│   └── isaaclab_quad_diag_observation/  # 可独立打包的诊断库
│
├── docs/                   # 文档（活跃维护）
│   ├── README.md           # 文档索引
│   ├── repository_architecture.md
│   ├── atomic_problems.md  # 原子问题跟踪
│   ├── maintenance_standard.md
│   ├── traditional_control/
│   │   └── PROJECT_MAP.md
│   └── archive/            # 历史文档（只读参考）
│       └── taili/
│
├── locomotion-console-ui/  # 前端源码
│   ├── src/
│   └── dist/               # 构建产物（gitignore）
│
├── tests/                  # 测试源码
│   ├── autotuner/
│   └── products/
│
├── output/                 # 所有运行产物（gitignore）
│   ├── assets/
│   ├── runtimes/
│   ├── payloads/
│   └── runs/
│
├── .git/
│   └── hooks/
│       ├── commit-msg      # Git 钩子
│       └── pre-commit
│
├── .github/
│   └── workflows/
│       └── ci.yml          # CI 配置
│
├── README.md               # 项目入口文档
├── pyproject.toml
├── requirements.txt
└── docker/
    └── docker-compose.yml
```

## 重组步骤

### 阶段 1：配置文件集中（低风险）

```bash
# 1. 确保 config/ 目录存在
mkdir -p config/products

# 2. 移动产品合同（如果不在 config/products/ 下）
# 已在正确位置，跳过

# 3. 验证所有配置文件路径
python tools/check_repository_structure.py
```

### 阶段 2：工具脚本整理（低风险）

```bash
# 所有工具已在 tools/ 下，无需移动
# 确保可执行权限
chmod +x tools/*.py
chmod +x tools/traditional_control/*.py
```

### 阶段 3：产物目录隔离（中风险）

当前 `output/`、`frontend/dist/` 已通过 `.gitignore` 排除，无需移动。
确认以下目录在 `.gitignore` 中：

```gitignore
output/
frontend/dist/
locomotion-console-ui/dist/
.pytest_cache/
__pycache__/
*.pyc
node_modules/
.pt/
```

### 阶段 4：历史文件明确归档（低风险）

```bash
# 1. 确保 docs/archive/ 结构清晰
# 当前 docs/archive/taili/ 已存在

# 2. 检查是否有对归档文档的不当引用
python tools/check_doc_redundancy.py

# 3. 在 docs/archive/README.md 中添加说明
cat > docs/archive/README.md << 'EOF'
# 历史归档文档

此目录包含历史参考文档，不作为当前系统的权威源。

- **用途**：追溯历史决策、比较方案、查找已废弃的实现
- **维护**：只读，不应被活跃文档引用
- **引用规则**：仅在必要时作为"历史证据"引用，不作为实现依据

活跃文档位于 `docs/` 根目录。
EOF
```

## 强制规则

**禁止行为**：

1. 在 `autotuner/` 或 `products/` 下创建 `output/`、`cache/` 等运行产物目录
2. 在 `docs/` 根目录引用 `docs/archive/` 中的文档作为实现依据
3. 在 `tools/` 外创建临时脚本（应统一放入 `tools/`）
4. 混合配置文件（SSH、产品合同、结构定义应在 `config/`）

**检查工具**：

```bash
# 运行完整检查
python tools/check_repository_structure.py
python tools/check_doc_redundancy.py
```

## 验收标准

1. `python tools/check_repository_structure.py` 通过
2. `python tools/check_doc_redundancy.py` 无 error 级别警告
3. `git status` 不显示临时文件或生成目录
4. 所有配置文件在 `config/`，所有工具在 `tools/`
5. `docs/archive/` 有明确说明文件，无活跃引用

## 执行时机

**立即执行**：阶段 1、2、4（配置、工具、归档说明）
**下次重构时执行**：阶段 3（仅在有产物目录冲突时）

## 相关原子问题

- X.3 文档冗余消除
- P2.2 Payload 完整性（涉及 payload 源映射路径）
