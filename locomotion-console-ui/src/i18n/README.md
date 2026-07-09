# 前端汉化维护说明

本目录只放显示层文案和格式化工具，不放业务逻辑。

规则：

- 所有用户可见中文优先放在 `zh-CN.ts`，组件里通过 `t.xxx` 引用。
- 常见后端状态、错误和提示的展示兜底放在 `format.ts`。
- 文件保持 UTF-8 编码；不要用 PowerShell 的默认编码重写中文文件。
- 不要把内部沟通中的非工程称呼写入产品文案或 LLM prompt；界面和提示词使用工程术语：LLM、框架、ConfigSet、Adapter、诊断、守门。
- 新增页面文案后运行：

```bash
npm run check:i18n
npm run build
```

`check:i18n` 会检查：

- 非 i18n 的 `.tsx` 文件是否出现直接英文 JSX 文本；
- 系统代码中是否误写入不应进入产品的非工程称呼。
