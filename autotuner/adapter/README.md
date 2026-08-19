# 产品适配层

`autotuner/adapter/` 提供产品无关的 URDF 校验、执行器派生、数值缩放、
配置副本物化和部署计划算法。机器人资产、经验锚点、字段映射、参考生成器
与远程目标均来自 `config/products/<product>.yaml` 和对应产品插件。

## 边界

- 系统层不导入 `products.<id>`，只按产品合同加载声明的插件入口。
- 通用算法不会猜测自由度、关节名称或训练文件路径。
- 旧文件级部署只保留兼容能力，默认关闭；版本化 product payload 是权威交付路径。
- 适配预览只写临时副本，不连接远端，也不启动训练。

## 命令

```bash
python -m autotuner.adapter [product-id] [--composition ID]
                            [--save ID|--load ID|--list|--list-products]
                            [--work-dir DIR]
```

注册表中只有一个产品时可以省略 `product-id`。存在多个产品时必须显式指定，
系统不会回退到任意默认机器人。

## 验证

```bash
python -m pytest tests/autotuner/adapter tests/autotuner/product \
  tests/autotuner/framework_library --basetemp .pytest-tmp-adapter -q
```
