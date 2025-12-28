# Sticker Selector

<p align="center">
  <img src="./assets/StickerSelector-Banner.png" alt="Logo"  width="500" />
</p>

## 介绍

**AI 很会说话，但大多数时候并不会用表情包。**

**StickerSelector 就是为了解决这个问题而存在的。**

它不会靠固定规则去凑关键词，而是用语义模型去理解一句话真正想表达的感觉，再从已有的表情包中选出那一张——现在用，刚刚好。

> 这是很多 AI 聊天应用都会卡住的一道坎，越过去，聊天会立刻变得更像人。

StickerSelector 本身非常轻量，即使运行在 `1 核心 / 1GB 内存 / 3Mbps` 的小型服务器上也可以正常使用。  

如果条件允许，你也可以在自己的 PC 上部署更高性能的版本，配合各种聊天应用接入工具，或直接搭配 QQSafeChat，打造一个真正“拟真”的 AI 聊天体验。


## 1. 环境准备

- Python 3.10+（推荐 3.10.x）
- Windows/本地运行均可

安装依赖：

```bash
pip install -r requirements.runtime.txt
```

## 2. 启动服务

```bash
uvicorn sticker_service.app:app --reload
```

启动后访问：

- 前台试用：`http://127.0.0.1:8000/try`
- 管理后台：`http://127.0.0.1:8000/admin`

## 3. 从 0 开始的使用流程

1. 进入「系列管理」新建系列（例如：猫猫 / memes）
2. 进入「批量上传」选择系列并上传图片
3. 上传完成后跳转到「批量打 Tag」，为每张表情包补充 tags
4. 进入「表情包管理」检查、批量启用/禁用或移动系列
5. 在「试用」页面输入 tags 进行检索验证
6. 需要迁移/备份时，使用「系列管理」的导入/导出功能

## 4. 目录结构

```
sticker_service/
  app.py            # FastAPI 入口
  db.py             # SQLite 数据操作
  templates/        # 页面模板
  static/           # 前端资源
  data_runtime/     # 运行时数据（DB、日志、表情包文件）
docs/               # 文档
```

## 5. 模型下载说明

首次启动项目需在 Web 页面中选择模型并下载。
项目会在非首次启动时加载模型，用于语义检索：

- 如果网络正常，首次启动会自动下载。
- 如果网络受限，请配置代理或提前下载模型缓存。
- 若出现 `SSLError`/`EOF` 等错误，多为网络或代理问题，建议检查 TLS/代理设置或使用镜像源。
