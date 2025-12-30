# Sticker Selector

<p align="center">
  <img src="./assets/StickerSelector-Banner.png" alt="Logo"  width="500" />
</p>

<p align="center">
  <img src="./assets/StickerSelector.png" height="28"/>
  <img src="https://img.shields.io/badge/platform-Windows-blue?style=for-the-badge" alt="Windows Support" />
  <img src="https://img.shields.io/badge/platform-macOS-lightgrey?style=for-the-badge" alt="macOS Support" />
  <img src="https://img.shields.io/badge/platform-Linux-green?style=for-the-badge" alt="Linux Support" />

</p>

## 介绍

**AI 很会说话，但大多数时候并不会用表情包。**

**StickerSelector 就是为了解决这个问题而存在的。**

它不会靠固定规则去凑关键词，而是用语义模型去理解一句话真正想表达的感觉，再从已有的表情包中选出那一张——现在用，刚刚好。

> 这是很多 AI 聊天应用都会卡住的一道坎，越过去，聊天会立刻变得更像人。

StickerSelector 本身非常轻量，即使运行在 `1 核心 / 1GB 内存 / 3Mbps` 的小型服务器上也可以正常使用。

如果条件允许，你也可以在自己的 PC 上部署更高性能的版本，配合各种聊天应用接入工具，或直接搭配 QQSafeChat，打造一个真正“拟真”的 AI 聊天体验。

## 演示

| 描述                                   | 演示                                      |
| -------------------------------------- | ----------------------------------------- |
| 使用 QQSafeChat 并接入 StickerSelector | <img src="./assets/a.png" height="300" /> |
| WebUI 试用                             | <img src="./assets/b.png" height="300"/>  |
| 模型选择页                             | <img src="./assets/c.png" height="300"/>  |

## 0. 克隆本项目

运行该命令进行克隆：

```bash
git clone https://github.com/TheD0ubleC/StickerSelector.git
```

进入：

```bash
cd .\StickerSelector
```

## 1. 环境准备

- Python 3.10+（推荐 3.10.x）
- Windows / Linux / macOS / 本地或服务器均可

> 推荐使用虚拟环境运行，避免依赖冲突（尤其是服务器环境）。

---

### （可选但推荐）使用虚拟环境

创建虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境（请根据平台选择命令）：

```bash
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.runtime.txt
```

## 2. 启动服务

```bash
# 如果你不需要直接公开在局域网或互联网 请使用
uvicorn sticker_service.app:app --port 8000
# 如果你需要公开在局域网或互联网 请使用
uvicorn sticker_service.app:app --host 0.0.0.0 --port 8000
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
- 若出现 `SSLError`/`EOF` 等错误，多为网络或代理问题，请确保可以正常访问 Hugging Face 并建议检查 TLS/代理设置或使用镜像源。
