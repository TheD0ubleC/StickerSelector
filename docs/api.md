# API 调用说明

本文档面向接口调用方，包含参数、返回结构与示例。默认基地址为：

```
http://127.0.0.1:8000
```

## 1. 约定与说明

- 所有接口均为本地服务，不含鉴权。
- JSON 接口需设置 `Content-Type: application/json`。
- 时间与日志为服务端本地时区。

### 1.1 匹配度（match_rate / fit_rate）

`match_rate` 为**语义相似度分数**（由向量余弦相似度映射到 0~100）：

```
match_rate = clamp(cosine_similarity, 0, 1) * 100
```

`fit_rate` 为**TopK 内 softmax 分布**（与 k 强相关）：

```
fit_rate = softmax(raw_scores * fit_scale) * 100
```

### 1.2 排序（order）

`order` 可选：

- `desc`：按匹配度从大到小
- `asc`：按匹配度从小到大
- `none`：不重新排序（保留向量检索顺序）

## 2. 核心检索接口

### 2.1 POST /api/select

按 tags 检索最匹配的表情包。

请求体：

```json
{
  "tags": "开心 兴奋",
  "k": 6,
  "series": "default",
  "order": "desc"
}
```

说明：

- `tags` 可以是字符串（空格分隔）或数组 `["开心","兴奋"]`
- `k` 为返回条数（1~50）
- `series` 为空时表示全系列

响应示例：

```json
{
  "items": [
    {
      "id": 123,
      "series": "default",
      "url": "/stickers/default/xxx.png",
      "raw": 0.312,
      "fit_rate": 66.7,
      "match_rate": 66.7,
      "tags": ["开心", "兴奋"]
    }
  ],
  "meta": {
    "k": 6,
    "series": "",
    "order": "desc",
    "match_mode": "embed_cosine",
    "fit_mode": "softmax_topk",
    "fit_scale": 55.0
  }
}
```

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/select" \
  -H "Content-Type: application/json" \
  -d '{"tags":"开心 兴奋","k":6,"series":"","order":"desc"}'
```

## 3. 系列相关接口

### 3.1 GET /api/series

获取所有系列。

响应示例：

```json
{
  "items": [{ "id": 1, "name": "default", "enabled": true }]
}
```

### 3.2 POST /api/series/exists

判断系列是否存在。

请求体：

```json
{ "name": "default" }
```

响应示例：

```json
{ "exists": true, "id": 1, "name": "default" }
```

### 3.3 POST /api/series/count

获取系列下的表情包数量。

请求体（二选一）：

```json
{ "series_id": 1 }
```

```json
{ "name": "default" }
```

响应示例（单个）：

```json
{ "id": 1, "name": "default", "count": 128 }
```

如果不传参数，将返回全部系列数量：

```json
{
  "items": [{ "id": 1, "name": "default", "count": 128 }],
  "meta": { "total_series": 1 }
}
```

## 4. 待打 Tag / 批次接口

### 4.1 GET /api/stats

获取当前待打 Tag 总数。

响应示例：

```json
{ "pending": 42 }
```

### 4.2 GET /api/pending

获取待打 Tag 列表。

查询参数：

- `batch_id`：可选
- `limit`：默认 2000

响应示例：

```json
{
  "items": [
    {
      "id": 1,
      "series": "default",
      "batch_id": 10,
      "filename": "xxx.png",
      "url": "/stickers/default/xxx.png",
      "enabled": true,
      "tags": []
    }
  ],
  "meta": { "batch_id": 10, "count": 1 }
}
```

### 4.3 POST /api/stickers/bulk_update

批量更新 tag 与启用状态（批次打 Tag 用）。

请求体：

```json
{
  "items": [{ "id": 1, "tags": ["开心", "兴奋"], "enabled": true }]
}
```

响应示例：

```json
{ "ok": true, "updated": 1, "cleaned_batches": [] }
```

### 4.4 POST /api/batch/delete

删除批次及其下所有表情包。

请求体：

```json
{ "batch_id": 10 }
```

### 4.5 POST /api/batch/cleanup_if_done

如果批次已全部打完 Tag，则自动脱离批次并清理。

请求体：

```json
{ "batch_id": 10 }
```

## 5. 表情包管理接口

### 5.1 POST /api/stickers/bulk_action

批量操作（启用/禁用/移动/删除）。

请求体示例：

```json
{ "action": "enable", "ids": [1, 2, 3] }
```

```json
{ "action": "disable", "ids": [1, 2, 3] }
```

```json
{ "action": "move", "ids": [1, 2, 3], "series_id": 2 }
```

```json
{ "action": "delete", "ids": [1, 2, 3] }
```

### 5.2 POST /api/sticker/delete

删除单张表情包（包含图片文件）。

请求体：

```json
{ "id": 123 }
```

## 6. 系列包导入/导出（管理接口）

### 6.1 GET /admin/series/export?series_id=ID

导出单个系列 ZIP 包。

### 6.2 POST /admin/series/export_bulk

批量导出系列 ZIP（内部会将多个系列包打成一个 ZIP）。

请求体：

```json
{ "series_ids": [1, 2, 3] }
```

### 6.3 POST /admin/series/import

导入系列 ZIP 包（multipart/form-data）。

表单字段：

- `file`：ZIP 文件
- `mode`：`ask` 或 `merge`

当重名且 `mode=ask` 时返回 409：

```json
{ "ok": false, "conflict": true, "series_name": "default", "series_id": 1 }
```

### 6.4 POST /admin/series/delete_bulk

批量删除系列（会同时删除系列下所有表情包文件）。

请求体：

```json
{ "series_ids": [1, 2, 3] }
```
