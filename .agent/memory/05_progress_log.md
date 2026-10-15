# 05 — 进度日志

> 每人完工一个阶段就追加一条（最新在最上）。格式：
> `## YYYY-MM-DD 名字 — 做了什么｜产出路径｜遗留问题`

## 2026-10-15 Daniel — hdk 目录归属确认并迁移

- `hdk` = Caron（误用真名缩写建目录），其 repo clone + conda env 已迁移至 `/home/caron@Caron/`，附 MIGRATION_NOTE.md（prefix 环境重建提醒、`src/encoding/` 代码落点约定、remote 已改代理前缀）

## 2026-10-15 Daniel — 成员目录实名制

- `/home/` 成员目录改名 `<缩写>@<群昵称>`（fish@🐠、ash@Prick1yAsh、panda@空腹panda、xw@谢武、arc@Arc、caron@Caron），新增 `/home/README.md` 总览表
- GitHub 已打通（deploy key + ssh.github.com:443）：Mac 可 push；服务器仓库 origin 配 gh-proxy.com 前缀只读拉取
- 遗留：`hdk` 目录归属未登记；`daniel` 目录群昵称待确认

## 2026-10-15 Daniel — 收录环境配置指南 + 服务器加速落地

- 《云端GPU服务器环境配置指南》（[gpu-rsi-plugin](https://github.com/miracle-techlink/gpu-rsi-plugin)）精华版收录为 `skills/env-setup/SKILL.md`
- 服务器落地加速项：全局安装 aria2c + uv；`HF_ENDPOINT`/`UV_INDEX_URL`/`UV_CACHE_DIR`(数据盘) 写入 `/root/.bashrc`
- 定位网络问题：服务器直连 GitHub git 端口 TLS 不稳 → clone 用 ghproxy 前缀（push 不可走代理）
- 遗留：BBR/sysctl 容器内无权限跳过；GitHub push 仍需凭据

## 2026-10-15 Daniel — 项目初始化

- 服务器环境：安装 Miniconda（`/opt/miniconda3`，清华镜像），conda 环境与数据集规划到数据盘 `/root/gpufree-data`
- 建立 `/home/` 下 7 个成员目录（fish / ash / panda / xw / arc / caron / daniel）+ `/home/shared`
- 仓库初始化：README、AGENTS.md、`.agent/`（memory×6 + skills×7）、`src/` 模块骨架
- **遗留问题**：
  1. ⚠️ 服务器 GPU 当前未挂载进容器（`nvidia-smi -L` 无输出）——训练前需在 gpufree 控制台确认
  2. 数据集尚未下载到 `/root/gpufree-data/datasets/`
  3. 《训练环境配置优化》文档链接待补充进 `04_resources.md`
