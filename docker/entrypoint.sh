#!/bin/bash
# 清除代理环境变量
export http_proxy=
export https_proxy=

# 执行 Docker CMD 中的命令
exec "$@"

