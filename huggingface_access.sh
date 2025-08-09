#!/bin/bash

# 重启服务
 echo "正在执行 restart.sh..."
 bash restart.sh

# 检查特定端口是否在监听
 echo -e "\n检查端口 6006 和 7890-7899 是否在监听..."
 lsof -i -P -n | grep LISTEN | grep -E ':6006|:789[0-9]'

# 启用代理
 echo -e "\n启用代理..."
 source proxy_on.sh

# 下载Hugging Face网页
 echo -e "\n正在下载Hugging Face网页..."
 wget https://huggingface.co/
 echo "下载完成，文件保存在当前目录下的 index.html"