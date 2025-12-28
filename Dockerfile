# Programmatic Tool Calling Sandbox Image
# 安全隔离的代码执行环境

FROM python:3.11-slim

# 安全设置：创建非 root 用户
RUN useradd -m -s /bin/bash -u 1000 sandbox && \
    mkdir -p /workspace /sandbox && \
    chown -R sandbox:sandbox /workspace /sandbox

# 安装基础依赖（可根据需要扩展）
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    python-dateutil \
    && rm -rf /root/.cache

# 安全加固
RUN chmod 755 /workspace && \
    chmod 755 /sandbox

# 切换到非 root 用户
USER sandbox

# 设置工作目录
WORKDIR /workspace

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('healthy')" || exit 1

# 默认命令
CMD ["python"]
