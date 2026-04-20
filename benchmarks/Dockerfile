FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ gfortran make cmake ninja-build \
        autoconf automake libtool bison flex texinfo \
        python3 python3-pip python3-venv python3-dev \
        nodejs npm \
        git curl wget ca-certificates gnupg \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libffi-dev libgmp-dev libmpfr-dev libmpc-dev \
        linux-headers-generic \
        jq procps tini \
        gdb strace \
    && rm -rf /var/lib/apt/lists/*

# Pin Claude Code version for reproducibility (matches host: 2.1.105)
# yarn is needed by react's monorepo workspace (link: protocol).
RUN npm install -g @anthropic-ai/claude-code@2.1.105 yarn

# Non-interactive git identity (required for some build systems that invoke git commit)
RUN git config --global user.email "bench@local" \
 && git config --global user.name "bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory '*'

COPY benchmarks/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Bookmark MCP server (agent-managed memory probe).
# Stdio JSON-RPC, zero npm deps. Activated only when ENABLE_BOOKMARKS=1.
COPY mcp /opt/mcp
RUN chmod +r /opt/mcp/*.mjs

# /scratch is the per-run writable copy of the fixture. Make it world-writable
# (like /tmp) so containers launched with --user <host-uid> can populate it.
RUN mkdir -p /scratch && chmod 1777 /scratch

# /tmp/.claude is Claude Code's config/state directory when HOME=/tmp. It
# needs to exist with permissive mode BEFORE Docker bind-mounts the
# credentials file into it; otherwise Docker creates the parent as root and
# the UID-1000 process can't write session-env/ there, which causes every
# Bash tool call to return EACCES on its first line.
RUN mkdir -p /tmp/.claude && chmod 1777 /tmp/.claude

WORKDIR /scratch
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
