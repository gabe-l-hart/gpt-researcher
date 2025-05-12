FROM registry.access.redhat.com/ubi9/python-311:9.5-1743079393 as base
COPY --from=ghcr.io/astral-sh/uv:0.6.8 /uv /uvx /bin/
WORKDIR /app

## Build ##
FROM base as build
USER 0:0

# Install dependencies for the agent wrapper
ENV UV_LINK_MODE=copy
COPY beeai-agent/README.md .
COPY beeai-agent/pyproject.toml .
RUN mkdir granite_researcher \
    && touch granite_researcher/__init__.py
RUN uv pip install -e .

# Install the local version of gpt-researcher
COPY pyproject.toml gpt-researcher/pyproject.toml
COPY README.md gpt-researcher/README.md
RUN mkdir -p gpt-researcher/gpt_researcher \
    && touch gpt-researcher/gpt_researcher/__init__.py
RUN uv pip install -e gpt-researcher
RUN uv pip install langchain_ollama

# Add the actual code and re-sync the venv with uv
COPY gpt_researcher gpt-researcher/gpt_researcher
COPY beeai-agent/granite_researcher granite_researcher
RUN uv sync --no-cache --locked --link-mode copy

# Open up permissions for things that will be copied to release
RUN chmod ugo+r -R /app
RUN chmod ugo+r -R /opt/app-root
RUN chmod ugo+w -R /opt/app-root/src

## Release ##
FROM base as release
USER 1001:1001
COPY --from=build /app /app
COPY --from=build /opt/app-root/ /opt/app-root/
ENV UV_OFFLINE="1"
ENV PATH="/opt/app-root/bin:$PATH"

# Add the necessary env for beeai
ARG RELEASE_VERSION="main"
ENV PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

CMD ["uv", "run", "--no-sync", "server"]
