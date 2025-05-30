FROM quay.io/jupyter/minimal-notebook:python-3.12.10

COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

COPY pyproject.toml .
RUN uv sync
RUN rm pyproject.toml

RUN fix-permissions "${CONDA_DIR}" && \
  fix-permissions "/home/${NB_USER}"
