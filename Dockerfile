FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN python -m pip install --no-cache-dir .

ENTRYPOINT ["ng-opt"]
CMD ["demo", "--output-dir", "/artifacts/demo"]
