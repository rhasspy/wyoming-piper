FROM python:3 AS builder

WORKDIR /src

COPY . .

RUN ./script/setup && ./script/package

FROM python:3

ARG PIPER_VERSION=v1.2.0
ARG PIPER_ARCH=amd64
RUN curl -L -s "https://github.com/rhasspy/piper/releases/download/$PIPER_VERSION/piper_$PIPER_ARCH.tar.gz" | tar -zxvf - -C /usr/share

RUN --mount=type=bind,from=builder,target=/mnt/builder pip3 install /mnt/builder/src/dist/*.whl

WORKDIR /src
COPY docker-entrypoint.sh .

EXPOSE 10300/tcp
VOLUME /data
LABEL org.opencontainers.image.source=https://github.com/rhasspy/wyoming-piper

ENTRYPOINT ["./docker-entrypoint.sh"]
