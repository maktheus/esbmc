FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends yosys python3 python3-pip python3-z3 time ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /w
