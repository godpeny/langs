SHELL := /bin/bash
REGISTRY ?= idock.daumkakao.io/peny_official
VERSION ?= v0.0.1

.PHONY: build
build:
	@echo "build"
	docker build --build-arg OPENAI_API_KEY --platform linux/amd64 -t $(REGISTRY)/test-lang:$(VERSION) -f deploy/Dockerfile .