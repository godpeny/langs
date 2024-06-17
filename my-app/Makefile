SHELL := /bin/bash
REGISTRY ?= idock.daumkakao.io/peny_official
VERSION ?= v0.0.2
OPENAI_API_KEY ?= ""

.PHONY: build
build:
	@echo "update dependency lock"
	poetry lock
	@echo "build"
	docker build --no-cache --build-arg $(OPENAI_API_KEY) --platform linux/amd64 -t $(REGISTRY)/test-lang:$(VERSION) -f Dockerfile .