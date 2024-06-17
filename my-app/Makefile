SHELL := /bin/bash
REGISTRY ?= idock.daumkakao.io/peny_official
VERSION ?= v0.0.3
OPENAI_API_KEY ?= "" # must be set!

.PHONY: build
build:
	@echo "update dependency lock"
	poetry lock
	@echo "build"
	docker build --no-cache --build-arg="OPENAI_API_KEY=$(OPENAI_API_KEY)" --platform linux/amd64 -t $(REGISTRY)/test-lang:$(VERSION) -f Dockerfile .

.phony: run
run:
	export OPENAI_API_KEY=$(OPENAI_API_KEY); poetry run langchain serve --port=8080

.phony: docker-run
docker-run:
	docker run -p 8080:8080 $(REGISTRY)/test-lang:$(VERSION)