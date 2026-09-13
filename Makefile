BINARY   := notion-proxy-search-mcp
PKG      := ./cmd/notion-proxy-search-mcp
VERSION  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
LDFLAGS  := -s -w -X main.Version=$(VERSION)

# Hugging Face checkpoint the default weight blob is built from.
MODEL_REPO     ?= intfloat/multilingual-e5-small
MODEL_REVISION ?= 614241f622f53c4eeff9890bdc4f31cfecc418b3
MODEL_DIR      ?= build/model-src
MODEL_OUT      ?= build/model.npse

.PHONY: all build arm64 test vet bench bench-model fmt model clean

all: vet test build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o build/$(BINARY) $(PKG)

arm64:
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags "$(LDFLAGS)" -o build/$(BINARY)-arm64 $(PKG)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -l -w .

# Kernel throughput is the thing to re-measure on a new board.
bench:
	go test ./internal/embed/ -run '^$$' -bench 'DotI8|Linear|Forward' -benchtime 200x

# Real forward-pass latency by sequence length; needs NPS_TEST_MODEL.
bench-model:
	go test ./internal/embed/ -run '^$$' -bench RealModelForward -benchtime 10x

# Download the checkpoint and quantize it into a weight blob.
model: build
	mkdir -p $(MODEL_DIR)
	for f in config.json tokenizer.json model.safetensors; do \
	  curl -fsSL -o $(MODEL_DIR)/$$f \
	    "https://huggingface.co/$(MODEL_REPO)/resolve/$(MODEL_REVISION)/$$f"; \
	done
	./build/$(BINARY) convert -model-dir $(MODEL_DIR) -out $(MODEL_OUT) -name $(MODEL_REPO)

clean:
	rm -rf build
