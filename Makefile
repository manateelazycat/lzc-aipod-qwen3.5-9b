.PHONY: all build test vet package release clean

all: test build

test:
	go test -v -race ./... -count=1

vet:
	go vet ./...

build:
	mkdir -p dist
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags '-s -w' -o dist/qwen35-9b-lpk-amd64 ./cmd/controller

package:
	python3 scripts/release.py --stage build_lpk

release:
	python3 scripts/release.py

clean:
	rm -rf dist/qwen35-9b-lpk* dist/*.lpk
