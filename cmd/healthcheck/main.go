package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Minimal static ARM64 healthcheck binary for FROM scratch llama-server images.
func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: healthcheck <url> <expected-substring>")
		os.Exit(2)
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 256*1024))
	if err != nil || resp.StatusCode != http.StatusOK || !strings.Contains(string(body), os.Args[2]) {
		os.Exit(1)
	}
}
