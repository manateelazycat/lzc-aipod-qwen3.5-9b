package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"qwen35-9b-deployer/internal/config"
	"qwen35-9b-deployer/internal/lpk"
)

func main() {
	port := flag.Int("port", envInt("QWEN35_9B_LPK_PORT", config.DefaultPort), "HTTP 端口")
	backendURL := flag.String("backend-url", os.Getenv("AIPOD_BACKEND_URL"), "AI Pod 主项目桥接地址")
	dataDir := flag.String("data-dir", os.Getenv("LZC_AGENT_DATA_DIR"), "部署记录保存目录")
	flag.Parse()

	server, err := lpk.NewServer(*backendURL, *dataDir)
	if err != nil {
		log.Fatal(err)
	}
	defer server.Close()

	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *port),
		Handler: server.Handler(),
	}
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
		<-ch
		_ = httpServer.Close()
	}()

	log.Printf("%s controller listening on %s", config.PublicServiceName, httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func envInt(name string, fallback int) int {
	if v := os.Getenv(name); v != "" {
		var parsed int
		if _, err := fmt.Sscanf(v, "%d", &parsed); err == nil && parsed > 0 {
			return parsed
		}
	}
	return fallback
}
