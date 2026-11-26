package lpk

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

const (
	piAgentConfigTarget    = "~/.pi/agent/models.json"
	piAgentConfigDirectory = "pi-agent"
)

// BuildPiAgentConfig returns only the public OpenAI-compatible connection
// settings for Pi Agent. Runtime image names, quantization formats, speculative
// decoding flags, and internal paths must never appear here.
func BuildPiAgentConfig(record DeploymentRecord) map[string]any {
	contextLength := record.ContextLength
	if contextLength < config.MinContextLength {
		contextLength = config.DefaultContextLength
	}
	maxTokens := 16384
	if contextLength/2 < maxTokens {
		maxTokens = contextLength / 2
	}
	baseURL := strings.TrimRight(strings.TrimSpace(record.APIURL), "/")
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	return map[string]any{
		"providers": map[string]any{
			"qwen35-9b-defiant-fable": map[string]any{
				"baseUrl": baseURL,
				"api":     "openai-completions",
				"apiKey":  config.DefaultAPIKey,
				"compat": map[string]any{
					"supportsStore":         false,
					"supportsDeveloperRole": false,
					"maxTokensField":        "max_tokens",
				},
				"models": []any{
					map[string]any{
						"id":            config.PublicModelID,
						"name":          config.PublicServiceName,
						"reasoning":     false,
						"input":         []string{"text", "image"},
						"contextWindow": contextLength,
						"maxTokens":     maxTokens,
						"cost": map[string]any{
							"input":      0,
							"output":     0,
							"cacheRead":  0,
							"cacheWrite": 0,
						},
					},
				},
			},
		},
	}
}

func (s *Server) persistPiAgentConfig(record DeploymentRecord) error {
	if strings.TrimSpace(record.ID) == "" || strings.TrimSpace(record.APIURL) == "" {
		return errors.New("Pi Agent 配置缺少部署 ID 或 API 地址")
	}
	if filepath.Base(record.ID) != record.ID || strings.ContainsAny(record.ID, "/\\\x00") {
		return errors.New("部署 ID 无效")
	}
	dir := filepath.Join(filepath.Dir(s.dataPath), piAgentConfigDirectory)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("创建 Pi Agent 配置目录失败: %w", err)
	}
	contents, err := json.MarshalIndent(BuildPiAgentConfig(record), "", "  ")
	if err != nil {
		return fmt.Errorf("编码 Pi Agent 配置失败: %w", err)
	}
	contents = append(contents, '\n')
	tmp, err := os.CreateTemp(dir, ".models-*.json")
	if err != nil {
		return fmt.Errorf("创建 Pi Agent 临时文件失败: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(contents); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	target := filepath.Join(dir, record.ID+"-models.json")
	return os.Rename(tmpName, target)
}
