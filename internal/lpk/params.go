package lpk

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

type deployRequest struct {
	SerialNumber        string         `json:"serialNumber"`
	SerialNumbers       []string       `json:"serialNumbers"`
	Mode                string         `json:"mode"`
	Concurrency         *int           `json:"concurrency"`
	ContextLength       *int           `json:"contextLength"`
	MaxNumSeqs          *int           `json:"maxNumSeqs"`
	MaxModelLen         *int           `json:"maxModelLen"`
	ModelProvider       string         `json:"modelProvider"`
	Parameters          map[string]any `json:"parameters"`
	StopExisting        bool           `json:"stopExisting"`
	StopProtected       bool           `json:"stopProtected"`
	ProtectedProcessIDs []string       `json:"protectedProcessIds"`
}

type startServiceRequest struct {
	StopExisting        bool     `json:"stopExisting"`
	StopProtected       bool     `json:"stopProtected"`
	ProtectedProcessIDs []string `json:"protectedProcessIds"`
}

type deploymentSettings struct {
	ContextLength int
	Concurrency   int
	ModelProvider string
}

func validateContextLengthValue(v int) error {
	if v < config.MinContextLength || v > config.MaxContextLength {
		return fmt.Errorf("contextLength 必须在 %d 到 %d tokens 之间", config.MinContextLength, config.MaxContextLength)
	}
	if (v-config.MinContextLength)%config.ContextLengthStep != 0 {
		return fmt.Errorf("contextLength 必须以 %d 为步长对齐", config.ContextLengthStep)
	}
	return nil
}

func normalizeModelProvider(raw string) (string, error) {
	p := strings.ToLower(strings.TrimSpace(raw))
	if p == "" {
		return config.DefaultModelProvider, nil
	}
	if p != "modelscope" && p != "huggingface" {
		return "", errors.New("modelProvider 仅支持 modelscope 或 huggingface")
	}
	return p, nil
}

func parseStrictInt(v any, field string) (int, error) {
	switch typed := v.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) || math.Trunc(typed) != typed {
			return 0, fmt.Errorf("%s 必须为整数", field)
		}
		return int(typed), nil
	case json.Number:
		s := strings.TrimSpace(string(typed))
		n, err := strconv.Atoi(s)
		if err != nil {
			return 0, fmt.Errorf("%s 必须为整数", field)
		}
		return n, nil
	case string:
		s := strings.TrimSpace(typed)
		n, err := strconv.Atoi(s)
		if err != nil {
			return 0, fmt.Errorf("%s 必须为整数", field)
		}
		return n, nil
	default:
		return 0, fmt.Errorf("%s 类型无效，必须为整数", field)
	}
}

// ParseRequirementsParams applies the exact same descriptor parameter contract
// to GET /api/v1/deployments/requirements as POST /api/v1/deployments.
func ParseRequirementsParams(r *http.Request) (deploymentSettings, error) {
	q := r.URL.Query()
	for key := range q {
		switch key {
		case "serialNumber", "contextLength", "maxModelLen", "modelProvider", "concurrency":
		default:
			return deploymentSettings{}, fmt.Errorf("不支持的查询参数 %q", key)
		}
	}

	if vals := q["concurrency"]; len(vals) > 0 && strings.TrimSpace(vals[0]) != "" {
		c, err := parseStrictInt(vals[0], "concurrency")
		if err != nil {
			return deploymentSettings{}, err
		}
		if c != config.MaxConcurrency {
			return deploymentSettings{}, fmt.Errorf("concurrency 固定为 %d，不允许修改", config.MaxConcurrency)
		}
	}

	ctxLen := config.DefaultContextLength
	rawCtx := strings.TrimSpace(q.Get("contextLength"))
	rawMaxLen := strings.TrimSpace(q.Get("maxModelLen"))
	if rawCtx != "" && rawMaxLen != "" && rawCtx != rawMaxLen {
		return deploymentSettings{}, errors.New("contextLength 与 maxModelLen 参数不一致")
	}
	if rawCtx == "" {
		rawCtx = rawMaxLen
	}
	if rawCtx != "" {
		parsed, err := parseStrictInt(rawCtx, "contextLength")
		if err != nil {
			return deploymentSettings{}, err
		}
		ctxLen = parsed
	}
	if err := validateContextLengthValue(ctxLen); err != nil {
		return deploymentSettings{}, err
	}

	provider, err := normalizeModelProvider(q.Get("modelProvider"))
	if err != nil {
		return deploymentSettings{}, err
	}

	return deploymentSettings{
		ContextLength: ctxLen,
		Concurrency:   config.MaxConcurrency,
		ModelProvider: provider,
	}, nil
}

// NormalizeDeployRequest enforces the §6.1 parameter contract across create,
// redeploy, and stored-record replay paths.
func NormalizeDeployRequest(req deployRequest) (deploymentSettings, error) {
	if m := strings.TrimSpace(req.Mode); m != "" && m != "custom" {
		return deploymentSettings{}, fmt.Errorf("mode 仅支持 \"custom\"，不支持 %q", req.Mode)
	}

	if req.Concurrency != nil && *req.Concurrency != config.MaxConcurrency {
		return deploymentSettings{}, fmt.Errorf("concurrency 固定为 %d，不允许传入 %d", config.MaxConcurrency, *req.Concurrency)
	}
	if req.MaxNumSeqs != nil && *req.MaxNumSeqs != config.MaxConcurrency {
		return deploymentSettings{}, fmt.Errorf("maxNumSeqs 固定为 %d，不允许传入 %d", config.MaxConcurrency, *req.MaxNumSeqs)
	}

	for key := range req.Parameters {
		switch key {
		case "contextLength", "modelProvider":
		default:
			return deploymentSettings{}, fmt.Errorf("不支持的模型参数 %q", key)
		}
	}

	var topCtx *int
	if req.ContextLength != nil {
		v := *req.ContextLength
		topCtx = &v
	}
	if req.MaxModelLen != nil {
		if topCtx != nil && *topCtx != *req.MaxModelLen {
			return deploymentSettings{}, errors.New("contextLength 与 maxModelLen 不一致")
		}
		v := *req.MaxModelLen
		topCtx = &v
	}

	var nestedCtx *int
	if raw, ok := req.Parameters["contextLength"]; ok {
		parsed, err := parseStrictInt(raw, "parameters.contextLength")
		if err != nil {
			return deploymentSettings{}, err
		}
		nestedCtx = &parsed
	}
	if topCtx != nil && nestedCtx != nil && *topCtx != *nestedCtx {
		return deploymentSettings{}, errors.New("顶层 contextLength 与 parameters.contextLength 不一致")
	}

	ctxLen := config.DefaultContextLength
	if nestedCtx != nil {
		ctxLen = *nestedCtx
	} else if topCtx != nil {
		ctxLen = *topCtx
	}
	if err := validateContextLengthValue(ctxLen); err != nil {
		return deploymentSettings{}, err
	}

	topProvider := strings.TrimSpace(req.ModelProvider)
	nestedProvider := ""
	if raw, ok := req.Parameters["modelProvider"]; ok {
		str, ok := raw.(string)
		if !ok {
			return deploymentSettings{}, errors.New("parameters.modelProvider 必须为字符串")
		}
		nestedProvider = strings.TrimSpace(str)
	}
	if topProvider != "" && nestedProvider != "" && !strings.EqualFold(topProvider, nestedProvider) {
		return deploymentSettings{}, errors.New("顶层 modelProvider 与 parameters.modelProvider 不一致")
	}
	selectedProvider := nestedProvider
	if selectedProvider == "" {
		selectedProvider = topProvider
	}
	provider, err := normalizeModelProvider(selectedProvider)
	if err != nil {
		return deploymentSettings{}, err
	}

	return deploymentSettings{
		ContextLength: ctxLen,
		Concurrency:   config.MaxConcurrency,
		ModelProvider: provider,
	}, nil
}

// ParseDeleteDataParam strictly parses §4.2 persistentData deleteData query parameter.
// Omitted -> false; "true" -> true; "false" -> false; anything else or repeated -> error.
func ParseDeleteDataParam(r *http.Request) (bool, error) {
	vals, exists := r.URL.Query()["deleteData"]
	if !exists {
		return false, nil
	}
	if len(vals) != 1 {
		return false, errors.New("deleteData 参数不能重复传递")
	}
	switch vals[0] {
	case "true":
		return true, nil
	case "false":
		return false, nil
	default:
		return false, errors.New("deleteData 参数仅接受小写的 true 或 false")
	}
}
