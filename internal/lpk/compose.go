package lpk

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

func BuildCacheDirectoryName(target string, rt RuntimeTargetManifest, matchedCache *RuntimeCacheEntry, settings deploymentSettings) string {
	if matchedCache != nil && validSHA256Hex(matchedCache.Fingerprint) {
		return strings.ToLower(matchedCache.Fingerprint)
	}
	idPrefix := "unpinned"
	if len(rt.RuntimeImageSHA256) >= 12 {
		idPrefix = strings.ToLower(rt.RuntimeImageSHA256[:12])
	} else if len(rt.SHA256) >= 12 {
		idPrefix = strings.ToLower(rt.SHA256[:12])
	}
	return fmt.Sprintf("cold-%s-%s-%d-%d", target, idPrefix, settings.ContextLength, settings.Concurrency)
}

func BuildInferenceCompose(target string, rt RuntimeTargetManifest, groups []ModelGroupEntry, modelRoot, cacheDir string, settings deploymentSettings, budget MemoryBudget) ([]byte, error) {
	modelRoot = strings.TrimSpace(modelRoot)
	if modelRoot == "" || modelRoot == "/" || !filepath.IsAbs(modelRoot) || strings.ContainsAny(modelRoot, "\r\n\x00:") {
		return nil, errors.New("算力舱模型存储根目录无效")
	}
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" || cacheDir == "." || cacheDir == ".." || filepath.Base(cacheDir) != cacheDir || strings.ContainsAny(cacheDir, "\\/\r\n\x00:") {
		return nil, errors.New("运行缓存目录名称无效")
	}
	if strings.TrimSpace(rt.Image) == "" {
		return nil, fmt.Errorf("目标 %s 缺少运行镜像名称", target)
	}
	if err := validateContextLengthValue(settings.ContextLength); err != nil {
		return nil, err
	}

	lp := rt.LaunchProfile
	engine := strings.ToLower(strings.TrimSpace(lp.Engine))
	if engine == "" {
		engine = config.EngineLlamaCpp
	}
	if engine != config.EngineLlamaCpp {
		return nil, fmt.Errorf("不支持的推理引擎 %q", lp.Engine)
	}

	maxSeqs := lp.MaxNumSeqs
	if maxSeqs <= 0 {
		maxSeqs = config.MaxConcurrency
	}

	args, err := buildLlamaCppArgs(lp, groups, settings.ContextLength, maxSeqs)
	if err != nil {
		return nil, err
	}

	var out strings.Builder
	out.WriteString("services:\n")
	fmt.Fprintf(&out, "  %s:\n", config.DefaultServiceName)
	fmt.Fprintf(&out, "    image: %s\n", rt.Image)
	out.WriteString("    pull_policy: never\n")
	out.WriteString("    restart: unless-stopped\n")
	out.WriteString("    network_mode: host\n")
	out.WriteString("    init: true\n")
	out.WriteString("    gpus: all\n")
	out.WriteString("    ipc: host\n")
	out.WriteString("    shm_size: 4gb\n")
	out.WriteString("    ulimits:\n      memlock: -1\n      stack: 67108864\n")

	cmdJSON, _ := json.Marshal(args)
	fmt.Fprintf(&out, "    entrypoint: %s\n", cmdJSON)

	out.WriteString("    volumes:\n")
	for _, g := range groups {
		targetPath := g.MountTarget
		if targetPath == "" {
			targetPath = "/" + g.Role
		}
		mountSpec, _ := json.Marshal(filepath.Join(modelRoot, g.Role) + ":" + targetPath + ":ro")
		fmt.Fprintf(&out, "      - %s\n", mountSpec)
	}
	fmt.Fprintf(&out, "      - ${LZC_AGENT_CACHE_DIR}/%s/cache:/root/.cache\n", cacheDir)
	fmt.Fprintf(&out, "      - ${LZC_AGENT_CACHE_DIR}/%s/triton:/root/.triton\n", cacheDir)
	fmt.Fprintf(&out, "      - ${LZC_AGENT_CACHE_DIR}/%s/nv:/root/.nv\n", cacheDir)

	out.WriteString("    environment:\n")
	fmt.Fprintf(&out, "      AIPOD_METRICS_PORT: %q\n", strconv.Itoa(config.DefaultAPIPort))
	fmt.Fprintf(&out, "      PORT: %q\n", strconv.Itoa(config.DefaultAPIPort))
	out.WriteString("      CUDA_CACHE_PATH: /root/.nv\n")
	if _, hasOMP := lp.ExtraEnv["OMP_NUM_THREADS"]; !hasOMP {
		if target == config.TargetOrin {
			out.WriteString("      OMP_NUM_THREADS: \"4\"\n")
		} else {
			out.WriteString("      OMP_NUM_THREADS: \"8\"\n")
		}
	}
	if len(lp.ExtraEnv) > 0 {
		keys := make([]string, 0, len(lp.ExtraEnv))
		for k := range lp.ExtraEnv {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&out, "      %s: %q\n", k, lp.ExtraEnv[k])
		}
	}

	hcArgs := []string{"CMD", "/runtime/healthcheck", fmt.Sprintf("http://127.0.0.1:%d/v1/models", config.DefaultAPIPort), config.PublicModelID}
	hcJSON, _ := json.Marshal(hcArgs)
	fmt.Fprintf(&out, "    healthcheck:\n      test: %s\n      interval: 10s\n      timeout: 10s\n      retries: 18\n      start_period: 3m\n", hcJSON)

	return []byte(out.String()), nil
}

func buildLlamaCppArgs(lp RuntimeLaunchProfile, groups []ModelGroupEntry, contextLength, maxSeqs int) ([]string, error) {
	modelFile := strings.TrimSpace(lp.ModelFile)
	mmprojFile := strings.TrimSpace(lp.MMProjFile)
	draftFile := strings.TrimSpace(lp.DraftFile)
	draftBase := "/model/"
	for _, g := range groups {
		if g.Role == "draft" {
			mount := g.MountTarget
			if mount == "" {
				mount = "/draft"
			}
			draftBase = strings.TrimRight(mount, "/") + "/"
		}
		for _, f := range g.Files {
			lower := strings.ToLower(f.Path)
			if !strings.HasSuffix(lower, ".gguf") {
				continue
			}
			base := filepath.Base(f.Path)
			if strings.Contains(lower, "mmproj") {
				if mmprojFile == "" {
					mmprojFile = base
				}
			} else if g.Role == "draft" {
				if draftFile == "" {
					draftFile = base
				}
			} else if modelFile == "" {
				modelFile = base
			}
		}
	}
	if modelFile == "" {
		return nil, errors.New("llama.cpp 运行配置缺少主 GGUF 模型文件名")
	}
	args := []string{
		"/runtime/llama-server",
		"-m", "/model/" + modelFile,
		"--alias", config.PublicModelID,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(config.DefaultAPIPort),
		"-c", strconv.Itoa(contextLength),
		"-np", strconv.Itoa(maxSeqs),
		"-ngl", "999",
		"-fa", "on",
		"--kv-unified",
		"--jinja",
		"--metrics",
	}
	if lp.ReasoningFormat != "" {
		args = append(args, "--reasoning-format", lp.ReasoningFormat)
	}
	if mmprojFile != "" {
		args = append(args, "--mmproj", "/model/"+mmprojFile)
	}
	if draftFile != "" {
		args = append(args, "--model-draft", draftBase+draftFile)
	}
	if lp.SpeculativeMethod != "" && lp.NumSpeculativeTokens > 0 {
		args = append(args,
			"--spec-type", lp.SpeculativeMethod,
			"--spec-draft-n-max", strconv.Itoa(lp.NumSpeculativeTokens),
		)
	}
	if lp.KVCacheDtype != "" && !strings.EqualFold(lp.KVCacheDtype, "auto") && !hasSliceFlag(lp.ExtraArgs, "-ctk") {
		args = append(args, "-ctk", lp.KVCacheDtype, "-ctv", lp.KVCacheDtype)
	}
	args = append(args, lp.ExtraArgs...)
	return args, nil
}

func hasSliceFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}
