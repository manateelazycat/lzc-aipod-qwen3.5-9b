package lpk

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"qwen35-9b-deployer/internal/config"
	"qwen35-9b-deployer/manifests"
)

type ModelFileEntry struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type ModelSourcePin struct {
	Repository string           `json:"repository"`
	Revision   string           `json:"revision"`
	Files      []ModelFileEntry `json:"files,omitempty"`
}

type ModelGroupEntry struct {
	ID             string                    `json:"id"`
	Targets        []string                  `json:"targets"`
	Role           string                    `json:"role"`
	Destination    string                    `json:"destination"`
	MountTarget    string                    `json:"mountTarget"`
	Name           string                    `json:"name"`
	EstimatedBytes int64                     `json:"estimatedBytes"`
	Sources        map[string]ModelSourcePin `json:"sources"`
	Files          []ModelFileEntry          `json:"files"`
}

type ModelManifestDoc struct {
	Schema   string            `json:"schema"`
	Status   string            `json:"status"`
	ModelID  string            `json:"modelId"`
	Models   []ModelGroupEntry `json:"models"`
}

type RuntimeCompatibility struct {
	Architecture    string   `json:"architecture"`
	GPUArchitecture string   `json:"gpuArchitecture"`
	CUDAVersion     string   `json:"cudaVersion"`
	L4TPrefixes     []string `json:"l4tPrefixes"`
}

type RuntimeLaunchProfile struct {
	Engine               string            `json:"engine,omitempty"`
	Quantization         string            `json:"quantization"`
	KVCacheDtype         string            `json:"kvCacheDtype"`
	LinearBackend        string            `json:"linearBackend,omitempty"`
	MaxNumSeqs           int               `json:"maxNumSeqs"`
	MaxNumBatchedTokens  int               `json:"maxNumBatchedTokens"`
	EnforceEager         bool              `json:"enforceEager"`
	SpeculativeMethod    string            `json:"speculativeMethod"`
	NumSpeculativeTokens int               `json:"numSpeculativeTokens"`
	EnablePrefixCaching  bool              `json:"enablePrefixCaching"`
	ReasoningParser      string            `json:"reasoningParser,omitempty"`
	ReasoningFormat      string            `json:"reasoningFormat,omitempty"`
	ToolCallParser       string            `json:"toolCallParser,omitempty"`
	LimitMMPerPrompt     string            `json:"limitMmPerPrompt"`
	ModelFile            string            `json:"modelFile,omitempty"`
	MMProjFile           string            `json:"mmprojFile,omitempty"`
	DraftFile            string            `json:"draftFile,omitempty"`
	GPUMemoryUtilHigh    float64           `json:"gpuMemoryUtilHigh"`
	GPUMemoryUtil64GB    float64           `json:"gpuMemoryUtil64GB"`
	MountHostPtxas       bool              `json:"mountHostPtxas"`
	ExtraArgs            []string          `json:"extraArgs,omitempty"`
	ExtraEnv             map[string]string `json:"extraEnv,omitempty"`
}

type RuntimeHubSource struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Path       string `json:"path"`
}

type RuntimeTargetManifest struct {
	Target             string                      `json:"target"`
	DisplayName        string                      `json:"displayName"`
	Image              string                      `json:"image"`
	ImageID            string                      `json:"imageId"`
	ConfigImageID      string                      `json:"configImageId"`
	RuntimeImageSHA256 string                      `json:"runtimeImageSha256"`
	Name               string                      `json:"name"`
	Size               int64                       `json:"size"`
	UnpackedSize       int64                       `json:"unpackedSize"`
	SHA256             string                      `json:"sha256"`
	Compatibility      RuntimeCompatibility        `json:"compatibility"`
	LaunchProfile      RuntimeLaunchProfile        `json:"launchProfile"`
	Sources            map[string]RuntimeHubSource `json:"sources"`
}

type RuntimeManifestDoc struct {
	Schema  string                           `json:"schema"`
	Status  string                           `json:"status"`
	Targets map[string]RuntimeTargetManifest `json:"targets"`
}

type RuntimeCacheEntry struct {
	Target             string                    `json:"target"`
	Name               string                    `json:"name"`
	Size               int64                     `json:"size"`
	SHA256             string                    `json:"sha256"`
	Fingerprint        string                    `json:"fingerprint"`
	RuntimeImageSHA256 string                    `json:"runtimeImageSha256"`
	LayoutVersion      int                       `json:"layoutVersion"`
	ContextLength      int                       `json:"contextLength"`
	MaxNumSeqs         int                       `json:"maxNumSeqs"`
	Sources            map[string]ProviderSource `json:"sources"`
}

type RuntimeCacheManifestDoc struct {
	Schema string                       `json:"schema"`
	Status string                       `json:"status"`
	Caches map[string]RuntimeCacheEntry `json:"caches"`
}

type LoadedManifests struct {
	Models   ModelManifestDoc
	Runtimes RuntimeManifestDoc
	Caches   RuntimeCacheManifestDoc
}

func LoadManifestsFromBytes(modelsBytes, runtimesBytes, cachesBytes []byte) (*LoadedManifests, error) {
	var m ModelManifestDoc
	if err := json.Unmarshal(modelsBytes, &m); err != nil {
		return nil, fmt.Errorf("解析 models.json 失败: %w", err)
	}
	if m.Schema != "model-host.runtime-model-manifest.v1" {
		return nil, fmt.Errorf("不支持的 models.json schema: %s", m.Schema)
	}
	var r RuntimeManifestDoc
	if err := json.Unmarshal(runtimesBytes, &r); err != nil {
		return nil, fmt.Errorf("解析 runtimes.json 失败: %w", err)
	}
	if r.Schema != "model-host.runtime-images.v1" {
		return nil, fmt.Errorf("不支持的 runtimes.json schema: %s", r.Schema)
	}
	var c RuntimeCacheManifestDoc
	if err := json.Unmarshal(cachesBytes, &c); err != nil {
		return nil, fmt.Errorf("解析 caches.json 失败: %w", err)
	}
	return &LoadedManifests{Models: m, Runtimes: r, Caches: c}, nil
}

func LoadManifests() (*LoadedManifests, error) {
	modelsBytes := manifests.ModelsJSON
	runtimesBytes := manifests.RuntimesJSON
	cachesBytes := manifests.CachesJSON

	if p := strings.TrimSpace(os.Getenv("QWEN35_9B_MODELS_MANIFEST")); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		modelsBytes = b
	}
	if p := strings.TrimSpace(os.Getenv("QWEN35_9B_RUNTIMES_MANIFEST")); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		runtimesBytes = b
	}
	if p := strings.TrimSpace(os.Getenv("QWEN35_9B_CACHES_MANIFEST")); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		cachesBytes = b
	}
	return LoadManifestsFromBytes(modelsBytes, runtimesBytes, cachesBytes)
}

func (m *LoadedManifests) ModelBytesForTarget(target string) int64 {
	var total int64
	for _, group := range m.Models.Models {
		if !slices.Contains(group.Targets, target) {
			continue
		}
		if len(group.Files) > 0 {
			for _, f := range group.Files {
				total += f.Size
			}
		} else {
			total += group.EstimatedBytes
		}
	}
	return total
}

func (m *LoadedManifests) GroupsForTarget(target string) []ModelGroupEntry {
	var out []ModelGroupEntry
	for _, g := range m.Models.Models {
		if slices.Contains(g.Targets, target) {
			out = append(out, g)
		}
	}
	return out
}

func (m *LoadedManifests) ValidateTargetLocked(target string) error {
	rt, ok := m.Runtimes.Targets[target]
	if !ok {
		return fmt.Errorf("未定义目标运行配置 %q", target)
	}
	if !validSHA256Hex(rt.SHA256) || rt.Size <= 0 || !validPrefixedImageID(rt.ImageID) {
		return fmt.Errorf("目标 %s 运行镜像归档尚未完成双源发布与摘要锁定（status=%s）", target, m.Runtimes.Status)
	}
	for _, provider := range []string{"huggingface", "modelscope"} {
		src, exists := rt.Sources[provider]
		if !exists || strings.TrimSpace(src.Repository) == "" || !validCommitRevision(src.Revision) || strings.TrimSpace(src.Path) == "" {
			return fmt.Errorf("目标 %s 缺少 %s 固定提交运行镜像源", target, provider)
		}
		u := BuildRuntimeArchiveURL(provider, src)
		if err := ValidateAllowedSourceURL(u, provider); err != nil {
			return err
		}
	}
	groups := m.GroupsForTarget(target)
	if len(groups) == 0 {
		return fmt.Errorf("目标 %s 未定义任何模型权重清单", target)
	}
	for _, g := range groups {
		if g.Role == "" || filepath.Base(g.Role) != g.Role {
			return fmt.Errorf("模型角色 %q 无效", g.Role)
		}
		for _, provider := range []string{"huggingface", "modelscope"} {
			pin, exists := g.Sources[provider]
			if !exists || strings.TrimSpace(pin.Repository) == "" || !validCommitRevision(pin.Revision) {
				return fmt.Errorf("模型 %s 缺少 %s 固定提交源", g.ID, provider)
			}
			files := pin.Files
			if len(files) == 0 {
				files = g.Files
			}
			if len(files) == 0 {
				return fmt.Errorf("模型 %s (%s) 尚未锁定逐文件 SHA256 清单", g.ID, provider)
			}
			for _, f := range files {
				if f.Size <= 0 || !validSHA256Hex(f.SHA256) || cleanRelPath(f.Path) == "" {
					return fmt.Errorf("模型文件 %s/%s 清单无效", g.ID, f.Path)
				}
				u := BuildModelFileURL(provider, pin, f.Path)
				if err := ValidateAllowedSourceURL(u, provider); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (m *LoadedManifests) ModelArtifactsForTarget(target, provider string) (map[string][]ArtifactFile, int64, error) {
	provider, err := normalizeModelProvider(provider)
	if err != nil {
		return nil, 0, err
	}
	if err := m.ValidateTargetLocked(target); err != nil {
		return nil, 0, err
	}
	artifacts := make(map[string][]ArtifactFile)
	var total int64
	for _, g := range m.GroupsForTarget(target) {
		pin := g.Sources[provider]
		files := pin.Files
		if len(files) == 0 {
			files = g.Files
		}
		for _, f := range files {
			fileURL := BuildModelFileURL(provider, pin, f.Path)
			if err := ValidateAllowedSourceURL(fileURL, provider); err != nil {
				return nil, 0, err
			}
			artifacts[g.Role] = append(artifacts[g.Role], ArtifactFile{
				Name:   path.Base(f.Path),
				Size:   f.Size,
				SHA256: strings.ToLower(f.SHA256),
				URL:    fileURL,
				URLs:   []string{fileURL},
			})
			total += f.Size
		}
	}
	return artifacts, total, nil
}

func (m *LoadedManifests) RuntimeImageArtifactForTarget(target, provider string) (RuntimeTargetManifest, ArtifactFile, error) {
	provider, err := normalizeModelProvider(provider)
	if err != nil {
		return RuntimeTargetManifest{}, ArtifactFile{}, err
	}
	if err := m.ValidateTargetLocked(target); err != nil {
		return RuntimeTargetManifest{}, ArtifactFile{}, err
	}
	rt := m.Runtimes.Targets[target]
	src := rt.Sources[provider]
	archiveURL := BuildRuntimeArchiveURL(provider, src)
	if err := ValidateAllowedSourceURL(archiveURL, provider); err != nil {
		return RuntimeTargetManifest{}, ArtifactFile{}, err
	}
	return rt, ArtifactFile{
		Name:   rt.Name,
		Size:   rt.Size,
		SHA256: strings.ToLower(rt.SHA256),
		URL:    archiveURL,
		URLs:   []string{archiveURL},
	}, nil
}

func (m *LoadedManifests) MatchingWarmCache(target, provider string, rt RuntimeTargetManifest, contextLength int) (*RuntimeCacheEntry, *ArtifactFile, error) {
	for _, entry := range m.Caches.Caches {
		if entry.Target != target || entry.ContextLength != contextLength || entry.MaxNumSeqs != config.MaxConcurrency {
			continue
		}
		if entry.LayoutVersion != 1 || !strings.EqualFold(entry.RuntimeImageSHA256, rt.RuntimeImageSHA256) {
			continue
		}
		if !validSHA256Hex(entry.Fingerprint) || !validSHA256Hex(entry.SHA256) || entry.Size <= 0 {
			continue
		}
		src, ok := entry.Sources[provider]
		if !ok || !validCommitRevision(src.Revision) || src.Path == "" {
			continue
		}
		cacheURL := BuildRuntimeArchiveURL(provider, RuntimeHubSource{
			Repository: src.Repository,
			Revision:   src.Revision,
			Path:       src.Path,
		})
		if err := ValidateAllowedSourceURL(cacheURL, provider); err != nil {
			return nil, nil, err
		}
		af := ArtifactFile{
			Name:   entry.Name,
			Size:   entry.Size,
			SHA256: strings.ToLower(entry.SHA256),
			URL:    cacheURL,
			URLs:   []string{cacheURL},
		}
		cp := entry
		return &cp, &af, nil
	}
	return nil, nil, nil
}

func BuildModelFileURL(provider string, pin ModelSourcePin, relPath string) string {
	switch provider {
	case "huggingface":
		return "https://huggingface.co/" + escapeURLPath(pin.Repository) + "/resolve/" + pin.Revision + "/" + escapeURLPath(relPath)
	case "modelscope":
		return "https://modelscope.cn/models/" + escapeURLPath(pin.Repository) + "/resolve/" + pin.Revision + "/" + escapeURLPath(relPath)
	default:
		return ""
	}
}

func BuildRuntimeArchiveURL(provider string, src RuntimeHubSource) string {
	switch provider {
	case "huggingface":
		return "https://huggingface.co/" + escapeURLPath(src.Repository) + "/resolve/" + src.Revision + "/" + escapeURLPath(src.Path)
	case "modelscope":
		return "https://modelscope.cn/models/" + escapeURLPath(src.Repository) + "/resolve/" + src.Revision + "/" + escapeURLPath(src.Path)
	default:
		return ""
	}
}

// ValidateAllowedSourceURL enforces the strict §2.5 & §3 allowlist:
// HTTPS only, matching the selected dual-source hub (Hugging Face or ModelScope),
// never Registry CDN.
func ValidateAllowedSourceURL(rawURL, provider string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme != "https" {
		return fmt.Errorf("制品 URL 必须使用 HTTPS: %q", rawURL)
	}
	host := strings.ToLower(parsed.Hostname())
	if strings.Contains(host, "lazycat") || strings.Contains(host, "registry") {
		return fmt.Errorf("禁止使用 Registry CDN 作为模型或运行归档下载源: %s", host)
	}
	switch provider {
	case "huggingface":
		if host != "huggingface.co" {
			return fmt.Errorf("Hugging Face 来源域名非法: %s", host)
		}
	case "modelscope":
		if host != "modelscope.cn" && host != "www.modelscope.cn" {
			return fmt.Errorf("ModelScope 来源域名非法: %s", host)
		}
	default:
		return fmt.Errorf("未知下载源: %s", provider)
	}
	if strings.Contains(parsed.Path, "..") {
		return errors.New("制品路径不允许包含上级目录跳转")
	}
	return nil
}

func escapeURLPath(v string) string {
	parts := strings.Split(strings.TrimLeft(v, "/"), "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

func cleanRelPath(p string) string {
	cleaned := path.Clean(strings.TrimSpace(p))
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "/") || strings.HasPrefix(cleaned, "../") || strings.ContainsAny(cleaned, "\\\r\n\x00") {
		return ""
	}
	return cleaned
}

func validSHA256Hex(v string) bool {
	if len(v) != 64 {
		return false
	}
	_, err := hex.DecodeString(v)
	return err == nil
}

func validPrefixedImageID(v string) bool {
	if !strings.HasPrefix(strings.ToLower(v), "sha256:") || len(v) != len("sha256:")+64 {
		return false
	}
	return validSHA256Hex(v[len("sha256:"):])
}

func validCommitRevision(v string) bool {
	if len(v) != 40 {
		return false
	}
	_, err := hex.DecodeString(v)
	return err == nil
}
