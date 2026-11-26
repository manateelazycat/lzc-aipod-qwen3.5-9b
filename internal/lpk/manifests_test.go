package lpk

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"qwen35-9b-deployer/internal/config"
)

func TestDualSourceAllowlist(t *testing.T) {
	for _, bad := range []struct{ url, provider string }{
		{"https://registry-static.lazycat.cloud/docker/registry/v2/blobs/sha256/ab/abcd/data", "modelscope"},
		{"https://registry.lazycat.cloud/test.tar", "huggingface"},
		{"http://huggingface.co/a/b/resolve/commit/f", "huggingface"},
		{"https://huggingface.co/a/b/resolve/commit/f", "modelscope"},
		{"https://modelscope.cn/models/a/b/resolve/commit/f", "huggingface"},
	} {
		if err := ValidateAllowedSourceURL(bad.url, bad.provider); err == nil {
			t.Fatalf("expected error for %s (%s)", bad.url, bad.provider)
		}
	}

	for _, good := range []struct{ url, provider string }{
		{"https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/resolve/" + strings.Repeat("a", 40) + "/model.gguf", "huggingface"},
		{"https://modelscope.cn/models/manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF/resolve/" + strings.Repeat("b", 40) + "/model.gguf", "modelscope"},
	} {
		if err := ValidateAllowedSourceURL(good.url, good.provider); err != nil {
			t.Fatalf("expected valid URL %s, got err: %v", good.url, err)
		}
	}
}

func TestDualImageIDPreflight(t *testing.T) {
	ociID := "sha256:" + strings.Repeat("1", 64)
	cfgID := "sha256:" + strings.Repeat("2", 64)

	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model-host/v1/devices/sn-1/images/status" {
			var req ImageStatusRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.ImageID == cfgID {
				writeJSON(w, http.StatusOK, ImageStatusResponse{Image: req.Image, ImageID: cfgID, Ready: true})
				return
			}
			writeJSON(w, http.StatusOK, ImageStatusResponse{Image: req.Image, Ready: false})
			return
		}
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
	}))

	rt := RuntimeTargetManifest{
		Target:        config.TargetThor,
		Image:         "local/qwen35-9b-defiant-fable-thor:0.1.0",
		ImageID:       ociID,
		ConfigImageID: cfgID,
	}
	ready, matched, err := srv.deviceRuntimeImageReady(t.Context(), "sn-1", "u1", rt)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ready || matched != cfgID {
		t.Fatalf("expected match on configImageId %s, got ready=%v matched=%q", cfgID, ready, matched)
	}
}

func TestComposeGenerationAndSanitization(t *testing.T) {
	rt := RuntimeTargetManifest{
		Target:             config.TargetThor,
		Image:              "local/qwen35-9b-defiant-fable-thor:0.1.0",
		RuntimeImageSHA256: strings.Repeat("a", 64),
		LaunchProfile: RuntimeLaunchProfile{
			Engine:               config.EngineLlamaCpp,
			KVCacheDtype:         "q8_0",
			ModelFile:            "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf",
			MMProjFile:           "mmproj-F16.gguf",
			SpeculativeMethod:    "draft-mtp",
			NumSpeculativeTokens: 2,
			ReasoningFormat:      "deepseek",
			MaxNumSeqs:           8,
			ExtraArgs:            []string{"-b", "4096", "-ub", "2048", "--temp", "0.7", "--top-p", "0.85", "--top-k", "20"},
			ExtraEnv:             map[string]string{"GGML_CUDA_PDL": "1"},
		},
	}
	groups := []ModelGroupEntry{
		{
			Role:           "model",
			MountTarget:    "/model",
			Targets:        []string{config.TargetThor},
			EstimatedBytes: 7898141472,
			Files: []ModelFileEntry{
				{Path: "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf", Size: 6979975392},
				{Path: "mmproj-F16.gguf", Size: 918166080},
			},
		},
	}
	manifest := &LoadedManifests{
		Models:   ModelManifestDoc{Models: groups},
		Runtimes: RuntimeManifestDoc{Targets: map[string]RuntimeTargetManifest{config.TargetThor: rt}},
	}
	budget := ComputeMemoryBudget(config.TargetThor, 122<<30, 131072, manifest)
	if budget.TotalCapacityTooSmall || budget.RequiredBytes > 20<<30 {
		t.Fatalf("unexpected Qwen3.5-9B q8_0 128K budget: %+v", budget)
	}
	composeBytes, err := BuildInferenceCompose(config.TargetThor, rt, groups, "/var/lib/lzc-ai-agent/data/qwen35-9b", "cold-thor-1234", deploymentSettings{
		ContextLength: 131072,
		Concurrency:   8,
		ModelProvider: "modelscope",
	}, budget)
	if err != nil {
		t.Fatalf("BuildInferenceCompose error: %v", err)
	}
	yamlStr := string(composeBytes)
	for _, mustContain := range []string{
		"pull_policy: never",
		"gpus: all",
		"AIPOD_METRICS_PORT: \"18156\"",
		"/runtime/llama-server",
		"/model/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf",
		"--mmproj",
		"/model/mmproj-F16.gguf",
		"--alias",
		config.PublicModelID,
		"--spec-type",
		"draft-mtp",
		"--spec-draft-n-max",
		`"-ctk","q8_0","-ctv","q8_0"`,
		`"/runtime/healthcheck"`,
		":/model:ro",
		"/v1/models",
	} {
		if !strings.Contains(yamlStr, mustContain) {
			t.Fatalf("compose missing %q:\n%s", mustContain, yamlStr)
		}
	}
	for _, mustNotContain := range []string{"--model-draft", "--api-key", "Authorization"} {
		if strings.Contains(yamlStr, mustNotContain) {
			t.Fatalf("compose should not contain %q:\n%s", mustNotContain, yamlStr)
		}
	}

	pi := BuildPiAgentConfig(DeploymentRecord{
		APIURL:        "http://192.168.1.199:18156/v1",
		ContextLength: 131072,
	})
	piRaw, _ := json.Marshal(pi)
	piStr := string(piRaw)
	for _, forbidden := range []string{"draft-mtp", "local/qwen35", "/var/lib/"} {
		if strings.Contains(piStr, forbidden) {
			t.Fatalf("Pi Agent config leaked internal detail %q: %s", forbidden, piStr)
		}
	}

	sanitized := SanitizePublicText("failed local/qwen35-9b-defiant-fable-thor:0.1.0 at /user_home/foo --spec-type draft-mtp -ctk q8_0 -ctv q8_0 https://hf.co/f?X-Amz-Signature=secret")
	for _, forbidden := range []string{"local/qwen35", "/user_home", "--spec-type", "-ctk", "-ctv", "X-Amz-Signature"} {
		if strings.Contains(sanitized, forbidden) {
			t.Fatalf("SanitizePublicText leaked %q: %s", forbidden, sanitized)
		}
	}

	// Verify Orin compose generates single OMP_NUM_THREADS=4 and GGML_CUDA_PDL=0 (no duplicate YAML key).
	orinRT := rt
	orinRT.Target = config.TargetOrin
	orinRT.Image = "local/qwen35-9b-defiant-fable-orin:0.2.0"
	orinRT.LaunchProfile.ExtraEnv = map[string]string{
		"GGML_CUDA_PDL":   "0",
		"OMP_NUM_THREADS": "4",
	}
	orinCompose, err := BuildInferenceCompose(config.TargetOrin, orinRT, groups, "/var/lib/lzc-ai-agent/data/qwen35-9b", "cold-orin-1234", deploymentSettings{
		ContextLength: 131072,
		Concurrency:   8,
		ModelProvider: "modelscope",
	}, budget)
	if err != nil {
		t.Fatalf("BuildInferenceCompose(orin) error: %v", err)
	}
	orinYAML := string(orinCompose)
	if strings.Count(orinYAML, "OMP_NUM_THREADS:") != 1 {
		t.Fatalf("expected exactly one OMP_NUM_THREADS entry in Orin compose:\n%s", orinYAML)
	}
	if !strings.Contains(orinYAML, `GGML_CUDA_PDL: "0"`) || !strings.Contains(orinYAML, `OMP_NUM_THREADS: "4"`) {
		t.Fatalf("expected Orin PDL=0 and OMP_NUM_THREADS=4:\n%s", orinYAML)
	}
}

func newTestServer(t *testing.T, handler http.Handler) *Server {
	t.Helper()
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	dir := t.TempDir()
	srv, err := NewServer(ts.URL, dir)
	if err != nil {
		t.Fatalf("NewServer error: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}
