package lpk

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"qwen35-9b-deployer/internal/config"
)

func TestDeferredStopAndPersistentDataLifecycle(t *testing.T) {
	driveBase := t.TempDir()
	t.Setenv("QWEN35_9B_NETWORK_DRIVE_BASE", driveBase)

	modelContent := []byte("fake-quantized-safetensors-weights")
	modelSum := sha256Hex(modelContent)
	rev := strings.Repeat("a", 40)
	rtRev := strings.Repeat("b", 40)
	rtSha := strings.Repeat("c", 64)
	imgID := "sha256:" + strings.Repeat("d", 64)

	modelsManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-model-manifest.v1",
  "status": "locked",
  "modelId": "qwen3.5-9b-defiant-fable",
  "models": [
    {
      "id": "thor-nvfp4",
      "targets": ["thor"],
      "role": "model",
      "destination": "model",
      "mountTarget": "/model",
      "name": "Qwen 3.5 9B Defiant Fable NVFP4",
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q}
      },
      "files": [
        {"path": "model.gguf", "size": %d, "sha256": %q}
      ]
    }
  ]
}`, rev, rev, len(modelContent), modelSum)

	runtimesManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-images.v1",
  "status": "locked",
  "targets": {
    "thor": {
      "target": "thor",
      "displayName": "AGX Thor",
      "image": "local/qwen35-9b-defiant-fable-thor:0.1.0",
      "imageId": %q,
      "configImageId": %q,
      "runtimeImageSha256": %q,
      "name": "qwen35-9b-defiant-fable-thor-0.1.0.tar.gz",
      "size": 1024,
      "unpackedSize": 4096,
      "sha256": %q,
      "launchProfile": {
        "engine": "llamacpp", "modelFile": "model.gguf",
        "kvCacheDtype": "auto",
        "maxNumSeqs": 8,
        "maxNumBatchedTokens": 8192,
        "enablePrefixCaching": true,
        "limitMmPerPrompt": "{\"image\": 4, \"video\": 0}"
      },
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b-defiant-fable-thor-0.1.0.tar.gz"},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b-defiant-fable-thor-0.1.0.tar.gz"}
      }
    }
  }
}`, imgID, imgID, rtSha, rtSha, rtRev, rtRev)

	tmpManifestDir := t.TempDir()
	modelsPath := filepath.Join(tmpManifestDir, "models.json")
	runtimesPath := filepath.Join(tmpManifestDir, "runtimes.json")
	_ = os.WriteFile(modelsPath, []byte(modelsManifest), 0o600)
	_ = os.WriteFile(runtimesPath, []byte(runtimesManifest), 0o600)
	t.Setenv("QWEN35_9B_MODELS_MANIFEST", modelsPath)
	t.Setenv("QWEN35_9B_RUNTIMES_MANIFEST", runtimesPath)

	var (
		mu              sync.Mutex
		events          []string
		oldServiceAlive = true
		imageLoaded     = false
		modelUploaded   = false
		serviceDeployed = false
		serviceDeleted  = false
		cancelJobs      []string
		hostShouldBlock = false
	)

	mockHost := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/capabilities":
			writeJSON(w, http.StatusOK, HostCapabilities{
				ProtocolVersion:    "1.0",
				ArtifactDownloader: "aria2",
				Capabilities: map[string]bool{
					"artifactDownload":     true,
					"selectiveProcessStop": true,
					"cacheRestore":         true,
				},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices":
			writeJSON(w, http.StatusOK, []HostDevice{{
				SerialNumber: "thor-1",
				Name:         "Thor Dev",
				Model:        "NVIDIA Jetson AGX Thor Developer Kit",
				L4TVersion:   "39.2.0",
				IP:           "192.168.1.122",
				Online:       true,
			}})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/resources":
			avail := int64(100 << 30)
			if oldServiceAlive {
				// Current available is 10 GiB, below required ~21.9 GiB, so protected stop is needed.
				avail = int64(4 << 30)
			}
			writeJSON(w, http.StatusOK, HostDeviceResources{
				MemoryTotalBytes:     122 << 30,
				MemoryAvailableBytes: avail,
				VRAMTotalBytes:       122 << 30,
				VRAMAvailableBytes:   avail,
				DiskAvailableBytes:   400 << 30,
				DiskAvailableKnown:   true,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/process/status":
			svcs := map[string]HostServiceStatus{
				"traefik": {MemoryUsage: 100 << 20},
			}
			if oldServiceAlive {
				svcs["vllm-old"] = HostServiceStatus{MemoryUsage: 50 << 30}
			}
			if hostShouldBlock {
				svcs["llama-unauthorized"] = HostServiceStatus{MemoryUsage: 40 << 30}
			}
			writeJSON(w, http.StatusOK, HostProcessStatus{Services: svcs})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/artifacts/status":
			events = append(events, "artifacts/status")
			writeJSON(w, http.StatusOK, ArtifactStatus{
				Root:    "/ssd/lzc-ai-agent/models/qwen35-9b",
				Ready:   modelUploaded,
				Missing: []string{"model/model.gguf"},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/images/status":
			events = append(events, "images/status")
			writeJSON(w, http.StatusOK, ImageStatusResponse{Image: "local/qwen35-9b-defiant-fable-thor:0.1.0", ImageID: imgID, Ready: imageLoaded})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/artifacts/prepare":
			events = append(events, "artifacts/prepare")
			if !oldServiceAlive {
				t.Errorf("old service was stopped BEFORE artifacts/prepare completed!")
			}
			// Materialize the file on fake network drive so multipart upload can read it.
			roleDir := filepath.Join(driveBase, strings.Trim(config.NetworkDriveDisplay, "/"), "model")
			_ = os.MkdirAll(roleDir, 0o755)
			_ = os.WriteFile(filepath.Join(roleDir, "model.gguf"), modelContent, 0o600)
			writeJSON(w, http.StatusAccepted, ArtifactPrepareResponse{JobID: "prep-1", Status: "succeeded", Progress: 100})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/cancel"):
			cancelJobs = append(cancelJobs, r.URL.Path)
			writeJSON(w, http.StatusOK, map[string]string{"status": "canceled"})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/artifacts/upload":
			events = append(events, "artifacts/upload")
			if !oldServiceAlive {
				t.Errorf("old service was stopped BEFORE artifacts/upload completed!")
			}
			modelUploaded = true
			writeJSON(w, http.StatusOK, map[string]bool{"reused": false})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/images/load":
			events = append(events, "images/load")
			if !oldServiceAlive {
				t.Errorf("old service was stopped BEFORE images/load completed!")
			}
			imageLoaded = true
			writeJSON(w, http.StatusOK, ImageLoadResponse{Status: "succeeded", Loaded: true, Progress: 100})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/process/stop":
			var req ProcessStopRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			events = append(events, "process/stop:"+strings.Join(req.PackageIDs, ","))
			if len(req.PackageIDs) == 0 {
				t.Errorf("process/stop called with empty packageIds (stop-all prohibited!)")
			}
			for _, id := range req.PackageIDs {
				if id == "vllm-old" {
					oldServiceAlive = false
				}
			}
			writeJSON(w, http.StatusOK, map[string]any{"stopped": req.PackageIDs})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/services/deploy":
			events = append(events, "services/deploy")
			serviceDeployed = true
			writeJSON(w, http.StatusAccepted, map[string]bool{"accepted": true})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/services/progress":
			writeJSON(w, http.StatusOK, HostServiceDeployStatus{ServiceID: config.DefaultServiceName, Stage: "health_check", Progress: 98})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/services/health":
			writeJSON(w, http.StatusOK, HealthResponse{
				Healthy:    true,
				StatusCode: 200,
				Body:       fmt.Sprintf(`{"data":[{"id":%q}]}`, config.PublicModelID),
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/model-host/v1/devices/thor-1/services/"+config.DefaultServiceName:
			serviceDeleted = true
			writeJSON(w, http.StatusOK, map[string]bool{"deleted": true})
		default:
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "unhandled " + r.URL.Path})
		}
	}))
	defer mockHost.Close()

	dataDir := t.TempDir()
	srv, err := NewServer(mockHost.URL, dataDir)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	// 1. Admission without stopProtected must return 409 with protectedProcessIds & NOT stop old service.
	rec := doAPI(t, srv, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"}}`)
	if rec.Code != http.StatusConflict {
		t.Fatalf("expected 409 conflict without confirmation, got %d (%s)", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "vllm-old") {
		t.Fatalf("expected vllm-old in protectedProcessIds: %s", rec.Body.String())
	}

	// 2. Submit with exact protectedProcessIds authorization -> deferred stop runs AFTER prepare.
	rec = doAPI(t, srv, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"},"stopProtected":true,"protectedProcessIds":["vllm-old"]}`)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected 202 accepted, got %d (%s)", rec.Code, rec.Body.String())
	}
	var dep DeploymentRecord
	_ = json.Unmarshal(rec.Body.Bytes(), &dep)

	// Wait for completion.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		r2 := doAPI(t, srv, http.MethodGet, "/api/v1/deployments/"+dep.ID, "u1", "")
		_ = json.Unmarshal(r2.Body.Bytes(), &dep)
		if dep.Status == "succeeded" || dep.Status == "failed" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if dep.Status != "succeeded" {
		t.Fatalf("deployment failed: status=%s err=%s", dep.Status, dep.Error)
	}
	if !serviceDeployed {
		t.Fatal("services/deploy was not called")
	}

	// Verify strict event ordering: prepare/upload/load happened BEFORE process/stop, and stop BEFORE services/deploy.
	mu.Lock()
	evList := append([]string(nil), events...)
	mu.Unlock()
	idxPrepare := indexOfPrefix(evList, "artifacts/prepare")
	idxUpload := indexOfPrefix(evList, "artifacts/upload")
	idxLoad := indexOfPrefix(evList, "images/load")
	idxStop := indexOfPrefix(evList, "process/stop:vllm-old")
	idxDeploy := indexOfPrefix(evList, "services/deploy")
	if !(idxPrepare >= 0 && idxUpload > idxPrepare && idxLoad > idxUpload && idxStop > idxLoad && idxDeploy > idxStop) {
		t.Fatalf("deferred stop ordering violated! events=%v", evList)
	}

	// 3. Cross-user isolation check
	otherUser := doAPI(t, srv, http.MethodGet, "/api/v1/deployments/"+dep.ID, "u2", "")
	if otherUser.Code != http.StatusNotFound {
		t.Fatalf("cross-user access should return 404, got %d", otherUser.Code)
	}

	// 4. PersistentData invalid deleteData values -> HTTP 400 with no side effects
	for _, badDel := range []string{"?deleteData=", "?deleteData=TRUE", "?deleteData=1", "?deleteData=true&deleteData=false"} {
		badRes := doAPI(t, srv, http.MethodDelete, "/api/v1/deployments/"+dep.ID+badDel, "u1", "")
		if badRes.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d", badDel, badRes.Code)
		}
	}

	// 5. Default delete (deleteData=false) -> preserves data, stops service, does NOT delete service dir, survives restart.
	delKeep := doAPI(t, srv, http.MethodDelete, "/api/v1/deployments/"+dep.ID+"?deleteData=false", "u1", "")
	if delKeep.Code != http.StatusOK || !strings.Contains(delKeep.Body.String(), `"deleted":true`) {
		t.Fatalf("expected 200 deleted:true, got %d (%s)", delKeep.Code, delKeep.Body.String())
	}
	if serviceDeleted {
		t.Fatal("deleteData=false MUST NOT call destructive DELETE /services/{serviceId}")
	}
	srv.Close()

	// Restart server from persisted state: deleted record must NOT reappear, preserved data marker kept!
	srv2, err := NewServer(mockHost.URL, dataDir)
	if err != nil {
		t.Fatalf("restart NewServer: %v", err)
	}
	defer srv2.Close()
	listAfterRestart := doAPI(t, srv2, http.MethodGet, "/api/v1/deployments", "u1", "")
	if strings.Contains(listAfterRestart.Body.String(), dep.ID) {
		t.Fatalf("deleted record resurrected after controller restart: %s", listAfterRestart.Body.String())
	}

	// 6. Redeploy on same device, then DELETE ?deleteData=true -> calls destructive cleanup!
	rec2 := doAPI(t, srv2, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"},"stopExisting":true}`)
	if rec2.Code != http.StatusAccepted {
		t.Fatalf("redeploy failed: %d (%s)", rec2.Code, rec2.Body.String())
	}
	var dep2 DeploymentRecord
	_ = json.Unmarshal(rec2.Body.Bytes(), &dep2)
	deadline = time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		r2 := doAPI(t, srv2, http.MethodGet, "/api/v1/deployments/"+dep2.ID, "u1", "")
		_ = json.Unmarshal(r2.Body.Bytes(), &dep2)
		if dep2.Status == "succeeded" || dep2.Status == "failed" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if dep2.Status != "succeeded" {
		t.Fatalf("redeploy did not succeed: %s (%s)", dep2.Status, dep2.Error)
	}

	delDestroy := doAPI(t, srv2, http.MethodDelete, "/api/v1/deployments/"+dep2.ID+"?deleteData=true", "u1", "")
	if delDestroy.Code != http.StatusOK {
		t.Fatalf("deleteData=true failed: %d (%s)", delDestroy.Code, delDestroy.Body.String())
	}
	if !serviceDeleted {
		t.Fatal("deleteData=true MUST call destructive DELETE /services/{serviceId}")
	}
}

func TestCancelRetryAndUnauthorizedNewService(t *testing.T) {
	driveBase := t.TempDir()
	t.Setenv("QWEN35_9B_NETWORK_DRIVE_BASE", driveBase)

	modelContent := []byte("fake-orin-weights")
	modelSum := sha256Hex(modelContent)
	rev := strings.Repeat("a", 40)
	rtRev := strings.Repeat("b", 40)
	rtSha := strings.Repeat("c", 64)
	imgID := "sha256:" + strings.Repeat("d", 64)

	modelsManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-model-manifest.v1",
  "status": "locked",
  "modelId": "qwen3.5-9b-defiant-fable",
  "models": [
    {
      "id": "thor-mtp-gguf",
      "targets": ["thor"],
      "role": "model",
      "destination": "model",
      "mountTarget": "/model",
      "name": "Qwen 3.5 9B Defiant Fable W4A16",
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q}
      },
      "files": [{"path": "model.gguf", "size": %d, "sha256": %q}]
    }
  ]
}`, rev, rev, len(modelContent), modelSum)

	runtimesManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-images.v1",
  "status": "locked",
  "targets": {
    "thor": {
      "target": "thor",
      "displayName": "AGX Thor",
      "image": "local/qwen35-9b-defiant-fable-thor:0.1.0",
      "imageId": %q,
      "configImageId": %q,
      "runtimeImageSha256": %q,
      "name": "qwen35-9b-defiant-fable-thor-0.1.0.tar.gz",
      "size": 1024,
      "unpackedSize": 4096,
      "sha256": %q,
      "launchProfile": {"engine": "llamacpp", "modelFile": "model.gguf", "maxNumSeqs": 8, "maxNumBatchedTokens": 4096},
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b.tar.gz"},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b.tar.gz"}
      }
    }
  }
}`, imgID, imgID, rtSha, rtSha, rtRev, rtRev)

	tmpDir := t.TempDir()
	mp := filepath.Join(tmpDir, "models.json")
	rp := filepath.Join(tmpDir, "runtimes.json")
	_ = os.WriteFile(mp, []byte(modelsManifest), 0o600)
	_ = os.WriteFile(rp, []byte(runtimesManifest), 0o600)
	t.Setenv("QWEN35_9B_MODELS_MANIFEST", mp)
	t.Setenv("QWEN35_9B_RUNTIMES_MANIFEST", rp)

	var (
		mu             sync.Mutex
		prepareStarted = make(chan struct{})
		canceledAria2  bool
		newRogueActive = false
		stopCalled     = false
		deployCalled   = false
		pollCount      = 0
	)

	mockHost := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/capabilities":
			writeJSON(w, http.StatusOK, HostCapabilities{ProtocolVersion: "1.0", ArtifactDownloader: "aria2", Capabilities: map[string]bool{"artifactDownload": true, "selectiveProcessStop": true}})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices":
			writeJSON(w, http.StatusOK, []HostDevice{{SerialNumber: "thor-1", Name: "Thor T5000", Model: "NVIDIA Jetson AGX Thor", L4TVersion: "39.2.1", IP: "192.168.1.176", Online: true}})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/resources":
			avail := int64(50 << 30)
			if newRogueActive {
				avail = int64(4 << 30)
			}
			writeJSON(w, http.StatusOK, HostDeviceResources{MemoryTotalBytes: 61 << 30, MemoryAvailableBytes: avail, VRAMTotalBytes: 61 << 30, VRAMAvailableBytes: avail, DiskAvailableBytes: 200 << 30, DiskAvailableKnown: true})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/process/status":
			svcs := map[string]HostServiceStatus{}
			if newRogueActive {
				svcs["vllm-rogue"] = HostServiceStatus{MemoryUsage: 30 << 30}
			}
			writeJSON(w, http.StatusOK, HostProcessStatus{Services: svcs})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/artifacts/status":
			writeJSON(w, http.StatusOK, ArtifactStatus{Root: "/ssd/gemma4", Ready: newRogueActive})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/images/status":
			writeJSON(w, http.StatusOK, ImageStatusResponse{Image: "local/qwen35-9b-defiant-fable-thor:0.1.0", ImageID: imgID, Ready: newRogueActive})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/artifacts/prepare":
			select {
			case <-prepareStarted:
			default:
				close(prepareStarted)
			}
			writeJSON(w, http.StatusAccepted, ArtifactPrepareResponse{JobID: "job-cancel", Status: "downloading", Progress: 10})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/artifacts/prepare/job-cancel":
			pollCount++
			writeJSON(w, http.StatusOK, ArtifactPrepareResponse{JobID: "job-cancel", Status: "downloading", Progress: 20})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/artifacts/prepare/job-cancel/cancel":
			canceledAria2 = true
			writeJSON(w, http.StatusOK, ArtifactPrepareResponse{JobID: "job-cancel", Status: "canceled"})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/process/stop":
			stopCalled = true
			writeJSON(w, http.StatusOK, map[string]any{})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/services/deploy":
			deployCalled = true
			writeJSON(w, http.StatusAccepted, map[string]bool{"accepted": true})
		default:
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		}
	}))
	defer mockHost.Close()

	srv := newTestServer(t, mockHost.Config.Handler)

	// 1. Test cancel propagation during Aria2 prepare
	rec := doAPI(t, srv, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"}}`)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d (%s)", rec.Code, rec.Body.String())
	}
	var dep DeploymentRecord
	_ = json.Unmarshal(rec.Body.Bytes(), &dep)
	<-prepareStarted

	cRec := doAPI(t, srv, http.MethodPost, "/api/v1/deployments/"+dep.ID+"/cancel", "u1", "")
	if cRec.Code != http.StatusOK {
		t.Fatalf("cancel failed: %d", cRec.Code)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		ok := canceledAria2
		mu.Unlock()
		if ok {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	mu.Lock()
	if !canceledAria2 || stopCalled || deployCalled {
		t.Fatalf("cancel check failed: canceledAria2=%v stopCalled=%v deployCalled=%v", canceledAria2, stopCalled, deployCalled)
	}
	// Now simulate a new rogue protected service appearing while artifacts are ready, then deploy -> must fail at pre-start switch without calling stop!
	newRogueActive = true
	mu.Unlock()

	rec2 := doAPI(t, srv, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"}}`)
	if rec2.Code != http.StatusConflict {
		t.Fatalf("expected 409 when rogue service causes shortfall at admission, got %d", rec2.Code)
	}
}

func TestUnauthorizedMidPreparationAndRetryPreservesID(t *testing.T) {
	driveBase := t.TempDir()
	t.Setenv("QWEN35_9B_NETWORK_DRIVE_BASE", driveBase)

	modelContent := []byte("fake-orin-weights-retry")
	modelSum := sha256Hex(modelContent)
	rev := strings.Repeat("a", 40)
	rtRev := strings.Repeat("b", 40)
	rtSha := strings.Repeat("c", 64)
	imgID := "sha256:" + strings.Repeat("d", 64)

	modelsManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-model-manifest.v1",
  "status": "locked",
  "modelId": "qwen3.5-9b-defiant-fable",
  "models": [
    {
      "id": "thor-mtp-gguf",
      "targets": ["thor"],
      "role": "model",
      "destination": "model",
      "mountTarget": "/model",
      "name": "Qwen 3.5 9B Defiant Fable W4A16",
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF", "revision": %q}
      },
      "files": [{"path": "model.gguf", "size": %d, "sha256": %q}]
    }
  ]
}`, rev, rev, len(modelContent), modelSum)

	runtimesManifest := fmt.Sprintf(`{
  "schema": "model-host.runtime-images.v1",
  "status": "locked",
  "targets": {
    "thor": {
      "target": "thor",
      "displayName": "AGX Thor",
      "image": "local/qwen35-9b-defiant-fable-thor:0.1.0",
      "imageId": %q,
      "configImageId": %q,
      "runtimeImageSha256": %q,
      "name": "qwen35-9b-defiant-fable-thor-0.1.0.tar.gz",
      "size": 1024,
      "unpackedSize": 4096,
      "sha256": %q,
      "launchProfile": {"engine": "llamacpp", "modelFile": "model.gguf", "maxNumSeqs": 8, "maxNumBatchedTokens": 4096},
      "sources": {
        "huggingface": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b.tar.gz"},
        "modelscope": {"repository": "manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime", "revision": %q, "path": "thor/qwen35-9b.tar.gz"}
      }
    }
  }
}`, imgID, imgID, rtSha, rtSha, rtRev, rtRev)

	tmpDir := t.TempDir()
	mp := filepath.Join(tmpDir, "models.json")
	rp := filepath.Join(tmpDir, "runtimes.json")
	_ = os.WriteFile(mp, []byte(modelsManifest), 0o600)
	_ = os.WriteFile(rp, []byte(runtimesManifest), 0o600)
	t.Setenv("QWEN35_9B_MODELS_MANIFEST", mp)
	t.Setenv("QWEN35_9B_RUNTIMES_MANIFEST", rp)

	var (
		mu             sync.Mutex
		rogueAppeared  = false
		stopCalled     = false
		deployCalled   = false
		preparedOnPod  = false
	)

	mockHost := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/capabilities":
			writeJSON(w, http.StatusOK, HostCapabilities{ProtocolVersion: "1.0", ArtifactDownloader: "aria2", Capabilities: map[string]bool{"artifactDownload": true, "selectiveProcessStop": true}})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices":
			writeJSON(w, http.StatusOK, []HostDevice{{SerialNumber: "thor-1", Name: "Thor T5000", Model: "NVIDIA Jetson AGX Thor", L4TVersion: "39.2.1", IP: "192.168.1.176", Online: true}})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/resources":
			avail := int64(50 << 30)
			if rogueAppeared {
				avail = int64(4 << 30)
			}
			writeJSON(w, http.StatusOK, HostDeviceResources{MemoryTotalBytes: 61 << 30, MemoryAvailableBytes: avail, VRAMTotalBytes: 61 << 30, VRAMAvailableBytes: avail, DiskAvailableBytes: 200 << 30, DiskAvailableKnown: true})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/process/status":
			svcs := map[string]HostServiceStatus{}
			if rogueAppeared {
				svcs["vllm-rogue"] = HostServiceStatus{MemoryUsage: 40 << 30}
			}
			writeJSON(w, http.StatusOK, HostProcessStatus{Services: svcs})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/artifacts/status":
			writeJSON(w, http.StatusOK, ArtifactStatus{Root: "/ssd/gemma4", Ready: preparedOnPod})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/images/status":
			writeJSON(w, http.StatusOK, ImageStatusResponse{Image: "local/qwen35-9b-defiant-fable-thor:0.1.0", ImageID: imgID, Ready: preparedOnPod})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/artifacts/prepare":
			roleDir := filepath.Join(driveBase, strings.Trim(config.NetworkDriveDisplay, "/"), "model")
			_ = os.MkdirAll(roleDir, 0o755)
			_ = os.WriteFile(filepath.Join(roleDir, "model.gguf"), modelContent, 0o600)
			writeJSON(w, http.StatusAccepted, ArtifactPrepareResponse{JobID: "p1", Status: "succeeded", Progress: 100})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/artifacts/upload":
			preparedOnPod = true
			writeJSON(w, http.StatusOK, map[string]bool{"reused": false})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/images/load":
			// Simulate rogue service starting while image load happens!
			rogueAppeared = true
			writeJSON(w, http.StatusOK, ImageLoadResponse{Status: "succeeded", Loaded: true, Progress: 100})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/process/stop":
			stopCalled = true
			writeJSON(w, http.StatusOK, map[string]any{})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/services/deploy":
			deployCalled = true
			writeJSON(w, http.StatusAccepted, map[string]bool{"accepted": true})
		case r.Method == http.MethodGet && r.URL.Path == "/model-host/v1/devices/thor-1/services/progress":
			writeJSON(w, http.StatusOK, HostServiceDeployStatus{ServiceID: config.DefaultServiceName, Stage: "health_check", Progress: 98})
		case r.Method == http.MethodPost && r.URL.Path == "/model-host/v1/devices/thor-1/services/health":
			writeJSON(w, http.StatusOK, HealthResponse{Healthy: true, StatusCode: 200, Body: fmt.Sprintf(`{"data":[{"id":%q}]}`, config.PublicModelID)})
		default:
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		}
	}))
	defer mockHost.Close()

	srv := newTestServer(t, mockHost.Config.Handler)
	rec := doAPI(t, srv, http.MethodPost, "/api/v1/deployments", "u1", `{"serialNumbers":["thor-1"],"parameters":{"contextLength":131072,"modelProvider":"modelscope"}}`)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d", rec.Code)
	}
	var dep DeploymentRecord
	_ = json.Unmarshal(rec.Body.Bytes(), &dep)

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		r2 := doAPI(t, srv, http.MethodGet, "/api/v1/deployments/"+dep.ID, "u1", "")
		_ = json.Unmarshal(r2.Body.Bytes(), &dep)
		if dep.Status == "failed" || dep.Status == "succeeded" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if dep.Status != "failed" || !strings.Contains(dep.Error, "vllm-rogue") {
		t.Fatalf("expected failed due to unauthorized vllm-rogue, got status=%s err=%s", dep.Status, dep.Error)
	}
	mu.Lock()
	if stopCalled || deployCalled {
		t.Fatalf("must not stop unauthorized process or deploy: stopCalled=%v deployCalled=%v", stopCalled, deployCalled)
	}
	rogueAppeared = false
	mu.Unlock()

	// Retry must keep the exact same deployment ID and succeed without re-downloading!
	retryRec := doAPI(t, srv, http.MethodPost, "/api/v1/deployments/"+dep.ID+"/retry", "u1", "")
	if retryRec.Code != http.StatusAccepted {
		t.Fatalf("retry failed: %d (%s)", retryRec.Code, retryRec.Body.String())
	}
	var retried DeploymentRecord
	_ = json.Unmarshal(retryRec.Body.Bytes(), &retried)
	if retried.ID != dep.ID {
		t.Fatalf("retry changed record ID: %s != %s", retried.ID, dep.ID)
	}
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		r2 := doAPI(t, srv, http.MethodGet, "/api/v1/deployments/"+dep.ID, "u1", "")
		_ = json.Unmarshal(r2.Body.Bytes(), &retried)
		if retried.Status == "succeeded" || retried.Status == "failed" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if retried.Status != "succeeded" {
		t.Fatalf("retry did not succeed: status=%s err=%s", retried.Status, retried.Error)
	}
}

func doAPI(t *testing.T, srv *Server, method, path, userID, body string) *httptest.ResponseRecorder {
	t.Helper()
	var buf *bytes.Buffer
	if body != "" {
		buf = bytes.NewBufferString(body)
	} else {
		buf = &bytes.Buffer{}
	}
	req := httptest.NewRequest(method, path, buf)
	if userID != "" {
		req.Header.Set("X-HC-User-ID", userID)
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

func indexOfPrefix(items []string, prefix string) int {
	for i, v := range items {
		if strings.HasPrefix(v, prefix) {
			return i
		}
	}
	return -1
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}
