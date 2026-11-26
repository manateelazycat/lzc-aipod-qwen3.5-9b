package lpk

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"qwen35-9b-deployer/internal/config"
)

const (
	defaultHTTPTimeout        = 20 * time.Second
	processStopTimeout        = 2 * time.Minute
	memoryReleaseTimeout      = 120 * time.Second
	memoryReleasePollInterval = 2 * time.Second
	serviceStartupTimeout     = 40 * time.Minute
	deploymentTimeout         = 12 * time.Hour

	progressCheckStart        = 0
	progressCheckSpan         = 5
	progressDownloadStart     = 5
	progressDownloadSpan      = 30
	progressVerifyStart       = 35
	progressVerifySpan        = 5
	progressTransferStart     = 40
	progressTransferSpan      = 15
	progressImageLoadStart    = 55
	progressImageLoadSpan     = 13
	progressCacheRestoreStart = 68
	progressCacheRestoreSpan  = 4
	progressStopStart         = 72
	progressStopSpan          = 3
	progressServiceStart      = 75
	progressServiceStartSpan  = 8
	progressWeightsStart      = 83
	progressWeightsSpan       = 8
	progressKVStart           = 91
	progressKVSpan            = 5
	progressKernelStart       = 96
	progressKernelSpan        = 2
	progressHealthStart       = 98
	progressHealthSpan        = 1
)

func (s *Server) hostPath(path string) string {
	return s.backendURL + "/model-host/v1" + path
}

func (s *Server) newRequestWithClient(ctx context.Context, client *http.Client, method, targetURL string, body io.Reader, contentType, userID string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, targetURL, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if strings.TrimSpace(userID) != "" {
		req.Header.Set("X-HC-User-ID", strings.TrimSpace(userID))
	}
	return client.Do(req)
}

func (s *Server) doJSONWithClient(ctx context.Context, client *http.Client, method, targetURL string, payload any, userID string) (*http.Response, error) {
	var body io.Reader
	ct := ""
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(raw)
		ct = "application/json"
	}
	return s.newRequestWithClient(ctx, client, method, targetURL, body, ct, userID)
}

func (s *Server) doJSON(ctx context.Context, method, targetURL string, payload any, userID string) (*http.Response, error) {
	return s.doJSONWithClient(ctx, s.client, method, targetURL, payload, userID)
}

func (s *Server) postHostJSON(ctx context.Context, path string, body any, userID string, result any) error {
	return s.postHostJSONWithClient(ctx, s.client, path, body, userID, result)
}

func (s *Server) postHostJSONWithClient(ctx context.Context, client *http.Client, path string, body any, userID string, result any) error {
	resp, err := s.doJSONWithClient(ctx, client, http.MethodPost, s.hostPath(path), body, userID)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return hostResponseError(resp.Status, data)
	}
	if result != nil {
		return json.NewDecoder(resp.Body).Decode(result)
	}
	return nil
}

func hostResponseError(status string, data []byte) error {
	message := strings.TrimSpace(string(data))
	var payload struct {
		Detail string `json:"detail"`
		Error  string `json:"error"`
		Err    string `json:"err"`
	}
	if len(data) > 0 && json.Unmarshal(data, &payload) == nil {
		switch {
		case strings.TrimSpace(payload.Detail) != "":
			message = payload.Detail
		case strings.TrimSpace(payload.Error) != "":
			message = payload.Error
		case strings.TrimSpace(payload.Err) != "":
			message = payload.Err
		}
	}
	if message == "" {
		message = status
	}
	return errors.New(SanitizePublicText(message))
}

func (s *Server) fetchHostCapabilities(ctx context.Context, userID string) (HostCapabilities, error) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/capabilities"), nil, userID)
	if err != nil {
		return HostCapabilities{}, fmt.Errorf("读取主项目能力失败: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return HostCapabilities{}, hostResponseError(resp.Status, body)
	}
	var caps HostCapabilities
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&caps); err != nil {
		return HostCapabilities{}, err
	}
	return caps, nil
}

func (s *Server) requireAria2Downloader(ctx context.Context, userID string) error {
	caps, err := s.fetchHostCapabilities(ctx, userID)
	if err != nil {
		return err
	}
	if caps.ArtifactDownloader != "aria2" || !caps.Capabilities["artifactDownload"] {
		return errors.New("主项目未启用必需的 Aria2 下载能力，无法准备模型与运行归档")
	}
	return nil
}

func (s *Server) requireSelectiveProcessStop(ctx context.Context, userID string) error {
	caps, err := s.fetchHostCapabilities(ctx, userID)
	if err != nil {
		return err
	}
	if !caps.Capabilities["selectiveProcessStop"] {
		return errors.New("主项目不支持选择性停止冲突模型（selectiveProcessStop），拒绝执行全局停服，请先升级 AI Pod")
	}
	return nil
}

func (s *Server) fetchHostDevices(ctx context.Context, userID string) ([]HostDevice, error) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices"), nil, userID)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, hostResponseError(resp.Status, body)
	}
	var hosts []HostDevice
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&hosts); err != nil {
		return nil, err
	}
	return hosts, nil
}

func (s *Server) fetchDeviceResources(ctx context.Context, serialNumber, userID string) (HostDeviceResources, error) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serialNumber)+"/resources"), nil, userID)
	if err != nil {
		return HostDeviceResources{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return HostDeviceResources{}, hostResponseError(resp.Status, body)
	}
	var res HostDeviceResources
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&res); err != nil {
		return HostDeviceResources{}, err
	}
	if res.MemoryTotalBytes <= 0 && res.VRAMTotalBytes > 0 {
		res.MemoryTotalBytes = res.VRAMTotalBytes
		res.MemoryUsedBytes = res.VRAMUsedBytes
		res.MemoryAvailableBytes = res.VRAMAvailableBytes
	}
	return res, nil
}

func (s *Server) fetchProcessStatus(ctx context.Context, serialNumber, userID string) (HostProcessStatus, error) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serialNumber)+"/process/status"), nil, userID)
	if err != nil {
		return HostProcessStatus{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return HostProcessStatus{}, hostResponseError(resp.Status, body)
	}
	var status HostProcessStatus
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&status); err != nil {
		return HostProcessStatus{}, err
	}
	return status, nil
}

func (s *Server) fetchServiceResources(ctx context.Context, serialNumber, userID string) (HostServiceResourceUsage, error) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serialNumber)+"/services/"+url.PathEscape(config.DefaultServiceName)+"/resources"), nil, userID)
	if err != nil {
		return HostServiceResourceUsage{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return HostServiceResourceUsage{}, fmt.Errorf("service resources returned %s", resp.Status)
	}
	var usage HostServiceResourceUsage
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&usage); err != nil {
		return HostServiceResourceUsage{}, err
	}
	return usage, nil
}

// deviceRuntimeImageReady checks both OCI manifest digest (imageId) and classic
// Docker config digest (configImageId) to support both JP6 overlay2 and JP7/Thor
// containerd image stores without falsely failing exact-identity preflight.
func (s *Server) deviceRuntimeImageReady(ctx context.Context, serial, userID string, rt RuntimeTargetManifest) (ready bool, matchedID string, err error) {
	candidates := []string{}
	if rt.ImageID != "" {
		candidates = append(candidates, rt.ImageID)
	}
	if rt.ConfigImageID != "" && !strings.EqualFold(rt.ConfigImageID, rt.ImageID) {
		candidates = append(candidates, rt.ConfigImageID)
	}
	if len(candidates) == 0 {
		return false, "", nil
	}
	for _, cid := range candidates {
		resp, reqErr := s.doJSON(ctx, http.MethodPost, s.hostPath("/devices/"+url.PathEscape(serial)+"/images/status"), ImageStatusRequest{
			Image:   rt.Image,
			ImageID: cid,
		}, userID)
		if reqErr != nil {
			return false, "", reqErr
		}
		var status ImageStatusResponse
		code := resp.StatusCode
		_ = json.NewDecoder(io.LimitReader(resp.Body, 16<<10)).Decode(&status)
		resp.Body.Close()
		if code == http.StatusNotFound {
			// Legacy Agent 404 falls back to archive download & image load path.
			return false, "", nil
		}
		if code != http.StatusOK {
			continue
		}
		if status.Ready && (strings.EqualFold(status.ImageID, cid) || status.ImageID == "") {
			return true, cid, nil
		}
	}
	return false, "", nil
}

func (s *Server) fetchDeviceModelsStatus(ctx context.Context, serial, userID, target, provider string, verify bool) (ArtifactStatus, map[string][]ArtifactFile, int64, error) {
	artifacts, total, err := s.manifests.ModelArtifactsForTarget(target, provider)
	if err != nil {
		return ArtifactStatus{}, nil, 0, err
	}
	var status ArtifactStatus
	client := s.client
	if verify {
		client = s.deploymentClient
	}
	err = s.postHostJSONWithClient(ctx, client, "/devices/"+url.PathEscape(serial)+"/artifacts/status", ArtifactStatusRequest{
		Namespace:    config.DeviceModelNamespace,
		Artifacts:    artifacts,
		VerifySHA256: verify,
	}, userID, &status)
	return status, artifacts, total, err
}

func missingBytesFromStatus(status ArtifactStatus, artifacts map[string][]ArtifactFile, total int64) int64 {
	if status.Ready {
		return 0
	}
	if len(status.Missing) == 0 {
		return total
	}
	missing := map[string]bool{}
	for _, m := range status.Missing {
		missing[filepath.ToSlash(filepath.Clean(m))] = true
	}
	var sum int64
	for role, files := range artifacts {
		for _, f := range files {
			if missing[role+"/"+f.Name] || missing[f.Name] {
				sum += f.Size
			}
		}
	}
	if sum == 0 {
		return total
	}
	return sum
}

func networkDriveRoot(userID string) (string, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" || userID == "." || userID == ".." || filepath.Base(userID) != userID || strings.ContainsAny(userID, "/\\\x00\r\n") {
		return "", errors.New("用户身份无效")
	}
	base := strings.TrimSpace(os.Getenv("QWEN35_9B_NETWORK_DRIVE_BASE"))
	if base == "" {
		base = "/lzcapp/run/mnt/home"
	}
	rel := strings.Trim(config.NetworkDriveDisplay, "/")
	return filepath.Join(base, rel), nil
}

func (s *Server) prepareAria2Artifacts(ctx context.Context, userID string, artifacts map[string][]ArtifactFile, onProgress func(ArtifactProgress)) error {
	if len(artifacts) == 0 {
		return nil
	}
	if err := s.requireAria2Downloader(ctx, userID); err != nil {
		return err
	}
	req := ArtifactPrepareRequest{
		Namespace:    config.ArtifactNamespace,
		StoragePath:  strings.Trim(config.NetworkDriveDisplay, "/"),
		Artifacts:    artifacts,
		VerifySHA256: true,
	}
	resp, err := s.doJSON(ctx, http.MethodPost, s.hostPath("/artifacts/prepare"), req, userID)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return hostResponseError(resp.Status, body)
	}
	var state ArtifactPrepareResponse
	if err := json.NewDecoder(resp.Body).Decode(&state); err != nil {
		return err
	}

	for {
		if onProgress != nil {
			p := ArtifactProgress{
				TotalFiles:          state.TotalFiles,
				CompletedFiles:      state.CompletedFiles,
				CurrentFile:         state.CurrentFile,
				CurrentFileIndex:    state.CurrentFileIndex,
				Progress:            state.Progress,
				Phase:               state.Phase,
				CurrentBytes:        state.CurrentBytes,
				TotalBytes:          state.TotalBytes,
				SpeedBytesPerSecond: state.SpeedBytesPerSecond,
				ETASeconds:          state.ETASeconds,
				Estimated:           state.Estimated,
				Message:             SanitizePublicText(state.Message),
				Files:               append([]ArtifactFileProgress(nil), state.Files...),
			}
			if p.Phase == "" {
				p.Phase = "downloading"
			}
			onProgress(p)
		}
		if state.Status == "succeeded" {
			return nil
		}
		if state.Status == "failed" || state.Status == "canceled" {
			msg := strings.TrimSpace(state.Error)
			if msg == "" {
				msg = strings.TrimSpace(state.Message)
			}
			if msg == "" {
				msg = "下载或校验模型制品失败"
			}
			if state.Aria2Log != "" {
				msg += "\n" + SanitizePublicText(state.Aria2Log)
			}
			return errors.New(SanitizePublicText(msg))
		}
		if strings.TrimSpace(state.JobID) == "" {
			return errors.New("主项目未返回下载任务 ID")
		}

		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			cancelCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			r, _ := s.doJSON(cancelCtx, http.MethodPost, s.hostPath("/artifacts/prepare/"+url.PathEscape(state.JobID)+"/cancel"), nil, userID)
			if r != nil {
				r.Body.Close()
			}
			cancel()
			return ctx.Err()
		case <-timer.C:
		}

		pollResp, pollErr := s.doJSON(ctx, http.MethodGet, s.hostPath("/artifacts/prepare/"+url.PathEscape(state.JobID)), nil, userID)
		if pollErr != nil {
			return pollErr
		}
		var next ArtifactPrepareResponse
		code := pollResp.StatusCode
		if code != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(pollResp.Body, 1<<20))
			pollResp.Body.Close()
			return hostResponseError(pollResp.Status, body)
		}
		err = json.NewDecoder(pollResp.Body).Decode(&next)
		pollResp.Body.Close()
		if err != nil {
			return err
		}
		state = next
	}
}

type uploadProgressReader struct {
	reader     io.Reader
	read       int64
	total      int64
	lastEmit   time.Time
	onProgress func(int64, int64)
}

func (r *uploadProgressReader) Read(buf []byte) (int, error) {
	n, err := r.reader.Read(buf)
	r.read += int64(n)
	if r.onProgress != nil && (err != nil || r.read == r.total || time.Since(r.lastEmit) >= time.Second) {
		r.onProgress(r.read, r.total)
		r.lastEmit = time.Now()
	}
	return n, err
}

func (s *Server) uploadModelArtifact(ctx context.Context, serialNumber, userID, driveRoot, role string, file ArtifactFile, onProgress func(int64, int64)) error {
	cleanRoot := filepath.Clean(driveRoot)
	wantRoot, err := networkDriveRoot(userID)
	if err != nil {
		return err
	}
	if cleanRoot != filepath.Clean(wantRoot) || filepath.Base(role) != role || filepath.Base(file.Name) != file.Name {
		return errors.New("模型制品路径无效")
	}
	filePath := filepath.Join(cleanRoot, role, file.Name)
	input, err := os.Open(filePath)
	if err != nil {
		// Fallback for flat layout if a previous host version wrote without role subdir.
		altPath := filepath.Join(cleanRoot, file.Name)
		altInput, altErr := os.Open(altPath)
		if altErr != nil {
			return fmt.Errorf("打开网盘模型文件失败: %w", err)
		}
		input = altInput
	}
	defer input.Close()

	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)
	ct := mw.FormDataContentType()
	prodDone := make(chan error, 1)

	go func() {
		fail := func(e error) {
			_ = pw.CloseWithError(e)
			prodDone <- e
		}
		for _, field := range []struct{ k, v string }{
			{"namespace", config.DeviceModelNamespace},
			{"role", role},
			{"name", file.Name},
			{"size", strconv.FormatInt(file.Size, 10)},
			{"sha256", file.SHA256},
		} {
			if err := mw.WriteField(field.k, field.v); err != nil {
				fail(err)
				return
			}
		}
		part, err := mw.CreateFormFile("file", file.Name)
		if err != nil {
			fail(err)
			return
		}
		tracked := &uploadProgressReader{reader: io.LimitReader(input, file.Size+1), total: file.Size, onProgress: onProgress}
		if _, err := io.Copy(part, tracked); err != nil {
			fail(err)
			return
		}
		if err := mw.Close(); err != nil {
			fail(err)
			return
		}
		if err := pw.Close(); err != nil {
			prodDone <- err
			return
		}
		prodDone <- nil
	}()

	resp, err := s.newRequestWithClient(ctx, s.deploymentClient, http.MethodPost,
		s.hostPath("/devices/"+url.PathEscape(serialNumber)+"/artifacts/upload"), pr, ct, userID)
	if err != nil {
		_ = pr.CloseWithError(err)
		<-prodDone
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		_ = pr.Close()
		<-prodDone
		return hostResponseError(resp.Status, body)
	}
	_ , _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	return <-prodDone
}

func (s *Server) loadRuntimeImage(ctx context.Context, serial, userID string, rt RuntimeTargetManifest, onProgress func(ArtifactProgress)) error {
	req := ImageLoadRequest{
		StoragePath:       strings.Trim(config.NetworkDriveDisplay, "/"),
		Image:             rt.Image,
		ImageID:           rt.ImageID,
		Name:              rt.Name,
		Size:              rt.Size,
		SHA256:            strings.ToLower(rt.SHA256),
		RequiredDiskBytes: rt.Size + rt.UnpackedSize + (4 << 30),
		Async:             true,
	}
	var state ImageLoadResponse
	if err := s.postHostJSONWithClient(ctx, s.deploymentClient, "/devices/"+url.PathEscape(serial)+"/images/load", req, userID, &state); err != nil {
		return err
	}
	for {
		if onProgress != nil {
			phase := state.Phase
			if phase == "" {
				phase = "transferring"
			}
			if state.Loaded || state.Reused || state.Status == "succeeded" {
				phase = "loading"
			}
			onProgress(ArtifactProgress{
				Phase:               phase,
				Progress:            state.Progress,
				CurrentBytes:        state.CurrentBytes,
				TotalBytes:          state.TotalBytes,
				SpeedBytesPerSecond: state.SpeedBytesPerSecond,
				ETASeconds:          state.ETASeconds,
				Estimated:           state.Estimated || phase == "loading",
				CurrentFile:         rt.Name,
				TotalFiles:          1,
				Message:             SanitizePublicText(state.Message),
				ImageLoaded:         state.Loaded || state.Reused || state.Status == "succeeded",
			})
		}
		if state.Loaded || state.Reused || state.Status == "succeeded" {
			return nil
		}
		if state.Status == "failed" || state.Status == "canceled" {
			msg := strings.TrimSpace(state.Error)
			if msg == "" {
				msg = strings.TrimSpace(state.Message)
			}
			if msg == "" {
				msg = "导入运行环境镜像失败"
			}
			return errors.New(SanitizePublicText(msg))
		}
		if strings.TrimSpace(state.JobID) == "" {
			return errors.New("主项目未返回镜像导入任务 ID")
		}

		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			cancelCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			r, _ := s.doJSON(cancelCtx, http.MethodPost, s.hostPath("/devices/"+url.PathEscape(serial)+"/images/load/"+url.PathEscape(state.JobID)+"/cancel"), nil, userID)
			if r != nil {
				r.Body.Close()
			}
			cancel()
			return ctx.Err()
		case <-timer.C:
		}

		resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serial)+"/images/load/"+url.PathEscape(state.JobID)), nil, userID)
		if err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			resp.Body.Close()
			return hostResponseError(resp.Status, body)
		}
		err = json.NewDecoder(resp.Body).Decode(&state)
		resp.Body.Close()
		if err != nil {
			return err
		}
	}
}

func (s *Server) restoreWarmCache(ctx context.Context, serial, userID string, cache RuntimeCacheEntry) error {
	var resp CacheRestoreResponse
	err := s.postHostJSONWithClient(ctx, s.deploymentClient, "/devices/"+url.PathEscape(serial)+"/cache/restore", CacheRestoreRequest{
		StoragePath:        strings.Trim(config.NetworkDriveDisplay, "/"),
		ServiceID:          config.DefaultServiceName,
		Name:               cache.Name,
		Size:               cache.Size,
		SHA256:             strings.ToLower(cache.SHA256),
		Fingerprint:        strings.ToLower(cache.Fingerprint),
		RuntimeImageSHA256: strings.ToLower(cache.RuntimeImageSHA256),
	}, userID, &resp)
	if err != nil {
		return err
	}
	if !resp.Restored && !resp.Reused {
		return errors.New("主项目未确认预热缓存已恢复")
	}
	return nil
}

// prepareTargetArtifacts runs BEFORE any existing model service is stopped (§7.1).
// Existing services keep serving real requests throughout download, verification,
// LAN upload, image load, and cache restore.
func (s *Server) prepareTargetArtifacts(ctx context.Context, record DeploymentRecord, target string, settings deploymentSettings) (string, string, error) {
	rt, rtFile, err := s.manifests.RuntimeImageArtifactForTarget(target, settings.ModelProvider)
	if err != nil {
		return "", "", err
	}

	modelStatus, artifacts, totalModelBytes, err := s.fetchDeviceModelsStatus(ctx, record.SerialNumber, record.UserID, target, settings.ModelProvider, true)
	if err != nil {
		return "", "", fmt.Errorf("检查设备模型文件失败: %w", err)
	}
	imageReady, _, err := s.deviceRuntimeImageReady(ctx, record.SerialNumber, record.UserID, rt)
	if err != nil {
		return "", "", fmt.Errorf("检查设备运行镜像失败: %w", err)
	}

	matchedCache, cacheFile, err := s.manifests.MatchingWarmCache(target, settings.ModelProvider, rt, settings.ContextLength)
	if err != nil {
		return "", "", err
	}
	cacheDir := BuildCacheDirectoryName(target, rt, matchedCache, settings)

	// Fast read-only hit skips lock contention on other targets/downloads (§7.2).
	if modelStatus.Ready && imageReady && matchedCache == nil {
		return modelStatus.Root, cacheDir, nil
	}

	lockKey := record.UserID + ":" + settings.ModelProvider + ":" + target
	unlock := s.acquireKeyedLock(ctx, lockKey)
	if unlock == nil {
		return "", "", ctx.Err()
	}
	defer unlock()

	// Re-check status after lock in case a concurrent deployment already prepared the artifacts.
	modelStatus, artifacts, totalModelBytes, err = s.fetchDeviceModelsStatus(ctx, record.SerialNumber, record.UserID, target, settings.ModelProvider, true)
	if err != nil {
		return "", "", err
	}
	imageReady, _, err = s.deviceRuntimeImageReady(ctx, record.SerialNumber, record.UserID, rt)
	if err != nil {
		return "", "", err
	}

	prepareMap := map[string][]ArtifactFile{}
	if !modelStatus.Ready {
		for role, files := range artifacts {
			prepareMap[role] = files
		}
	}
	if !imageReady {
		prepareMap["runtime-image"] = []ArtifactFile{rtFile}
	}
	if matchedCache != nil && cacheFile != nil {
		prepareMap["runtime-cache"] = []ArtifactFile{*cacheFile}
	}

	if len(prepareMap) > 0 {
		s.updateRecord(record.ID, func(rec *DeploymentRecord) {
			rec.Stage = "downloading"
			rec.Message = "准备并校验模型文件与运行归档（当前已有服务不受影响）"
			rec.Progress = progressDownloadStart
			rec.PhaseProgress = 0
		})
		if err := s.prepareAria2Artifacts(ctx, record.UserID, prepareMap, func(p ArtifactProgress) {
			s.updateRecord(record.ID, func(rec *DeploymentRecord) {
				stage := "downloading"
				if strings.EqualFold(p.Phase, "verifying") {
					stage = "verifying"
				}
				rec.Stage = stage
				if p.Message != "" {
					rec.Message = p.Message
				}
				rec.PhaseProgress = clampPct(p.Progress)
				if stage == "verifying" {
					rec.Progress = progressVerifyStart + rec.PhaseProgress*progressVerifySpan/100
				} else {
					rec.Progress = progressDownloadStart + rec.PhaseProgress*progressDownloadSpan/100
				}
				rec.CurrentBytes = p.CurrentBytes
				rec.TotalBytes = p.TotalBytes
				rec.SpeedBytesPerSecond = p.SpeedBytesPerSecond
				rec.ETASeconds = p.ETASeconds
				rec.Estimated = p.Estimated
				rec.CurrentFile = p.CurrentFile
				rec.TotalFiles = p.TotalFiles
				rec.CompletedFiles = p.CompletedFiles
				rec.CurrentFileIndex = p.CurrentFileIndex
				rec.Files = p.Files
			})
		}); err != nil {
			return "", "", err
		}
	}

	if !modelStatus.Ready {
		driveRoot, err := networkDriveRoot(record.UserID)
		if err != nil {
			return "", "", err
		}
		missingSet := map[string]bool{}
		for _, m := range modelStatus.Missing {
			missingSet[filepath.ToSlash(filepath.Clean(m))] = true
		}
		roles := make([]string, 0, len(artifacts))
		for r := range artifacts {
			roles = append(roles, r)
		}
		sort.Strings(roles)

		var uploadedBytes int64
		for _, role := range roles {
			for _, f := range artifacts[role] {
				if len(missingSet) > 0 && !missingSet[role+"/"+f.Name] && !missingSet[f.Name] {
					uploadedBytes += f.Size
					continue
				}
				s.updateRecord(record.ID, func(rec *DeploymentRecord) {
					rec.Stage = "transferring"
					rec.Message = "网盘 → 算力舱：传输模型文件 " + f.Name
					rec.CurrentFile = f.Name
					rec.TotalBytes = totalModelBytes
				})
				base := uploadedBytes
				if err := s.uploadModelArtifact(ctx, record.SerialNumber, record.UserID, driveRoot, role, f, func(cur, _ int64) {
					s.updateRecord(record.ID, func(rec *DeploymentRecord) {
						pct := 0
						if totalModelBytes > 0 {
							pct = clampPct(int((base + cur) * 100 / totalModelBytes))
						}
						rec.Stage = "transferring"
						rec.PhaseProgress = pct
						rec.Progress = progressTransferStart + pct*progressTransferSpan/100
						rec.CurrentBytes = base + cur
						rec.TotalBytes = totalModelBytes
					})
				}); err != nil {
					return "", "", fmt.Errorf("传输模型文件 %s 失败: %w", f.Name, err)
				}
				uploadedBytes += f.Size
			}
		}
		modelStatus, _, _, err = s.fetchDeviceModelsStatus(ctx, record.SerialNumber, record.UserID, target, settings.ModelProvider, false)
		if err != nil {
			return "", "", err
		}
		if !modelStatus.Ready {
			return "", "", fmt.Errorf("模型文件传输后仍不完整: %s", strings.Join(modelStatus.Missing, ", "))
		}
	}

	if !imageReady {
		s.updateRecord(record.ID, func(rec *DeploymentRecord) {
			rec.Stage = "loading"
			rec.Message = "导入算力舱运行镜像"
			rec.Progress = progressImageLoadStart
			rec.PhaseProgress = 0
		})
		if err := s.loadRuntimeImage(ctx, record.SerialNumber, record.UserID, rt, func(p ArtifactProgress) {
			s.updateRecord(record.ID, func(rec *DeploymentRecord) {
				rec.Stage = "loading"
				rec.PhaseProgress = clampPct(p.Progress)
				rec.Progress = progressImageLoadStart + rec.PhaseProgress*progressImageLoadSpan/100
				rec.CurrentFile = p.CurrentFile
				rec.Estimated = p.Estimated
				if p.Message != "" {
					rec.Message = p.Message
				}
			})
		}); err != nil {
			return "", "", err
		}
	}

	if matchedCache != nil {
		s.updateRecord(record.ID, func(rec *DeploymentRecord) {
			rec.Stage = "restoring_cache"
			rec.Message = "恢复匹配 profile 的预热编译缓存"
			rec.Progress = progressCacheRestoreStart
			rec.PhaseProgress = 0
		})
		if err := s.restoreWarmCache(ctx, record.SerialNumber, record.UserID, *matchedCache); err != nil {
			return "", "", fmt.Errorf("恢复预热缓存失败: %w", err)
		}
	}

	return modelStatus.Root, cacheDir, nil
}

// switchResourcesBeforeStart executes the strict §7.1 deferred-stop boundary
// after all off-container preparation is finished and Compose is validated.
// Per-device mutex serializes resource switches on the same device.
func (s *Server) switchResourcesBeforeStart(ctx context.Context, record DeploymentRecord, target string, settings deploymentSettings) (memoryAssessment, error) {
	unlock := s.acquireDeviceSwitchLock(ctx, record.SerialNumber)
	if unlock == nil {
		return memoryAssessment{}, ctx.Err()
	}
	defer unlock()

	if err := ctx.Err(); err != nil {
		return memoryAssessment{}, err
	}

	res, err := s.fetchDeviceResources(ctx, record.SerialNumber, record.UserID)
	if err != nil {
		return memoryAssessment{}, fmt.Errorf("启动前复核算力舱资源失败: %w", err)
	}
	procStatus, err := s.fetchProcessStatus(ctx, record.SerialNumber, record.UserID)
	if err != nil {
		return memoryAssessment{}, fmt.Errorf("启动前复核算力舱进程失败: %w", err)
	}

	// Re-check disk remaining after artifacts & image are already prepared (imageReady=true, missingModel=0).
	if res.DiskAvailableKnown {
		rt := s.manifests.Runtimes.Targets[target]
		remainingDiskNeed := ComputeRequiredDiskBytes(true, 0, rt, nil)
		if res.DiskAvailableBytes < remainingDiskNeed {
			return memoryAssessment{}, fmt.Errorf("启动前复核算力舱磁盘空间不足：剩余 %s，启动及编译缓存需至少 %s",
				formatBytesGiB(res.DiskAvailableBytes), formatBytesGiB(remainingDiskNeed))
		}
	}

	assessment := assessMemoryAndProcesses(target, res, procStatus, settings.ContextLength, s.manifests)
	if assessment.totalCapacityInsufficient() {
		return assessment, errors.New(assessment.insufficientMemoryMessage())
	}

	stopIDs := append([]string(nil), assessment.ExistingStopIDs...)
	if len(stopIDs) > 0 && !record.StopExisting && !record.StopProtected {
		return assessment, errors.New("检测到当前已有同类实例运行，请确认停止后重试")
	}

	if assessment.ShortfallBytes > 0 {
		if len(assessment.ProtectedStopIDs) == 0 {
			return assessment, errors.New(assessment.insufficientMemoryMessage())
		}
		approved := map[string]bool{}
		if record.StopProtected {
			for _, id := range record.ProtectedProcessIDs {
				approved[strings.TrimSpace(id)] = true
			}
		}
		for _, pid := range assessment.ProtectedStopIDs {
			if !approved[pid] {
				return assessment, fmt.Errorf("准备完成后检测到未授权的冲突模型服务 (%s)，已保留已准备文件，请重新确认停止后启动或重试", pid)
			}
		}
		stopIDs = append(stopIDs, assessment.ProtectedStopIDs...)
	}

	stopIDs = dedupeAndSort(stopIDs)
	if len(stopIDs) == 0 {
		if assessment.AvailableBytes < assessment.Budget.RequiredBytes {
			return assessment, errors.New(assessment.insufficientMemoryMessage())
		}
		return assessment, nil
	}

	if err := s.requireSelectiveProcessStop(ctx, record.UserID); err != nil {
		return assessment, err
	}

	s.updateRecord(record.ID, func(rec *DeploymentRecord) {
		rec.Stage = "stopping"
		rec.Message = "准备完成，正在停止已确认的冲突模型并等待内存释放"
		rec.Progress = progressStopStart
		rec.PhaseProgress = 0
	})

	if err := s.postHostJSONWithClient(ctx, s.stopClient, "/devices/"+url.PathEscape(record.SerialNumber)+"/process/stop", ProcessStopRequest{PackageIDs: stopIDs}, record.UserID, nil); err != nil {
		return assessment, fmt.Errorf("停止已授权冲突模型失败: %w", err)
	}

	releasedResources, err := s.waitForMemoryRelease(ctx, record.SerialNumber, record.UserID, stopIDs, assessment.Budget.RequiredBytes)
	if err != nil {
		return assessment, err
	}
	assessment.Resources = releasedResources
	assessment.AvailableBytes = releasedResources.MemoryAvailableBytes
	assessment.ProjectedAvailableBytes = releasedResources.MemoryAvailableBytes
	assessment.ShortfallBytes = 0
	return assessment, nil
}

func (s *Server) waitForMemoryRelease(ctx context.Context, serial, userID string, stoppedIDs []string, requiredBytes int64) (HostDeviceResources, error) {
	waitCtx, cancel := context.WithTimeout(ctx, memoryReleaseTimeout)
	defer cancel()

	wantStop := map[string]bool{}
	for _, id := range stoppedIDs {
		wantStop[id] = true
	}

	var lastRes HostDeviceResources
	var lastProc HostProcessStatus
	var lastErr error
	for {
		proc, procErr := s.fetchProcessStatus(waitCtx, serial, userID)
		res, resErr := s.fetchDeviceResources(waitCtx, serial, userID)
		if procErr == nil {
			lastProc = proc
		} else {
			lastErr = procErr
		}
		if resErr == nil {
			lastRes = res
		} else {
			lastErr = resErr
		}

		if procErr == nil && resErr == nil {
			stillRunning := false
			for _, p := range proc.OllamaPs {
				if wantStop[strings.TrimSpace(p.PackageID)] {
					stillRunning = true
					break
				}
			}
			if !stillRunning {
				for sid := range proc.Services {
					if wantStop[strings.TrimSpace(sid)] {
						stillRunning = true
						break
					}
				}
			}
			avail := res.MemoryAvailableBytes
			if res.VRAMAvailableBytes > avail {
				avail = res.VRAMAvailableBytes
			}
			if !stillRunning && avail >= requiredBytes {
				return res, nil
			}
		}

		timer := time.NewTimer(memoryReleasePollInterval)
		select {
		case <-waitCtx.Done():
			timer.Stop()
			if errors.Is(ctx.Err(), context.Canceled) {
				return HostDeviceResources{}, ctx.Err()
			}
			var remaining []string
			for _, p := range lastProc.OllamaPs {
				if wantStop[strings.TrimSpace(p.PackageID)] {
					remaining = append(remaining, p.PackageID)
				}
			}
			for sid := range lastProc.Services {
				if wantStop[strings.TrimSpace(sid)] {
					remaining = append(remaining, sid)
				}
			}
			avail := lastRes.MemoryAvailableBytes
			if lastRes.VRAMAvailableBytes > avail {
				avail = lastRes.VRAMAvailableBytes
			}
			if len(remaining) > 0 {
				return HostDeviceResources{}, fmt.Errorf("等待冲突服务退出超时（残留: %s，最后错误: %v），未启动新推理服务", strings.Join(remaining, ","), lastErr)
			}
			return HostDeviceResources{}, fmt.Errorf("等待统一内存释放超时：当前可用 %s，至少需要 %s（最后错误: %v），未启动新推理服务",
				formatBytesGiB(avail), formatBytesGiB(requiredBytes), lastErr)
		case <-timer.C:
		}
	}
}

// registerServiceOwnership registers service ownership with AI Pod host before
// service stop/deploy so host OwnershipManager admits OwnerPackageID deployments.
// Legacy AI Pod hosts without this route (404) are tolerated cleanly.
func (s *Server) registerServiceOwnership(ctx context.Context, serialNumber, userID string) error {
	resp, err := s.doJSON(ctx, http.MethodPost, s.hostPath("/devices/"+url.PathEscape(serialNumber)+"/services/ownership"), ServiceOwnershipRequest{
		PackageID: config.PackageID,
		ServiceID: config.DefaultServiceName,
	}, userID)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return hostResponseError(resp.Status, body)
	}
	var out ServiceOwnershipResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&out); err != nil {
		return err
	}
	if !out.Registered {
		return errors.New("主项目未确认模型服务归属登记")
	}
	return nil
}

func (s *Server) deployAndWaitHealthy(ctx context.Context, record DeploymentRecord, compose []byte) error {
	startCtx, cancel := context.WithTimeout(ctx, serviceStartupTimeout)
	defer cancel()

	s.updateRecord(record.ID, func(rec *DeploymentRecord) {
		rec.Stage = "starting"
		rec.Message = "提交模型服务并拉起容器"
		rec.Progress = progressServiceStart
		rec.PhaseProgress = 0
		rec.Estimated = true
	})

	if err := s.registerServiceOwnership(startCtx, record.SerialNumber, record.UserID); err != nil {
		return err
	}

	deployReq := DeployServiceRequest{
		ServiceID:      config.DefaultServiceName,
		OwnerPackageID: config.PackageID,
		Compose:        string(compose),
		ForceRecreate:  false,
		DirOverWrite:   true,
	}

	done := make(chan error, 1)
	go func() {
		done <- s.postHostJSONWithClient(startCtx, s.deploymentClient, "/devices/"+url.PathEscape(record.SerialNumber)+"/services/deploy", deployReq, record.UserID, nil)
	}()

	started := time.Now()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

DeployWaitLoop:
	for {
		select {
		case err := <-done:
			if err != nil {
				detail := s.collectDeployFailureDiagnostic(record.SerialNumber, record.UserID, err)
				return errors.New(detail)
			}
			break DeployWaitLoop
		case <-ticker.C:
			s.pollServiceDeployProgress(startCtx, record, started)
		case <-startCtx.Done():
			return startCtx.Err()
		}
	}

	for {
		var h HealthResponse
		err := s.postHostJSON(startCtx, "/devices/"+url.PathEscape(record.SerialNumber)+"/services/health", HealthRequest{
			Port: config.DefaultAPIPort,
			Path: "/v1/models",
		}, record.UserID, &h)
		if err == nil && h.Healthy && responseContainsPublicModel(h.Body) {
			return nil
		}
		if diag := s.checkFatalServiceProgress(startCtx, record.SerialNumber, record.UserID, started); diag != nil {
			return diag
		}
		s.pollServiceDeployProgress(startCtx, record, started)

		timer := time.NewTimer(2 * time.Second)
		select {
		case <-startCtx.Done():
			timer.Stop()
			return errors.New(s.collectDeployFailureDiagnostic(record.SerialNumber, record.UserID, startCtx.Err()))
		case <-timer.C:
		}
	}
}

func responseContainsPublicModel(body string) bool {
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal([]byte(body), &payload) != nil {
		return false
	}
	for _, m := range payload.Data {
		if m.ID == config.PublicModelID {
			return true
		}
	}
	return false
}

func (s *Server) pollServiceDeployProgress(ctx context.Context, record DeploymentRecord, started time.Time) {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(record.SerialNumber)+"/services/progress?serviceId="+url.QueryEscape(config.DefaultServiceName)), nil, record.UserID)
	if err == nil {
		var st HostServiceDeployStatus
		code := resp.StatusCode
		decErr := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&st)
		resp.Body.Close()
		if code == http.StatusOK && decErr == nil {
			stage, msg, pct, phasePct := mapServicePhase(st, time.Since(started))
			s.updateRecord(record.ID, func(rec *DeploymentRecord) {
				rec.Stage = stage
				rec.Message = msg
				rec.Progress = pct
				rec.PhaseProgress = phasePct
				rec.Estimated = st.Estimated || st.AggregateProgress == false
			})
			return
		}
	}
	stage, msg, pct, phasePct := mapElapsedStartupPhase(time.Since(started))
	s.updateRecord(record.ID, func(rec *DeploymentRecord) {
		rec.Stage = stage
		rec.Message = msg
		rec.Progress = pct
		rec.PhaseProgress = phasePct
		rec.Estimated = true
	})
}

func (s *Server) checkFatalServiceProgress(ctx context.Context, serial, userID string, started time.Time) error {
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serial)+"/services/progress?serviceId="+url.QueryEscape(config.DefaultServiceName)), nil, userID)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var st HostServiceDeployStatus
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&st) != nil {
		return nil
	}
	lowerStage := strings.ToLower(strings.TrimSpace(st.Stage))
	lowerStatus := strings.ToLower(strings.TrimSpace(st.Status))
	if (lowerStage == "failed" || lowerStatus == "failed") && st.FinishedAt >= started.Unix()-30 {
		msg := strings.TrimSpace(st.Error)
		if msg == "" {
			msg = strings.TrimSpace(st.Message)
		}
		if msg == "" {
			msg = "容器启动或模型加载失败"
		}
		return errors.New(SanitizePublicText(msg))
	}
	return nil
}

func (s *Server) collectDeployFailureDiagnostic(serial, userID string, fallback error) string {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := s.doJSON(ctx, http.MethodGet, s.hostPath("/devices/"+url.PathEscape(serial)+"/services/progress?serviceId="+url.QueryEscape(config.DefaultServiceName)), nil, userID)
	if err == nil {
		var st HostServiceDeployStatus
		code := resp.StatusCode
		decErr := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&st)
		resp.Body.Close()
		if code == http.StatusOK && decErr == nil && strings.TrimSpace(st.Error) != "" {
			return SanitizePublicText(fmt.Sprintf("%s (state=%s health=%s)", st.Error, st.State, st.Health))
		}
	}
	if fallback != nil {
		return SanitizePublicText(fallback.Error())
	}
	return "模型服务启动失败"
}

func mapServicePhase(st HostServiceDeployStatus, elapsed time.Duration) (stage, msg string, pct, phasePct int) {
	lower := strings.ToLower(st.Stage + " " + st.Message + " " + st.CurrentLayer)
	switch {
	case strings.Contains(lower, "weight") || strings.Contains(lower, "safetensor") || st.Stage == "loading_weights":
		return "loading_weights", "加载模型权重", progressWeightsStart + clampPct(st.PhaseProgress)*progressWeightsSpan/100, clampPct(st.PhaseProgress)
	case strings.Contains(lower, "kv") || st.Stage == "initializing_kv":
		return "initializing_kv", "初始化 KV Cache", progressKVStart + clampPct(st.PhaseProgress)*progressKVSpan/100, clampPct(st.PhaseProgress)
	case strings.Contains(lower, "compile") || strings.Contains(lower, "triton") || strings.Contains(lower, "graph") || st.Stage == "compiling_kernels":
		return "compiling_kernels", "编译推理内核与图捕获", progressKernelStart + clampPct(st.PhaseProgress)*progressKernelSpan/100, clampPct(st.PhaseProgress)
	case st.Stage == "health_check" || st.Stage == "waiting" || strings.Contains(lower, "health"):
		return "health_check", "验证模型推理接口就绪", progressHealthStart, 50
	default:
		return mapElapsedStartupPhase(elapsed)
	}
}

func mapElapsedStartupPhase(elapsed time.Duration) (string, string, int, int) {
	sec := elapsed.Seconds()
	switch {
	case sec < 20:
		p := clampPct(int(sec / 20 * 100))
		return "starting", "启动模型容器", progressServiceStart + p*progressServiceStartSpan/100, p
	case sec < 90:
		p := clampPct(int((sec - 20) / 70 * 100))
		return "loading_weights", "加载模型权重", progressWeightsStart + p*progressWeightsSpan/100, p
	case sec < 150:
		p := clampPct(int((sec - 90) / 60 * 100))
		return "initializing_kv", "初始化 KV Cache", progressKVStart + p*progressKVSpan/100, p
	case sec < 240:
		p := clampPct(int((sec - 150) / 90 * 100))
		return "compiling_kernels", "编译 CUDA / Triton 内核", progressKernelStart + p*progressKernelSpan/100, p
	default:
		return "health_check", "等待模型服务通过健康检查", progressHealthStart, 50
	}
}

func clampPct(v int) int {
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return v
}

func dedupeAndSort(vals []string) []string {
	set := map[string]struct{}{}
	var out []string
	for _, v := range vals {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if _, ok := set[v]; !ok {
			set[v] = struct{}{}
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

func validLANIP(ip string) bool {
	return net.ParseIP(strings.TrimSpace(ip)) != nil
}
