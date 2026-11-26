package lpk

import (
	"testing"

	"qwen35-9b-deployer/internal/config"
)

func TestClassifyDeviceTarget(t *testing.T) {
	tests := []struct {
		name    string
		dev     HostDevice
		want    string
		wantErr bool
	}{
		{
			name: "thor jp7.2",
			dev:  HostDevice{SerialNumber: "s1", Model: "NVIDIA Jetson AGX Thor Developer Kit", L4TVersion: "39.2.0"},
			want: config.TargetThor,
		},
		{
			name: "thor jp7.1 t4000",
			dev:  HostDevice{SerialNumber: "s2", Model: "AGX Thor T4000", L4TVersion: "38.4.0"},
			want: config.TargetThor,
		},
		{
			name:    "thor unsupported l4t rejected",
			dev:     HostDevice{SerialNumber: "s3", Model: "NVIDIA Jetson AGX Thor", L4TVersion: "36.4.3"},
			wantErr: true,
		},
		{
			name: "orin jp7.2.1 supported",
			dev:  HostDevice{SerialNumber: "s4", Model: "NVIDIA Jetson AGX Orin Developer Kit", L4TVersion: "39.2.1"},
			want: config.TargetOrin,
		},
		{
			name:    "orin jp6 rejected",
			dev:     HostDevice{SerialNumber: "s4b", Model: "NVIDIA Jetson AGX Orin", L4TVersion: "36.4.3"},
			wantErr: true,
		},
		{
			name:    "orin nano rejected",
			dev:     HostDevice{SerialNumber: "s4c", Model: "NVIDIA Jetson Orin Nano", L4TVersion: "39.2.1"},
			wantErr: true,
		},
		{
			name:    "unknown x86 rejected",
			dev:     HostDevice{SerialNumber: "s5", Model: "Generic x86_64"},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ClassifyDeviceTarget(tc.dev)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got target %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMemoryBudgetAndCapacity(t *testing.T) {
	// Thor 64GB T4000 (~61 GiB usable) at default 128K must fit comfortably.
	t4kBudget := ComputeMemoryBudget(config.TargetThor, 61<<30, 131072, nil)
	if t4kBudget.TotalCapacityTooSmall {
		t.Fatalf("expected Thor T4000 64GB 128K to fit, got error: %s", t4kBudget.CapacityError)
	}
	if t4kBudget.RequiredBytes > 24<<30 {
		t.Fatalf("expected Qwen3.5-9B 128K required <= 24 GiB, got %d", t4kBudget.RequiredBytes)
	}

	// 32GB device must fail total capacity check without stop-all fallback.
	smallBudget := ComputeMemoryBudget(config.TargetThor, 31<<30, 131072, nil)
	if !smallBudget.TotalCapacityTooSmall {
		t.Fatal("expected 31 GiB device to be rejected for total capacity")
	}

	// Thor 122GB at 256K fits easily.
	thorBudget := ComputeMemoryBudget(config.TargetThor, 122<<30, 262144, nil)
	if thorBudget.TotalCapacityTooSmall {
		t.Fatalf("expected Thor 122GB 256K to fit, got: %s", thorBudget.CapacityError)
	}
}
