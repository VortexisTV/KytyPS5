#ifndef KYTY_COMMON_PERFSTATS_H_
#define KYTY_COMMON_PERFSTATS_H_

#include "common/common.h"

#include <array>
#include <cstdint>
#include <string>

// Per-interval performance statistics for finding the part of the emulator a slow scene loads.
// Collection is off unless the environment variable KYTY_PERF_STATS=1 is set.
// When on, a summary is printed after every second of guest frames and each interval is 
// appended as one row to _PerfStats.csv.

namespace PerfStats {

// Wall-clock time with an occurrence count and the longest single occurrence. Spans nest: the draw
// phases are part of Draw, and Draw, GpuWait and FlipWait are part of GpuThreadBusy.
enum class SpanId : uint8_t {
	GpuThreadBusy,      // emulated GPU thread executing submissions or queued commands
	GpuThreadIdle,      // emulated GPU thread waiting for work
	GpuThreadBlocked,   // every queued submission suspended on a wait packet
	GpuThreadCommands,  // commands other threads ran on the emulated GPU thread
	GameWaitGpuIdle,    // guest thread blocked at a suspend point until the GPU thread drains
	GameWaitGpuCommand, // guest thread blocked on a synchronous GPU-thread command
	Draw,
	DrawTargets,        // render-target resolution and attachment acquisition
	DrawShaders,        // shader program lookup, including resource materialization
	DrawBindings,       // textures, samplers, storage buffers and DMA preparation
	DrawVertexIndex,    // vertex and index buffer acquisition
	DrawPipeline,       // graphics pipeline lookup or creation
	DrawRecord,         // descriptor commit, dynamic state and draw recording
	Dispatch,
	BdaPrepare,         // synchronizing every cached buffer before a DMA draw or dispatch
	BufferDownload,     // GPU-to-guest buffer readback, including the GPU drain
	QueueSubmit,
	GpuWait,            // emulated GPU thread blocked on host GPU completion
	FlipWait,           // emulated GPU thread blocked until a flip is presented
	Present,            // presentation, including FIFO vsync blocking
	PageFault,          // guest access faults on GPU-tracked pages
	GarbageCollect,
	ShaderCompileSync,  // translation and compilation on the emulated GPU thread
	ShaderCompileAsync, // translation and compilation on worker threads
	PipelineCreateSync,
	PipelineCreateAsync,
	PipelineCacheSave,
	Count
};

enum class CounterId : uint8_t {
	DrawsSkippedShader,
	DrawsSkippedPipeline,
	BdaBuffersVisited,
	BufferUploads,
	BufferUploadBytes,
	StreamUploads, // small CPU-modified bindings copied into the stream buffer
	StreamUploadBytes,
	BufferCreates,
	BufferDownloadBytes,
	BuffersEvicted,
	ImageCreates,
	ImageUploads,
	ImageUploadBytes,
	ImagesEvicted,
	WriteFaults,
	ReadFaults,
	ShadersCompiled,
	PipelinesCreated,
	Count
};

// Most recent value; gauges are not reset between intervals.
enum class GaugeId : uint8_t {
	GpuMemoryMb,
	BufferGcTriggerMb,
	TextureGcTriggerMb,
	CachedBuffers,
	Count
};

inline constexpr size_t SpanCount    = static_cast<size_t>(SpanId::Count);
inline constexpr size_t CounterCount = static_cast<size_t>(CounterId::Count);
inline constexpr size_t GaugeCount   = static_cast<size_t>(GaugeId::Count);

namespace Detail {

// Written during initialization, before emulation threads start.
inline bool g_enabled = false;

void RecordSpan(SpanId id, uint64_t ticks) noexcept;
void AddCounter(CounterId id, uint64_t value) noexcept;
void SetGauge(GaugeId id, uint64_t value) noexcept;

} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled;
}

// Common::Timer::QueryPerformanceCounter ticks.
[[nodiscard]] uint64_t Now() noexcept;
[[nodiscard]] uint64_t TicksPerSecond() noexcept;

inline void Add(CounterId id, uint64_t value = 1) noexcept {
	if (Enabled()) {
		Detail::AddCounter(id, value);
	}
}

inline void Set(GaugeId id, uint64_t value) noexcept {
	if (Enabled()) {
		Detail::SetGauge(id, value);
	}
}

// Records the time from construction to Stop() or destruction, whichever comes first. A false
// condition makes the span inert.
class Span final {
public:
	explicit Span(SpanId id, bool condition = true) noexcept
	    : m_id(id), m_active(condition && Enabled()), m_start(m_active ? Now() : 0) {}
	~Span() { Stop(); }

	void Stop() noexcept {
		if (m_active) {
			m_active = false;
			Detail::RecordSpan(m_id, Now() - m_start);
		}
	}

	KYTY_CLASS_NO_COPY(Span);

private:
	SpanId   m_id;
	bool     m_active;
	uint64_t m_start;
};

struct SpanTotal {
	uint64_t ticks     = 0;
	uint64_t count     = 0;
	uint64_t max_ticks = 0;
};

struct Snapshot {
	double                             elapsed_seconds = 0.0; // since collection started
	uint64_t                           interval_ticks  = 0;
	uint64_t                           frames          = 0;
	uint64_t                           max_frame_ticks = 0;
	uint64_t                           cpu_ns          = 0; // process CPU time in the interval
	std::array<SpanTotal, SpanCount>   spans {};
	std::array<uint64_t, CounterCount> counters {};
	std::array<uint64_t, GaugeCount>   gauges {};

	[[nodiscard]] const SpanTotal& Get(SpanId id) const { return spans[static_cast<size_t>(id)]; }
	[[nodiscard]] uint64_t Get(CounterId id) const { return counters[static_cast<size_t>(id)]; }
	[[nodiscard]] uint64_t Get(GaugeId id) const { return gauges[static_cast<size_t>(id)]; }
};

// Returns the spans and counters accumulated since the previous call and resets them. Gauges keep
// their values; the frame and interval fields are left for the caller.
[[nodiscard]] Snapshot CollectAndReset() noexcept;

[[nodiscard]] std::string FormatSummary(const Snapshot& snapshot, uint64_t ticks_per_second);
[[nodiscard]] std::string FormatCsvHeader();
[[nodiscard]] std::string FormatCsvRow(const Snapshot& snapshot, uint64_t ticks_per_second);

// Present thread, after a guest flip is shown. Closes an interval once a second has passed.
void OnGuestFrame() noexcept;

void Initialize();
void Shutdown();
void EmergencyShutdown();

struct Lifecycle {
	static constexpr const char* name               = "PerfStats";
	static constexpr auto        initialize         = PerfStats::Initialize;
	static constexpr auto        shutdown           = PerfStats::Shutdown;
	static constexpr auto        emergency_shutdown = PerfStats::EmergencyShutdown;
};

} // namespace PerfStats

#endif /* KYTY_COMMON_PERFSTATS_H_ */
