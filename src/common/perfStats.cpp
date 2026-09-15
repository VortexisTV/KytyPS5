#include "common/perfStats.h"

#include "common/file.h"
#include "common/timer.h"

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include <iterator>
#include <mutex>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <cstdlib>
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace PerfStats {

namespace {

constexpr const char* CsvFileName = "_PerfStats.csv";

constexpr std::array<std::string_view, SpanCount> SpanNames = {
    "gpu_thread_busy",       "gpu_thread_idle",      "gpu_thread_blocked",
    "gpu_thread_commands",   "game_wait_gpu_idle",   "game_wait_gpu_command",
    "draw",                  "draw_targets",         "draw_shaders",
    "draw_bindings",         "draw_vertex_index",    "draw_pipeline",
    "draw_record",           "dispatch",             "bda_prepare",
    "buffer_download",       "queue_submit",         "gpu_wait",
    "flip_wait",             "present",              "page_fault",
    "garbage_collect",       "shader_compile_sync",  "shader_compile_async",
    "pipeline_create_sync",  "pipeline_create_async", "pipeline_cache_save",
};

constexpr std::array<std::string_view, CounterCount> CounterNames = {
    "draws_skipped_shader", "draws_skipped_pipeline", "bda_buffers_visited",
    "buffer_uploads",       "buffer_upload_bytes",    "stream_uploads",
    "stream_upload_bytes",  "buffer_creates",         "buffer_download_bytes",
    "buffers_evicted",      "image_creates",          "image_uploads",
    "image_upload_bytes",   "images_evicted",         "write_faults",
    "read_faults",          "shaders_compiled",       "pipelines_created",
};

constexpr std::array<std::string_view, GaugeCount> GaugeNames = {
    "gpu_memory_mb",
    "buffer_gc_trigger_mb",
    "texture_gc_trigger_mb",
    "cached_buffers",
};

template <size_t N>
constexpr bool AllNamed(const std::array<std::string_view, N>& names) {
	return std::ranges::none_of(names, [](std::string_view name) { return name.empty(); });
}

// A name missing from a table leaves an empty entry at its end.
static_assert(AllNamed(SpanNames) && AllNamed(CounterNames) && AllNamed(GaugeNames));

struct AtomicSpan {
	std::atomic<uint64_t> ticks {0};
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> max_ticks {0};
};

std::array<AtomicSpan, SpanCount>               g_spans;
std::array<std::atomic<uint64_t>, CounterCount> g_counters {};
std::array<std::atomic<uint64_t>, GaugeCount>   g_gauges {};

// Interval bookkeeping, touched by initialization, the present thread and shutdown.
struct Report {
	std::mutex   mutex;
	uint64_t     ticks_per_second = 0;
	uint64_t     start            = 0;
	uint64_t     interval_start   = 0;
	uint64_t     last_frame       = 0;
	uint64_t     max_frame        = 0;
	uint64_t     frames           = 0;
	uint64_t     cpu_start_ns     = 0;
	Common::File csv;
	bool         csv_open = false;
};

Report& GetReport() {
	static Report report;
	return report;
}

uint64_t ProcessCpuNs() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME creation {};
	FILETIME exited {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetProcessTimes(GetCurrentProcess(), &creation, &exited, &kernel, &user) == 0) {
		return 0;
	}
	const auto to_ns = [](const FILETIME& time) {
		return ((static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime) * 100u;
	};
	return to_ns(kernel) + to_ns(user);
#else
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0;
	}
	const auto to_ns = [](const timeval& time) {
		return static_cast<uint64_t>(time.tv_sec) * 1000000000u +
		       static_cast<uint64_t>(time.tv_usec) * 1000u;
	};
	return to_ns(usage.ru_utime) + to_ns(usage.ru_stime);
#endif
}

bool EnvironmentRequestsStats() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	char       buffer[8] {};
	const auto length = GetEnvironmentVariableA("KYTY_PERF_STATS", buffer, sizeof(buffer));
	if (length == 0 || length >= sizeof(buffer)) {
		return false;
	}
	const std::string_view value {buffer, length};
#else
	const char* variable = std::getenv("KYTY_PERF_STATS");
	if (variable == nullptr) {
		return false;
	}
	const std::string_view value {variable};
#endif
	return value == "1" || value == "true";
}

double Milliseconds(uint64_t ticks, uint64_t ticks_per_second) {
	return ticks_per_second == 0
	           ? 0.0
	           : static_cast<double>(ticks) * 1000.0 / static_cast<double>(ticks_per_second);
}

double Megabytes(uint64_t bytes) {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double IntervalSeconds(const Snapshot& snapshot, uint64_t ticks_per_second) {
	return Milliseconds(snapshot.interval_ticks, ticks_per_second) / 1000.0;
}

double FramesPerSecond(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto seconds = IntervalSeconds(snapshot, ticks_per_second);
	return seconds > 0.0 ? static_cast<double>(snapshot.frames) / seconds : 0.0;
}

double AverageFrameMs(const Snapshot& snapshot, uint64_t ticks_per_second) {
	return snapshot.frames == 0 ? 0.0
	                            : Milliseconds(snapshot.interval_ticks, ticks_per_second) /
	                                  static_cast<double>(snapshot.frames);
}

double CpuCores(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto seconds = IntervalSeconds(snapshot, ticks_per_second);
	return seconds > 0.0 ? static_cast<double>(snapshot.cpu_ns) / 1e9 / seconds : 0.0;
}

void CloseCsv(Report& report) {
	if (report.csv_open) {
		report.csv.Flush();
		report.csv.Close();
		report.csv_open = false;
	}
}

} // namespace

namespace Detail {

void RecordSpan(SpanId id, uint64_t ticks) noexcept {
	auto& span = g_spans[static_cast<size_t>(id)];
	span.ticks.fetch_add(ticks, std::memory_order_relaxed);
	span.count.fetch_add(1, std::memory_order_relaxed);
	auto longest = span.max_ticks.load(std::memory_order_relaxed);
	while (ticks > longest &&
	       !span.max_ticks.compare_exchange_weak(longest, ticks, std::memory_order_relaxed)) {
	}
}

void AddCounter(CounterId id, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(id)].fetch_add(value, std::memory_order_relaxed);
}

void SetGauge(GaugeId id, uint64_t value) noexcept {
	g_gauges[static_cast<size_t>(id)].store(value, std::memory_order_relaxed);
}

} // namespace Detail

uint64_t Now() noexcept {
	return Common::Timer::QueryPerformanceCounter();
}

uint64_t TicksPerSecond() noexcept {
	static const uint64_t ticks_per_second = Common::Timer::QueryPerformanceFrequency();
	return ticks_per_second;
}

Snapshot CollectAndReset() noexcept {
	Snapshot snapshot;
	for (size_t i = 0; i < SpanCount; i++) {
		snapshot.spans[i] = {
		    .ticks     = g_spans[i].ticks.exchange(0, std::memory_order_relaxed),
		    .count     = g_spans[i].count.exchange(0, std::memory_order_relaxed),
		    .max_ticks = g_spans[i].max_ticks.exchange(0, std::memory_order_relaxed),
		};
	}
	for (size_t i = 0; i < CounterCount; i++) {
		snapshot.counters[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	for (size_t i = 0; i < GaugeCount; i++) {
		snapshot.gauges[i] = g_gauges[i].load(std::memory_order_relaxed);
	}
	return snapshot;
}

std::string FormatSummary(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto frames = static_cast<double>(std::max<uint64_t>(snapshot.frames, 1));
	const auto ms     = [&](SpanId id) {
		return Milliseconds(snapshot.Get(id).ticks, ticks_per_second) / frames;
	};
	const auto count = [&](SpanId id) {
		return static_cast<double>(snapshot.Get(id).count) / frames;
	};
	const auto per_frame = [&](CounterId id) {
		return static_cast<double>(snapshot.Get(id)) / frames;
	};
	const auto mb = [&](CounterId id) { return Megabytes(snapshot.Get(id)) / frames; };
	const auto bda_passes = snapshot.Get(SpanId::BdaPrepare).count;
	const auto buffers_per_pass =
	    bda_passes == 0 ? 0.0
	                    : static_cast<double>(snapshot.Get(CounterId::BdaBuffersVisited)) /
	                          static_cast<double>(bda_passes);

	std::string out;
	auto        it = std::back_inserter(out);
	fmt::format_to(it,
	               "[perf] {:.0f}s: {} frames in {:.2f}s ({:.1f} fps), frame avg {:.1f} ms, max {:.1f} "
	               "ms, CPU {:.2f} cores, GPU memory {} MB (GC above {}/{} MB), {} cached buffers\n",
	               snapshot.elapsed_seconds, snapshot.frames,
	               IntervalSeconds(snapshot, ticks_per_second),
	               FramesPerSecond(snapshot, ticks_per_second),
	               AverageFrameMs(snapshot, ticks_per_second),
	               Milliseconds(snapshot.max_frame_ticks, ticks_per_second),
	               CpuCores(snapshot, ticks_per_second), snapshot.Get(GaugeId::GpuMemoryMb),
	               snapshot.Get(GaugeId::BufferGcTriggerMb),
	               snapshot.Get(GaugeId::TextureGcTriggerMb), snapshot.Get(GaugeId::CachedBuffers));
	fmt::format_to(it,
	               "[perf] per frame: GPU thread busy {:.1f} ms, idle {:.1f} ms, blocked {:.1f} ms | "
	               "game waiting on GPU thread {:.1f} ms, on GPU commands {:.1f} ms ({:.1f}) | flip "
	               "wait {:.1f} ms | present {:.1f} ms\n",
	               ms(SpanId::GpuThreadBusy), ms(SpanId::GpuThreadIdle),
	               ms(SpanId::GpuThreadBlocked), ms(SpanId::GameWaitGpuIdle),
	               ms(SpanId::GameWaitGpuCommand), count(SpanId::GameWaitGpuCommand),
	               ms(SpanId::FlipWait), ms(SpanId::Present));
	fmt::format_to(it,
	               "[perf] per frame: {:.0f} draws {:.1f} ms (targets {:.1f}, shaders {:.1f}, "
	               "bindings {:.1f}, vertex/index {:.1f}, pipeline {:.1f}, record {:.1f}; skipped "
	               "{:.1f} for shaders, {:.1f} for pipelines) | {:.0f} dispatches {:.1f} ms\n",
	               count(SpanId::Draw), ms(SpanId::Draw), ms(SpanId::DrawTargets),
	               ms(SpanId::DrawShaders), ms(SpanId::DrawBindings), ms(SpanId::DrawVertexIndex),
	               ms(SpanId::DrawPipeline), ms(SpanId::DrawRecord),
	               per_frame(CounterId::DrawsSkippedShader),
	               per_frame(CounterId::DrawsSkippedPipeline), count(SpanId::Dispatch),
	               ms(SpanId::Dispatch));
	fmt::format_to(it,
	               "[perf] per frame: BDA {:.1f} passes {:.1f} ms ({:.0f} buffers per pass) | {:.1f} submits "
	               "{:.1f} ms | {:.1f} GPU waits {:.1f} ms (longest {:.1f} ms) | {:.1f} page faults "
	               "{:.1f} ms ({:.1f} write) | GC {:.1f} ms\n",
	               count(SpanId::BdaPrepare), ms(SpanId::BdaPrepare),
	               buffers_per_pass, count(SpanId::QueueSubmit),
	               ms(SpanId::QueueSubmit), count(SpanId::GpuWait), ms(SpanId::GpuWait),
	               Milliseconds(snapshot.Get(SpanId::GpuWait).max_ticks, ticks_per_second),
	               count(SpanId::PageFault), ms(SpanId::PageFault),
	               per_frame(CounterId::WriteFaults), ms(SpanId::GarbageCollect));
	fmt::format_to(it,
	               "[perf] per frame: uploads {:.1f} buffer ({:.2f} MB), {:.1f} stream ({:.2f} MB), "
	               "{:.1f} image ({:.2f} MB) | created {:.1f} buffers, {:.1f} images | evicted "
	               "{:.1f} buffers, {:.1f} images | {:.1f} downloads ({:.2f} MB) {:.1f} ms | "
	               "compiled {:.1f} shaders ({:.1f} ms blocking), {:.1f} pipelines ({:.1f} ms "
	               "blocking)\n",
	               per_frame(CounterId::BufferUploads), mb(CounterId::BufferUploadBytes),
	               per_frame(CounterId::StreamUploads), mb(CounterId::StreamUploadBytes),
	               per_frame(CounterId::ImageUploads), mb(CounterId::ImageUploadBytes),
	               per_frame(CounterId::BufferCreates), per_frame(CounterId::ImageCreates),
	               per_frame(CounterId::BuffersEvicted), per_frame(CounterId::ImagesEvicted),
	               count(SpanId::BufferDownload), mb(CounterId::BufferDownloadBytes),
	               ms(SpanId::BufferDownload), per_frame(CounterId::ShadersCompiled),
	               ms(SpanId::ShaderCompileSync), per_frame(CounterId::PipelinesCreated),
	               ms(SpanId::PipelineCreateSync));
	return out;
}

std::string FormatCsvHeader() {
	std::string out = "elapsed_s,interval_s,frames,fps,frame_avg_ms,frame_max_ms,cpu_cores";
	auto        it  = std::back_inserter(out);
	for (const auto name: SpanNames) {
		fmt::format_to(it, ",{0}_ms,{0}_n,{0}_max_ms", name);
	}
	for (const auto name: CounterNames) {
		fmt::format_to(it, ",{}", name);
	}
	for (const auto name: GaugeNames) {
		fmt::format_to(it, ",{}", name);
	}
	out += '\n';
	return out;
}

std::string FormatCsvRow(const Snapshot& snapshot, uint64_t ticks_per_second) {
	std::string out;
	auto        it = std::back_inserter(out);
	fmt::format_to(it, "{:.3f},{:.3f},{},{:.3f},{:.3f},{:.3f},{:.3f}", snapshot.elapsed_seconds,
	               IntervalSeconds(snapshot, ticks_per_second), snapshot.frames,
	               FramesPerSecond(snapshot, ticks_per_second),
	               AverageFrameMs(snapshot, ticks_per_second),
	               Milliseconds(snapshot.max_frame_ticks, ticks_per_second),
	               CpuCores(snapshot, ticks_per_second));
	for (const auto& span: snapshot.spans) {
		fmt::format_to(it, ",{:.3f},{},{:.3f}", Milliseconds(span.ticks, ticks_per_second),
		               span.count, Milliseconds(span.max_ticks, ticks_per_second));
	}
	for (const auto value: snapshot.counters) {
		fmt::format_to(it, ",{}", value);
	}
	for (const auto value: snapshot.gauges) {
		fmt::format_to(it, ",{}", value);
	}
	out += '\n';
	return out;
}

void OnGuestFrame() noexcept {
	if (!Enabled()) {
		return;
	}
	const auto  now    = Now();
	auto&       report = GetReport();
	std::string summary;
	{
		std::lock_guard lock(report.mutex);
		if (report.last_frame != 0) {
			report.max_frame = std::max(report.max_frame, now - report.last_frame);
		}
		report.last_frame = now;
		report.frames++;
		if (now - report.interval_start < report.ticks_per_second) {
			return;
		}
		auto snapshot            = CollectAndReset();
		snapshot.elapsed_seconds = static_cast<double>(now - report.start) /
		                           static_cast<double>(report.ticks_per_second);
		snapshot.interval_ticks  = now - report.interval_start;
		snapshot.frames          = report.frames;
		snapshot.max_frame_ticks = report.max_frame;
		const auto cpu_ns        = ProcessCpuNs();
		snapshot.cpu_ns          = cpu_ns - std::min(cpu_ns, report.cpu_start_ns);
		report.cpu_start_ns      = cpu_ns;
		report.interval_start    = now;
		report.frames            = 0;
		report.max_frame         = 0;

		summary = FormatSummary(snapshot, report.ticks_per_second);
		if (report.csv_open) {
			const auto row = FormatCsvRow(snapshot, report.ticks_per_second);
			report.csv.Write(row.data(), static_cast<uint32_t>(row.size()));
			report.csv.Flush();
		}
	}
	std::fwrite(summary.data(), 1, summary.size(), stdout);
	std::fflush(stdout);
}

void Initialize() {
	if (!Config::PerfStatsEnabled() && !EnvironmentRequestsStats()) {
		return;
	}
	auto&           report = GetReport();
	std::lock_guard lock(report.mutex);
	report.ticks_per_second = TicksPerSecond();
	report.start            = Now();
	report.interval_start   = report.start;
	report.cpu_start_ns     = ProcessCpuNs();
	report.csv_open         = report.csv.Create(CsvFileName);
	if (report.csv_open) {
		const auto header = FormatCsvHeader();
		report.csv.Write(header.data(), static_cast<uint32_t>(header.size()));
		report.csv.Flush();
	}
	std::printf("Performance statistics enabled: a summary follows every second of guest frames%s\n",
	            report.csv_open ? "; intervals are written to _PerfStats.csv"
	                            : "; _PerfStats.csv could not be created");
	std::fflush(stdout);
	Detail::g_enabled = true;
}

void Shutdown() {
	auto&           report = GetReport();
	std::lock_guard lock(report.mutex);
	CloseCsv(report);
}

void EmergencyShutdown() {
	// A crashing thread may hold the report lock; losing the last row beats a hang.
	auto&            report = GetReport();
	std::unique_lock lock(report.mutex, std::try_to_lock);
	if (lock.owns_lock()) {
		CloseCsv(report);
	}
}

} // namespace PerfStats
