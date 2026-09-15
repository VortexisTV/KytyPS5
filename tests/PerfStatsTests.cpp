#include "common/perfStats.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using PerfStats::CounterId;
using PerfStats::GaugeId;
using PerfStats::SpanId;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PerfStatsTests: failed: %s\n", message);
		std::abort();
	}
}

size_t Columns(const std::string& line) {
	size_t columns = 1;
	for (const auto c: line) {
		columns += c == ',' ? 1u : 0u;
	}
	return columns;
}

void TestDisabledHooksRecordNothing() {
	PerfStats::Detail::g_enabled = false;
	PerfStats::Add(CounterId::BufferUploads, 5);
	PerfStats::Set(GaugeId::CachedBuffers, 7);
	{
		PerfStats::Span span(SpanId::Draw);
	}
	const auto snapshot = PerfStats::CollectAndReset();
	Check(snapshot.Get(CounterId::BufferUploads) == 0, "a disabled counter was recorded");
	Check(snapshot.Get(GaugeId::CachedBuffers) == 0, "a disabled gauge was recorded");
	Check(snapshot.Get(SpanId::Draw).count == 0, "a disabled span was recorded");
}

void TestIntervalsAccumulateAndReset() {
	PerfStats::Detail::g_enabled = true;
	PerfStats::Add(CounterId::BufferUploads);
	PerfStats::Add(CounterId::BufferUploadBytes, 4096);
	PerfStats::Add(CounterId::BufferUploadBytes, 4096);
	PerfStats::Set(GaugeId::CachedBuffers, 3);
	PerfStats::Set(GaugeId::CachedBuffers, 9);
	PerfStats::Detail::RecordSpan(SpanId::Draw, 10);
	PerfStats::Detail::RecordSpan(SpanId::Draw, 30);
	{
		PerfStats::Span span(SpanId::BdaPrepare);
		span.Stop();
		span.Stop();
	}
	{
		PerfStats::Span span(SpanId::GpuWait, false);
	}

	const auto first = PerfStats::CollectAndReset();
	Check(first.Get(CounterId::BufferUploads) == 1, "a counter increment was lost");
	Check(first.Get(CounterId::BufferUploadBytes) == 8192, "counter values were not summed");
	Check(first.Get(GaugeId::CachedBuffers) == 9, "a gauge did not keep its latest value");
	const auto& draw = first.Get(SpanId::Draw);
	Check(draw.ticks == 40 && draw.count == 2 && draw.max_ticks == 30, "span totals are wrong");
	Check(first.Get(SpanId::BdaPrepare).count == 1, "stopping a span twice recorded it twice");
	Check(first.Get(SpanId::GpuWait).count == 0, "a span with a false condition was recorded");

	const auto second = PerfStats::CollectAndReset();
	Check(second.Get(CounterId::BufferUploads) == 0 &&
	          second.Get(CounterId::BufferUploadBytes) == 0,
	      "counters were not reset between intervals");
	Check(second.Get(SpanId::Draw).count == 0 && second.Get(SpanId::Draw).max_ticks == 0,
	      "spans were not reset between intervals");
	Check(second.Get(GaugeId::CachedBuffers) == 9, "a gauge was reset between intervals");
	PerfStats::Detail::g_enabled = false;
}

void TestCsvRowsMatchHeader() {
	const auto          header = PerfStats::FormatCsvHeader();
	PerfStats::Snapshot snapshot;
	snapshot.frames         = 3;
	snapshot.interval_ticks = 1000;
	const auto row          = PerfStats::FormatCsvRow(snapshot, 1000);
	Check(!header.empty() && header.back() == '\n' && !row.empty() && row.back() == '\n',
	      "CSV lines are not newline terminated");
	Check(Columns(header) == Columns(row), "CSV row and header column counts differ");
	Check(header.find(",draw_ms,draw_n,draw_max_ms,") != std::string::npos,
	      "span columns are missing from the CSV header");
	Check(header.find(",bda_buffers_visited,") != std::string::npos,
	      "counter columns are missing from the CSV header");
	Check(header.find(",cached_buffers\n") != std::string::npos,
	      "gauge columns are missing from the CSV header");
	Check(row.rfind("0.000,1.000,3,3.000,333.333,", 0) == 0, "CSV frame columns are wrong");
}

void TestSummaryReportsPerFrameValues() {
	PerfStats::Snapshot snapshot;
	snapshot.frames                                        = 4;
	snapshot.interval_ticks                                = 1000;
	snapshot.spans[static_cast<size_t>(SpanId::Draw)]       = {1000, 400, 10};
	snapshot.spans[static_cast<size_t>(SpanId::BdaPrepare)] = {600, 40, 20};
	snapshot.counters[static_cast<size_t>(CounterId::BdaBuffersVisited)] = 8000;
	const auto summary = PerfStats::FormatSummary(snapshot, 1000);
	Check(summary.find("4 frames in 1.00s (4.0 fps)") != std::string::npos,
	      "the summary does not report the frame rate");
	Check(summary.find("100 draws 250.0 ms") != std::string::npos,
	      "the summary does not report per-frame draw cost");
	Check(summary.find("BDA 10.0 passes 150.0 ms (200 buffers per pass)") != std::string::npos,
	      "the summary does not report per-frame BDA cost");
}

} // namespace

int main() {
	TestDisabledHooksRecordNothing();
	TestIntervalsAccumulateAndReset();
	TestCsvRowsMatchHeader();
	TestSummaryReportsPerFrameValues();
	std::printf("PerfStatsTests: ok\n");
	return 0;
}
