#include "graphics/host_gpu/renderer/frameDump.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <share.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr const char* TRIGGER_FILE    = "kyty_dump.trigger";
constexpr uint64_t    MAX_IMAGE_BYTES = 512ull * 1024ull * 1024ull;
constexpr uint32_t    PREVIEW_WIDTH   = 960;

struct TexelLayout {
	uint32_t bytes     = 0;
	uint32_t channels  = 0;
	bool     is_float  = false;
	bool     is_snorm  = false;
	bool     has_alpha = false;
};

TexelLayout LayoutOf(vk::Format format) {
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Srgb:
		case vk::Format::eR8Uint: return {1, 1, false, false, false};
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8Srgb: return {2, 2, false, false, false};
		case vk::Format::eR8G8Snorm: return {2, 2, false, true, false};
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: return {4, 4, false, false, true};
		case vk::Format::eR8G8B8A8Snorm:
		case vk::Format::eB8G8R8A8Snorm: return {4, 4, false, true, true};
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eE5B9G9R9UfloatPack32: return {4, 3, true, false, false};
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Uint: return {2, 1, false, false, false};
		case vk::Format::eR16Sfloat: return {2, 1, true, false, false};
		case vk::Format::eR16G16Unorm: return {4, 2, false, false, false};
		case vk::Format::eR16G16Snorm: return {4, 2, false, true, false};
		case vk::Format::eR16G16Sfloat: return {4, 2, true, false, false};
		case vk::Format::eR16G16B16A16Unorm: return {8, 4, false, false, true};
		case vk::Format::eR16G16B16A16Snorm: return {8, 4, false, true, true};
		case vk::Format::eR16G16B16A16Sfloat: return {8, 4, true, false, true};
		case vk::Format::eR32Sfloat: return {4, 1, true, false, false};
		case vk::Format::eR32Uint: return {4, 1, false, false, false};
		case vk::Format::eR32G32Sfloat: return {8, 2, true, false, false};
		case vk::Format::eR32G32B32A32Sfloat: return {16, 4, true, false, true};
		default: return {};
	}
}

template <typename T>
T ReadAs(const uint8_t* source) {
	T value {};
	std::memcpy(&value, source, sizeof(T));
	return value;
}

float HalfToFloat(uint16_t value) {
	const uint32_t sign     = (value >> 15u) & 0x1u;
	const uint32_t exponent = (value >> 10u) & 0x1fu;
	const uint32_t mantissa = value & 0x3ffu;
	float          result   = 0.0f;
	if (exponent == 0) {
		result = std::ldexp(static_cast<float>(mantissa), -24);
	} else if (exponent == 0x1fu) {
		result = mantissa == 0 ? std::numeric_limits<float>::infinity()
		                       : std::numeric_limits<float>::quiet_NaN();
	} else {
		result = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
	}
	return sign != 0 ? -result : result;
}

float PackedFloat(uint32_t mantissa, uint32_t exponent, int mantissa_bits) {
	if (exponent == 0) {
		return std::ldexp(static_cast<float>(mantissa), -14 - mantissa_bits);
	}
	if (exponent == 0x1fu) {
		return mantissa == 0 ? std::numeric_limits<float>::infinity()
		                     : std::numeric_limits<float>::quiet_NaN();
	}
	return std::ldexp(static_cast<float>(mantissa + (1u << static_cast<uint32_t>(mantissa_bits))),
	                  static_cast<int>(exponent) - 15 - mantissa_bits);
}

void DecodeTexel(vk::Format format, const uint8_t* s, std::array<float, 4>& out) {
	out              = {0.0f, 0.0f, 0.0f, 1.0f};
	const auto un8   = [s](uint32_t i) { return static_cast<float>(s[i]) / 255.0f; };
	const auto sn8   = [s](uint32_t i) {
        return std::max(static_cast<float>(static_cast<int8_t>(s[i])) / 127.0f, -1.0f);
	};
	const auto un16  = [s](uint32_t i) { return static_cast<float>(ReadAs<uint16_t>(s + i)) / 65535.0f; };
	const auto sn16  = [s](uint32_t i) {
        return std::max(static_cast<float>(ReadAs<int16_t>(s + i)) / 32767.0f, -1.0f);
	};
	const auto half  = [s](uint32_t i) { return HalfToFloat(ReadAs<uint16_t>(s + i)); };
	const auto fl32  = [s](uint32_t i) { return ReadAs<float>(s + i); };
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Srgb:
		case vk::Format::eR8Uint: out[0] = un8(0); break;
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8Srgb:
			out[0] = un8(0);
			out[1] = un8(1);
			break;
		case vk::Format::eR8G8Snorm:
			out[0] = sn8(0);
			out[1] = sn8(1);
			break;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eR8G8B8A8Uint: out = {un8(0), un8(1), un8(2), un8(3)}; break;
		case vk::Format::eR8G8B8A8Snorm: out = {sn8(0), sn8(1), sn8(2), sn8(3)}; break;
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: out = {un8(2), un8(1), un8(0), un8(3)}; break;
		case vk::Format::eB8G8R8A8Snorm: out = {sn8(2), sn8(1), sn8(0), sn8(3)}; break;
		case vk::Format::eA2B10G10R10UnormPack32: {
			const auto p = ReadAs<uint32_t>(s);
			out = {static_cast<float>(p & 0x3ffu) / 1023.0f,
			       static_cast<float>((p >> 10u) & 0x3ffu) / 1023.0f,
			       static_cast<float>((p >> 20u) & 0x3ffu) / 1023.0f,
			       static_cast<float>(p >> 30u) / 3.0f};
			break;
		}
		case vk::Format::eA2R10G10B10UnormPack32: {
			const auto p = ReadAs<uint32_t>(s);
			out = {static_cast<float>((p >> 20u) & 0x3ffu) / 1023.0f,
			       static_cast<float>((p >> 10u) & 0x3ffu) / 1023.0f,
			       static_cast<float>(p & 0x3ffu) / 1023.0f, static_cast<float>(p >> 30u) / 3.0f};
			break;
		}
		case vk::Format::eB10G11R11UfloatPack32: {
			const auto p = ReadAs<uint32_t>(s);
			out[0]       = PackedFloat(p & 0x3fu, (p >> 6u) & 0x1fu, 6);
			out[1]       = PackedFloat((p >> 11u) & 0x3fu, (p >> 17u) & 0x1fu, 6);
			out[2]       = PackedFloat((p >> 22u) & 0x1fu, (p >> 27u) & 0x1fu, 5);
			break;
		}
		case vk::Format::eE5B9G9R9UfloatPack32: {
			const auto p        = ReadAs<uint32_t>(s);
			const int  exponent = static_cast<int>((p >> 27u) & 0x1fu) - 24;
			out[0]              = std::ldexp(static_cast<float>(p & 0x1ffu), exponent);
			out[1]              = std::ldexp(static_cast<float>((p >> 9u) & 0x1ffu), exponent);
			out[2]              = std::ldexp(static_cast<float>((p >> 18u) & 0x1ffu), exponent);
			break;
		}
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Uint: out[0] = un16(0); break;
		case vk::Format::eR16Sfloat: out[0] = half(0); break;
		case vk::Format::eR16G16Unorm:
			out[0] = un16(0);
			out[1] = un16(2);
			break;
		case vk::Format::eR16G16Snorm:
			out[0] = sn16(0);
			out[1] = sn16(2);
			break;
		case vk::Format::eR16G16Sfloat:
			out[0] = half(0);
			out[1] = half(2);
			break;
		case vk::Format::eR16G16B16A16Unorm: out = {un16(0), un16(2), un16(4), un16(6)}; break;
		case vk::Format::eR16G16B16A16Snorm: out = {sn16(0), sn16(2), sn16(4), sn16(6)}; break;
		case vk::Format::eR16G16B16A16Sfloat: out = {half(0), half(2), half(4), half(6)}; break;
		case vk::Format::eR32Sfloat: out[0] = fl32(0); break;
		case vk::Format::eR32Uint: out[0] = static_cast<float>(ReadAs<uint32_t>(s)); break;
		case vk::Format::eR32G32Sfloat:
			out[0] = fl32(0);
			out[1] = fl32(4);
			break;
		case vk::Format::eR32G32B32A32Sfloat: out = {fl32(0), fl32(4), fl32(8), fl32(12)}; break;
		default: break;
	}
}

struct DumpMeta {
	std::filesystem::path dir;
	std::string           name;
	vk::Format            format  = vk::Format::eUndefined;
	uint32_t              width   = 0;
	uint32_t              height  = 0;
	uint32_t              depth   = 0;
	uint64_t              bytes   = 0;
};

std::mutex g_index_mutex;

void AppendIndex(const std::filesystem::path& dir, const std::string& line) {
	std::lock_guard lock(g_index_mutex);
	const auto      path = (dir / "index.txt").string();
#ifdef _WIN32
	FILE* file = _fsopen(path.c_str(), "a", _SH_DENYNO);
#else
	FILE* file = std::fopen(path.c_str(), "a");
#endif
	if (file != nullptr) {
		std::fputs(line.c_str(), file);
		std::fputc('\n', file);
		std::fclose(file);
	}
}

void WriteBinary(const std::filesystem::path& path, const std::string& header,
                 const std::vector<uint8_t>& data) {
#ifdef _WIN32
	FILE* file = _fsopen(path.string().c_str(), "wb", _SH_DENYNO);
#else
	FILE* file = std::fopen(path.string().c_str(), "wb");
#endif
	if (file == nullptr) {
		return;
	}
	std::fwrite(header.data(), 1, header.size(), file);
	std::fwrite(data.data(), 1, data.size(), file);
	std::fclose(file);
}

uint8_t ToByte(float value) {
	return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

void WriteImage(const DumpMeta& meta, const uint8_t* data) {
	const auto     layout = LayoutOf(meta.format);
	const uint64_t texels = static_cast<uint64_t>(meta.width) * meta.height * meta.depth;

	std::array<double, 4>   sum {};
	std::array<float, 4>    min_value {};
	std::array<float, 4>    max_value {};
	std::array<uint64_t, 4> finite {};
	std::array<uint64_t, 4> non_finite {};
	std::array<uint64_t, 4> above_one {};
	min_value.fill(std::numeric_limits<float>::max());
	max_value.fill(std::numeric_limits<float>::lowest());
	double               luminance_sum   = 0.0;
	uint64_t             luminance_count = 0;
	std::array<float, 4> texel {};
	for (uint64_t index = 0; index < texels; index++) {
		DecodeTexel(meta.format, data + index * layout.bytes, texel);
		for (uint32_t c = 0; c < layout.channels; c++) {
			const float v = texel[c];
			if (!std::isfinite(v)) {
				non_finite[c]++;
				continue;
			}
			finite[c]++;
			sum[c] += v;
			min_value[c] = std::min(min_value[c], v);
			max_value[c] = std::max(max_value[c], v);
			above_one[c] += v > 1.0f ? 1u : 0u;
		}
		if (layout.channels >= 3 && std::isfinite(texel[0]) && std::isfinite(texel[1]) &&
		    std::isfinite(texel[2])) {
			luminance_sum += std::max(0.2126f * texel[0] + 0.7152f * texel[1] + 0.0722f * texel[2], 0.0f);
			luminance_count++;
		} else if (layout.channels < 3 && std::isfinite(texel[0])) {
			luminance_sum += std::max(texel[0], 0.0f);
			luminance_count++;
		}
	}
	const double mean_luminance = luminance_count != 0 ? luminance_sum / luminance_count : 0.0;

	std::string stats = fmt::format("{} fmt={} extent={}x{}x{} meanlum={:.6g}", meta.name,
	                                VulkanToString(meta.format), meta.width, meta.height,
	                                meta.depth, mean_luminance);
	static constexpr std::array channel_names {'R', 'G', 'B', 'A'};
	for (uint32_t c = 0; c < layout.channels; c++) {
		const double mean = finite[c] != 0 ? sum[c] / finite[c] : 0.0;
		stats += fmt::format(" | {}: min={:.5g} max={:.5g} mean={:.5g} >1={} nonfinite={}",
		                     channel_names[c], finite[c] != 0 ? min_value[c] : 0.0f,
		                     finite[c] != 0 ? max_value[c] : 0.0f, mean, above_one[c],
		                     non_finite[c]);
	}
	AppendIndex(meta.dir, stats);

	// Volumes are tiled slice by slice; 2D images are point-sampled down to the preview width.
	const bool     volume  = meta.depth > 1;
	const uint32_t columns = volume ? static_cast<uint32_t>(std::ceil(std::sqrt(meta.depth))) : 1u;
	const uint32_t rows    = volume ? (meta.depth + columns - 1u) / columns : 1u;
	const uint32_t step    = volume ? 1u : std::max((meta.width + PREVIEW_WIDTH - 1u) / PREVIEW_WIDTH, 1u);
	const uint32_t tile_w  = (meta.width + step - 1u) / step;
	const uint32_t tile_h  = (meta.height + step - 1u) / step;
	const uint32_t out_w   = tile_w * columns;
	const uint32_t out_h   = tile_h * rows;

	const float auto_scale =
	    mean_luminance > 1e-8 ? static_cast<float>(0.18 / mean_luminance) : 1.0f;
	const auto map_value = [&layout](float v, float scale) {
		if (layout.is_float) {
			const float scaled = std::max(v * scale, 0.0f);
			return ToByte(std::pow(scaled / (1.0f + scaled), 1.0f / 2.2f));
		}
		if (layout.is_snorm) {
			return ToByte(v * 0.5f + 0.5f);
		}
		return ToByte(v);
	};

	const uint32_t variants = layout.is_float ? 2u : 1u;
	for (uint32_t variant = 0; variant < variants; variant++) {
		const float          scale = variant == 0 ? 1.0f : auto_scale;
		std::vector<uint8_t> rgb(static_cast<size_t>(out_w) * out_h * 3u, 0);
		std::vector<uint8_t> alpha;
		if (layout.has_alpha && variant == 0) {
			alpha.assign(static_cast<size_t>(out_w) * out_h, 0);
		}
		for (uint32_t z = 0; z < meta.depth; z++) {
			const uint32_t origin_x = (z % columns) * tile_w;
			const uint32_t origin_y = (z / columns) * tile_h;
			for (uint32_t y = 0; y < tile_h; y++) {
				for (uint32_t x = 0; x < tile_w; x++) {
					const uint64_t source = (static_cast<uint64_t>(z) * meta.height + y * step) *
					                            meta.width +
					                        x * step;
					DecodeTexel(meta.format, data + source * layout.bytes, texel);
					const size_t out = (static_cast<size_t>(origin_y) + y) * out_w + origin_x + x;
					const bool   bad = !std::isfinite(texel[0]) || !std::isfinite(texel[1]) ||
					                 !std::isfinite(texel[2]);
					if (bad) {
						rgb[out * 3u]      = 255;
						rgb[out * 3u + 1u] = 0;
						rgb[out * 3u + 2u] = 255;
					} else if (layout.channels == 1) {
						const auto value   = map_value(texel[0], scale);
						rgb[out * 3u]      = value;
						rgb[out * 3u + 1u] = value;
						rgb[out * 3u + 2u] = value;
					} else {
						rgb[out * 3u]      = map_value(texel[0], scale);
						rgb[out * 3u + 1u] = map_value(texel[1], scale);
						rgb[out * 3u + 2u] = layout.channels >= 3 ? map_value(texel[2], scale) : 0;
					}
					if (!alpha.empty()) {
						alpha[out] = std::isfinite(texel[3]) ? ToByte(texel[3]) : 255;
					}
				}
			}
		}
		const auto suffix = variant == 0 ? "" : "_auto";
		WriteBinary(meta.dir / fmt::format("{}{}.ppm", meta.name, suffix),
		            fmt::format("P6\n{} {}\n255\n", out_w, out_h), rgb);
		if (!alpha.empty()) {
			WriteBinary(meta.dir / fmt::format("{}_alpha.pgm", meta.name),
			            fmt::format("P5\n{} {}\n255\n", out_w, out_h), alpha);
		}
	}
}

} // namespace

void FrameDumpOnFlip(RenderContext& context, uint64_t surface_address) {
	static uint32_t flip_count = 0;
	if ((++flip_count % 15u) != 0) {
		return;
	}
	std::error_code error;
	if (!std::filesystem::exists(TRIGGER_FILE, error)) {
		return;
	}
	std::filesystem::remove(TRIGGER_FILE, error);

	static uint32_t dump_index = 0;
	const auto      dir        = std::filesystem::path("_FrameDump") / fmt::format("{:02}", dump_index++);
	std::filesystem::create_directories(dir, error);
	AppendIndex(dir, fmt::format("flip={} surface=0x{:010x}", flip_count, surface_address));

	auto&    cache     = context.GetTextureCache();
	auto&    scheduler = context.GetCommandScheduler();
	auto&    graphics  = context.GetGraphics();
	uint32_t ordinal   = 0;
	cache.DebugForEachImage([&](ImageId /*id*/, Image& image) {
		const auto& backing = image.backing;
		const auto  layout  = LayoutOf(backing.format);
		const bool  wanted  = image.usage.render_target || image.usage.storage ||
		                    image.usage.video_out || image.IsGpuModified();
		const bool  volume  = backing.image_type == vk::ImageType::e3D;
		const uint32_t depth = volume ? backing.extent.depth : 1u;
		const uint64_t bytes = static_cast<uint64_t>(backing.extent.width) * backing.extent.height *
		                       depth * layout.bytes;
		std::string reason;
		if (!wanted) {
			return;
		}
		if (backing.image == nullptr || backing.samples != 1 || image.info.IsDepth()) {
			reason = "not a single-sample color image";
		} else if (layout.bytes == 0) {
			reason = "format not decoded";
		} else if (!(backing.usage & vk::ImageUsageFlagBits::eTransferSrc)) {
			reason = "no transfer-src usage";
		} else if (bytes == 0 || bytes > MAX_IMAGE_BYTES) {
			reason = "size out of range";
		}
		const auto name = fmt::format(
		    "{:03}_{:010x}_{}x{}x{}_{}{}{}{}{}{}", ordinal++, image.info.data.address,
		    backing.extent.width, backing.extent.height, depth, VulkanToString(backing.format),
		    image.usage.render_target ? "_rt" : "", image.usage.storage ? "_storage" : "",
		    image.usage.texture ? "_tex" : "", image.usage.video_out ? "_videoout" : "",
		    image.info.data.address == surface_address ? "_PRESENTED" : "");
		if (!reason.empty()) {
			AppendIndex(dir, fmt::format("{} skipped: {}", name, reason));
			return;
		}
		auto buffer = std::make_shared<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                       vk::BufferUsageFlagBits::eTransferDst, bytes);
		vk::BufferImageCopy copy {};
		copy.bufferOffset     = 0;
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent      = {backing.extent.width, backing.extent.height, depth};
		image.Download(std::span {&copy, 1}, buffer->Handle(), 0, bytes);
		DumpMeta meta {dir, name, backing.format, backing.extent.width, backing.extent.height, depth,
		               bytes};
		scheduler.DeferPriorityOperation([buffer, meta] {
			buffer->Invalidate(0, meta.bytes);
			WriteImage(meta, buffer->Mapped().data());
		});
	});
}

} // namespace Libs::Graphics
