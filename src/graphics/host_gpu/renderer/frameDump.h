#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_

#include <cstdint>

namespace Libs::Graphics {

class RenderContext;

// Debug capture: when "kyty_dump.trigger" exists in the working directory, writes a preview and
// channel statistics of every live render target to _FrameDump/NN, then deletes the trigger.
void FrameDumpOnFlip(RenderContext& context, uint64_t surface_address);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
