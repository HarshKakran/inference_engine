#pragma once

#include "llama.h"

#include <cstdio>
#include <memory>

// RAII wrappers for llama.cpp's C-style opaque handles: each pairs a raw
// pointer type with a deleter functor so std::unique_ptr calls the right
// free function automatically on scope exit, from every return path.

struct LlamaModelDeleter
{
    void operator()(llama_model *model) const { llama_model_free(model); }
};
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;

struct LlamaContextDeleter
{
    void operator()(llama_context *ctx) const { llama_free(ctx); }
};
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

struct FileDeleter
{
    void operator()(FILE *file) const { fclose(file); }
};
using FilePtr = std::unique_ptr<FILE, FileDeleter>;

// Guards the one-time global ggml/Metal backend init: construction calls
// llama_backend_init(), destruction calls llama_backend_free().
struct LlamaBackend
{
    LlamaBackend() { llama_backend_init(); }
    ~LlamaBackend() { llama_backend_free(); }
};
