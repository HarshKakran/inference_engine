#pragma once

#include "llama.h"

#include <cstdio>

// ggml_log_callback: writes every llama.cpp/ggml log line to the log file
// instead of stderr, keeping the terminal clean during dev.
void log_to_file(ggml_log_level level, const char *text, void *user_data);

// Our own diagnostics: print to stderr (so you see it live) AND to the
// log file (so it's kept alongside the library's own log lines).
void report_error(FILE *log_file, const char *msg);
