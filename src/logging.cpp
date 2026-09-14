#include "logging.h"

void log_to_file(ggml_log_level level, const char *text, void *user_data)
{
    FILE *log_file = static_cast<FILE *>(user_data);
    fprintf(log_file, "%s", text);
}

void report_error(FILE *log_file, const char *msg)
{
    fprintf(stderr, "%s", msg);
    fprintf(log_file, "%s", msg);
}
