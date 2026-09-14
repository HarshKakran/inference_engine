#include "llama.h"
#include "llama_raii.h"
#include "logging.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int n_ctx = 512;

int main()
{
    FilePtr log_file(fopen("engine.log", "a"));
    if (!log_file)
    {
        fprintf(stderr, "failed to open log file.\n");
        return 1;
    }

    // logging to file to keep the terminal clean
    llama_log_set(log_to_file, log_file.get());

    // RAII: llama_backend_init() now, llama_backend_free() on scope exit
    LlamaBackend backend;

    // pre-filled default struct
    llama_model_params mparams = llama_model_default_params();
    // offload all the layers to GPU
    mparams.n_gpu_layers = -1;

    // load the model; returns an opaque handle (nullptr on failure, C-style error signaling)
    LlamaModelPtr model(llama_model_load_from_file("models/qwen2.5-3b-instruct-q4_k_m.gguf", mparams));
    if (!model)
    {
        report_error(log_file.get(), "failed to load the model\n");
        return 1;
    }

    printf("model loaded: %.2f B params\n\n\n", llama_model_n_params(model.get()) / 1e9);

    // tmpl for chat templating
    const char *tmpl = llama_model_chat_template(model.get(), nullptr);
    if (!tmpl)
    {
        report_error(log_file.get(), "model metadata does not have the chat template");
        return 1;
    }

    // vocab handle lives inside the model but is fetched separately (tokenization doesn't need the compute context)
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    // size of the vocab = length of the logits array (one float score per possible next token)
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    // this is the expensive call: allocates the KV cache and sets up the compute graph
    LlamaContextPtr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx)
    {
        report_error(log_file.get(), "failed to init the context from the model.\n");
        return 1;
    }

    // import the system prompt
    std::ifstream system_prompt_file("prompts/system_prompt.txt");
    if (!system_prompt_file)
    {
        report_error(log_file.get(), "failed to open system prompt file.\n");
        return 1;
    }

    std::stringstream rbuf;
    rbuf << system_prompt_file.rdbuf(); // rdbuf() = the stream's internal buffer; << reads it all in one shot
    std::string system_prompt = rbuf.str();

    system_prompt_file.close();

    std::string user_prompt;
    std::cout << "How can I help you today?" << std::endl;
    std::getline(std::cin, user_prompt);

    llama_chat_message chat[2] = {
        {"system", system_prompt.c_str()},
        {"user", user_prompt.c_str()}};

    int32_t buf_capacity = 2 * (system_prompt.size() + user_prompt.size());
    std::unique_ptr<char[]> buf(new char[buf_capacity]);
    int buf_len = llama_chat_apply_template(tmpl, chat, 2, true, buf.get(), buf_capacity);
    while (buf_len > buf_capacity)
    {
        buf_capacity = 2 * buf_len;
        buf.reset(new char[buf_capacity]);
        buf_len = llama_chat_apply_template(tmpl, chat, 2, true, buf.get(), buf_capacity);
    }

    for (int i = 0; i < buf_len; i++)
    {
        std::cout << buf[i];
    }

    // caller-owns-the-buffer idiom: we allocate tokens[] and pass its capacity (64);
    // llama_tokenize writes into it and returns the count used (negative = buffer too small)
    llama_token tokens[n_ctx];
    int n = llama_tokenize(vocab, buf.get(), buf_len, tokens, n_ctx, true, true);
    if (n < 0)
    {
        report_error(log_file.get(), "token buffer too small.\n");
        return 1;
    }

    // batch = per-step scratch describing which tokens go in, at which positions, for which sequence;
    // get_one() is the convenience constructor for "single sequence, these n tokens"
    llama_batch batch = llama_batch_get_one(tokens, n);

    std::string res = "";

    for (int it = n; it < n_ctx; it++)
    {
        // the forward pass (prefill: all n tokens in one compute-bound shot); fills the KV cache.
        // return value is just success/failure (0/nonzero) - NOT a token, decode doesn't pick anything
        if (llama_decode(ctx.get(), batch) != 0)
        {
            report_error(log_file.get(), "decode failed.\n\n");
            return 1;
        }

        // borrowed pointer into ctx's internal buffer - we don't own it, and it's only valid
        // until the next llama_decode call. -1 = logits for the last position (prediction after the prompt)
        float *logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits)
        {
            report_error(log_file.get(), "failed to get the logits.\n\n");
            return 1;
        }

        int next_token = pick_logit_topk(logits, n_vocab);
        if (next_token == -1)
        {
            report_error(log_file.get(), "failed to get the next token.\n\n");
            return 1;
        }

        // detokenize the logit
        char token_buf[128];
        int len = llama_token_to_piece(vocab, next_token, token_buf, sizeof(token_buf), 0, false);
        res.append(token_buf, len);

        // break after model stops
        if (llama_vocab_is_eog(vocab, next_token))
        {
            break;
        }

        tokens[it] = next_token;
        batch = llama_batch_get_one(&tokens[it], 1);
    }

    std::cout << "\nFinal Response:\n"
              << res << std::endl;

    return 0;
}
