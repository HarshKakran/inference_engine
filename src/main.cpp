#include "llama.h"
#include <cstring>
#include <cstdio>
#include <iostream>

int main()
{
    llama_backend_init();
    // printf("SYSTEM INFO: \n%s\n\n", llama_print_system_info());

    // pre-filled default struct
    llama_model_params mparams = llama_model_default_params();
    // offload all the layers to GPU
    mparams.n_gpu_layers = -1;

    // load the model
    llama_model *model = llama_model_load_from_file("models/qwen2.5-3b-instruct-q4_k_m.gguf", mparams);
    if (!model)
    {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    printf("model loaded: %.2f B params\n\n\n", llama_model_n_params(model) / 1e9);

    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_context_params cparams = llama_context_default_params();
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx)
    {
        fprintf(stderr, "failed to init the context from the model.\n");
        return 1;
    }

    const char *prompt = "The capital of India is";
    llama_token tokens[64];
    int n = llama_tokenize(vocab, prompt, strlen(prompt), tokens, 64, true, false);
    if (n < 0)
    {
        fprintf(stderr, "token buffer too small.\n");
        return 1;
    }

    std::cout << "Token Count: " << n << std::endl
              << "Tokens: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << tokens[i] << " ";
    }

    llama_batch batch = llama_batch_get_one(tokens, n);
    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "decode failed.\n\n");
        return 1;
    }

    int n_vocab = llama_vocab_n_tokens(vocab);
    float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits)
    {
        fprintf(stderr, "failed to get the logits.\n\n");
        return 1;
    }

    // Argmax(greedy pick)
    int max_logit = 0;
    for (int i = 1; i < n_vocab; i++)
    {
        if (logits[i] > logits[max_logit])
        {
            max_logit = i;
        }
    }

    // detokenize the logit
    char buf[128];
    int len = llama_token_to_piece(vocab, max_logit, buf, sizeof(buf), 0, false);
    printf("\nnext token: '%.*s'  (id=%d, logit=%.2f)\n", len, buf, max_logit, logits[max_logit]);

    // free the memory - garbage collection
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}