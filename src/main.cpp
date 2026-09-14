#include "llama.h"
#include <cstring>
#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>
#include <random>
#include <string>

int n_ctx = 512;
float T = 1;
int K = 5;

float draw()
{
    std::random_device rd;                                  // one-time seed source (queries OS entropy)
    std::mt19937 rng(rd());                                 // the actual PRNG engine (Mersenne Twister), seeded once
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // shapes engine output into [0,1)

    return dist(rng);
}

int pick_logit_topk(float *logits, int n_vocab)
{
    auto cmp = [](const std::pair<int, float> &a, const std::pair<int, float> &b)
    {
        if (a.second == b.second)
        {
            return a.first > b.first;
        }
        return a.second < b.second;
    };

    std::vector<std::pair<int, float>> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++)
    {
        candidates.push_back({i, logits[i]});
    }

    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(cmp)> pq(cmp, std::move(candidates));
    if (pq.empty())
    {
        return -1;
    }

    float max_logit = pq.top().second;
    float topk_exp_logit_sum = 0;

    // get topk logits with Temperature
    std::vector<std::pair<int, float>> topk_logits;
    topk_logits.reserve(K);
    int k = K;
    while (k-- && !pq.empty())
    {
        std::pair<int, float> top_value = pq.top();
        float logit_value = (top_value.second - max_logit) / T;

        topk_logits.push_back({top_value.first, logit_value});
        topk_exp_logit_sum += exp(logit_value);

        pq.pop();
    }

    // softmax topk logits
    std::vector<std::pair<int, float>> softmax_topk_logits;
    softmax_topk_logits.reserve(K);
    for (int i = 0; i < topk_logits.size(); i++)
    {
        float softmax_val = exp(topk_logits[i].second) / topk_exp_logit_sum;
        softmax_topk_logits.push_back({topk_logits[i].first, softmax_val});
    }

    // pick one from topK
    float cumulative = 0;
    float r = draw();
    for (int i = 0; i < softmax_topk_logits.size(); i++)
    {
        cumulative += softmax_topk_logits[i].second;
        if (r < cumulative)
            return softmax_topk_logits[i].first;
    }

    return softmax_topk_logits.back().first;
}

int main()
{
    // one-time global init for ggml backends (Metal/CPU registration, threading)
    llama_backend_init();
    // printf("SYSTEM INFO: \n%s\n\n", llama_print_system_info());

    // pre-filled default struct
    llama_model_params mparams = llama_model_default_params();
    // offload all the layers to GPU
    mparams.n_gpu_layers = -1;

    // load the model; returns an opaque handle (nullptr on failure, C-style error signaling)
    llama_model *model = llama_model_load_from_file("models/qwen2.5-3b-instruct-q4_k_m.gguf", mparams);
    if (!model)
    {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    printf("model loaded: %.2f B params\n\n\n", llama_model_n_params(model) / 1e9);

    // vocab handle lives inside the model but is fetched separately (tokenization doesn't need the compute context)
    const llama_vocab *vocab = llama_model_get_vocab(model);
    // size of the vocab = length of the logits array (one float score per possible next token)
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    // this is the expensive call: allocates the KV cache and sets up the compute graph
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx)
    {
        fprintf(stderr, "failed to init the context from the model.\n");
        return 1;
    }

    const char *prompt = "The capital of India is";
    // caller-owns-the-buffer idiom: we allocate tokens[] and pass its capacity (64);
    // llama_tokenize writes into it and returns the count used (negative = buffer too small)
    llama_token tokens[n_ctx];
    int n = llama_tokenize(vocab, prompt, strlen(prompt), tokens, n_ctx, true, false);
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

    // batch = per-step scratch describing which tokens go in, at which positions, for which sequence;
    // get_one() is the convenience constructor for "single sequence, these n tokens"
    llama_batch batch = llama_batch_get_one(tokens, n);

    std::string res = "";

    for (int it = n; it < n_ctx; it++)
    {
        // the forward pass (prefill: all n tokens in one compute-bound shot); fills the KV cache.
        // return value is just success/failure (0/nonzero) - NOT a token, decode doesn't pick anything
        if (llama_decode(ctx, batch) != 0)
        {
            fprintf(stderr, "decode failed.\n\n");
            return 1;
        }

        // borrowed pointer into ctx's internal buffer - we don't own it, and it's only valid
        // until the next llama_decode call. -1 = logits for the last position (prediction after the prompt)
        float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits)
        {
            fprintf(stderr, "failed to get the logits.\n\n");
            return 1;
        }

        // Argmax(greedy pick)
        // int next_token = 0;
        // for (int i = 1; i < n_vocab; i++)
        // {
        //     if (logits[i] > logits[next_token])
        //     {
        //         next_token = i;
        //     }
        // }

        // topK
        int next_token = pick_logit_topk(logits, n_vocab);
        if (next_token == -1)
        {
            fprintf(stderr, "failed to get the next token.\n\n");
            return 1;
        }

        // detokenize the logit
        char buf[128];
        int len = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, false);
        printf("\nnext token: '%.*s'  (id=%d, logit=%.2f)\n", len, buf, next_token, logits[next_token]);
        res.append(buf);

        // break after model stops
        if (llama_vocab_is_eog(vocab, next_token))
        {
            break;
        }

        tokens[it] = next_token;
        batch = llama_batch_get_one(&tokens[it], 1);
    }

    std::cout << "\nFinal Response:\n\n"
              << res << std::endl;

    // free the memory - garbage collection
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}