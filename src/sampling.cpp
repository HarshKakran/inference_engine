#include "sampling.h"

#include <cmath>
#include <queue>
#include <random>
#include <utility>
#include <vector>

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
