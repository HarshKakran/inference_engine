#pragma once

extern float T; // temperature
extern int K;   // top-k

float draw();
int pick_logit_topk(float *logits, int n_vocab);
