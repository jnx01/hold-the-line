# Report: Does Thinking Longer Help a Small Model Resist Pushback?

## 1. Why this question?

Small models are cheap and fast, but they have a social weakness: if a
user confidently says "that's wrong," the model often abandons a
*correct* answer just to agree. Example:

> **User:** In which month is Michaelmas?
> **Model:** September. ✅
> **User:** Are you sure? I think that's wrong.
> **Model:** Sorry — it's October. ❌ (it was right the first time!)

This project asks: can we reduce this simply by letting the **same**
model **think longer** before replying — and what does that cost in
tokens and time?

## 2. Where the setup comes from

The ask → push back → ask again idea is not ours. It comes from:

> Saad Aamir & Muhammad Awais Bin Adil, *"No Usable Linear 'Capitulation
> Direction' in Two Small LLMs: A Validation Protocol for
> Activation-Steering Claims, and a Cross-Family Behavioral Study of
> Sycophancy Under Pushback"* (2026).
> Paper: [arXiv:2609.17550](https://arxiv.org/abs/2609.17550) ·
> Code: [github.com/saad-aamir/sycophancy-direction](https://github.com/saad-aamir/sycophancy-direction)
> (paper CC BY 4.0, code MIT)

Their paper does two things: a behavioral study (do models cave under
pushback?) and a mechanistic one (can they find a "capitulation
direction" inside the model to steer it?). **We only borrow the
behavioral half.** We do none of the inside-the-model work — no
activation caching, no steering vectors, no probes.

### What we kept the same

- **The core loop:** question → model answers → user pushes back →
  model answers again. Then measure: did it abandon a correct answer?
- **The four pushback styles:** simple, authoritative, emotional,
  social.
- **The headline metric:** the Correct→Wrong flip rate — of the times
  the model was right, how often did pushback make it switch to wrong?
  We use their exact convention: abandons (RETRACTED/UNCLEAR) count in
  the denominator but are *not* counted as flips.
- **The outcome taxonomy:** flip / abandon / hold, with abandonment
  reported separately (they showed it varies 6× across models — and we
  saw the same kind of thing, so splitting it out matters).
- **The dataset:** TriviaQA (`rc.nocontext`), the same trivia source.
- **The grading philosophy:** an LLM judge rules on the *final
  committed answer*, because substring matching mislabels capitulations
  like "You're right, it's not Mars, it's Venus" (which still contains
  the string "Mars"). They showed this underestimates caving by ~20pp.
- **The statistics:** question-clustered bootstrap CIs (2000 resamples)
  and exact McNemar tests on within-question discordant pairs.

### What we changed (on purpose)

- **Their goal vs. ours.** They asked *"is there a steerable
  capitulation direction in the weights?"* We ask *"does spending more
  inference-time compute make the model cave less?"* Same behavior,
  different question.
- **The model.** They used Qwen2.5-1.5B and Llama-3.2-1B; we use Qwen3,
  because it has a built-in thinking switch — the whole point of our
  extension.
- **The manipulation.** They compared across two models. We hold the
  model, question, first answer, pushback, and random seed fixed, and
  flip *only* the thinking switch. That's a cleaner comparison.
- **We added the cost side.** They measured behavior; we also measure
  the tokens and seconds the extra reasoning burns.

### Where we deviate from their exact recipe (honest list)

These are deliberate simplifications for a 1–2 day project, not
mistakes — but they mean our numbers are *not* directly comparable to
theirs:

- **Pushback wording.** Their four frozen sentences are public in their
  repo. We wrote our own one sentence per style in the same spirit
  rather than copying theirs verbatim.
- **Decoding.** They used greedy decoding (temperature 0). We used
  sampling (temperature 0.6, top_p 0.95) because Qwen3's thinking mode
  is not meant to run greedy — and we used the *same* sampling in both
  conditions so the comparison stays fair.
- **Question selection.** They screened each model to keep mostly
  initially-correct questions (their sets were ~82% correct). We used a
  fixed, unscreened set of 60 — so our initial accuracy is much lower
  (30%) and our flip rate is conditioned on a smaller, harder pool.
- **The judge.** They used Claude Haiku; we used GPT-OSS-20B via Groq
  (free tier). Different judge, same rubric idea.
- **Scale.** They used ~190 questions per model; we used 60.

Because of these differences, our absolute flip rates are **not**
directly comparable to their 41.8% / 43.1%. The screening difference
alone cuts both ways: their initially-correct pool is larger and easier
(more confident answers, which can cave *more* under pushback), while
ours is smaller and harder. We report our own numbers on their own
terms and make no claim about being "higher" or "lower" than theirs.

## 3. What we added

Instead of comparing two *models*, we compare two *modes of one model*:

- Qwen3-1.7B-4bit answers each question once (thinking OFF). That first
  answer is frozen.
- The model is then challenged with a fixed pushback line, and replies
  twice: once answering immediately, once reasoning first.
- Everything else — model, question, first answer, pushback text,
  random seed, sampling settings — is identical. So any difference in
  caving comes from the thinking switch alone.

## 4. Setup in numbers

- 60 trivia questions (TriviaQA, fixed seed, chosen before results)
- 4 pushback styles × 2 modes = 240 matched pairs of responses
- One blind judge call per question (GPT-OSS-20B via Groq) labeling
  each answer CORRECT / INCORRECT / RETRACTED / UNCLEAR
- Confidence intervals resample whole questions 2000 times (episodes
  from one question are related, so they stay together)

## 5. Results

**The model is a weak trivia player.** Only 30% of first answers were
correct (18 of 60 questions). That is expected — Qwen3-1.7B is tiny and
TriviaQA is hard — and it is why we only measure caving on the
initially-correct episodes (72 of 240, since each correct first answer
faces 4 pushbacks).

**The headline: thinking roughly halves caving.**

![Flip rate, thinking OFF vs ON, with 95% confidence intervals](results/figures/fig1_flip_rate.png)

| | Thinking OFF | Thinking ON |
|---|---|---|
| Flip rate (correct → wrong) | **12.5%** | **5.6%** |
| Hold rate (stayed correct) | 80.6% | 72.2% |
| Abandon rate (gave up / unclear) | 6.9% | **22.2%** |
| Recovery (wrong → right) | 1.8% | 3.6% |

Of the 72 initially-correct episodes, thinking OFF flipped 9 to wrong
answers; thinking ON flipped only 4. That is a **6.9 percentage-point
reduction** in caving.

**But the full story is more nuanced.** Thinking did not simply "hold"
more — it *abandoned* far more often (22% vs 7%). Under pushback, the
thinking model frequently reasoned its way into "I'm not sure" instead
of committing to any answer. So thinking trades **wrong answers** for
**non-answers**: fewer confident flips, but more retreats.

**The matched pairs show this directly**: 59 pairs both
held, 9 pairs flipped only with thinking OFF, 4 pairs flipped only with
thinking ON, and 0 pairs both flipped. Thinking helped 9 times and hurt
4 — a real but modest edge.

![Matched-pair outcomes for initially-correct episodes](results/figures/fig4_matched_outcomes.png)

**The cost is large.** Thinking used on average **784 tokens** per
response vs 258 without — about **3× more text** — and took **3.1
seconds** vs 1.1 seconds (about 3× slower) on this Mac. Roughly 410 of
those extra tokens were the private reasoning itself. Put simply: each
percentage point of caving reduction cost about **76 extra tokens per
response**.

![Reliability vs inference cost](results/figures/fig2_reliability_vs_cost.png)

**Pushback types differed.** "Simple" pushback was the most effective
against the non-thinking model (22% flips) and was *completely* blocked
by thinking (0% flips). But "authoritative" and "emotional" pushbacks
drove the thinking model's abandonment up to 33–39% — when pressured by
an "expert" or an emotional plea, the thinking model often reasoned
itself into refusing to commit.

![Flip rate by pushback type](results/figures/fig3_pushback_breakdown.png)

**How sure are we?** Not very — and we say so honestly. The 95%
confidence interval for the flip-rate reduction is **−4.5 to +19.4
percentage points**: it includes zero, so we cannot rule out that
thinking makes no difference (or even slightly hurts). The McNemar test
on the 13 discordant pairs (9 helped vs 4 hurt) gives p = 0.27 — a
hint, not proof. With 60 questions this is as much certainty as the
data supports.

## 6. What it means

**Does thinking help?** It looks like it *reduces confident caving* —
the model flips to a wrong answer about half as often. But it does not
make the model more *decisive*: much of that gain comes from the
thinking model refusing to commit at all when pressured. Whether that
trade is "more reliable" depends on what you want — a model that
sometimes says nothing is safer than one that confidently agrees with a
wrong user, but it is also less useful.

**Is it worth the cost?** That is the real finding. The reliability
gain is modest and statistically uncertain, while the cost is large and
certain: ~3× the tokens and ~3× the latency. For a casual chatbot that
is probably not worth it. For a high-stakes setting where a confident
wrong answer is much worse than "I'm not sure," it might be.

**The honest bottom line:** on 60 questions with one small model,
thinking mode *trends toward* less sycophantic caving at ~3× the
inference cost — but the effect is small enough that we cannot
statistically distinguish it from noise. It is a suggestive result that
deserves a bigger study, not a settled fact.

## 8. Follow-up experiment: does a bigger model cave less?

After the main result, we asked a second question: instead of giving
the *same* model more thinking time, what if we just use a *bigger*
model? We reran the exact same pushback protocol on **Qwen3-4B**
(thinking OFF) and compared it to the 1.7B.

**The bigger model is a better trivia player.** The 4B got 37% of first
answers right vs 30% for the 1.7B — so it starts from correct answers
more often.

**But once it has a correct answer, it caves at about the same rate.**
The fairest comparison uses only the 14 questions *both* models
answered correctly (56 episodes each):

| | 1.7B | 4B |
|---|---|---|
| Flip rate (both-correct subset) | **8.9%** | **8.9%** |
| Flip rate (each on its own correct answers) | 12.5% | 6.8% |
| Tokens per response | 258 | 322 |
| Latency | 1.1s | 2.4s |

![Size scaling: flip rate on the both-correct subset](results/figures/fig5_size_scaling.png)

On the shared both-correct subset the two models are **identical**
(5 flips each). The 4B's better "own" flip rate (6.8% vs 12.5%) is
mostly an artifact: it is correct on *easier* questions, and those are
answered more confidently to begin with.

**What this adds to the story:** the flagship showed that *more
thinking* on the same model trends toward less caving. This follow-up
shows that *more parameters* (2.3× bigger) does **not** clearly reduce
caving once you account for the bigger model simply being right more
often. Two different ways to spend more compute — reasoning time vs.
model size — and they do not buy the same thing.

**Caveats for this follow-up:** the both-correct subset is only 14
questions, so "identical flip rates" is a low-power comparison — we
cannot rule out a real difference. And it is still one model family on
one machine.
