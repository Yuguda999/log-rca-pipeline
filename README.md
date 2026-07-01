# Log Root-Cause Classification & Summary Pipeline

A lightweight AI pipeline that classifies system error logs into one of eight
predefined root-cause categories, generates a structured issue summary per log,
and reports evaluation metrics. The core pipeline runs fully offline with no GPU
and no external API calls. An optional local-LLM summary path (via Ollama) is
available behind a flag.

## Quick start

```bash
pip install -r requirements.txt

python cli.py train       # train, cross-validate, write metrics + confusion matrix
python cli.py evaluate    # print saved metrics
python cli.py predict --text "ERROR [db-pool] all 15 connections exhausted."
python cli.py predict --file logs.txt            # one log per line
python cli.py predict --text "..." --llm         # optional local-LLM summary
```

## Project structure

```
log-rca-pipeline/
├── cli.py                 # train / evaluate / predict entry points
├── requirements.txt
├── data/
│   ├── logs.csv           # 120 labeled log entries
│   └── labels.csv         # the 8 root-cause categories + metadata
├── src/
│   ├── config.py          # paths, seed, hyperparameters, thresholds
│   ├── data_loader.py     # load + validate; drops the noisy service column
│   ├── preprocess.py      # text normalization
│   ├── model.py           # word + char TF-IDF -> logistic regression
│   ├── train.py           # cross-validation + fit + persist
│   ├── inference.py       # load model, predict with confidence + review flag
│   ├── summarize.py       # deterministic summary + optional Ollama path
│   └── evaluate.py        # metrics, per-class report, confusion matrix
├── models/
│   └── model.joblib       # trained pipeline
└── outputs/
    ├── metrics.json
    ├── confusion_matrix.png
    └── sample_predictions.json
```

## Model approach and reasoning

The classifier is **word + character TF-IDF features feeding a multinomial
logistic regression**. This was a deliberate choice over a fine-tuned transformer
or an LLM-as-classifier, driven by what the data actually looks like.

Three properties of the dataset shaped the design:

1. **It is small (120 rows) and balanced** (12–18 examples per class). On data
   this size, a high-capacity neural model would overfit and its accuracy
   estimates would be unstable. A linear model over sparse text features is the
   right complexity for the sample size.
2. **The messages are short and heavily templated** (~12 words on average, ~67
   distinct templates after masking volatile tokens). The vocabulary is highly
   discriminative — phrases like *"schema validation failed"*, *"Redis memory
   limit reached"*, or *"429 Too Many Requests"* map almost deterministically to
   a category. A linear model captures this cleanly and remains interpretable:
   you can inspect which tokens drove any prediction.
3. **Character n-grams matter here specifically.** Much of the signal lives in
   structured tokens — `502`, `429`, `401`, `OOM` — that word tokenizers
   fragment or drop. `char_wb` 3–5-grams preserve them, which is why both
   feature families are unioned together.

Choosing classical ML here is the engineering judgment, not a shortcut. It trains
in well under a second, needs no GPU, is fully reproducible, and has zero network
dependencies — matching both the "lightweight prototype" brief and a
locally-hosted, no-external-dependency operating model. A transformer would add
cost and fragility for no measurable accuracy gain on this data.

### Where the LLM belongs: the summary, not the classification

Classification is a deterministic, supervised problem → classical ML. Generating
a readable issue summary is a generation problem → a natural fit for an LLM. The
summary generator is therefore two-tier:

- **Default (deterministic, zero dependencies):** joins the predicted label to the
  category catalog (definition, severity, typical resolution) and extracts
  entities from the raw log with regex — component, HTTP status, latency,
  provider, and affected IDs. Reliable, testable, and instant.
- **Optional (`--llm`, local Ollama):** sends the log + predicted category to a
  small local model (default `llama3.2:3b`) for a natural-language summary. It
  runs on CPU and is **off by default**; if Ollama is not reachable the pipeline
  falls back to the structured summary and records why. The grader can run the
  entire project with nothing installed beyond `requirements.txt`.

## Data preprocessing

- **Dropped the `service` column.** In 108 of 120 rows (90%) the service that
  *emitted* the log does not match the component the error is *about* (e.g.
  `service=fx-rate-fetcher` on a `payment-gateway returned 502` message). The
  root cause lives in the message text; `service` is noise and was excluded as a
  feature.
- **Used `log_message` as the sole input feature**, per the dataset guide.
- **Normalized text** before vectorization (`src/preprocess.py`): masked volatile,
  high-cardinality tokens that carry no class signal and would only add sparse
  noise — IP addresses, entity IDs (`client_…`, `txn_…`), key prefixes, and
  measurements (latency, memory, throughput) → placeholders. **HTTP status codes
  and other small integers are preserved** because they are discriminative across
  categories. Text is lowercased and whitespace-collapsed.

## Evaluation results

Headline metrics come from **stratified 5-fold cross-validation** rather than a
single held-out split. With 120 rows across 8 classes, one test split would have
only ~3 examples per class, making any single-split number unreliable.
Cross-validation uses every row for evaluation exactly once and gives a far more
stable estimate.

Metrics are **macro-averaged** (each class weighted equally), which is the honest
choice for multi-class problems and exposes weak classes that a micro/weighted
average would mask.

| Metric | Score |
| --- | --- |
| Accuracy | 0.925 |
| Precision (macro) | 0.933 |
| Recall (macro) | 0.921 |
| F1 (macro) | 0.922 |
| F1 (weighted) | 0.924 |

Per-class F1: RC-05 1.00, RC-06 0.96, RC-03 0.93, RC-04 0.93, RC-01 0.91,
RC-02 0.91, RC-07 0.91, **RC-08 0.83**. See `outputs/metrics.json` and
`outputs/confusion_matrix.png` for the full breakdown.

## Observed tradeoffs

- **Linear model vs. neural model.** Traded a small theoretical ceiling on
  accuracy for interpretability, speed, reproducibility, and zero infra cost. On
  this data that ceiling is not actually being hit by anything heavier.
- **Cross-validation vs. fixed test split.** Traded a simple one-line "test
  accuracy" for a more trustworthy estimate. The right call given the sample size.
- **`C=10` and aggressive n-grams.** Tuned toward fitting the templated vocabulary
  closely. This is reasonable for a closed, templated log space but is the main
  thing to revisit if the log distribution broadens (see Limitations).
- **Deterministic vs. LLM summary.** Deterministic summaries are reliable and free
  but rigid; the LLM path reads better but adds a dependency and latency. Keeping
  it optional gets both without forcing the cost on anyone.

## Limitations

- **RC-08 (Network/Connectivity) is the weakest class** (0.83 F1, 0.71 recall). Its
  language ("failed to connect", "TLS handshake timeout") overlaps with RC-02
  (DB timeout) and RC-03 (third-party failure), so it is occasionally absorbed by
  those classes. More RC-08 examples or a disambiguating feature would help.
- **Closed-world assumption.** The model only knows these 8 categories and will
  confidently assign one even to a genuinely novel failure mode. The confidence
  threshold (below) is the mitigation, not a fix.
- **Small dataset.** 92.5% on 120 synthetic, templated rows will not transfer
  one-to-one to messier production logs with new templates, languages, and
  multi-cause entries.
- **Single-label only.** Real logs sometimes have compound causes; this is
  single-label classification.
- **Regex entity extraction** in the deterministic summary is tuned to the
  observed formats and will miss novel ones.

## Productionizing this system

**Monitoring.** Log every prediction with its confidence and emit (a) the rolling
predicted-class distribution, (b) the confidence histogram, and (c) the
human-review rate. A sudden shift in class mix or a rising low-confidence rate is
the earliest signal that something upstream changed. Track classification latency
and error rates as standard service metrics.

**Drift detection.** Two complementary signals. *Input drift:* fingerprint each
incoming log into a template (the same masking used in preprocessing) and alarm
when the share of never-before-seen templates crosses a threshold — new templates
are exactly what this closed-world model cannot handle. *Embedding distance:*
periodically compare the distribution of recent logs to the training set and
alert on divergence. *Performance drift:* sample low-confidence and human-reviewed
predictions for labeling and track live accuracy over time.

**Scaling.** Inference is a stateless function over a tiny model artifact, so it
scales horizontally trivially — replicate behind a load balancer and autoscale on
CPU/queue depth. Use batched `predict_proba` for high-throughput streams and put a
queue (e.g. Kafka) in front so spikes are absorbed rather than dropped. The
optional LLM summary, being heavier, should run as a separate async worker pool
(or be reserved for low-confidence/high-severity logs) so it never blocks
classification.

**Reliability.** A confidence threshold (`CONFIDENCE_THRESHOLD` in `config.py`)
routes uncertain predictions to a human-review queue instead of acting on them
automatically — this is the autonomous-investigation-with-human-fallback pattern,
and the routed examples become future training data. Version the model and data
together, keep a pinned environment, and make redeploys reversible. The LLM path
already fails closed (falls back to the structured summary), and the same
principle should apply system-wide: a summary or LLM outage must never take down
classification.

## Reproducibility

Fixed random seed (`config.SEED = 42`), pinned dependencies, and committed
model + metrics artifacts. `python cli.py train` regenerates every number in this
README.
