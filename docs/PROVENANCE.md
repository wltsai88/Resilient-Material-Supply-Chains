# Provenance and reconstruction choices

Source: the supplied `AAI_Manuscript_Full_Unblinded-v2.docx`, especially §3.6 and §4.1–4.5, Tables 6–8. The source document hash is recorded in SOURCE.json. Its original generation script and supporting files were not supplied with the manuscript.

The source describes a **newly generated** synthetic benchmark with this population. Earlier conversation references to unavailable historical data do not replace this document's explicit specification. This package implements that specification but does not recover the exact original records.

## Explicit manuscript parameters

|Item|Specification|
|---|---|
|Seed|20260924, NumPy|
|Population|3,255 unique requests; 2,604 compliant; 651 single-violation requests|
|Families|Seven; 465 requests per family|
|Violation counts|Seven; 93 per category|
|Policies|28 release policies + 56 notes, four revisions; requests use v1/v2|
|Process lower bound|0.12 + 0.035f + 0.035v|
|Process upper bound|0.90 - 0.02f - 0.025v|
|Quota|600 - 100v - 20f|
|Territory|Odd revision: zone_a; even: zone_b|
|Redistribution|(f+v) odd permits it|
|Composition|Three nonnegative fractions; sum within 0.001|
|Compliance generation|Integer quantity uniform 1..quota; process uniform within bounds; Dirichlet(2,3,4), six decimals|
|Violation magnitudes|Quota +1..80; process 0.002..0.08 outside bounds; composition +0.015..0.15 to first fraction|
|Folds|15 groups of 217; five folds each hold out three groups|
|BM25|k1=1.5; b=0.75; positive IDF; binary query occurrence; doc-ID ties|
|TF-IDF|1,800 training-document-frequency features; unigrams/bigrams; pure numbers -> NUM; log TF, smoothed IDF, L2 normalization|
|Logistic fit|250 steps; LR=1; L2=.002; class balance; zero initialization; probability threshold .5|
|Missing policies|Titanium alloy, aluminum alloy, ceramic: remove v2 release policies|
|Bootstrap|1,000 resamples of complete template groups; seed 20260925|
|Cost|N=3,255; c_tx=21,000+512*68=55,816; c_pre=4,500; q=.05,.10,.20,.30,.40|
|Bargaining|N=3,255; 20 rounds; b U(.55,1); s U(.1,.7); p0 U(.4,.9); both decays U(.8,.98)|
|Timing|1.25t+2+0.8 ln(1+c); conditional on agreement|
|Loads and repetitions|10,50,100,500,1000,2000; 10 runs/load; seed 42+repetition+concurrency|

## Choices needed to make an executable implementation

|Unspecified detail|Choice in this repository|Expected effect|
|---|---|---|
|NumPy RNG engine/API|default_rng / PCG64|Same seed alone cannot recover another engine's outputs|
|Family-by-label allocation|372 positive and 93 negative/family; 13/category/family plus cyclic two extras|Meets all global counts; revisions and removed-policy counts may differ|
|Revision indexing|v0,v1,v2,v3; requests v1,v2|Documented resolution of zero-based v=0..3 notation|
|Normal revision distribution|Uniform choice v1/v2; prohibited revision chosen for redistribution violations|Affects version ablation and missing-evidence counts|
|Random draw order|Family loop, shuffled anomaly assignments, per-request fields, global shuffle|Changes individual records and sampled model results|
|Compliant redistribution requests|Bernoulli(0.5) if permitted; false otherwise|Affects text distributions|
|Lower quantity bound|1|No zero/negative production requests generated|
|Context-note rules|Same family/version rule schema copied into each note; explicit nonauthorizing type|Unguarded comparator can consume nonauthorizing text; guard blocks it|
|Full policy wording|Newly authored deterministic policy strings|BM25 results can differ substantially|
|Full text templates|Five headers × three phrasings; rotated/reversed field order|Learned baseline performance can differ substantially|
|Group assignment|Global shuffled records assigned cyclically to 15 groups|Exactly 217/group; three consecutive groups/fold|
|Keyword list|Literal phrase `safety absent` only|Restricted comparator; original keyword list unavailable|
|Equal-frequency feature order|Lexicographic tie-break by feature|Can affect truncated vocabulary|
|Logistic intercept|Unpenalized learned intercept initialized at zero|Original intercept details unavailable|
|Unscoped comparison|Apply predicates of top-ranked document without scope/type guard|Isolates explicit guard; notes share a rule schema|
|Wrong-version injection|Same family v0 policy for every request|Always mismatched with v1/v2|
|Unsafe fallback|Immediately approve missing/mismatched evidence|Deliberately unsafe ablation; no rule check afterward|
|Bargaining draws|One full vector each for b,s,p0,buyer decay,seller decay, in that order|Historical conditional means can differ|
|Rounding|Python round (ties to even)|Can affect integer invalid counts|

Neither policy text nor templates were tuned to match manuscript results. `report.py` contains reported claim counts only to compare outputs after inference; the generator, retriever, classifiers, and rule decisions never consume these counts.

## Evidence boundaries

This implementation uses structured predicates directly associated with fictional policies. It does not extract rules from free-form policies with an LLM. Perfect scoped/direct results are expected on a generator sharing the same specified policy model and establish an internal consistency property, not independent industrial validation.

The cost model uses gas-equivalent accounting for an off-chain check, not EVM gas consumed by that check. The timing model adds concurrency as a logarithmic penalty; it does not implement network scheduling, consensus, queue dynamics, or empirical throughput. Bargaining simulation samples are separate from labeled compliance requests.

There is no field trial, expert annotation, independent real-policy test, measured gas benchmark, or original historical dataset in this package. Use the newly computed tables if reporting this implementation, or supply the original script/data/templates before claiming exact reproduction of historical manuscript tables.
