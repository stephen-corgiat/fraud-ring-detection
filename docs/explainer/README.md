# How the detector works, in seven short lessons

A plain-language course on the seeded engine — no formulas you have to derive, an analogy per idea,
two quiz questions per lesson, and an "in one breath" summary you can say out loud to a colleague.
Each lesson ends by pointing at the exact place in the code it describes.

| # | Lesson | The idea |
|---|---|---|
| 1 | [Narrow, then Rank](lessons/0001-narrow-then-rank.html) | The spine of the detector: a cheap wide net, then careful ranking. |
| 2 | [Only Rare Clues Count](lessons/0002-only-rare-clues-count.html) | Why a shared coffee-shop Wi-Fi proves nothing and a shared obscure device proves a lot. |
| 3 | [Four Lines of Evidence](lessons/0003-four-lines-of-evidence.html) | How the rank step blends linkage, behavior, co-target and action mix into one score. |
| 4 | [Acting Alike](lessons/0004-acting-alike.html) | The behavioral signal — the one that sounds like scary math, defused. |
| 5 | [The Scorecard](lessons/0005-the-scorecard.html) | Precision and recall, and the ceiling retrieval puts on recall. |
| 6 | [When "Wrong" Isn't Wrong](lessons/0006-when-wrong-isnt-wrong.html) | Why flagging a member of *another* ring isn't a false positive. |
| 7 | [Following the Money](lessons/0007-following-the-money.html) | What happens when the adversary gets careful, and why the detector degrades gracefully. |

The lessons are standalone HTML pages (GitHub shows their source rather than rendering them). Read
them on the [project site](https://stephen-corgiat.github.io/fraud-ring-detection/explainer/), or serve them locally after cloning:

```sh
python -m http.server --directory docs/explainer 8000   # then open http://localhost:8000/lessons/
```
