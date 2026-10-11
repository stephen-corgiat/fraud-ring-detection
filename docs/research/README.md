# Research

The ideas behind the project, in the order you'd want them: how to measure a detector when you have
no labels, what the literature says, and what we'd tell anyone building one.

- [Measuring a fraud-ring detector when you have no labels](method.md): the method, end to end. Build a
  world you know the answer to, turn the adversary into a dial, hold rings out, and price the labels.
- [Literature review](literature-review.md): about thirty papers and industry sources, each with what it
  claims and what this project took from it, or deliberately didn't.
- [Lessons](lessons.md): working rules for detectors against adaptive adversaries. They apply to any
  platform with coordinated abuse.
- [Decisions](../adr/): the design decisions, one record each.

## What the results look like

Every result is a curve over how hard the adversary tries to look organic. Hover for the numbers; the full tables are in the [README](../../README.md#results).

![Three panels across the evasion dial: per-ring AP falls from 1.00 to 0.50 while operational precision stays near 0.95; on held-out rings the learned rule holds recall at about 1.00 while the hand-weighted score falls to 0.37; the seedless ranker puts 100%, 92% and 50% of ring targets near the head.](../images/results/dial.png)<!-- chart: results-dial -->

![Learning curve at the sophisticated level: recall crosses 80% at about 80 random labels and legit false positives drop below 5%; uncertainty sampling crosses recall earlier but its false-positive line wobbles back over 5%.](../images/results/learning_curve.png)<!-- chart: results-learning-curve -->
