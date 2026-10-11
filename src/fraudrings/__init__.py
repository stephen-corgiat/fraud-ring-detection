"""fraudrings — simulate, detect and honestly measure coordinated fraud rings.

The package layout is the anti-tautology boundary:

    generator/  -> writes the event stream + a SEPARATE ground-truth file
    detector/   -> reads ONLY the event stream (never ground truth)
    eval/       -> reads ground truth + detector output to score

Because the world is simulated, every detector result comes with a real precision/recall
number against hidden ground truth, across an adversary that tries harder to look organic.
"""

__version__ = "0.0.1"
