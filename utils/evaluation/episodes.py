from typing import List, Tuple


def extract_episodes(
    binary_sequence: List[bool],
) -> List[Tuple[int, int]]:
    """
    Convert binary anomaly sequence into contiguous episodes.

    Example:
    [0,0,1,1,1,0,1,1]
    ->
    [(2,4), (6,7)]
    """

    episodes = []
    active = False
    start = None
    for i, val in enumerate(binary_sequence):
        if val and not active:
            start = i
            active = True
        elif not val and active:
            episodes.append((start, i - 1))
            active = False
    if active:
        episodes.append((start, len(binary_sequence) - 1))

    return episodes


def match_episodes(
    predicted,
    ground_truth,
):
    """
    Counts TP/FP/FN at episode level.

    A predicted episode is TP if it overlaps
    any GT episode.
    """

    tp = 0
    matched_gt = set()

    for ps, pe in predicted:
        found = False
        for gi, (gs, ge) in enumerate(ground_truth):
            overlap = not (pe < gs or ps > ge)
            if overlap:
                tp += 1
                matched_gt.add(gi)
                found = True
                break
        if not found:
            pass
    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn
