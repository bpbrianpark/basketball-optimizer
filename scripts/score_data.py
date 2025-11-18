def score_shot(stats: dict) -> dict:
    """
    Analyzes basketball shooting stats and return score with positive and negative messages

    Args:
        stats: a dictionary with keys such as 'elbow_angle', 'knee_bent_angle' etc.
            - stats attribute suggests:
                - https://www.youtube.com/watch?v=-MxVeYxY7fE
                - https://coachdavelove.com/launch-angle-and-velocity-in-basketball-shooting/

    Returns:
        a dictionary with keys such as 'score', 'strengths', 'weakness', 'metadata'
            - strengths: a list with all good things that were analyzed
            - weakness: a list with all the poor things that were analyzed
    
    """

    dummy_score = 50
    strengths = []
    weaknesses = []


    if 'elbow_angle' in stats:
        angle = stats['elbow_angle']
        if angle < 70:
            weaknesses.append("Elbow Angle Too Low.")
        elif angle > 110:
            weaknesses.append("Elbow Angle Too High")
        else:
            strengths.append("Elbow Angle Just Right")

    # TODO: Add additional scoring metrics here
    # Kuan: new scoring metrics can be determined by trained model through feature importance

    score = max(0, min(100, dummy_score))
    return {'score':score,
            'strengths':strengths,
            'weaknesses':weaknesses,
            'metadata':stats}

def main():
    dummy_stats = {"elbow_angle":30}
    result = score_shot(dummy_stats)
    print(result)

if __name__ == "__main__":
    main()