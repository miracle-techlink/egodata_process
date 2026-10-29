"""Bind sampled decisions to immutable source observations, not regenerated indices."""

def bind_quality_to_tracks(decisions,tracks):
    by_source={}
    known={t['tracklet_id'] for t in tracks}
    for track in tracks:
        source=track.get('source_observation_id')
        if source:by_source.setdefault(source,[]).append(track['tracklet_id'])
    result={}
    for decision in decisions:
        source=decision.get('source_observation_id')
        if source:
            targets=by_source.get(source,[])
        else:
            targets=[decision['tracklet_id']] if decision['tracklet_id'] in known else []
        for target in targets:result[target]=decision
    return result
