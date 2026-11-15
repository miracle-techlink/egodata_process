import pytest
from src.annotation.refine_semantics import refined_operations

@pytest.mark.parametrize('text,expected',[
 ('Screw the orange lid onto the transparent plastic jar',{'tighten'}),
 ('Continue to pair the jar with the lid and screw it tight',{'tighten'}),
 ('Load the new staples into the stapler',{'insert'}),
 ('Take the plastic container out of the microwave',{'extract'}),
 ('Cover the bowl with the lid',{'close'}),
 ('Lay the white paper flat on the table',{'place'}),
 ('Clip the blue clip to the edge of the tray',{'clamp'}),
 ('Disassemble the cardboard box and flatten it',{'disassemble','flatten'}),
 ('Divide the flour into containers with the blue scoop',{'portion'}),
 ('Reinforce the side seams on the top of the box with tape',{'attach'}),
 ('Scoop the black granules into the transparent bag',{'scoop'}),
 ])
def test_source_verbs_keep_exact_evidence_and_unknown_physical_targets(text,expected):
    rows=refined_operations(text);assert expected<={x['type'] for x in rows}
    for row in rows:
        assert row['object_id'] is None and not row['label_valid']
        for evidence in row['text_evidence']:
            a,b=evidence['span'];assert text[a:b]==evidence['text']


def test_tool_nouns_do_not_become_scoop_or_clamp_actions():
    assert not {'scoop','clamp'}&{x['type'] for x in refined_operations('Place the blue scoop beside the paper clip.')}
