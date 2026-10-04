import pytest

from stageload import Event, ListSink


def test_as_dict_drops_empty_fields():
    event = Event("load", name="flow", nbytes=10, t=1.5)
    assert event.as_dict() == {"kind": "load", "name": "flow", "nbytes": 10, "t": 1.5}


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown event kind"):
        Event("loaded")


def test_list_sink_keeps_events_in_order():
    sink = ListSink()
    sink.emit(Event("stage", stage="one"))
    sink.emit(Event("mark", note="x"))
    assert [e.kind for e in sink.events] == ["stage", "mark"]
