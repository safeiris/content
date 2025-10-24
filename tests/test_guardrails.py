from services.guardrails import GuardrailResult, parse_and_repair_jsonld


def test_parse_and_repair_jsonld_success():
    payload = '{"@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Q","acceptedAnswer":{"text":"A"}}]}'
    result = parse_and_repair_jsonld(payload)
    assert isinstance(result, GuardrailResult)
    assert result.ok is True
    assert result.faq_entries == [{"question": "Q", "answer": "A"}]
    assert result.degradation_flags == []


def test_parse_and_repair_jsonld_repairs_single_quotes():
    payload = "{'faq':[{'q':'What?','a':'Answer'}]}"
    result = parse_and_repair_jsonld(payload)
    assert result.ok is True
    assert result.degradation_flags == ["jsonld_repaired"]
    assert result.faq_entries == [{"question": "What?", "answer": "Answer"}]


def test_parse_and_repair_jsonld_missing_payload():
    result = parse_and_repair_jsonld("")
    assert result.ok is False
    assert "jsonld_missing" in result.degradation_flags
