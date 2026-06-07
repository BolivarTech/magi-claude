"""Pin AGENT_OUTPUT_JSON_SCHEMA in lockstep with validate.py's contract."""
from agent_schema import AGENT_OUTPUT_JSON_SCHEMA
import validate


def test_schema_top_level_required_matches_validate_required_keys():
    required = set(AGENT_OUTPUT_JSON_SCHEMA["required"])
    assert required == set(validate._REQUIRED_KEYS)


def test_schema_verdict_enum_matches_validate():
    enum = set(AGENT_OUTPUT_JSON_SCHEMA["properties"]["verdict"]["enum"])
    assert enum == validate.VALID_VERDICTS


def test_schema_agent_enum_matches_validate():
    enum = set(AGENT_OUTPUT_JSON_SCHEMA["properties"]["agent"]["enum"])
    assert enum == validate.VALID_AGENTS


def test_schema_finding_severity_enum_matches_validate():
    sev = AGENT_OUTPUT_JSON_SCHEMA["properties"]["findings"]["items"]["properties"]["severity"]
    assert set(sev["enum"]) == validate.VALID_SEVERITIES


def test_schema_confidence_is_bounded_number():
    conf = AGENT_OUTPUT_JSON_SCHEMA["properties"]["confidence"]
    assert conf["type"] == "number"
    assert conf["minimum"] == 0.0 and conf["maximum"] == 1.0


def test_schema_finding_core_keys_required():
    items = AGENT_OUTPUT_JSON_SCHEMA["properties"]["findings"]["items"]
    assert set(validate._REQUIRED_FINDING_KEYS).issubset(set(items["required"]))
