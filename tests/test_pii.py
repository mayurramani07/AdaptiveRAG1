from adaptive_rag.pii import redact_pii


def test_redact_pii_scrubs_email():
    assert redact_pii("contact me at john.doe@example.com please") == "contact me at [REDACTED_EMAIL] please"


def test_redact_pii_scrubs_phone_number():
    assert "[REDACTED_PHONE]" in redact_pii("call me at 415-555-2671 tomorrow")


def test_redact_pii_scrubs_ssn():
    assert "[REDACTED_SSN]" in redact_pii("my ssn is 123-45-6789")


def test_redact_pii_scrubs_credit_card():
    assert "[REDACTED_CARD]" in redact_pii("card number 4111 1111 1111 1111 expires soon")


def test_redact_pii_leaves_clean_text_untouched():
    assert redact_pii("what is the refund policy") == "what is the refund policy"


def test_redact_pii_handles_multiple_pii_types_in_one_string():
    text = "Email me at a@b.com or call 415-555-2671, SSN 123-45-6789"
    result = redact_pii(text)
    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_SSN]" in result
    assert "a@b.com" not in result
    assert "123-45-6789" not in result
