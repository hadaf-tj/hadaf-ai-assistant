"""Unit tests for Language Detection and Safety Guardrails."""

from app.services.guardrails import SafetyGuardrails
from app.services.language import detect_language

# ---------------------------------------------------------------------------
# 1. Language Detection Tests
# ---------------------------------------------------------------------------

def test_language_detection_tajik():
    """Test detection of Tajik language."""
    assert detect_language("Салом, платформаи Ҳадаф чӣ гуна кор мекунад?") == "tg"
    assert detect_language("Барои фонди хайрия чӣ ҳуҷҷатҳо лозиманд?") == "tg"
    assert detect_language("Кумаки мо ба кӣ мерасад?") == "tg"


def test_language_detection_russian():
    """Test detection of Russian language."""
    assert detect_language("Здравствуйте! Какая комиссия у платформы?") == "ru"
    assert detect_language("Как проходит верификация благотворительных фондов?") == "ru"
    assert detect_language("Покажите список активных сборов на лечение") == "ru"


def test_language_detection_english():
    """Test detection of English language."""
    assert detect_language("Hello, what is the platform fee for donations?") == "en"
    assert detect_language("How are charitable foundations verified in Tajikistan?") == "en"
    assert detect_language("Show me the latest statistics of helped people.") == "en"


def test_language_detection_fallback():
    """Test default fallback for empty, numeric, or neutral input."""
    assert detect_language("") == "ru"
    assert detect_language("12345") == "ru"
    assert detect_language("   ") == "ru"


# ---------------------------------------------------------------------------
# 2. Medical Advice Guardrail Tests
# ---------------------------------------------------------------------------

def test_guardrails_medical_advice_detection():
    """Test detection of medical diagnostic and treatment inquiries."""
    # Positive detection
    assert SafetyGuardrails.check_medical_advice("Поставь диагноз по моим симптомам") is True
    assert SafetyGuardrails.check_medical_advice("Какое лекарство мне принять от кашля?") is True
    assert SafetyGuardrails.check_medical_advice("Чӣ гуна табобат кардан лозим аст?") is True
    assert SafetyGuardrails.check_medical_advice("What medicine should I take for this pain?") is True
    assert SafetyGuardrails.check_medical_advice("What dosage of antibiotics should I take?") is True

    # Negative detection (legitimate platform queries)
    assert SafetyGuardrails.check_medical_advice("Как зарегистрировать благотворительный фонд?") is False
    assert SafetyGuardrails.check_medical_advice("Какая комиссия у платформы?") is False


def test_guardrails_medical_advice_multilingual_evaluation():
    """Test localized disclaimer responses for medical advice across RU, TG, and EN."""
    # Russian
    res_ru = SafetyGuardrails.evaluate_input(
        "Какое лекарство принять при болях в сердце?", language="ru"
    )
    assert res_ru is not None
    assert "не предоставляет медицинских консультаций" in res_ru
    assert "обратитесь к квалифицированному врачу" in res_ru

    # Tajik
    res_tg = SafetyGuardrails.evaluate_input(
        "Доруи табобат барои ин беморӣ чист?", language="tg"
    )
    assert res_tg is not None
    assert "маслиҳатҳои тиббӣ" in res_tg
    assert "ба духтури ботаҷриба муроҷиат кунед" in res_tg

    # English
    res_en = SafetyGuardrails.evaluate_input(
        "What dosage of antibiotics should I take?", language="en"
    )
    assert res_en is not None
    assert "does not provide medical advice" in res_en
    assert "consult a qualified healthcare professional" in res_en


# ---------------------------------------------------------------------------
# 3. Card Credentials Protection Tests
# ---------------------------------------------------------------------------

def test_guardrails_card_credentials_detection():
    """Test detection of payment card numbers, CVV, and PIN values."""
    # Card number detection (16-digit spaced and hyphenated)
    assert SafetyGuardrails.check_card_credentials("Моя карта 4400 1234 5678 9010") is True
    assert SafetyGuardrails.check_card_credentials("Рақами кортам 5058-2700-1111-2222") is True
    assert SafetyGuardrails.check_card_credentials("Continuous 5058270123456789 card") is True

    # CVV and PIN detection
    assert SafetyGuardrails.check_card_credentials("Вот карта и CVV 123") is True
    assert SafetyGuardrails.check_card_credentials("My CVC: 456") is True
    assert SafetyGuardrails.check_card_credentials("Пин-код от кабинета 1234") is True
    assert SafetyGuardrails.check_card_credentials("PIN code is 4321") is True

    # Negative detection (safe amounts and IDs)
    assert SafetyGuardrails.check_card_credentials("Сколько закрыто сборов?") is False
    assert SafetyGuardrails.check_card_credentials("Пожертвование 1000 сомони") is False
    assert SafetyGuardrails.check_card_credentials("Campaign ID: 1234") is False


def test_guardrails_card_credentials_multilingual_evaluation():
    """Test localized warning responses for card credentials across RU, TG, and EN."""
    # Russian
    res_ru = SafetyGuardrails.evaluate_input(
        "Моя карта для перевода: 4400 4321 8765 4321, cvv: 456", language="ru"
    )
    assert res_ru is not None
    assert "никогда не отправляйте номера банковских карт" in res_ru
    assert "Алиф, Тавхид" in res_ru

    # Tajik
    res_tg = SafetyGuardrails.evaluate_input(
        "Рақами кортам 5058 2700 1111 2222", language="tg"
    )
    assert res_tg is not None
    assert "ҳеҷ гоҳ рақамҳои кортҳои бонкӣ" in res_tg
    assert "Алиф, Тавҳид" in res_tg

    # English
    res_en = SafetyGuardrails.evaluate_input(
        "Here is my card 4000 1234 5678 9010 with pin 4321", language="en"
    )
    assert res_en is not None
    assert "never share credit card numbers" in res_en
    assert "Alif, Tavhid" in res_en


# ---------------------------------------------------------------------------
# 4. Credential Sanitization Tests
# ---------------------------------------------------------------------------

def test_guardrails_credential_sanitization():
    """Test redacting card numbers and CVV/PINs while leaving safe text intact."""
    # Spaced 16-digit card
    assert (
        SafetyGuardrails.sanitize_credentials("My card is 4400 1234 5678 9010")
        == "My card is [CARD REDACTED]"
    )

    # Hyphenated card with CVV
    assert (
        SafetyGuardrails.sanitize_credentials("Card: 5058-2700-1111-2222, CVV: 123")
        == "Card: [CARD REDACTED], CVV: [REDACTED]"
    )

    # Cyrillic PIN code
    assert (
        SafetyGuardrails.sanitize_credentials("ПИН-код: 1234")
        == "ПИН-код: [REDACTED]"
    )

    # Normal donation amounts, dates, and phone numbers must remain unchanged
    assert SafetyGuardrails.sanitize_credentials("Donation amount is 1000") == "Donation amount is 1000"
    assert SafetyGuardrails.sanitize_credentials("In year 2026") == "In year 2026"
    assert SafetyGuardrails.sanitize_credentials("Call +992 900 12 34 56") == "Call +992 900 12 34 56"


# ---------------------------------------------------------------------------
# 5. Unverified Transfers Guardrail Tests
# ---------------------------------------------------------------------------

def test_guardrails_unverified_transfers_detection():
    """Test detection of unverified personal card-to-card transfer attempts."""
    # Positive detection
    assert SafetyGuardrails.check_unverified_transfers("Скинь номер карты для прямого перевода") is True
    assert SafetyGuardrails.check_unverified_transfers("Рақами кортатро бидеҳ пул мегузаронам") is True
    assert SafetyGuardrails.check_unverified_transfers("Please send your card number to transfer directly") is True
    assert SafetyGuardrails.check_unverified_transfers("Переведу на личную карту") is True

    # Negative detection
    assert SafetyGuardrails.check_unverified_transfers("Как пожертвовать через приложение Алиф?") is False


def test_guardrails_unverified_transfers_multilingual_evaluation():
    """Test localized rejection responses for unverified transfers across RU, TG, and EN."""
    # Russian
    res_ru = SafetyGuardrails.evaluate_input(
        "Скинь номер карты для прямого перевода", language="ru"
    )
    assert res_ru is not None
    assert "не принимает переводы на личные карты" in res_ru
    assert "официальных чеков" in res_ru

    # Tajik
    res_tg = SafetyGuardrails.evaluate_input(
        "Рақами кортатро бидеҳ пул мегузаронам", language="tg"
    )
    assert res_tg is not None
    assert "ба кортҳои шахсии шахсони алоҳидаро қабул намекунад" in res_tg

    # English
    res_en = SafetyGuardrails.evaluate_input(
        "Please send your card number to transfer directly", language="en"
    )
    assert res_en is not None
    assert "does not accept transfers to private personal cards" in res_en


# ---------------------------------------------------------------------------
# 6. Donor Privacy (Sadaqah Principle) Guardrail Tests
# ---------------------------------------------------------------------------

def test_guardrails_donor_privacy_detection():
    """Test detection of inquiries attempting to expose individual donor identity."""
    # Positive detection
    assert SafetyGuardrails.check_donor_privacy("Кто пожертвовал деньги на лечение Зафара?") is True
    assert SafetyGuardrails.check_donor_privacy("Ба ин сбор кӣ пул дод?") is True
    assert SafetyGuardrails.check_donor_privacy("Who donated to this campaign?") is True
    assert SafetyGuardrails.check_donor_privacy("Покажи список имен доноров") is True

    # Negative detection (general inquiries about donating)
    assert SafetyGuardrails.check_donor_privacy("Как сделать пожертвование?") is False
    assert SafetyGuardrails.check_donor_privacy("Чӣ тавр хайрия кардан мумкин аст?") is False


def test_guardrails_donor_privacy_multilingual_evaluation():
    """Test localized responses protecting donor anonymity across RU, TG, and EN."""
    # Russian
    res_ru = SafetyGuardrails.evaluate_input(
        "Кто пожертвовал деньги на лечение Зафара?", language="ru"
    )
    assert res_ru is not None
    assert "анонимности (Садака)" in res_ru
    assert "имена доноров не раскрываются" in res_ru

    # Tajik
    res_tg = SafetyGuardrails.evaluate_input(
        "Ба ин сбор кӣ пул дод?", language="tg"
    )
    assert res_tg is not None
    assert "беном будани хайрия (Садақа)" in res_tg

    # English
    res_en = SafetyGuardrails.evaluate_input(
        "Who donated to this campaign?", language="en"
    )
    assert res_en is not None
    assert "Sadaqah principle" in res_en
    assert "never disclosed publicly" in res_en


# ---------------------------------------------------------------------------
# 7. Safe Input Evaluation Test
# ---------------------------------------------------------------------------

def test_guardrails_safe_input_evaluation():
    """Test that legitimate queries trigger no guardrail violations."""
    assert SafetyGuardrails.evaluate_input("Как зарегистрировать фонд?", language="ru") is None
    assert SafetyGuardrails.evaluate_input("Платформа чӣ гуна кор мекунад?", language="tg") is None
    assert SafetyGuardrails.evaluate_input("How does Hadaf verify charities?", language="en") is None
