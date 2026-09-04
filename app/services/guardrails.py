"""Safety guardrails and compliance filters for the Hadaf AI Assistant."""

import re

from app.services.language import SupportedLanguage

# Precompiled regular expressions for sensitive financial credentials
CARD_NUMBER_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
CARD_SANITIZATION_PATTERN = re.compile(r"\b\d(?:\s*-?\s*\d){12,18}\b")
CVV_PIN_PATTERN = re.compile(
    r"\b(cvv|cvc|cvv2|код безопасности|пин|пин[\s-]*код|pin|pin[\s-]*code|пароль|password)\b.*?\b\d{3,6}\b",
    re.IGNORECASE | re.DOTALL,
)
CVV_PIN_SANITIZATION_PATTERN = re.compile(
    r"(?i)\b(cvv2?|cvc|код\s+безопасности|пин[\s-]*код|пин|pin[\s-]*code|pin|пароль|password)\b(\s*[:=\-]?\s*)(\d{3,6})\b"
)

# Root stems indicating request for medical diagnosis, prescriptions, or treatment advice
MEDICAL_ADVICE_STEMS = [
    # Russian
    "диагноз", "лекарств", "таблетк", "дозировк", "вылечить", "лечить", "симптом",
    "антибиотик", "болезн", "препарат", "назначь",
    # Tajik
    "ташхис", "табобат", "дору", "дард", "симптом",
    # English
    "diagnos", "medicine", "prescri", "cure", "treatment", "dosage", "antibiotic",
    "symptom", "pill", "therapy", "disease", "illness",
]

# Keywords indicating request to send money to private cards or unverified channels
UNVERIFIED_TRANSFER_KEYWORDS = [
    # Russian
    "номер твоей карты", "скинь номер карты", "переведу на личную карту", "номер сбера",
    "личный кошелек", "перевод на карту напрямую", "номер карты скинь", "на твою карту",
    # Tajik
    "рақами кортатро бидеҳ", "ба корти шахсӣ мепардозам", "корти шахсӣ", "ба корти ту",
    # English
    "send your card number", "transfer to your personal card", "send to private wallet", "to your card",
]

# Keywords indicating attempt to expose private donor information
DONOR_PRIVACY_KEYWORDS = [
    # Russian
    "кто пожертвовал", "кто перевел деньги на", "список имен доноров", "имя того кто задонатил",
    "раскрой имя донора", "покажи кто оплатил", "кто задонатил", "кто скинул деньги",
    # Tajik
    "кӣ маблағ гузаронд", "кӣ пул дод", "номи донорҳо", "кӣ хайрия кард",
    # English
    "who donated to", "list of donor names", "reveal donor identity", "who gave money", "who donated",
]


class SafetyGuardrails:
    """Safety evaluation and guardrail enforcement engine."""

    @classmethod
    def sanitize_credentials(cls, text: str) -> str:
        """Redact sensitive card numbers and CVV/PIN values from text."""
        # Redact CVV/PIN values while preserving the surrounding label
        sanitized = CVV_PIN_SANITIZATION_PATTERN.sub(r"\1\2[REDACTED]", text)
        # Redact 13-19 digit payment card numbers
        sanitized = CARD_SANITIZATION_PATTERN.sub("[CARD REDACTED]", sanitized)
        return sanitized

    @classmethod
    def check_card_credentials(cls, text: str) -> bool:
        """Return True if message contains sensitive card numbers, CVV, or PINs."""
        if CVV_PIN_PATTERN.search(text):
            return True

        # Check for card number pattern (validate minimum length of continuous digits)
        digits_only = re.sub(r"\D", "", text)
        return bool(len(digits_only) >= 13 and CARD_NUMBER_PATTERN.search(text))

    @classmethod
    def check_medical_advice(cls, text: str) -> bool:
        """Return True if message requests medical diagnosis or treatment advice."""
        t_lower = text.lower()
        return any(stem in t_lower for stem in MEDICAL_ADVICE_STEMS)

    @classmethod
    def check_unverified_transfers(cls, text: str) -> bool:
        """Return True if message inquires about private/unverified financial transfers."""
        t_lower = text.lower()
        return any(k in t_lower for k in UNVERIFIED_TRANSFER_KEYWORDS)

    @classmethod
    def check_donor_privacy(cls, text: str) -> bool:
        """Return True if message attempts to extract private donor identity."""
        t_lower = text.lower()
        return any(k in t_lower for k in DONOR_PRIVACY_KEYWORDS)

    @classmethod
    def evaluate_input(cls, message: str, language: SupportedLanguage = "ru") -> str | None:
        """Evaluate incoming message against safety guardrails.

        Returns a canned, helpful, and safe response if a rule is triggered; otherwise None.
        """
        # Rule 1: Sensitive Card Credentials Protection
        if cls.check_card_credentials(message):
            if language == "tg":
                return (
                    "⚠️ Барои амнияти шумо, ҳеҷ гоҳ рақамҳои кортҳои бонкӣ, рамзҳои CVV/PIN ё паролҳоро "
                    "дар чат нафиристед. Платформаи Ҳадаф маълумоти махфии бонкии шуморо намепурсад. "
                    "Ҳамаи хайрияҳо танҳо тавассути шлюзҳои расмии бонкӣ (Алиф, Тавҳид) анҷом дода мешаванд."
                )
            elif language == "en":
                return (
                    "⚠️ For your security, never share credit card numbers, CVV/PIN codes, or passwords in chat. "
                    "The Hadaf platform never requests sensitive financial credentials. "
                    "All donations are conducted exclusively through official banking payment gateways (Alif, Tavhid)."
                )
            else:
                return (
                    "⚠️ В целях вашей безопасности никогда не отправляйте номера банковских карт, CVV/PIN-коды "
                    "или пароли в чат. Платформа Ҳадаф никогда не запрашивает ваши платёжные реквизиты. "
                    "Все пожертвования проводятся только через официальные платёжные шлюзы банков (Алиф, Тавхид)."
                )

        # Rule 2: No Medical Diagnosis or Treatment Advice
        if cls.check_medical_advice(message):
            if language == "tg":
                return (
                    "Платформаи Ҳадаф маслиҳатҳои тиббӣ, ташхис ва тавсияҳо оид ба табобат намедиҳад. "
                    "Мо танҳо ҳуҷҷатҳои расмии клиникаҳоро барои ҷамъоварии маблағ месанҷем. "
                    "Лутфан, барои масъалаҳои саломатӣ ба духтури ботаҷриба муроҷиат кунед."
                )
            elif language == "en":
                return (
                    "The Hadaf platform does not provide medical advice, diagnoses, or treatment prescriptions. "
                    "We verify official medical documentation and invoices from licensed clinics for fundraising campaigns. "
                    "Please consult a qualified healthcare professional for medical concerns."
                )
            else:
                return (
                    "Платформа Ҳадаф не предоставляет медицинских консультаций, диагнозов и рекомендаций по лечению. "
                    "Мы проверяем официальные документы и счета от лицензированных клиник для сборов. "
                    "По вопросам здоровья, пожалуйста, обратитесь к квалифицированному врачу."
                )

        # Rule 3: No Unverified Private Transfers
        if cls.check_unverified_transfers(message):
            if language == "tg":
                return (
                    "Платформаи Ҳадаф интиқоли маблағ ба кортҳои шахсии шахсони алоҳидаро қабул намекунад. "
                    "Ҳамаи ҷамъовариҳо танҳо тавассути фондҳои тасдиқшудаи хайрия бо пешниҳоди чекҳои расмӣ сурат мегиранд."
                )
            elif language == "en":
                return (
                    "The Hadaf platform does not accept transfers to private personal cards. "
                    "All charitable campaigns are conducted strictly through verified foundations with official fiscal receipts."
                )
            else:
                return (
                    "Платформа Ҳадаф не принимает переводы на личные карты физических лиц. "
                    "Все сборы ведутся строго через верифицированные благотворительные фонды с предоставлением официальных чеков."
                )

        # Rule 4: Donor Privacy (Sadaqah Principle)
        if cls.check_donor_privacy(message):
            if language == "tg":
                return (
                    "Мувофиқи принсипи беном будани хайрия (Садақа) ва сиёсати махфияти платформа, "
                    "номҳои донорҳо ба таври оммавӣ ошкор карда намешаванд. "
                    "Донор метавонад таърихи кумакҳои худро танҳо дар утоқи шахсии худ бинад."
                )
            elif language == "en":
                return (
                    "In accordance with public anonymity (the Sadaqah principle) and our privacy policy, "
                    "donor names are never disclosed publicly. "
                    "Donors can view their personal donation history exclusively inside their private dashboard."
                )
            else:
                return (
                    "В соответствии с принципом публичной анонимности (Садака) и политикой конфиденциальности платформы, "
                    "имена доноров не раскрываются публично. "
                    "Донор может видеть свою историю пожертвований только в своём личном кабинете."
                )

        return None
