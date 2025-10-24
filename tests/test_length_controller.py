from length_controller import ensure_article_length
from validators import length_no_spaces


def _base_article() -> str:
    faq_pairs = []
    for idx in range(1, 6):
        faq_pairs.append(f"**Вопрос {idx}.** Что-то?")
        faq_pairs.append(f"**Ответ.** Ответ {idx}.")
        faq_pairs.append("")
    faq_section = "\n".join(["## FAQ", "<!--FAQ_START-->"] + faq_pairs[:-1] + ["<!--FAQ_END-->"])
    return (
        "# Заголовок\n\n"
        "## Раздел 1\n\n"
        "Короткий абзац с базовой мыслью.\n\n"
        "## Раздел 2\n\n"
        "Ещё один короткий абзац.\n\n"
        f"{faq_section}\n\n"
        "## Заключение\n\n"
        "Финальный вывод.\n"
    )


def test_ensure_article_length_extends_short_article():
    article = _base_article()
    result = ensure_article_length(article, min_chars=800, max_chars=1200, protected_blocks=[])

    assert result.success is True
    assert result.adjusted is True
    assert result.length >= 800
    assert length_no_spaces(result.text) == result.length


def test_ensure_article_length_trims_long_article():
    extra = "\n\n".join(f"Дополнительный абзац {idx}." for idx in range(12))
    article = _base_article().replace("## FAQ", f"{extra}\n\n## FAQ")
    result = ensure_article_length(article, min_chars=500, max_chars=600, protected_blocks=[])

    assert result.success is True
    assert result.length <= 600
    assert length_no_spaces(result.text) == result.length
