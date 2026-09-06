import os
import io
import re
import html
import asyncio
import difflib
import hashlib
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError, TelegramBadRequest
from aiogram.types import (
    ReactionTypeEmoji,
    InlineQueryResultArticle,
    InputTextMessageContent,
    InputRichMessage,
    InputRichBlockTable,
    InputRichBlockThinking,
    InputRichBlockParagraph,
    InputRichBlockPreformatted,
    InputRichBlockSectionHeading,
    RichBlockTableCell,
)
from aiogram.utils.chat_action import ChatActionSender
import httpx
from openai import AsyncOpenAI

import db

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY") or os.getenv("PROXY")

if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
    print("⚠️ ВНИМАНИЕ: Укажите реальный BOT_TOKEN в файле .env!")

# Настройка прокси
session = None
if TELEGRAM_PROXY and TELEGRAM_PROXY.strip():
    proxy_url = TELEGRAM_PROXY.strip()
    print(f"🌐 Используется прокси: {proxy_url}")
    session = AiohttpSession(proxy=proxy_url)
    http_client = httpx.AsyncClient(proxy=proxy_url)
else:
    http_client = None

bot = Bot(token=BOT_TOKEN or "DUMMY_TOKEN", session=session)
dp = Dispatcher()

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY or "DUMMY_KEY",
    http_client=http_client,
)

# ===================== ПРАВИЛА И СТИЛИ ФОРМАТИРОВАНИЯ ДЛЯ QA =====================
FORMATTING_RULES = """
ПРАВИЛА ОФОРМЛЕНИЯ ОТВЕТОВ ДЛЯ QA MANUAL (RICH MESSAGES ДЛЯ TELEGRAM BOT API 10.1):
1. Структура: Дели ответ на логические секции с аккуратными заголовками (например: '### 📋 Тест-кейсы', '### 🐛 Баг-репорт').
2. Акценты: Выделяй ключевые термины (Severity, Priority, Ожидаемый результат) **жирным шрифтом**.
3. Тестовые данные и локаторы: URL, эндпоинты, селекторы, тестовые строки и пейлоады ВСЕГДА оборачивай в `моноширинный шрифт` или блоки ```code```.
4. Тест-кейсы: Оформляй четко:
   - ID и Название (Title)
   - Тип (Позитивный / Негативный / Граничный)
   - Предусловия (Preconditions)
   - Шаги воспроизведения (Steps) с нумерацией 1, 2, 3
   - Ожидаемый результат (Expected Result) — выделяй **жирным**
5. Баг-репорты: Используй золотой стандарт 'Что? Где? При каких условиях?', указывай Severity / Priority, шаги воспроизведения, Фактический и Ожидаемый результаты.
6. Таблицы: Если генерируешь матрицу тест-дизайна, классы эквивалентности или граничные значения — используй Markdown-таблицы (| Параметр | Валидные | Невалидные |).
7. Списки: Используй аккуратные буллеты (•, ✔️, ❌, ⚠️, 💡).
8. Цитаты: Важные предупреждения и замечания оформляй в цитаты через '> Текст цитаты'.
9. Язык: Отвечай на русском языке, профессионально, дружелюбно и понятно для начинающего QA.
"""

MODES = {
    "mentor": {
        "title": "🧑‍🏫 QA Mentor (Теория, тест-дизайн, собесы)",
        "prompt": (
            "Ты — опытный Lead QA Engineer и доброжелательный наставник для начинающего специалиста по ручному тестированию (Manual QA). "
            "Твоя цель: помогать осваивать профессию QA с нуля. Понятно объясняй теорию тестирования (ISTQB, жизненный цикл дефекта, "
            "виды и уровни тестирования, клиент-серверную архитектуру, DevTools, снифферы Fiddler/Charles, Postman и REST API, SQL для тестировщика). "
            "Разбирай реальные кейсы, давай практические советы, учи грамотно мыслить как тестировщик и готовь к собеседованиям на позицию Junior QA. "
            + FORMATTING_RULES
        ),
    },
    "edge_hunter": {
        "title": "🔍 Bug Hunter (Негативные сценарии и корнер-кейсы)",
        "prompt": (
            "Ты — въедливый Senior QA Engineer, эксперт по исследовательскому тестированию, поиску неочевидных багов и нестандартных сценариев. "
            "Твоя задача — находить самые каверзные краевые случаи (edge cases), уязвимости валидации, проблемы с concurrency, "
            "граничные значения, спецсимволы, SQL-инъекции, XSS-строки, падения при обрыве сети и стрессовые сценарии для любого функционала. "
            + FORMATTING_RULES
        ),
    },
    "fast_qa": {
        "title": "⚡ Fast QA Tool (Быстрый генератор по ISTQB & Jira)",
        "prompt": (
            "Ты — быстрый и строгий генератор QA-артефактов. "
            "Без лишних вступительных слов сразу выдавай готовые профессиональные тест-кейсы, чек-листы, таблицы классов эквивалентности "
            "и баг-репорты по стандартам ISTQB и Jira. Используй четкую структуру и краткие формулировки. "
            + FORMATTING_RULES
        ),
    },
}

MAX_HISTORY_MESSAGES = 10

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".sql", ".html",
    ".xml", ".log", ".py", ".js", ".ts", ".jsx", ".tsx", ".sh"
}

LANG_EXTENSIONS = {
    "csv": ".csv", "markdown": ".md", "md": ".md", "json": ".json",
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "sql": ".sql", "bash": ".sh", "sh": ".sh",
    "html": ".html", "yaml": ".yaml", "yml": ".yml", "xml": ".xml",
}


# ===================== УТИЛИТЫ ДЛЯ ФАЙЛОВ И ТЕСТОВ =====================

def extract_primary_code_block(text: str) -> tuple[str, str]:
    """Извлекает основной блок кода или данных и его язык из ответа модели."""
    matches = re.findall(r"```([a-zA-Z0-9_\+\-\#]*)\n?(.*?)```", text, flags=re.DOTALL)
    if not matches:
        return "", ""
    best_lang, best_code = max(matches, key=lambda m: len(m[1].strip()))
    return best_lang.strip().lower(), best_code.strip("\r\n")


def extract_csv_block(text: str) -> str:
    """Извлекает блок CSV из ответа модели для экспорта в TestRail / Qase."""
    matches = re.findall(r"```(?:csv)?\n?(\"?ID\"?,.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    m_csv = re.findall(r"```csv\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m_csv:
        return m_csv[0].strip()
    return ""


def generate_visual_diff(old_code: str, new_code: str, filename: str = "solution.py") -> str:
    """Генерирует аккуратный unified diff («Было / Стало») для Telegram."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename} (Оригинал)",
        tofile=f"b/{filename} (Рефакторинг)",
        n=2
    ))
    if not diff:
        return ""
    diff_text = "".join(diff)
    if len(diff_text) > 3500:
        diff_text = diff_text[:3500] + "\n... [diff сокращен по лимиту]"
    return diff_text



# ===================== ПАРСЕР MARKDOWN -> TELEGRAM HTML =====================

def escape_telegram_html(s: str) -> str:
    """Telegram HTML поддерживает ТОЛЬКО &lt;, &gt;, &amp;. Кавычки ' и \" экранировать нельзя!"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def convert_markdown_tables(text: str) -> str:
    """
    Преобразует Markdown-таблицы (| Заголовок 1 | Заголовок 2 | ...)
    в элегантные блоки Telegram цитат с аккуратными карточками параметров.
    """
    lines = text.splitlines()
    in_table = False
    table_lines = []
    output_lines = []

    def format_table(tbl):
        if len(tbl) < 2:
            return tbl
        headers = [c.strip() for c in tbl[0].strip().strip("|").split("|")]
        start_row = 1
        if start_row < len(tbl) and re.match(r"^[\s\|:\-]+$", tbl[start_row].strip()):
            start_row = 2
        cards = []
        for row in tbl[start_row:]:
            if not row.strip():
                continue
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if not any(cells):
                continue
            row_items = []
            for i, cell in enumerate(cells):
                hdr = headers[i] if i < len(headers) and headers[i] else f"Параметр {i+1}"
                row_items.append(f"<b>{hdr}:</b> {cell}")
            cards.append("• " + " — ".join(row_items))
        if cards:
            return ["<blockquote>" + "\n".join(cards) + "</blockquote>"]
        return []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            table_lines.append(line)
            in_table = True
        else:
            if in_table:
                output_lines.extend(format_table(table_lines))
                table_lines = []
                in_table = False
            output_lines.append(line)
    if in_table:
        output_lines.extend(format_table(table_lines))

    return "\n".join(output_lines)


def markdown_to_telegram_html(text: str) -> str:
    """
    Преобразует стандартный Markdown модели в валидный Telegram HTML:
    - Блоки кода с указанием языка <pre><code class="language-...">
    - Сворачиваемые цитаты <blockquote expandable> и обычные <blockquote>
    - Автоматическая трансформация таблиц (| ... |) в красивые блоки с карточками
    - Списки (- пункт или * пункт) в аккуратные буллеты (• пункт)
    - Жирный <b>, курсив <i>, зачеркнутый <s>, код <code>
    - Экранирование спецсимволов (&, <, >) без ломающих Telegram сущностей (&quot;, &#x27;)
    """
    code_blocks = []
    emoji_tags = []

    # 1. Сохраняем кастомные премиум эмодзи <tg-emoji ...>...</tg-emoji>
    def replace_emoji(m):
        idx = len(emoji_tags)
        emoji_tags.append(m.group(0))
        return f"___EMOJI_TAG_{idx}___"

    text = re.sub(r"<tg-emoji[^>]*>.*?</tg-emoji>", replace_emoji, text, flags=re.DOTALL)

    # 2. Сохраняем блоки кода, экранируя только &, <, >
    def replace_code_block(m):
        raw_lang = m.group(1).strip().lower() if m.group(1) else ""
        code = m.group(2).strip("\r\n")
        escaped_code = escape_telegram_html(code)
        idx = len(code_blocks)
        if raw_lang:
            tag = f'<pre><code class="language-{raw_lang}">{escaped_code}</code></pre>'
        else:
            tag = f'<pre>{escaped_code}</pre>'
        code_blocks.append(tag)
        return f"___CODE_BLOCK_{idx}___"

    text = re.sub(r"```([a-zA-Z0-9_\+\-\#]*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)

    # 3. Сохраняем инлайн-код `code` до обработки курсива/жирного/экранирования
    inline_code_blocks = []
    def replace_inline_code(m):
        code_content = escape_telegram_html(m.group(1))
        idx = len(inline_code_blocks)
        inline_code_blocks.append(f"<code>{code_content}</code>")
        return f"___INLINE_CODE_{idx}___"

    text = re.sub(r"`([^`\n]+)`", replace_inline_code, text)

    # 4. Преобразуем Markdown таблицы
    text = convert_markdown_tables(text)

    # 5. Преобразуем маркеры списков (- пункт или * пункт) в аккуратный буллет •
    text = re.sub(r"(?m)^[\-\*]\s+", "• ", text)

    # 6. Экранируем &, <, > в обычном тексте (но защищаем уже созданные <blockquote> и <b> из таблиц)
    blockquote_blocks = []
    def save_bq(m):
        idx = len(blockquote_blocks)
        blockquote_blocks.append(m.group(0))
        return f"___TEMP_BQ_{idx}___"

    text = re.sub(r"<blockquote>.*?</blockquote>", save_bq, text, flags=re.DOTALL)

    # Экранируем оставшийся текст
    text = escape_telegram_html(text)

    # Возвращаем сохраненные блоки таблиц
    for idx, bq in enumerate(blockquote_blocks):
        text = text.replace(f"___TEMP_BQ_{idx}___", bq)

    # 7. Заголовки (без дублирования эмодзи)
    def format_header(m):
        header_text = m.group(1).strip()
        if any(header_text.startswith(e) for e in ("📌", "💡", "🚀", "⚠️", "📂", "🔍", "⚡", "🧪", "📊")):
            return f"<b>{header_text}</b>"
        return f"📌 <b>{header_text}</b>"

    text = re.sub(r"(?m)^#{1,4}\s*(.*?)$", format_header, text)

    # 8. Жирный шрифт (**text**) с поддержкой многострочности и жадности
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)

    # 9. Курсив (*text* или _text_)
    text = re.sub(r"(?<!\w)\*([^\*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # 10. Зачеркнутый (~~text~~)
    text = re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

    # 11. Цитаты Telegram (> текст)
    def format_bq(m):
        lines = m.group(0).splitlines()
        cleaned = [re.sub(r"^\s*&gt;\s*", "", line) for line in lines]
        content = "\n".join(cleaned).strip()
        return f"<blockquote>{content}</blockquote>\n"

    text = re.sub(r"(?m)^(?:&gt;\s*.*(?:\n|$))+", format_bq, text)

    # 12. Возвращаем сохраненные блоки кода, инлайн-кода и эмодзи
    for idx, ic in enumerate(inline_code_blocks):
        text = text.replace(f"___INLINE_CODE_{idx}___", ic)

    for idx, cb in enumerate(code_blocks):
        text = text.replace(f"___CODE_BLOCK_{idx}___", cb)

    for idx, em in enumerate(emoji_tags):
        text = text.replace(f"___EMOJI_TAG_{idx}___", em)

    return text.strip()


def split_markdown_into_chunks(text: str, max_chars: int = 3500) -> list[str]:
    """Разбивает Markdown по смысловым параграфам ДО конвертации, чтобы не разрывать HTML-теги."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= max_chars:
            current_chunk += (("\n\n" if current_chunk else "") + p)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def set_safe_reaction(message: types.Message, emoji: str):
    """Ставит реакцию на сообщение (фича Bot API)."""
    try:
        await message.react([ReactionTypeEmoji(emoji=emoji)])
    except Exception:
        pass


async def ask_model(messages: list[dict], enable_thinking: bool = False) -> tuple[str, str]:
    """Запрос к Nemotron-120B."""
    extra_body = {}
    if enable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    response_stream = await client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=messages,
        temperature=0.5,
        top_p=0.9,
        max_tokens=8192,
        extra_body=extra_body,
        stream=True,
    )

    full_reasoning = ""
    full_content = ""

    async for chunk in response_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            full_reasoning += reasoning

        if delta.content is not None:
            full_content += delta.content

    return full_reasoning.strip(), full_content.strip()


def build_rich_blocks_from_markdown(text: str) -> list:
    """
    Разбирает Markdown в нативные блоки Telegram Bot API 10.1 (Rich Messages):
    - InputRichBlockSectionHeading: нативные заголовки (#, ##, ###)
    - InputRichBlockTable: НАСТОЯЩИЕ нативные таблицы (| Header 1 | Header 2 |)
    - InputRichBlockPreformatted: нативные блоки кода с языком программирования
    - InputRichBlockParagraph: обычные текстовые параграфы
    """
    blocks = []
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 1. Нативные таблицы Telegram Bot API 10.1
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                headers = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
                start_row = 1
                if start_row < len(table_lines) and re.match(r"^[\s\|:\-]+$", table_lines[start_row].strip()):
                    start_row = 2
                grid = []
                hdr_row = [types.RichBlockTableCell(text=h, align="left", valign="top", is_header=True) for h in headers]
                grid.append(hdr_row)
                for r in table_lines[start_row:]:
                    cells = [c.strip() for c in r.strip().strip("|").split("|")]
                    row_cells = [types.RichBlockTableCell(text=c, align="left", valign="top") for c in cells]
                    grid.append(row_cells)
                blocks.append(types.InputRichBlockTable(cells=grid, is_bordered=True))
                continue

        # 2. Нативные блоки кода (Preformatted)
        if stripped.startswith("```"):
            lang = stripped.lstrip("`").strip().lower()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            blocks.append(types.InputRichBlockPreformatted(text="\n".join(code_lines), language=lang or None))
            continue

        # 3. Нативные заголовки разделов
        if stripped.startswith("#"):
            h_text = stripped.lstrip("#").strip()
            level = min(max(len(stripped) - len(stripped.lstrip("#")), 1), 3)
            blocks.append(types.InputRichBlockSectionHeading(text=h_text, size=level))
            i += 1
            continue

        # 4. Текстовые параграфы
        para_lines = []
        while i < len(lines):
            l_strip = lines[i].strip()
            if not l_strip or l_strip.startswith("#") or l_strip.startswith("```") or (l_strip.startswith("|") and l_strip.endswith("|") and l_strip.count("|") >= 2):
                break
            para_lines.append(lines[i])
            i += 1

        if para_lines:
            blocks.append(types.InputRichBlockParagraph(text="\n".join(para_lines)))
        else:
            i += 1

    return blocks


async def send_formatted_response(chat_id: int, reasoning: str, content: str, show_thinking: bool):
    """
    Отправляет ответ пользователю через Telegram Bot API 10.1 (Rich Messages) ПО УМОЛЧАНИЮ:
    - Нативный блок InputRichBlockThinking для рассуждений модели.
    - Нативные таблицы InputRichBlockTable, заголовки, блоки кода.
    - Автоматический fallback на классический HTML при старых клиентах/ошибках.
    """
    # 1. Если включен режим рассуждений (Thinking)
    if show_thinking and reasoning:
        sent_rich_thinking = False
        try:
            # Нативный блок рассуждений Bot API 10.1
            thinking_block = types.InputRichBlockThinking(text=reasoning[:3000])
            rich_thinking_msg = types.InputRichMessage(blocks=[thinking_block])
            await bot.send_rich_message(chat_id=chat_id, rich_message=rich_thinking_msg)
            sent_rich_thinking = True
        except Exception:
            sent_rich_thinking = False

        if not sent_rich_thinking:
            # Fallback на красивую сворачиваемую цитату
            escaped_reasoning = escape_telegram_html(reasoning)
            spoiler_text = (
                "🧠 <b>Ход рассуждений модели (Chain of Thought):</b>\n"
                f"<blockquote expandable>{escaped_reasoning}</blockquote>"
            )
            try:
                await bot.send_message(chat_id=chat_id, text=spoiler_text, parse_mode=ParseMode.HTML)
            except Exception:
                clean_reasoning = re.sub(r"<[^>]+>", "", spoiler_text)
                await bot.send_message(chat_id=chat_id, text=clean_reasoning)

    # 2. Основной ответ
    if not content:
        await bot.send_message(chat_id=chat_id, text="⚠️ Модель вернула пустой ответ.")
        return

    # Отправляем с полноценным форматированием Telegram HTML (жирный, курсив, моноширинный, цитаты, буллеты)
    chunks = split_markdown_into_chunks(content, max_chars=3500)
    for chunk in chunks:
        formatted_html = markdown_to_telegram_html(chunk)
        try:
            await bot.send_message(chat_id=chat_id, text=formatted_html, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            # Если Telegram отклонил разметку (например, незакрытый тег в коде) — отправляем безопасный чистый текст
            clean_text = re.sub(r"<[^>]+>", "", formatted_html)
            clean_text = html.unescape(clean_text)
            await bot.send_message(chat_id=chat_id, text=clean_text)


# ===================== НАСТОЯЩИЕ ЦВЕТНЫЕ КНОПКИ (BOT API 9.4) ДЛЯ QA =====================

def get_qa_keyboard():
    """Настоящие цветные инлайн-кнопки по спецификации Bot API 9.4 для QA Manual."""
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [
            types.InlineKeyboardButton(text="📋 Тест-кейсы / Чек-лист", callback_data="act_cases", style="success"),
            types.InlineKeyboardButton(text="🐛 Баг-репорт (Jira)", callback_data="act_bugreport", style="danger"),
        ],
        [
            types.InlineKeyboardButton(text="🎲 Тестовые данные", callback_data="act_data", style="primary"),
            types.InlineKeyboardButton(text="⚠️ Граничные значения (BVA)", callback_data="act_bva", style="primary"),
        ],
        [
            types.InlineKeyboardButton(text="🔍 Анализ ТЗ и рисков", callback_data="act_analysis", style="primary"),
            types.InlineKeyboardButton(text="📥 Скачать CSV (TestRail)", callback_data="act_export", style="success"),
        ],
        [
            types.InlineKeyboardButton(text="🗑 Сбросить объект", callback_data="act_cancel", style="danger"),
        ],
    ])
    return keyboard

get_code_keyboard = get_qa_keyboard


def get_mode_keyboard():
    """Выбор режима с цветовым разделением (зеленый / красный / синий)."""
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🧑‍🏫 QA Ментор (Теория и собесы)", callback_data="setmode_mentor", style="success")],
        [types.InlineKeyboardButton(text="🔍 Bug Hunter (Негативные и Edge Cases)", callback_data="setmode_edge_hunter", style="danger")],
        [types.InlineKeyboardButton(text="⚡ Fast QA Tool (Быстрый генератор)", callback_data="setmode_fast_qa", style="primary")],
    ])
    return keyboard


# ===================== КОМАНДЫ БОТА =====================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await set_safe_reaction(message, "⚡")
    user_id = message.from_user.id
    mode, show_thinking = await db.get_user_settings(user_id)
    mode_name = MODES.get(mode, MODES["mentor"])["title"]
    thinking_state = "Включен ✅" if show_thinking else "Выключен ❌"

    welcome_text = (
        "👋 <b>Добро пожаловать в QA Manual Companion!</b>\n\n"
        "Я твой персональный AI-ментор и ассистент по ручному тестированию ПО на базе <code>NVIDIA Nemotron 120B</code>.\n\n"
        f"⚙️ <b>Режим работы:</b> {mode_name}\n"
        f"🧠 <b>Показ мыслей (Thinking):</b> {thinking_state}\n"
        "💾 <b>База данных SQLite:</b> Активна (история и контекст сохраняются)\n\n"
        "<b>📌 Чем я помогу начинающему QA:</b>\n"
        "• 📋 <b>Тест-дизайн:</b> Генерация позитивных и негативных тест-кейсов, чек-листов\n"
        "• 🐛 <b>Баг-репорты:</b> Оформление дефектов по стандарту «Что? Где? При каких условиях?»\n"
        "• 🎲 <b>Тестовые данные:</b> Граничные значения, спецсимволы, XSS/SQLi строки, JSON payload\n"
        "• 📥 <b>Экспорт в TMS:</b> Скачивание готового <code>.csv</code> файла для TestRail и Qase\n"
        "• 🧑‍🏫 <b>Теория и собеседования:</b> Понятные ответы на любые вопросы по тестированию с нуля\n\n"
        "<b>📌 Доступные команды:</b>\n"
        "• /mode — Выбрать роль (Ментор / Охотник за багами / Быстрый генератор)\n"
        "• /thinking — Вкл/выкл пошаговые рассуждения AI\n"
        "• /clear — Начать диалог с чистого листа\n"
        "• /help — Подробное руководство со шпаргалками\n\n"
        "💬 <i>Отправь описание фичи/ТЗ, форму, JSON или просто задай вопрос по QA!</i>"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data == "act_cancel")
async def cb_act_cancel(callback: types.CallbackQuery):
    await db.clear_code(callback.from_user.id)
    await callback.answer("Буфер объекта очищен")
    await callback.message.edit_text("🗑 <b>Объект тестирования успешно удален из памяти бота.</b>", parse_mode=ParseMode.HTML)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await set_safe_reaction(message, "💡")
    help_text = (
        "📚 <b>Руководство по работе с QA Manual Companion:</b>\n\n"
        "• <b>1. Обучение и теория тестирования:</b>\n"
        "Задавай любые вопросы начинающего тестировщика:\n"
        "  - <i>«Чем отличается Severity от Priority?»</i>\n"
        "  - <i>«Как составить классы эквивалентности для поля возраста от 18 до 65?»</i>\n"
        "  - <i>«Что проверять в Chrome DevTools во вкладке Network?»</i>\n"
        "  - <i>«Объясни разницу между Smoke, Sanity и Regression тестированием»</i>\n\n"
        "• <b>2. Тестирование фичи или требований:</b>\n"
        "Отправь текст требований, ТЗ, описание экрана или файл (<code>.txt</code>, <code>.md</code>, <code>.json</code>, <code>.csv</code> и др.).\n"
        "Появятся быстрые цветные кнопки:\n"
        "  - 📋 <b>Тест-кейсы / Чек-лист</b> — полный комплект позитивных и негативных проверок\n"
        "  - 🐛 <b>Баг-репорт (Jira)</b> — идеальный баг-репорт с шагами и фактическим/ожидаемым результатом\n"
        "  - 🎲 <b>Тестовые данные</b> — спецсимволы, XSS, пустые строки, длинные тексты, JSON payload\n"
        "  - ⚠️ <b>Граничные значения (BVA)</b> — таблица анализа границ и эквивалентных классов\n"
        "  - 🔍 <b>Анализ ТЗ и рисков</b> — поиск нестыковок и вопросов к аналитику\n"
        "  - 📥 <b>Скачать CSV (TestRail)</b> — готовый скачиваемый файл для импорта в TMS\n\n"
        "• <b>3. Быстрый вызов в любом чате (Inline Mode):</b>\n"
        "Введи в чате с коллегой <code>@имя_бота</code>:\n"
        "  - Появятся интерактивные шпаргалки по HTTP-кодам, тест-дизайну и шаблону баг-репорта.\n"
        "  - Напиши вопрос (например, <code>@имя_бота HTTP 403 vs 401</code>) и отправь ответ прямо в диалог!\n\n"
        "• <b>4. Команды управления:</b>\n"
        "/mode — Смена стиля и роли ассистента\n"
        "/thinking — Отображение хода рассуждений модели\n"
        "/clear — Сброс памяти текущей беседы"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)


@dp.message(Command("mode"))
async def cmd_mode(message: types.Message):
    await set_safe_reaction(message, "⚙️")
    await message.answer(
        "⚙️ <b>Выберите режим работы QA-помощника:</b>\n\n"
        "• 🧑‍🏫 <b>QA Ментор</b> — понятные объяснения с нуля, теория ISTQB, подготовка к собеседованиям.\n"
        "• 🔍 <b>Bug Hunter</b> — поиск коварных корнер-кейсов, негативные сценарии, стресс-тесты.\n"
        "• ⚡ <b>Fast QA Tool</b> — мгновенная генерация структурированных чек-листов и баг-репортов.",
        reply_markup=get_mode_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("setmode_"))
async def cb_set_mode(callback: types.CallbackQuery):
    new_mode = callback.data.replace("setmode_", "")
    # Совместимость со старыми сохраненными значениями
    if new_mode == "reviewer":
        new_mode = "edge_hunter"
    elif new_mode == "assistant":
        new_mode = "fast_qa"

    if new_mode in MODES:
        await db.set_user_mode(callback.from_user.id, new_mode)
        title = MODES[new_mode]["title"]
        await callback.answer(f"Режим: {title}")
        await callback.message.edit_text(
            f"✅ <b>Режим сохранен в базу:</b>\n{title}\n\n"
            "Все последующие ответы будут формироваться в этом стиле.",
            parse_mode=ParseMode.HTML,
        )


@dp.message(Command("thinking"))
async def cmd_thinking(message: types.Message):
    user_id = message.from_user.id
    new_state = await db.toggle_user_thinking(user_id)
    state_str = "ВКЛЮЧЕН ✅" if new_state else "ВЫКЛЮЧЕН ❌"
    desc = (
        "Теперь перед каждым ответом будет выводиться блок <blockquote expandable>Ход рассуждений AI</blockquote> со всеми внутренними шагами мышления модели."
        if new_state
        else "Ответы будут приходить сразу в готовом и чистом виде без внутренних монологов."
    )
    await message.answer(
        f"🧠 <b>Режим рассуждений (Thinking): {state_str}</b>\n\n{desc}",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    await set_safe_reaction(message, "🧹")
    user_id = message.from_user.id
    await db.clear_history(user_id)
    await db.clear_code(user_id)
    await message.answer("🧹 <b>Память диалога и буфер объекта очищены в SQLite!</b> Начинаем разговор с чистого листа.", parse_mode=ParseMode.HTML)


# ===================== ОБРАБОТКА ФАЙЛОВ И ТРЕБОВАНИЙ =====================

@dp.message(F.document)
async def handle_document(message: types.Message):
    file_name = message.document.file_name or ""
    _, ext = os.path.splitext(file_name)

    if ext.lower() not in SUPPORTED_EXTENSIONS:
        await message.answer(f"⚠️ Неподдерживаемый формат: <code>{html.escape(ext)}</code>. Отправьте файл требований, спецификацию или данные (<code>.txt</code>, <code>.md</code>, <code>.json</code>, <code>.csv</code> и др.).", parse_mode=ParseMode.HTML)
        return

    if message.document.file_size and message.document.file_size > 1024 * 1024:
        await message.answer("⚠️ Файл слишком большой. Лимит — 1 МБ.")
        return

    await set_safe_reaction(message, "📝")
    status_msg = await message.answer(f"📥 Загрузка <code>{html.escape(file_name)}</code>...", parse_mode=ParseMode.HTML)

    file_bytes = io.BytesIO()
    await bot.download(message.document, destination=file_bytes)
    code_content = file_bytes.getvalue().decode("utf-8", errors="replace")

    user_id = message.from_user.id
    await db.save_code(user_id, file_name, code_content)

    try:
        await status_msg.delete()
    except Exception:
        pass

    line_count = len(code_content.splitlines())
    await message.answer(
        f"📄 Объект тестирования <b>{html.escape(file_name)}</b> ({line_count} строк) сохранен в базу.\n"
        "Выберите желаемое QA-действие:",
        reply_markup=get_qa_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("act_"))
async def handle_code_action(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    orig_filename, code = await db.get_code(user_id)

    if not code:
        await callback.answer("⚠️ Объект тестирования не найден в базе. Отправьте файл или ТЗ заново.", show_alert=True)
        return

    action = callback.data.replace("act_", "")
    legacy_map = {
        "review": "analysis",
        "tests": "cases",
        "complexity": "bva",
        "docs": "data",
        "refactor": "bugreport",
        "diff": "export",
    }
    action = legacy_map.get(action, action)

    prompts = {
        "cases": (
            "Ты — ведущий QA Engineer. На основе следующего объекта (требования/функционал/код) "
            "составь профессиональный набор тест-кейсов и чек-лист для ручного тестирования:\n"
            "1. 🟢 Позитивные проверки (Happy Path)\n"
            "2. 🔴 Негативные проверки (невалидные данные, спецсимволы, пустые поля, лимиты)\n"
            "3. ⚠️ Граничные значения\n"
            "4. 🔒 Базовые проверки безопасности и прав доступа (если применимо)\n\n"
            "Каждый тест-кейс оформляй со структурой:\n"
            "- **ID**: TC-01\n"
            "- **Название**: Краткая цель проверки\n"
            "- **Предусловия**: Необходимое начальное состояние\n"
            "- **Шаги воспроизведения**: 1, 2, 3...\n"
            "- **Ожидаемый результат**: Четкое ожидаемое поведение (выдели **жирным**)\n\n"
            "В конце ответа ОБЯЗАТЕЛЬНО сформируй блок ```csv ... ``` с компактными тест-кейсами для скачивания.\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
        "bugreport": (
            "Ты — Senior QA Engineer. Составь исчерпывающий, профессиональный баг-репорт (Bug Report) "
            "по дефекту в данном функционале (или если в тексте описана проблема — оформи ее в идеальный отчет):\n\n"
            "1. 📌 **Summary (Заголовок)** по золотому стандарту 'Что? Где? При каких условиях?'\n"
            "2. ⚡ **Severity & Priority**: Выбери уровень (Blocker / Critical / Major / Minor) и обоснуй\n"
            "3. 💻 **Environment (Окружение)**: OS, Browser/Device, Version\n"
            "4. 🚪 **Preconditions (Предусловия)**: Начальное состояние системы и тестовый аккаунт\n"
            "5. 👣 **Steps to Reproduce (Шаги воспроизведения)**: 1, 2, 3...\n"
            "6. ❌ **Actual Result (Фактический результат)**: Что произошло ошибочно\n"
            "7. ✔️ **Expected Result (Ожидаемый результат)**: Как система должна работать по ТЗ\n"
            "8. 📎 **Attachments & Workaround**: Что приложить (скриншот, HAR-файл, логи) и есть ли обходной путь\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
        "data": (
            "Ты — эксперт по тестированию данных и безопасности. "
            "Сгенерируй всесторонний набор тестовых данных (Test Data) для проверки этой формы/поля/API:\n"
            "1. 🟢 **Валидные данные** (типичные значения, минимальная и максимальная длина, допустимые спецсимволы)\n"
            "2. 🔴 **Невалидные данные** (превышение лимита, пробелы в начале/конце, табы, эмодзи, переполнение int)\n"
            "3. 💣 **Строки для проверок безопасности** (XSS-векторы, SQL-инъекции, спецсимволы HTML/URL, кавычки)\n"
            "4. 📦 **Готовый JSON Payload** (если применимо для REST API)\n\n"
            "Оформляй все тестовые строки в моноширинном виде `значение` или в блоках ```код```, "
            "чтобы тестировщик мог скопировать их в 1 клик.\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
        "bva": (
            "Ты — эксперт по техникам тест-дизайна. Примени к данному объекту Equivalence Partitioning и BVA:\n"
            "1. 📐 **Классы эквивалентности**: разбей входные данные на валидные и невалидные классы.\n"
            "2. ⚠️ **Анализ граничных значений (BVA)**: определи границы диапазонов (Min-1, Min, Min+1, Max-1, Max, Max+1).\n"
            "3. 📊 **Сводная таблица**: оформи результат в наглядную Markdown-таблицу:\n"
            "| Поле/Параметр | Класс эквивалентности | Тестовое значение | Тип (Валид/Невалид) | Ожидаемый результат |\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
        "analysis": (
            "Ты — Senior QA Lead. Проведи анализ требований / ТЗ к этому объекту на тестируемость (Requirements Review):\n"
            "1. ❓ **Пробелы и неясности в ТЗ**: что не описано, какие сценарии забыли упомянуть?\n"
            "2. ⚠️ **Потенциальные риски**: где чаще всего будут возникать ошибки (UX, интеграции, валидация)?\n"
            "3. 💡 **Вопросы к аналитику / разработчику**: список конкретных вопросов для уточнения требований до релиза.\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
        "export": (
            "Ты — QA инженер. На основе этого объекта сгенерируй готовый валидный CSV файл для импорта в TestRail / Qase TMS.\n"
            "ОБЯЗАТЕЛЬНО выведи результат в блоке ```csv\n"
            "\"ID\",\"Title\",\"Type\",\"Preconditions\",\"Steps\",\"Expected Result\",\"Priority\"\n"
            "...тест-кейсы...\n"
            "```\n"
            "Составь не менее 6-8 подробных кейсов (позитивные, негативные, граничные). Кавычки экранируй удвоением.\n\n"
            "Объект тестирования:\n```\n" + code + "\n```"
        ),
    }

    prompt = prompts.get(action)
    if not prompt:
        return

    await callback.answer()
    status_msg = await callback.message.answer("⚡ QA-Ассистент анализирует объект, секунду...")

    mode, show_thinking = await db.get_user_settings(user_id)
    sys_prompt = MODES.get(mode, MODES["mentor"])["prompt"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        async with ChatActionSender.typing(chat_id=callback.message.chat.id, bot=bot, interval=4.0):
            reasoning, content = await ask_model(messages, enable_thinking=show_thinking)
            try:
                await status_msg.delete()
            except Exception:
                pass
            await send_formatted_response(callback.message.chat.id, reasoning, content, show_thinking)

            # ===================== АВТОГЕНЕРАЦИЯ СКАЧИВАЕМОГО ФАЙЛА =====================
            csv_content = extract_csv_block(content)
            base_name = os.path.splitext(orig_filename)[0] or "feature"

            if csv_content and action in ("export", "cases"):
                csv_filename = f"test_cases_{base_name}.csv"
                file_bytes = csv_content.encode("utf-8-sig")  # utf-8-sig для отличного открытия в Excel / Windows
                file_doc = types.BufferedInputFile(file_bytes, filename=csv_filename)
                await bot.send_document(
                    chat_id=callback.message.chat.id,
                    document=file_doc,
                    caption=f"📥 <b>Готовый CSV для TestRail / Qase:</b> <code>{csv_filename}</code>",
                    parse_mode=ParseMode.HTML
                )
            elif action == "cases":
                # Отправляем полный чек-лист в Markdown формате
                md_filename = f"checklist_{base_name}.md"
                file_bytes = content.encode("utf-8")
                file_doc = types.BufferedInputFile(file_bytes, filename=md_filename)
                await bot.send_document(
                    chat_id=callback.message.chat.id,
                    document=file_doc,
                    caption=f"📋 <b>Скачать чек-лист / тест-кейсы:</b> <code>{md_filename}</code>",
                    parse_mode=ParseMode.HTML
                )
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Ошибка при анализе: {html.escape(str(e))}")


# ===================== ОБЫЧНЫЙ ЧАТ =====================

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text

    # Проверка: прислал ли пользователь описание фичи / ТЗ / форму / JSON / объект для тестирования
    text_lower = text.lower()
    qa_keywords = [
        "фича", "feature", "тз", "требован", "форма", "кнопк", "эндпоинт",
        "тестир", "чек-лист", "баг", "сценари", "страниц", "поле", "авториз",
        "регистрац", "корзин", "валидац", "swagger", "postman", "api", "payload",
        "input", "button", "endpoint", "login", "signup", "checkout"
    ]
    has_qa_keywords = any(kw in text_lower for kw in qa_keywords)
    is_multiline_spec = (len(text.strip().splitlines()) >= 3 and (has_qa_keywords or any(c in text for c in [":", "->", "-", "*"])))
    is_code_or_json = ("```" in text) or (text.strip().startswith("{") and text.strip().endswith("}"))

    # Если это простой короткий вопрос начинающего QA:
    is_simple_question = text.strip().endswith("?") and len(text.strip().splitlines()) <= 2 and not is_code_or_json

    if (is_multiline_spec or is_code_or_json) and not is_simple_question and len(text.strip()) >= 20:
        await set_safe_reaction(message, "📝")
        await db.save_code(user_id, "requirement_spec.txt", text)
        await message.answer(
            "📋 <b>Объект тестирования сохранен в память!</b>\n\n"
            "Выберите необходимое QA-действие в меню ниже:",
            reply_markup=get_qa_keyboard(),
            parse_mode=ParseMode.HTML,
        )
        return

    # Обычный вопрос или консультация по QA
    await set_safe_reaction(message, "👀")
    await db.add_message(user_id, "user", text)
    history = await db.get_history(user_id, limit=MAX_HISTORY_MESSAGES)

    mode, show_thinking = await db.get_user_settings(user_id)
    sys_prompt = MODES.get(mode, MODES["mentor"])["prompt"]

    full_messages = [{"role": "system", "content": sys_prompt}] + history

    try:
        # Непрерывная анимация «печатает...» каждые 4 секунды до момента полного ответа
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot, interval=4.0):
            reasoning, content = await ask_model(full_messages, enable_thinking=show_thinking)
            if content:
                await db.add_message(user_id, "assistant", content)
            await send_formatted_response(message.chat.id, reasoning, content, show_thinking)
    except Exception as e:
        error_text = str(e)
        if "451" in error_text:
            await message.answer(
                "⚠️ <b>Ошибка доступа к AI (HTTP 451):</b>\n"
                "NVIDIA API блокирует запросы из вашего региона без VPN/прокси.\n"
                "Включите VPN или укажите <code>TELEGRAM_PROXY</code> в файле <code>.env</code>.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(f"⚠️ <b>Ошибка:</b> {html.escape(error_text)}", parse_mode=ParseMode.HTML)


# ===================== ИНЛАЙН РЕЖИМ (@bot в любом чате) =====================

@dp.inline_query()
async def inline_query_handler(inline_query: types.InlineQuery):
    """
    Позволяет вызывать бота в ЛЮБОМ чате с коллегой через:
    @bot_username <запрос>
    Например:
    @bot HTTP 401 vs 403
    @bot баг-репорт корзина
    @bot чек-лист формы логина
    """
    raw_query = inline_query.query.strip()
    results = []

    if not raw_query:
        # Быстрые шпаргалки для QA, когда пользователь только напечатал @bot
        hints = [
            (
                "🌐 Шпаргалка: HTTP-коды ответов API",
                "200, 201, 400, 401, 403, 404, 422, 500, 502",
                "🌐 <b>Шпаргалка QA: Основные HTTP-коды REST API</b>\n\n"
                "• <b>200 OK</b> — Успешный запрос с телом ответа\n"
                "• <b>201 Created</b> — Ресурс успешно создан (POST)\n"
                "• <b>204 No Content</b> — Успешно, но тела ответа нет (DELETE)\n"
                "• <b>400 Bad Request</b> — Ошибка валидации параметров клиентом\n"
                "• <b>401 Unauthorized</b> — Не авторизован (нет токена/куки)\n"
                "• <b>403 Forbidden</b> — Авторизован, но нет прав на действие\n"
                "• <b>404 Not Found</b> — Эндпоинт или ресурс не найден\n"
                "• <b>422 Unprocessable</b> — Семантическая ошибка валидации\n"
                "• <b>500 Internal Error</b> — Необработанное исключение бэкенда\n"
                "• <b>502 / 504 Gateway</b> — Сервер за шлюзом/прокси недоступен\n\n"
                "<i>Отправлено через QA Manual Companion</i>"
            ),
            (
                "🐛 Шаблон идеального баг-репорта",
                "Золотой стандарт оформления дефектов для Jira",
                "🐛 <b>Шаблон идеального баг-репорта (Jira / YouTrack):</b>\n\n"
                "<b>📌 Title (Что? Где? Когда?):</b>\n"
                "<code>[Авторизация] Ошибка 500 при вводе спецсимволов в поле Email</code>\n\n"
                "• <b>Severity:</b> Major | <b>Priority:</b> High\n"
                "• <b>Environment:</b> Chrome 124, Windows 11, стенд Staging\n"
                "• <b>Preconditions:</b> Пользователь не авторизован\n"
                "• <b>Steps to Reproduce:</b>\n"
                "  1. Открыть страницу /login\n"
                "  2. В поле Email ввести: <code>test'--@mail.com</code>\n"
                "  3. Нажать кнопку «Войти»\n"
                "• <b>Actual Result:</b> Отображается белый экран и ошибка 500\n"
                "• <b>Expected Result:</b> Валидационное сообщение «Неверный формат email»\n\n"
                "<i>Отправлено через QA Manual Companion</i>"
            ),
            (
                "📐 Тест-дизайн: BVA & Классы эквивалентности",
                "Памятка по граничным значениям с примером",
                "📐 <b>Тест-дизайн: Классы эквивалентности и BVA</b>\n\n"
                "<i>Пример: Поле 'Возраст' принимает от 18 до 65 лет включительно.</i>\n\n"
                "• <b>Классы эквивалентности:</b>\n"
                "  - Невалидный (&lt; 18)\n"
                "  - Валидный (18 .. 65)\n"
                "  - Невалидный (&gt; 65)\n\n"
                "• <b>Граничные значения (BVA):</b>\n"
                "  - Нижняя граница: <code>17</code> (невалид), <code>18</code> (валид), <code>19</code> (валид)\n"
                "  - Верхняя граница: <code>64</code> (валид), <code>65</code> (валид), <code>66</code> (невалид)\n\n"
                "<i>Отправлено через QA Manual Companion</i>"
            ),
        ]

        for idx, (title, desc, text_content) in enumerate(hints):
            results.append(
                InlineQueryResultArticle(
                    id=f"hint_{idx}",
                    title=title,
                    description=desc,
                    input_message_content=InputTextMessageContent(
                        message_text=text_content,
                        parse_mode=ParseMode.HTML,
                    ),
                )
            )

        await inline_query.answer(results, cache_time=30, is_personal=True)
        return

    # Если запрос введён: обращаемся к модели для ответа по QA
    qid = hashlib.md5(raw_query.encode("utf-8")).hexdigest()[:10]

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — Senior QA Manual Companion в Telegram. "
                    "Пользователь обратился к тебе через inline-запрос (@bot <запрос>) из группового или личного чата. "
                    "Дай максимально полезный, структурированный и понятный ответ для специалиста по ручному тестированию (до 1500 символов). "
                    "Оформи красиво в Markdown (акценты жирным, ключевые понятия и локаторы в `code`). "
                    "Отвечай на русском языке."
                ),
            },
            {"role": "user", "content": raw_query},
        ]

        _, raw_answer = await ask_model(messages, enable_thinking=False)
        html_answer = markdown_to_telegram_html(raw_answer)

        footer = f"\n\n<i>💬 QA Запрос: «{html.escape(raw_query)}»</i>"
        final_text = html_answer + footer

        if len(final_text) > 4000:
            final_text = final_text[:3950] + "\n...</i>"

        results.append(
            InlineQueryResultArticle(
                id=f"ans_{qid}",
                title=f"💡 QA Разбор: {raw_query[:40]}",
                description="Отправить структурированный ответ от QA-эксперта в текущий чат",
                input_message_content=InputTextMessageContent(
                    message_text=final_text,
                    parse_mode=ParseMode.HTML,
                ),
            )
        )
    except Exception as e:
        err_msg = f"⚠️ <b>Ошибка генерации:</b> <code>{html.escape(str(e))}</code>"
        results.append(
            InlineQueryResultArticle(
                id=f"err_{qid}",
                title="⚠️ Ошибка генерации ответа",
                description=str(e)[:60],
                input_message_content=InputTextMessageContent(
                    message_text=err_msg,
                    parse_mode=ParseMode.HTML,
                ),
            )
        )

    await inline_query.answer(results, cache_time=60, is_personal=True)


# ===================== РЕГИСТРАЦИЯ КОМАНД И СТАРТ =====================

async def setup_bot_commands():
    commands = [
        types.BotCommand(command="start", description="🚀 Перезапуск / Статус"),
        types.BotCommand(command="mode", description="⚙️ Выбрать режим QA-помощника"),
        types.BotCommand(command="thinking", description="🧠 Вкл/выкл показ рассуждений AI"),
        types.BotCommand(command="clear", description="🧹 Очистить контекст диалога"),
        types.BotCommand(command="help", description="📚 Справка и примеры для QA"),
    ]
    await bot.set_my_commands(commands)


async def main():
    if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
        print("❌ ОШИБКА: Пожалуйста, вставьте валидный BOT_TOKEN в файл .env!")
        return

    print("💾 Инициализация базы данных SQLite...")
    await db.init_db()

    print("🚀 Регистрация меню команд Telegram...")
    try:
        await setup_bot_commands()
    except Exception as e:
        print(f"Предупреждение при регистрации команд: {e}")

    print("🚀 QA Manual Companion готов к работе!")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except TelegramNetworkError:
        print("\n" + "=" * 60)
        print("❌ ОШИБКА СЕТИ TELEGRAM:")
        print("Провайдер блокирует прямой доступ к api.telegram.org:443.")
        print("Включите VPN или укажите TELEGRAM_PROXY в файле .env.")
        print("=" * 60 + "\n")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

