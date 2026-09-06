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
from aiogram.types import ReactionTypeEmoji, InlineQueryResultArticle, InputTextMessageContent
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

# ===================== ПРАВИЛА И СТИЛИ ФОРМАТИРОВАНИЯ =====================
FORMATTING_RULES = """
ПРАВИЛА ОФОРМЛЕНИЯ ОТВЕТОВ (MARKDOWN ТИПОГРАФИКА ДЛЯ TELEGRAM):
1. Структура: Дели ответ на логические секции с аккуратными заголовками (например: '### 📌 Заголовок').
2. Акценты: Выделяй ключевые термины, имена библиотек и главные выводы **жирным шрифтом**.
3. Код в тексте: Имена переменных, типов, функций, методов и параметров ВСЕГДА оборачивай в `моноширинный шрифт`.
4. Блоки кода: ВСЕГДА указывай язык программирования в начале блока (например, ```python, ```javascript, ```typescript, ```go, ```sql). Код должен быть чистым и с пояснениями.
5. Списки: Используй аккуратные маркированные списки с эмодзи-буллетами (•, ✔️, ❌, ⚡, 💡).
6. ТАБЛИЦЫ СТРОГО ЗАПРЕЩЕНЫ: Telegram НЕ умеет отображать Markdown-таблицы (символы | и ---). ВМЕСТО ТАБЛИЦ оформляй сравнительные данные списком карточек с буллетами (например: '• **Ситуация:** Параллельный запуск — **Как обработать:** `Promise.all(...)`').
7. Цитаты и сноски: Важные предупреждения, резюме или выводы оформляй в цитаты через '> Текст цитаты'.
8. Язык: Отвечай на русском языке, живо, профессионально, без воды.
"""

MODES = {
    "mentor": {
        "title": "🧑‍💻 Senior Ментор (по умолчанию)",
        "prompt": (
            "Ты — опытный, дружелюбный Senior Software Engineer и наставник. "
            "Помогай разработчику, подробно отвечай на вопросы, объясняй концепции на пальцах. "
            + FORMATTING_RULES
        ),
    },
    "reviewer": {
        "title": "🔍 Строгий Код-Ревьюер",
        "prompt": (
            "Ты — строгий Principal Code Reviewer. "
            "Проводи аудит кода: скрытые баги, утечки ресурсов, race conditions, "
            "оценка сложности O(N) по времени и памяти, рефакторинг по SOLID/DRY. "
            + FORMATTING_RULES
        ),
    },
    "assistant": {
        "title": "⚡ Быстрый IT-Ассистент",
        "prompt": (
            "Ты — лаконичный и точный AI-помощник разработчика. "
            "Давай краткие, точные ответы по коду и синтаксису без лишних вступлений. "
            + FORMATTING_RULES
        ),
    },
}

MAX_HISTORY_MESSAGES = 10

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".java", ".kt", ".cs", ".php", ".rb", ".sql", ".sh",
    ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt"
}

LANG_EXTENSIONS = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "go": ".go", "golang": ".go",
    "rust": ".rs", "rs": ".rs",
    "cpp": ".cpp", "c++": ".cpp", "c": ".c",
    "java": ".java", "kotlin": ".kt", "cs": ".cs", "csharp": ".cs",
    "sql": ".sql", "bash": ".sh", "sh": ".sh", "shell": ".sh",
    "html": ".html", "css": ".css", "json": ".json", "yaml": ".yaml", "yml": ".yml",
}


# ===================== УТИЛИТЫ ДЛЯ DIFF И КОДА =====================

def extract_primary_code_block(text: str) -> tuple[str, str]:
    """Извлекает основной блок кода и его язык из ответа модели."""
    matches = re.findall(r"```([a-zA-Z0-9_\+\-\#]*)\n?(.*?)```", text, flags=re.DOTALL)
    if not matches:
        return "", ""
    best_lang, best_code = max(matches, key=lambda m: len(m[1].strip()))
    return best_lang.strip().lower(), best_code.strip("\r\n")


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

    # 3. Преобразуем Markdown таблицы до экранирования HTML
    text = convert_markdown_tables(text)

    # 4. Преобразуем маркеры списков (- пункт или * пункт) в аккуратный буллет •
    text = re.sub(r"(?m)^[\-\*]\s+", "• ", text)

    # 5. Экранируем &, <, > в обычном тексте (но защищаем уже созданные <blockquote> и <b> из таблиц)
    # Чтобы не экранировать теги <blockquote>, временно сохраним их или аккуратно заменим
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

    # 6. Инлайн-код `code`
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # 7. Заголовки (без дублирования эмодзи)
    def format_header(m):
        header_text = m.group(1).strip()
        if any(header_text.startswith(e) for e in ("📌", "💡", "🚀", "⚠️", "📂", "🔍", "⚡", "🧪", "📊")):
            return f"<b>{header_text}</b>"
        return f"📌 <b>{header_text}</b>"

    text = re.sub(r"(?m)^#{1,4}\s*(.*?)$", format_header, text)

    # 8. Жирный шрифт (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # 9. Курсив (*text* или _text_)
    text = re.sub(r"(?<!\w)\*([^\*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # 10. Зачеркнутый (~~text~~)
    text = re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

    # 11. Цитаты Telegram (> текст)
    def format_bq(m):
        lines = m.group(0).splitlines()
        cleaned = [re.sub(r"^&gt;\s*", "", line) for line in lines]
        content = "\n".join(cleaned).strip()
        return f"<blockquote>{content}</blockquote>\n"

    text = re.sub(r"(?m)^(?:&gt;\s*.*(?:\n|$))+", format_bq, text)

    # 12. Возвращаем сохраненные блоки кода и эмодзи
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


async def send_formatted_response(chat_id: int, reasoning: str, content: str, show_thinking: bool):
    """
    Отправляет ответ пользователю с гарантированно валидной разметкой Telegram HTML.
    """
    # 1. Если включен режим рассуждений
    if show_thinking and reasoning:
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
    if content:
        chunks = split_markdown_into_chunks(content, max_chars=3500)
        for chunk in chunks:
            formatted_html = markdown_to_telegram_html(chunk)
            try:
                await bot.send_message(chat_id=chat_id, text=formatted_html, parse_mode=ParseMode.HTML)
            except TelegramBadRequest:
                clean_text = re.sub(r"<[^>]+>", "", formatted_html)
                clean_text = html.unescape(clean_text)
                await bot.send_message(chat_id=chat_id, text=clean_text)
    else:
        await bot.send_message(chat_id=chat_id, text="⚠️ Модель вернула пустой ответ.")


# ===================== НАСТОЯЩИЕ ЦВЕТНЫЕ КНОПКИ (BOT API 9.4) =====================

def get_code_keyboard():
    """Настоящие цветные инлайн-кнопки по спецификации Bot API 9.4."""
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [
            types.InlineKeyboardButton(text="🔍 Код-Ревью", callback_data="act_review", style="primary"),
            types.InlineKeyboardButton(text="⚡ Сложность O(N)", callback_data="act_complexity", style="primary"),
        ],
        [
            types.InlineKeyboardButton(text="🧪 Сгенерировать тесты", callback_data="act_tests", style="success"),
            types.InlineKeyboardButton(text="💡 Рефакторинг SOLID", callback_data="act_refactor", style="success"),
        ],
        [
            types.InlineKeyboardButton(text="📊 Показать Git Diff", callback_data="act_diff", style="primary"),
            types.InlineKeyboardButton(text="📝 Документация", callback_data="act_docs"),
        ],
        [
            types.InlineKeyboardButton(text="🗑 Сбросить буфер", callback_data="act_cancel", style="danger"),
        ],
    ])
    return keyboard


def get_mode_keyboard():
    """Выбор режима с цветовым разделением (зеленый / красный / синий)."""
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🧑‍💻 Senior Ментор (Дружелюбный)", callback_data="setmode_mentor", style="success")],
        [types.InlineKeyboardButton(text="🔍 Строгий Ревьюер (Аудит и баги)", callback_data="setmode_reviewer", style="danger")],
        [types.InlineKeyboardButton(text="⚡ Быстрый Ассистент (Лаконичный)", callback_data="setmode_assistant", style="primary")],
    ])
    return keyboard


# ===================== КОМАНДЫ БОТА =====================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await set_safe_reaction(message, "⚡")
    user_id = message.from_user.id
    mode, show_thinking = await db.get_user_settings(user_id)
    mode_name = MODES[mode]["title"]
    thinking_state = "Включен ✅" if show_thinking else "Выключен ❌"

    welcome_text = (
        "👋 <b>Добро пожаловать в Senior AI Code Companion!</b>\n\n"
        "Я ваш персональный AI-ментор по программированию на базе <code>NVIDIA Nemotron 120B</code>.\n\n"
        f"⚙️ <b>Режим работы:</b> {mode_name}\n"
        f"🧠 <b>Показ мыслей (Thinking):</b> {thinking_state}\n"
        "💾 <b>База данных SQLite:</b> Активна (история и настройки сохраняются)\n\n"
        "<b>📌 Доступные команды:</b>\n"
        "• /mode — Сменить стиль и режим работы\n"
        "• /thinking — Вкл/выкл пошаговые рассуждения модели\n"
        "• /clear — Очистить память диалога\n"
        "• /help — Подробная справка\n\n"
        "💬 <i>Отправьте вопрос текстом или прикрепите файл с кодом!</i>"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data == "act_cancel")
async def cb_act_cancel(callback: types.CallbackQuery):
    await db.clear_code(callback.from_user.id)
    await callback.answer("Буфер кода очищен")
    await callback.message.edit_text("🗑 <b>Код успешно удален из памяти бота.</b>", parse_mode=ParseMode.HTML)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await set_safe_reaction(message, "💡")
    help_text = (
        "📚 <b>Как работать с ботом:</b>\n\n"
        "• <b>Вопросы и консультации:</b>\n"
        "Задавайте любые вопросы по Python, JS/TS, Go, базам данных, Linux или алгоритмам. Бот помнит историю диалога навсегда благодаря SQLite базе.\n\n"
        "• <b>Анализ файлов с кодом:</b>\n"
        "Прикрепите файл (<code>.py</code>, <code>.js</code>, <code>.cpp</code> и т.д.) или пришлите код в сообщении. Появятся кнопки:\n"
        "  - 🔍 <b>Код-Ревью</b> — поиск багов, уязвимостей, edge cases\n"
        "  - ⚡ <b>Сложность O(N)</b> — точный расчет времени и памяти\n"
        "  - 🧪 <b>Сгенерировать тесты</b> — создаст готовый скачиваемый файл <code>test_*.py</code>!\n"
        "  - 💡 <b>Рефакторинг SOLID</b> — пришлет скачиваемый файл и Git Diff!\n"
        "  - 📊 <b>Показать Git Diff</b> — покажет наглядное сравнение «Было / Стало»\n"
        "  - 📝 <b>Документация</b> — docstrings и описание типов\n\n"
        "• <b>Вызов в любом чате (Inline Mode):</b>\n"
        "Наберите в чате с коллегой <code>@имя_бота свой вопрос</code> — бот мгновенно сгенерирует сниппет или ответ и позволит отправить его в один клик!\n\n"
        "• <b>Управление:</b>\n"
        "/mode — Выбор стиля ответов\n"
        "/thinking — Показ пошаговых рассуждений AI под спойлером\n"
        "/clear — Сброс памяти текущей беседы"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)


@dp.message(Command("mode"))
async def cmd_mode(message: types.Message):
    await set_safe_reaction(message, "⚙️")
    await message.answer(
        "⚙️ <b>Выберите режим работы бота:</b>\n\n"
        "• <b>Senior Ментор</b> — понятные, глубокие объяснения, дружелюбный стиль, примеры кода.\n"
        "• <b>Строгий Ревьюер</b> — придирчивый аудит, вычисление $O(N)$, архитектурные замечания.\n"
        "• <b>Быстрый Ассистент</b> — сверхкраткие и четкие ответы без предисловий.",
        reply_markup=get_mode_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("setmode_"))
async def cb_set_mode(callback: types.CallbackQuery):
    new_mode = callback.data.replace("setmode_", "")
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
    await message.answer("🧹 <b>Память диалога и буфер кода очищены в SQLite!</b> Начинаем разговор с чистого листа.", parse_mode=ParseMode.HTML)


# ===================== ОБРАБОТКА ФАЙЛОВ И КОДА =====================

@dp.message(F.document)
async def handle_document(message: types.Message):
    file_name = message.document.file_name or ""
    _, ext = os.path.splitext(file_name)

    if ext.lower() not in SUPPORTED_EXTENSIONS:
        await message.answer(f"⚠️ Неподдерживаемый формат: <code>{html.escape(ext)}</code>. Отправьте файл с исходным кодом.", parse_mode=ParseMode.HTML)
        return

    if message.document.file_size and message.document.file_size > 1024 * 1024:
        await message.answer("⚠️ Файл слишком большой. Лимит — 1 МБ.")
        return

    await set_safe_reaction(message, "👨‍💻")
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
        f"📄 Файл <b>{html.escape(file_name)}</b> ({line_count} строк) сохранен в базу.\n"
        "Выберите желаемое действие:",
        reply_markup=get_code_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("act_"))
async def handle_code_action(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    orig_filename, code = await db.get_code(user_id)

    if not code:
        await callback.answer("⚠️ Код не найден в базе. Отправьте файл заново.", show_alert=True)
        return

    action = callback.data.replace("act_", "")
    prompts = {
        "review": "Проведи подробный Code Review этого кода. Раздели ответ на секции: 1. Найденные баги и уязвимости. 2. Краевые случаи (Edge Cases). 3. Рекомендации по исправлению с кодом:\n\n```\n" + code + "\n```",
        "complexity": "Оцени алгоритмическую сложность этого кода: 1. Время выполнения O(...) с подробным объяснением циклов и рекурсии. 2. Память O(...) (Space complexity). 3. Как оптимизировать алгоритм:\n\n```\n" + code + "\n```",
        "tests": "Напиши профессиональный комплект Unit-тестов для этого кода. Включи happy path, граничные значения и исключения. Обязательно оформи весь код тестов в один полный блок ```код```:\n\n```\n" + code + "\n```",
        "docs": "Напиши документацию к этому коду: подробные docstrings для всех методов/классов, описание типов параметров и возвращаемых значений, а также пример использования:\n\n```\n" + code + "\n```",
        "refactor": "Выполни глубокий рефакторинг этого кода в соответствии с принципами SOLID, Clean Code и DRY. В ответе обязательно покажи полную обновленную версию кода в блоке ```код``` и объясни каждое изменение:\n\n```\n" + code + "\n```",
        "diff": "Сделай оптимизированную и чистую версию этого кода, исправив все баги и узкие места. Обязательно предоставь полный готовый код в блоке ```код```:\n\n```\n" + code + "\n```",
    }

    prompt = prompts.get(action)
    if not prompt:
        return

    await callback.answer()
    status_msg = await callback.message.answer("⚡ Senior AI анализирует код, секунду...")

    mode, show_thinking = await db.get_user_settings(user_id)
    sys_prompt = MODES[mode]["prompt"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        # Непрерывная анимация «печатает...» каждые 4 секунды до отправки сообщения
        async with ChatActionSender.typing(chat_id=callback.message.chat.id, bot=bot, interval=4.0):
            reasoning, content = await ask_model(messages, enable_thinking=show_thinking)
            try:
                await status_msg.delete()
            except Exception:
                pass
            await send_formatted_response(callback.message.chat.id, reasoning, content, show_thinking)

            # ===================== ФИЧА 2: АВТОГЕНЕРАЦИЯ СКАЧИВАЕМОГО ФАЙЛА =====================
            lang, extracted_code = extract_primary_code_block(content)
            if extracted_code and action in ("tests", "refactor", "diff"):
                ext = LANG_EXTENSIONS.get(lang) or os.path.splitext(orig_filename)[1] or ".py"
                base_name = os.path.splitext(orig_filename)[0] or "code"

                if action == "tests":
                    out_filename = f"test_{base_name}{ext}"
                    caption = f"🧪 <b>Готовый файл Unit-тестов:</b> <code>{out_filename}</code>"
                elif action == "refactor":
                    out_filename = f"refactored_{base_name}{ext}"
                    caption = f"💡 <b>Готовый файл с рефакторингом:</b> <code>{out_filename}</code>"
                else:
                    out_filename = f"improved_{base_name}{ext}"
                    caption = f"📦 <b>Готовое решение:</b> <code>{out_filename}</code>"

                file_bytes = extracted_code.encode("utf-8")
                file_doc = types.BufferedInputFile(file_bytes, filename=out_filename)
                await bot.send_document(
                    chat_id=callback.message.chat.id,
                    document=file_doc,
                    caption=caption,
                    parse_mode=ParseMode.HTML
                )

                # ===================== ФИЧА 3: ВИЗУАЛЬНЫЙ GIT DIFF =====================
                if action in ("refactor", "diff"):
                    diff_text = generate_visual_diff(code, extracted_code, filename=orig_filename or "code.py")
                    if diff_text:
                        escaped_diff = escape_telegram_html(diff_text)
                        diff_msg = (
                            "📊 <b>Визуальный Git Diff («Было / Стало»):</b>\n"
                            f"<pre><code class=\"language-diff\">{escaped_diff}</code></pre>"
                        )
                        await bot.send_message(
                            chat_id=callback.message.chat.id,
                            text=diff_msg,
                            parse_mode=ParseMode.HTML
                        )
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Ошибка при генерации: {html.escape(str(e))}")


# ===================== ОБЫЧНЫЙ ЧАТ =====================

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text

    # Проверка на отправку кода текстом
    is_code = (
        ("```" in text)
        or ("def " in text and ":" in text)
        or ("class " in text and ":" in text)
        or ("function" in text and "{" in text)
        or ("import " in text and "\n" in text)
        or ("{" in text and "}" in text and ";" in text)
    )

    if is_code and len(text.strip().splitlines()) >= 3:
        await set_safe_reaction(message, "👨‍💻")
        await db.save_code(user_id, "snippet.py", text)
        await message.answer(
            "💻 Код сохранен в базу! Выберите необходимое действие:",
            reply_markup=get_code_keyboard(),
        )
        return

    # Обычный вопрос
    await set_safe_reaction(message, "👀")
    await db.add_message(user_id, "user", text)
    history = await db.get_history(user_id, limit=MAX_HISTORY_MESSAGES)

    mode, show_thinking = await db.get_user_settings(user_id)
    sys_prompt = MODES[mode]["prompt"]

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
    @bot O(N) бинарный поиск
    @bot python singleton
    @bot что такое dead lock
    """
    raw_query = inline_query.query.strip()
    results = []

    if not raw_query:
        # Быстрые подсказки / шаблоны, когда пользователь только напечатал @bot
        hints = [
            (
                "⚡ Оценка сложности O(N)",
                "Пример: @bot O(N) binary search",
                "⚡ <b>Памятка: Оценка сложности алгоритмов $O(N)$</b>\n\n"
                "• <b>O(1)</b> — Константная: доступ по ключу в hash map / массиву по индексу.\n"
                "• <b>O(log N)</b> — Логарифмическая: бинарный поиск, сбалансированные деревья.\n"
                "• <b>O(N)</b> — Линейная: один проход по списку, поиск максимума.\n"
                "• <b>O(N log N)</b> — Квазилинейная: TimSort, MergeSort, QuickSort (avg).\n"
                "• <b>O(N²)</b> — Квадратичная: вложенные циклы, BubbleSort.\n\n"
                "<i>Вызовите бота с конкретным алгоритмом:</i> <code>@bot O(N) quicksort</code>"
            ),
            (
                "🐍 Паттерн Python: Singleton",
                "Быстрый сниппет потокобезопасного синглтона",
                "🐍 <b>Python Thread-Safe Singleton:</b>\n\n"
                "<pre><code class=\"language-python\">import threading\n\n"
                "class Singleton:\n"
                "    _instance = None\n"
                "    _lock = threading.Lock()\n\n"
                "    def __new__(cls, *args, **kwargs):\n"
                "        if not cls._instance:\n"
                "            with cls._lock:\n"
                "                if not cls._instance:\n"
                "                    cls._instance = super().__new__(cls)\n"
                "        return cls._instance</code></pre>\n"
                "<i>Отправлено через @AI_Companion_Bot</i>"
            ),
            (
                "💡 Архитектурный совет: SOLID",
                "Краткая памятка по принципам SOLID для коллег",
                "🏛 <b>Принципы SOLID в разработке:</b>\n\n"
                "• <b>S (Single Responsibility)</b> — один класс решает ровно одну задачу.\n"
                "• <b>O (Open/Closed)</b> — открыт для расширения, закрыт для модификации.\n"
                "• <b>L (Liskov Substitution)</b> — подкласс заменяет базовый класс без сюрпризов.\n"
                "• <b>I (Interface Segregation)</b> — много мелких интерфейсов лучше одного раздутого.\n"
                "• <b>D (Dependency Inversion)</b> — зависимость от абстракций, а не реализаций.\n\n"
                "<i>Отправлено через @AI_Companion_Bot</i>"
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

    # Если запрос введён: обращаемся к NVIDIA Nemotron для мгновенного ответа
    qid = hashlib.md5(raw_query.encode("utf-8")).hexdigest()[:10]

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — Senior AI Code Companion в Telegram. "
                    "Пользователь обратился к тебе через inline-запрос (@bot <запрос>) из группового или личного чата. "
                    "Дай максимально полезный, точный, компактный ответ (до 1500 символов). "
                    "Оформи красиво в Markdown (код в ```язык, акценты жирным, ключевые понятия в `code`). "
                    "Отвечай на русском языке."
                ),
            },
            {"role": "user", "content": raw_query},
        ]

        # Запрашиваем модель без рассуждений (быстрый лаконичный ответ для inline)
        _, raw_answer = await ask_model(messages, enable_thinking=False)
        html_answer = markdown_to_telegram_html(raw_answer)

        footer = f"\n\n<i>💬 Запрос: «{html.escape(raw_query)}»</i>"
        final_text = html_answer + footer

        # Если текст слишком длинный, обрезаем безопасно
        if len(final_text) > 4000:
            final_text = final_text[:3950] + "\n...</i>"

        results.append(
            InlineQueryResultArticle(
                id=f"ans_{qid}",
                title=f"💡 Ответ AI: {raw_query[:40]}",
                description="Отправить готовый разбор и сниппет от AI в текущий чат",
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
        types.BotCommand(command="mode", description="⚙️ Выбрать режим работы"),
        types.BotCommand(command="thinking", description="🧠 Вкл/выкл показ рассуждений AI"),
        types.BotCommand(command="clear", description="🧹 Очистить контекст диалога"),
        types.BotCommand(command="help", description="📚 Справка и примеры"),
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

    print("🚀 Senior AI Code Companion готов к работе!")
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
