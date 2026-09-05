import os
import io
import re
import html
import asyncio
from collections import defaultdict
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError, TelegramBadRequest
from aiogram.types import ReactionTypeEmoji
import httpx
from openai import AsyncOpenAI

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
ПРАВИЛА ОФОРМЛЕНИЯ ОТВЕТОВ (MARKDOWN ТИПОГРАФИКА):
1. Структура: Дели ответ на логические секции с аккуратными заголовками (например: '### 📌 Заголовок').
2. Акценты: Выделяй ключевые термины, имена библиотек и главные выводы **жирным шрифтом**.
3. Код в тексте: Имена переменных, типов, функций, методов и параметров ВСЕГДА оборачивай в `моноширинный шрифт`.
4. Блоки кода: ВСЕГДА указывай язык программирования в начале блока (например, ```python, ```go, ```typescript, ```bash, ```sql). Код должен быть чистым и с комментариями.
5. Списки: Используй аккуратные маркированные списки с эмодзи-буллетами (•, ✔️, ❌, ⚡, 💡).
6. Цитаты и сноски: Важные предупреждения, резюме или выводы оформляй в цитаты через '> Текст цитаты'.
7. Язык: Отвечай на русском языке, живо, профессионально, без воды.
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

user_modes = defaultdict(lambda: "mentor")
user_thinking = defaultdict(lambda: False)
user_history = defaultdict(list)
last_code_cache = {}

MAX_HISTORY_MESSAGES = 10

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".java", ".kt", ".cs", ".php", ".rb", ".sql", ".sh",
    ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt"
}


def escape_telegram_html(s: str) -> str:
    """Telegram HTML поддерживает ТОЛЬКО &lt;, &gt;, &amp;. Кавычки ' и \" экранировать нельзя!"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_telegram_html(text: str) -> str:
    """
    Преобразует стандартный Markdown модели в валидный Telegram HTML:
    - Блоки кода с указанием языка <pre><code class="language-...">
    - Сворачиваемые цитаты <blockquote expandable> и обычные <blockquote>
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
        lang = m.group(1).strip() if m.group(1) else ""
        code = m.group(2).strip("\r\n")
        escaped_code = escape_telegram_html(code)
        idx = len(code_blocks)
        if lang:
            tag = f'<pre><code class="language-{lang}">{escaped_code}</code></pre>'
        else:
            tag = f'<pre>{escaped_code}</pre>'
        code_blocks.append(tag)
        return f"___CODE_BLOCK_{idx}___"

    text = re.sub(r"```([a-zA-Z0-9_\+\-\#]*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)

    # 3. Экранируем &, <, > в обычном тексте
    text = escape_telegram_html(text)

    # 4. Инлайн-код `code`
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # 5. Заголовки (без дублирования эмодзи)
    def format_header(m):
        header_text = m.group(1).strip()
        if any(header_text.startswith(e) for e in ("📌", "💡", "🚀", "⚠️", "📂", "🔍", "⚡", "🧪")):
            return f"<b>{header_text}</b>"
        return f"📌 <b>{header_text}</b>"

    text = re.sub(r"(?m)^#{1,4}\s*(.*?)$", format_header, text)

    # 6. Жирный шрифт (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # 7. Курсив (*text* или _text_)
    text = re.sub(r"(?<!\w)\*([^\*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # 8. Зачеркнутый (~~text~~)
    text = re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

    # 9. Цитаты Telegram
    def format_bq(m):
        lines = m.group(0).splitlines()
        cleaned = [re.sub(r"^&gt;\s*", "", line) for line in lines]
        content = "\n".join(cleaned).strip()
        return f"<blockquote>{content}</blockquote>\n"

    text = re.sub(r"(?m)^(?:&gt;\s*.*(?:\n|$))+", format_bq, text)

    # 10. Возвращаем сохраненные блоки кода и эмодзи
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
        # Разбиваем текст по логическим блокам ДО парсинга, чтобы не рвать открытые теги
        chunks = split_markdown_into_chunks(content, max_chars=3500)
        for chunk in chunks:
            formatted_html = markdown_to_telegram_html(chunk)
            try:
                await bot.send_message(chat_id=chat_id, text=formatted_html, parse_mode=ParseMode.HTML)
            except TelegramBadRequest as e:
                # В случае непредвиденного сбоя парсинга Telegram очищаем теги, чтобы не показывать сырой HTML-код
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
            types.InlineKeyboardButton(text="🧪 Unit-тесты", callback_data="act_tests", style="success"),
            types.InlineKeyboardButton(text="💡 Рефакторинг SOLID", callback_data="act_refactor", style="success"),
        ],
        [
            types.InlineKeyboardButton(text="📝 Документация", callback_data="act_docs"),
            types.InlineKeyboardButton(text="🗑 Сбросить", callback_data="act_cancel", style="danger"),
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


# ===================== КОМАНДЫ =====================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await set_safe_reaction(message, "⚡")
    user_id = message.from_user.id
    mode_name = MODES[user_modes[user_id]]["title"]
    thinking_state = "Включен ✅" if user_thinking[user_id] else "Выключен ❌"

    welcome_text = (
        "👋 <b>Добро пожаловать в Senior AI Code Companion!</b>\n\n"
        "Я ваш персональный AI-ментор по программированию на базе <code>NVIDIA Nemotron 120B</code>.\n\n"
        f"⚙️ <b>Режим работы:</b> {mode_name}\n"
        f"🧠 <b>Показ мыслей (Thinking):</b> {thinking_state}\n\n"
        "<b>📌 Быстрые команды:</b>\n"
        "• /mode — Сменить режим работы бота\n"
        "• /thinking — Вкл/выкл показ рассуждений модели\n"
        "• /features — Новейшие фичи Telegram Bot API\n"
        "• /clear — Очистить память диалога\n"
        "• /help — Подробная справка\n\n"
        "💬 <i>Просто отправьте мне вопрос текстом или пришлите файл с кодом!</i>"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.message(Command("features"))
async def cmd_features(message: types.Message):
    """Демонстрация последних фич Telegram Bot API."""
    await set_safe_reaction(message, "🔥")
    features_text = (
        "🚀 <b>Главные фичи Telegram Bot API 9.4 (2024–2026):</b>\n\n"
        "1. <b>Настоящие цветные кнопки (style):</b>\n"
        "Попробуйте команду /demo94 — Telegram официально добавил стили <code>danger</code> (красный), <code>success</code> (зеленый) и <code>primary</code> (синий)!\n\n"
        "2. <b>Сворачиваемые цитаты (Expandable Blockquotes):</b>\n"
        "<blockquote expandable>Нажмите на этот блок! Он аккуратно сворачивается и разворачивается. В такие блоки наш бот прячет длинные рассуждения и детальные лог-файлы, чтобы не загромождать чат.</blockquote>\n\n"
        "3. <b>Реакции бота на сообщения (Bot Reactions):</b>\n"
        "Бот может ставить эмодзи-реакции на ваши сообщения (обратите внимание на реакцию 🔥 над этой командой)!\n\n"
        "4. <b>Темы (Topics) прямо в диалогах:</b>\n"
        "Команда /topics демонстрирует создание топиков методом <code>createForumTopic</code>.\n\n"
        "5. <b>Подсветка синтаксиса и копирование кода:</b>\n"
        "<pre><code class=\"language-python\">def solve_problem(code: str):\n"
        "    return 'Багов нет! $O(1)$'</code></pre>\n"
        "6. <b>Премиум эмодзи без Fragment:</b>\n"
        "Если у владельца бота есть Telegram Premium, бот может присылать кастомные анимированные эмодзи в тексте и кнопках!"
    )
    await message.answer(features_text, parse_mode=ParseMode.HTML)


@dp.message(Command("demo94"))
async def cmd_demo94(message: types.Message):
    """Демонстрация цветных кнопок Bot API 9.4 из статьи на Хабре."""
    await set_safe_reaction(message, "🔥")
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [
            types.InlineKeyboardButton(text="🔴 Опасное действие (danger)", callback_data="demo_danger", style="danger"),
            types.InlineKeyboardButton(text="🟢 Успешное действие (success)", callback_data="demo_success", style="success"),
        ],
        [
            types.InlineKeyboardButton(text="🔵 Основное действие (primary)", callback_data="demo_primary", style="primary"),
            types.InlineKeyboardButton(text="⚪ Стандартная серая", callback_data="demo_default"),
        ],
    ])
    demo_text = (
        "🎨 <b>Демонстрация Telegram Bot API 9.4:</b>\n\n"
        "В этом сообщении используются <b>настоящие цветные кнопки</b> (поле <code>style</code>):\n"
        "• 🔴 <code>danger</code> — красная кнопка\n"
        "• 🟢 <code>success</code> — зелёная кнопка\n"
        "• 🔵 <code>primary</code> — синяя кнопка\n"
        "• ⚪ <i>default</i> — стандартная серая кнопка\n\n"
        "Нажмите любую кнопку ниже для интерактивного ответа:"
    )
    await message.answer(demo_text, reply_markup=kb, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data.startswith("demo_"))
async def cb_demo(callback: types.CallbackQuery):
    kind = callback.data.replace("demo_", "")
    descriptions = {
        "danger": "🔴 Вы нажали кнопку со стилем <b>danger</b> (красный цвет в клиентах 9.4).",
        "success": "🟢 Вы нажали кнопку со стилем <b>success</b> (зелёный цвет в клиентах 9.4).",
        "primary": "🔵 Вы нажали кнопку со стилем <b>primary</b> (синий цвет в клиентах 9.4).",
        "default": "⚪ Вы нажали стандартную прозрачно-серую кнопку.",
    }
    msg = descriptions.get(kind, "Кнопка нажата!")
    await callback.answer(f"Стиль: {kind}")
    await callback.message.reply(msg, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data == "act_cancel")
async def cb_act_cancel(callback: types.CallbackQuery):
    last_code_cache.pop(callback.from_user.id, None)
    await callback.answer("Код удален из буфера")
    await callback.message.edit_text("🗑 <b>Код успешно удален из памяти бота.</b>", parse_mode=ParseMode.HTML)


@dp.message(Command("topics"))
async def cmd_topics(message: types.Message):
    """Создание тем (Topics) в чате (Bot API 9.4)."""
    await set_safe_reaction(message, "📌")
    try:
        topic1 = await bot.create_forum_topic(chat_id=message.chat.id, name="🔍 Код-Ревью и Баги", icon_color=0x6FB9F0)
        topic2 = await bot.create_forum_topic(chat_id=message.chat.id, name="💡 Вопросы и Архитектура", icon_color=0xFFD67E)
        await message.answer(
            f"✅ <b>Темы успешно созданы:</b>\n"
            f"• <code>{topic1.name}</code> (ID: {topic1.message_thread_id})\n"
            f"• <code>{topic2.name}</code> (ID: {topic2.message_thread_id})\n\n"
            "<i>(Примечание: для создания тем в личных диалогах у пользователя и клиента должна быть включена поддержка тем в личке Bot API 9.4)</i>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await message.answer(
            f"ℹ️ <b>Информация о темах (Topics):</b>\n"
            f"Telegram ответил: <code>{html.escape(str(e))}</code>\n\n"
            "Темы в личных чатах доступны, если в вашем клиенте Telegram включен режим форумов для личных чатов или если бот добавлен в супергруппу с темами.",
            parse_mode=ParseMode.HTML,
        )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await set_safe_reaction(message, "💡")
    help_text = (
        "📚 <b>Как работать с ботом:</b>\n\n"
        "• <b>Вопросы и консультации:</b>\n"
        "Задавайте любые вопросы по Python, JS/TS, Go, базам данных, Linux или алгоритмам. Бот помнит контекст предыдущих сообщений.\n\n"
        "• <b>Анализ файлов с кодом:</b>\n"
        "Прикрепите файл (<code>.py</code>, <code>.js</code>, <code>.cpp</code> и т.д.) или пришлите код в сообщении. Появятся кнопки:\n"
        "  - 🔍 <b>Код-Ревью</b> — поиск багов, уязвимостей, edge cases\n"
        "  - ⚡ <b>Сложность O(N)</b> — точный расчет времени и памяти\n"
        "  - 🧪 <b>Unit-тесты</b> — генерация тестового набора\n"
        "  - 💡 <b>Рефакторинг</b> — чистый код по SOLID/DRY\n"
        "  - 📝 <b>Документация</b> — docstrings и описание типов\n\n"
        "• <b>Управление:</b>\n"
        "/mode — Выбор одного из 3 стилей ответов\n"
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
        user_modes[callback.from_user.id] = new_mode
        title = MODES[new_mode]["title"]
        await callback.answer(f"Режим: {title}")
        await callback.message.edit_text(
            f"✅ <b>Режим успешно изменен на:</b>\n{title}\n\n"
            "Все последующие ответы будут формироваться в этом стиле.",
            parse_mode=ParseMode.HTML,
        )


@dp.message(Command("thinking"))
async def cmd_thinking(message: types.Message):
    user_id = message.from_user.id
    current = user_thinking[user_id]
    user_thinking[user_id] = not current
    state_str = "ВКЛЮЧЕН ✅" if not current else "ВЫКЛЮЧЕН ❌"
    desc = (
        "Теперь перед каждым ответом будет выводиться блок <blockquote expandable>Ход рассуждений AI</blockquote> со всеми внутренними шагами мышления модели."
        if not current
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
    user_history[user_id].clear()
    last_code_cache.pop(user_id, None)
    await message.answer("🧹 <b>Память диалога очищена!</b> Задайте новый вопрос.", parse_mode=ParseMode.HTML)


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
    last_code_cache[user_id] = code_content

    try:
        await status_msg.delete()
    except Exception:
        pass

    line_count = len(code_content.splitlines())
    await message.answer(
        f"📄 Файл <b>{html.escape(file_name)}</b> ({line_count} строк) загружен.\n"
        "Выберите желаемое действие:",
        reply_markup=get_code_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("act_"))
async def handle_code_action(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    code = last_code_cache.get(user_id)

    if not code:
        await callback.answer("⚠️ Код не найден в памяти. Отправьте файл заново.", show_alert=True)
        return

    action = callback.data.replace("act_", "")
    prompts = {
        "review": "Проведи подробный Code Review этого кода. Раздели ответ на секции: 1. Найденные баги и уязвимости. 2. Краевые случаи (Edge Cases). 3. Рекомендации по исправлению с кодом:\n\n```\n" + code + "\n```",
        "complexity": "Оцени алгоритмическую сложность этого кода: 1. Время выполнения O(...) с подробным объяснением циклов и рекурсии. 2. Память O(...) (Space complexity). 3. Как оптимизировать алгоритм:\n\n```\n" + code + "\n```",
        "tests": "Напиши профессиональный комплект Unit-тестов для этого кода с проверкой happy path, граничных значений и исключений:\n\n```\n" + code + "\n```",
        "docs": "Напиши документацию к этому коду: подробные docstrings для всех методов/классов, описание типов параметров и возвращаемых значений, а также пример использования:\n\n```\n" + code + "\n```",
        "refactor": "Выполни глубокий рефакторинг этого кода в соответствии с принципами SOLID, Clean Code и DRY. Покажи улучшенную версию кода и объясни каждое изменение:\n\n```\n" + code + "\n```",
    }

    prompt = prompts.get(action)
    if not prompt:
        return

    await callback.answer()
    status_msg = await callback.message.answer("⚡ Senior AI анализирует код, секунду...")
    await bot.send_chat_action(chat_id=callback.message.chat.id, action=ChatAction.TYPING)

    mode = user_modes[user_id]
    sys_prompt = MODES[mode]["prompt"]
    show_thinking = user_thinking[user_id]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        reasoning, content = await ask_model(messages, enable_thinking=show_thinking)
        try:
            await status_msg.delete()
        except Exception:
            pass
        await send_formatted_response(callback.message.chat.id, reasoning, content, show_thinking)
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
        last_code_cache[user_id] = text
        await message.answer(
            "💻 Код принят! Выберите необходимое действие:",
            reply_markup=get_code_keyboard(),
        )
        return

    # Обычный вопрос
    await set_safe_reaction(message, "👀")
    history = user_history[user_id]
    history.append({"role": "user", "content": text})

    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
        user_history[user_id] = history

    mode = user_modes[user_id]
    sys_prompt = MODES[mode]["prompt"]
    show_thinking = user_thinking[user_id]

    full_messages = [{"role": "system", "content": sys_prompt}] + history

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        reasoning, content = await ask_model(full_messages, enable_thinking=show_thinking)
        if content:
            history.append({"role": "assistant", "content": content})
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


# ===================== РЕГИСТРАЦИЯ КОМАНД И СТАРТ =====================

async def setup_bot_commands():
    commands = [
        types.BotCommand(command="start", description="🚀 Перезапуск / Статус"),
        types.BotCommand(command="demo94", description="🎨 Демо цветных кнопок 9.4"),
        types.BotCommand(command="mode", description="⚙️ Выбрать режим работы"),
        types.BotCommand(command="thinking", description="🧠 Вкл/выкл показ рассуждений AI"),
        types.BotCommand(command="topics", description="📌 Создать темы в чате"),
        types.BotCommand(command="features", description="🔥 Фичи Telegram Bot API"),
        types.BotCommand(command="clear", description="🧹 Очистить контекст диалога"),
        types.BotCommand(command="help", description="📚 Справка и примеры"),
    ]
    await bot.set_my_commands(commands)


async def main():
    if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
        print("❌ ОШИБКА: Пожалуйста, вставьте валидный BOT_TOKEN в файл .env!")
        return

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
