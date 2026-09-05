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


# ===================== ПАРСЕР MARKDOWN -> TELEGRAM HTML =====================

def markdown_to_telegram_html(text: str) -> str:
    """
    Преобразует стандартный Markdown модели в валидный Telegram HTML:
    - Блоки кода с указанием языка <pre><code class="language-...">
    - Сворачиваемые цитаты <blockquote expandable> и обычные <blockquote> (фича Bot API)
    - Жирный <b>, курсив <i>, зачеркнутый <s>, код <code>
    - Заголовки с эмодзи
    - Экранирование спецсимволов HTML вне тегов
    """
    code_blocks = []

    # 1. Сохраняем блоки кода, экранируя их содержимое
    def replace_code_block(m):
        lang = m.group(1).strip() if m.group(1) else ""
        code = m.group(2).strip("\r\n")
        escaped_code = html.escape(code)
        idx = len(code_blocks)
        if lang:
            tag = f'<pre><code class="language-{lang}">{escaped_code}</code></pre>'
        else:
            tag = f'<pre>{escaped_code}</pre>'
        code_blocks.append(tag)
        return f"___CODE_BLOCK_{idx}___"

    # Ищем тройные бэктики
    text = re.sub(r"```([a-zA-Z0-9_\+\-\#]*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)

    # 2. Экранируем HTML в обычном тексте
    text = html.escape(text)

    # 3. Инлайн-код `code`
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # 4. Заголовки (###, ##, #)
    text = re.sub(r"(?m)^#{1,4}\s*(.*?)$", r"📌 <b>\1</b>", text)

    # 5. Жирный шрифт (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # 6. Курсив (*text* или _text_)
    text = re.sub(r"(?<!\w)\*([^\*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # 7. Зачеркнутый (~~text~~)
    text = re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

    # 8. Цитаты Telegram (обычные и сворачиваемые)
    # Если цитата многострочная:
    def format_blockquote(match):
        content = match.group(1).strip()
        return f"<blockquote>{content}</blockquote>"

    text = re.sub(r"(?m)^(?:&gt;\s*.*(?:\n|$))+", lambda m: f"<blockquote>{m.group(0).replace('&gt;', '').strip()}</blockquote>\n", text)

    # 9. Возвращаем сохраненные блоки кода
    for idx, cb in enumerate(code_blocks):
        text = text.replace(f"___CODE_BLOCK_{idx}___", cb)

    return text.strip()


def split_text(text: str, max_chars: int = 3900) -> list[str]:
    """Разбивка длинных текстов на части с сохранением структуры."""
    chunks = []
    while len(text) > max_chars:
        split_idx = text.rfind("\n", 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars
        chunks.append(text[:split_idx])
        text = text[split_idx:].strip()
    if text:
        chunks.append(text)
    return chunks


async def set_safe_reaction(message: types.Message, emoji: str):
    """Ставит реакцию на сообщение (новая фича Telegram Bot API)."""
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
    Отправляет ответ с премиальным форматированием:
    - Рассуждения модели в сворачиваемом блоке <blockquote expandable> (фича Telegram 2024-2026).
    - Основной текст с красивой типографикой HTML.
    """
    # 1. Если включен режим рассуждений — выводим его в новом формате сворачиваемой цитаты
    if show_thinking and reasoning:
        escaped_reasoning = html.escape(reasoning)
        spoiler_text = (
            "🧠 <b>Ход рассуждений модели (Chain of Thought):</b>\n"
            f"<blockquote expandable>{escaped_reasoning}</blockquote>"
        )
        for chunk in split_text(spoiler_text, max_chars=3500):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            except Exception:
                await bot.send_message(chat_id=chat_id, text=f"🧠 Ход рассуждений:\n{reasoning}")

    # 2. Основной ответ
    if content:
        formatted_html = markdown_to_telegram_html(content)
        for chunk in split_text(formatted_html, max_chars=3900):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            except TelegramBadRequest:
                # Фоллбэк на обычный текст при непредвиденных ошибках вложенности тегов
                await bot.send_message(chat_id=chat_id, text=chunk)
    else:
        await bot.send_message(chat_id=chat_id, text="⚠️ Модель вернула пустой ответ.")


# ===================== КЛАВИАТУРЫ =====================

def get_code_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="🔍 Код-Ревью и баги", callback_data="act_review")
    builder.button(text="⚡ Оценка O(N) и скорость", callback_data="act_complexity")
    builder.button(text="🧪 Написать Unit-тесты", callback_data="act_tests")
    builder.button(text="📝 Документация", callback_data="act_docs")
    builder.button(text="💡 Рефакторинг по SOLID", callback_data="act_refactor")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def get_mode_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="🧑‍💻 Senior Ментор", callback_data="setmode_mentor")
    builder.button(text="🔍 Строгий Ревьюер", callback_data="setmode_reviewer")
    builder.button(text="⚡ Быстрый Ассистент", callback_data="setmode_assistant")
    builder.adjust(1)
    return builder.as_markup()


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
        "🚀 <b>Главные фичи Telegram Bot API (2024–2026):</b>\n\n"
        "1. <b>Сворачиваемые цитаты (Expandable Blockquotes):</b>\n"
        "<blockquote expandable>Нажмите на этот блок! Он аккуратно сворачивается и разворачивается. В такие блоки наш бот прячет длинные рассуждения и детальные лог-файлы, чтобы не загромождать чат.</blockquote>\n\n"
        "2. <b>Реакции бота на сообщения (Bot Reactions):</b>\n"
        "Бот может ставить эмодзи-реакции на ваши сообщения (обратите внимание на реакцию 🔥 над этой командой)!\n\n"
        "3. <b>Подсветка синтаксиса и копирование кода:</b>\n"
        "<pre><code class=\"language-python\">def solve_problem(code: str):\n"
        "    return 'Багов нет! $O(1)$'</code></pre>\n"
        "В современных клиентах Telegram при клике на блок кода появляется название языка и удобная кнопка копирования.\n\n"
        "4. <b>Telegram Stars & Paid Media:</b>\n"
        "Встроенная платежная система Telegram Stars (валюта <code>XTR</code>) для продажи подписок и цифрового контента прямо в боте.\n\n"
        "5. <b>Telegram Business Bots:</b>\n"
        "Возможность подключать AI-бота к вашему личному Telegram-аккаунту, чтобы бот отвечал вашим клиентам от вашего имени.\n\n"
        "6. <b>Telegram Mini Apps 2.0:</b>\n"
        "Полноэкранный режим, вибрации (Haptic Feedback), доступ к геопозиции и сохранение состояния в Cloud Storage."
    )
    await message.answer(features_text, parse_mode=ParseMode.HTML)


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
        types.BotCommand(command="mode", description="⚙️ Выбрать режим работы"),
        types.BotCommand(command="thinking", description="🧠 Вкл/выкл показ рассуждений AI"),
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
