import os
import io
import html
import asyncio
from collections import defaultdict
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramNetworkError
import httpx
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY") or os.getenv("PROXY")

if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
    print("⚠️ ВНИМАНИЕ: Укажите реальный BOT_TOKEN в файле .env!")

# Настройка прокси для Telegram
session = None
if TELEGRAM_PROXY and TELEGRAM_PROXY.strip():
    proxy_url = TELEGRAM_PROXY.strip()
    print(f"🌐 Используется прокси для Telegram и AI: {proxy_url}")
    session = AiohttpSession(proxy=proxy_url)
    http_client = httpx.AsyncClient(proxy=proxy_url)
else:
    http_client = None

bot = Bot(token=BOT_TOKEN or "DUMMY_TOKEN", session=session)
dp = Dispatcher()

# Клиент OpenAI с поддержкой прокси
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY or "DUMMY_KEY",
    http_client=http_client,
)

# Режимы работы
MODES = {
    "mentor": {
        "title": "🧑‍💻 Senior Ментор (по умолчанию)",
        "prompt": (
            "Ты — дружелюбный, опытный Senior Software Engineer и наставник. "
            "Твоя цель — помогать разработчику: отвечать на любые вопросы по программированию, "
            "архитектуре, алгоритмам, багам и инструментам. "
            "Отвечай структурированно, понятно, с примерами кода на русском языке. "
            "Если пользователь просто здоровается или общается — общайся естественно, без лишнего академизма."
        ),
    },
    "reviewer": {
        "title": "🔍 Строгий Код-Ревьюер",
        "prompt": (
            "Ты — строгий Principal Code Reviewer. "
            "Твоя задача — проводить бескомпромиссный аудит кода: находить скрытые баги, "
            "утечки памяти, race conditions, оценивать алгоритмическую сложность O(...) "
            "и предлагать чистый рефакторинг по SOLID/DRY. Отвечай на русском языке."
        ),
    },
    "assistant": {
        "title": "⚡ Быстрый IT-Ассистент",
        "prompt": (
            "Ты — лаконичный и точный AI-помощник разработчика. "
            "Давай краткие, точные ответы по коду, командам терминала и синтаксису без лишних рассуждений."
        ),
    },
}

# Пользовательские настройки и память:
# user_modes: {user_id: "mentor" | "reviewer" | "assistant"}
# user_thinking: {user_id: bool} (включен ли вывод мыслей модели)
# user_history: {user_id: list of {"role": str, "content": str}}
# last_code_cache: {user_id: str}
user_modes = defaultdict(lambda: "mentor")
user_thinking = defaultdict(lambda: False)  # По умолчанию выключено, чтобы не загромождать чат
user_history = defaultdict(list)
last_code_cache = {}

MAX_HISTORY_MESSAGES = 10

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".java", ".kt", ".cs", ".php", ".rb", ".sql", ".sh",
    ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt"
}


def get_code_keyboard():
    """Интерактивные кнопки действий над кодом."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🔍 Код-Ревью и баги", callback_data="act_review")
    builder.button(text="⚡ Оценка O(N) и скорость", callback_data="act_complexity")
    builder.button(text="🧪 Написать Unit-тесты", callback_data="act_tests")
    builder.button(text="📝 Документация", callback_data="act_docs")
    builder.button(text="💡 Рефакторинг по SOLID", callback_data="act_refactor")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def get_mode_keyboard():
    """Клавиатура выбора режима."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🧑‍💻 Senior Ментор", callback_data="setmode_mentor")
    builder.button(text="🔍 Строгий Ревьюер", callback_data="setmode_reviewer")
    builder.button(text="⚡ Быстрый Ассистент", callback_data="setmode_assistant")
    builder.adjust(1)
    return builder.as_markup()


def split_text(text: str, max_chars: int = 3900) -> list[str]:
    """Разбивка длинных текстов на части с учетом лимитов Telegram."""
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


async def ask_model(messages: list[dict], enable_thinking: bool = False) -> tuple[str, str]:
    """Запрос к модели Nemotron-120B."""
    extra_body = {}
    if enable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    response_stream = await client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=messages,
        temperature=0.6,
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


async def send_response(chat_id: int, reasoning: str, content: str, show_thinking: bool):
    """Отправка ответа пользователю с опциональным блоком рассуждений."""
    if show_thinking and reasoning:
        escaped_reasoning = html.escape(reasoning)
        spoiler_text = f"🧠 <b>Ход мыслей AI:</b>\n<tg-spoiler>{escaped_reasoning}</tg-spoiler>"
        for chunk in split_text(spoiler_text, max_chars=3500):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            except Exception:
                await bot.send_message(chat_id=chat_id, text=f"🧠 Ход мыслей AI:\n{reasoning}")

    if content:
        for chunk in split_text(content, max_chars=3900):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await bot.send_message(chat_id=chat_id, text=chunk)
    else:
        await bot.send_message(chat_id=chat_id, text="⚠️ Модель вернула пустой ответ.")


# ===================== КОМАНДЫ БОТА =====================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    mode = user_modes[user_id]
    mode_name = MODES[mode]["title"]
    thinking_state = "Включен ✅" if user_thinking[user_id] else "Выключен ❌"

    await message.answer(
        "👋 <b>Добро пожаловать в Senior AI Code Companion!</b>\n\n"
        "Я твой персональный AI-ассистент и ментор по разработке ПО на базе модели <code>Nemotron 120B</code>.\n\n"
        f"⚙️ <b>Текущий режим:</b> {mode_name}\n"
        f"🧠 <b>Режим рассуждений (Thinking):</b> {thinking_state}\n\n"
        "<b>📌 Доступные команды:</b>\n"
        "/mode — Выбрать режим работы бота\n"
        "/thinking — Вкл/выкл показ внутренних мыслей модели\n"
        "/clear — Очистить память диалога\n"
        "/help — Справка и советы по использованию\n\n"
        "💬 <b>Как пользоваться:</b>\n"
        "• Задавай любые вопросы текстом (как живому ментору).\n"
        "• Присылай файлы с кодом (<code>.py</code>, <code>.js</code>, <code>.cpp</code> и др.) или текст кода — я предложу ревью, тесты, оценку O(N) и рефакторинг!",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "📚 <b>Справка по использованию бота:</b>\n\n"
        "1. <b>Общение и вопросы:</b>\n"
        "Пишите любые вопросы по программированию, алгоритмам, багам или библиотекам. Бот помнит контекст диалога.\n\n"
        "2. <b>Анализ кода:</b>\n"
        "Отправьте файл с кодом или код в сообщении. Появятся интерактивные кнопки:\n"
        "• 🔍 <i>Код-Ревью</i> — поиск багов, утечек, проблем безопасности\n"
        "• ⚡ <i>Сложность O(N)</i> — расчет времени и памяти алгоритма\n"
        "• 🧪 <i>Unit-тесты</i> — генерация тестовых сценариев\n"
        "• 💡 <i>Рефакторинг</i> — улучшение архитектуры по SOLID/DRY\n"
        "• 📝 <i>Документация</i> — docstrings и описания функций\n\n"
        "3. <b>Команды управления:</b>\n"
        "/mode — Сменить характер и режим ответов бота\n"
        "/thinking — Включить отображение того, как модель размышляет\n"
        "/clear — Забыть предыдущие сообщения и начать диалог заново",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("mode"))
async def cmd_mode(message: types.Message):
    await message.answer(
        "⚙️ <b>Выберите режим работы бота:</b>\n\n"
        "• <b>Senior Ментор</b> — подробные, понятные ответы, дружелюбный тон, помощь новичкам и профи.\n"
        "• <b>Строгий Ревьюер</b> — бескомпромиссный аудит кода, $O(N)$ анализ, архитектурные замечания.\n"
        "• <b>Быстрый Ассистент</b> — краткие и точные ответы без вступлений.",
        reply_markup=get_mode_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("setmode_"))
async def cb_set_mode(callback: types.CallbackQuery):
    new_mode = callback.data.replace("setmode_", "")
    if new_mode in MODES:
        user_modes[callback.from_user.id] = new_mode
        title = MODES[new_mode]["title"]
        await callback.answer(f"Режим изменен на: {title}")
        await callback.message.edit_text(
            f"✅ <b>Режим успешно изменен на:</b> {title}\n\n"
            "Теперь бот будет отвечать в соответствии с выбранным профилем.",
            parse_mode=ParseMode.HTML,
        )


@dp.message(Command("thinking"))
async def cmd_thinking(message: types.Message):
    user_id = message.from_user.id
    current = user_thinking[user_id]
    user_thinking[user_id] = not current
    state_str = "ВКЛЮЧЕН ✅" if not current else "ВЫКЛЮЧЕН ❌"
    desc = (
        "Теперь перед ответом будет показываться скрытый спойлер с ходом рассуждений модели."
        if not current
        else "Теперь ответы будут чистыми, без длинных внутренних монологов модели."
    )
    await message.answer(
        f"🧠 <b>Режим рассуждений (Thinking): {state_str}</b>\n\n{desc}",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    user_id = message.from_user.id
    user_history[user_id].clear()
    last_code_cache.pop(user_id, None)
    await message.answer("🧹 <b>Память диалога очищена!</b> Начинаем разговор с чистого листа.", parse_mode=ParseMode.HTML)


# ===================== ОБРАБОТКА КОДА И ФАЙЛОВ =====================

@dp.message(F.document)
async def handle_document(message: types.Message):
    file_name = message.document.file_name or ""
    _, ext = os.path.splitext(file_name)

    if ext.lower() not in SUPPORTED_EXTENSIONS:
        await message.answer(f"⚠️ Неподдерживаемый формат: <code>{html.escape(ext)}</code>. Отправьте файл с исходным кодом.", parse_mode=ParseMode.HTML)
        return

    if message.document.file_size and message.document.file_size > 1024 * 1024:
        await message.answer("⚠️ Файл слишком большой. Максимальный размер — 1 МБ.")
        return

    status_msg = await message.answer(f"📥 Загружаю <code>{html.escape(file_name)}</code>...", parse_mode=ParseMode.HTML)

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
        "Какое действие выполнить?",
        reply_markup=get_code_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.callback_query(F.data.startswith("act_"))
async def handle_code_action(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    code = last_code_cache.get(user_id)

    if not code:
        await callback.answer("⚠️ Код не найден в памяти. Отправьте файл или код заново.", show_alert=True)
        return

    action = callback.data.replace("act_", "")
    prompts = {
        "review": "Проведи подробный профессиональный Code Review следующего кода. Укажи на скрытые баги, краевые случаи, уязвимости и предложи конкретные исправления:\n\n```\n" + code + "\n```",
        "complexity": "Оцени алгоритмическую сложность следующего кода: временную O(...) и пространственную O(...). Подробно объясни расчет и предложи, как ускорить выполнение:\n\n```\n" + code + "\n```",
        "tests": "Напиши полный комплект Unit-тестов для следующего кода с проверкой краевых случаев и исключений:\n\n```\n" + code + "\n```",
        "docs": "Напиши подробную документацию и docstrings для этого кода с описанием аргументов, типов и примером использования:\n\n```\n" + code + "\n```",
        "refactor": "Выполни качественный рефакторинг этого кода по принципам Clean Code, SOLID и DRY. Покажи итоговый код и объясни улучшения:\n\n```\n" + code + "\n```",
    }

    prompt = prompts.get(action)
    if not prompt:
        return

    await callback.answer()
    status_msg = await callback.message.answer("⏳ Анализирую код, пожалуйста, подождите...")
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
        await send_response(callback.message.chat.id, reasoning, content, show_thinking)
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Ошибка при анализе: {html.escape(str(e))}")


# ===================== ОБЫЧНЫЙ ДИАЛОГ =====================

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text

    # Если пользователь вставил большой блок кода (от 3 строк с признаками кода)
    is_code = (
        ("```" in text)
        or ("def " in text and ":" in text)
        or ("class " in text and ":" in text)
        or ("function" in text and "{" in text)
        or ("import " in text and "\n" in text)
        or ("{" in text and "}" in text and ";" in text)
    )

    if is_code and len(text.strip().splitlines()) >= 3:
        last_code_cache[user_id] = text
        await message.answer(
            "💻 Код принят в буфер! Выберите действие:",
            reply_markup=get_code_keyboard(),
        )
        return

    # Обычный диалог с сохранением истории
    history = user_history[user_id]
    history.append({"role": "user", "content": text})

    # Ограничиваем историю диалога
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
        user_history[user_id] = history

    mode = user_modes[user_id]
    sys_prompt = MODES[mode]["prompt"]
    show_thinking = user_thinking[user_id]

    # Полный список сообщений для модели
    full_messages = [{"role": "system", "content": sys_prompt}] + history

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        reasoning, content = await ask_model(full_messages, enable_thinking=show_thinking)
        if content:
            history.append({"role": "assistant", "content": content})
        await send_response(message.chat.id, reasoning, content, show_thinking)
    except Exception as e:
        error_text = str(e)
        if "451" in error_text:
            await message.answer(
                "⚠️ <b>Ошибка доступа к AI (HTTP 451):</b>\n"
                "NVIDIA API блокирует запросы из вашего региона без прокси/VPN.\n"
                "Включите VPN или укажите рабочий прокси в параметре <code>TELEGRAM_PROXY</code> файла <code>.env</code>.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(f"⚠️ <b>Ошибка:</b> {html.escape(error_text)}", parse_mode=ParseMode.HTML)


# ===================== ЗАПУСК =====================

async def setup_bot_commands():
    """Регистрация команд в синей кнопке 'Меню' Telegram."""
    commands = [
        types.BotCommand(command="start", description="🚀 Перезапустить / Приветствие"),
        types.BotCommand(command="mode", description="⚙️ Выбрать режим (Ментор / Ревьюер / Чат)"),
        types.BotCommand(command="thinking", description="🧠 Вкл/выкл показ мыслей модели"),
        types.BotCommand(command="clear", description="🧹 Очистить историю диалога"),
        types.BotCommand(command="help", description="📚 Справка и примеры"),
    ]
    await bot.set_my_commands(commands)


async def main():
    if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
        print("❌ ОШИБКА: Пожалуйста, вставьте валидный BOT_TOKEN в файл .env!")
        return

    print("🚀 Регистрация команд в меню Telegram...")
    try:
        await setup_bot_commands()
    except Exception as e:
        print(f"Предупреждение при регистрации команд: {e}")

    print("🚀 Senior AI Code Companion запущен и ожидает сообщений...")
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
