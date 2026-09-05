import os
import io
import html
import asyncio
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramNetworkError
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY")
TELEGRAM_API_SERVER = os.getenv("TELEGRAM_API_SERVER")

if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
    print("⚠️ ВНИМАНИЕ: Укажите реальный BOT_TOKEN в файле .env перед запуском!")

# Настройка сессии для обхода блокировок Telegram (Proxy / Реверс-прокси)
session = None
if TELEGRAM_PROXY and TELEGRAM_PROXY.strip():
    print(f"🌐 Используется прокси для Telegram: {TELEGRAM_PROXY.strip()}")
    session = AiohttpSession(proxy=TELEGRAM_PROXY.strip())

if TELEGRAM_API_SERVER and TELEGRAM_API_SERVER.strip():
    api_server = TelegramAPIServer.from_base(TELEGRAM_API_SERVER.strip())
    bot = Bot(token=BOT_TOKEN or "DUMMY_TOKEN", session=session, api_server=api_server)
else:
    bot = Bot(token=BOT_TOKEN or "DUMMY_TOKEN", session=session)

dp = Dispatcher()

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY or "DUMMY_KEY",
)

# Системный промпт для позиции Senior / Principal Engineer
SYSTEM_PROMPT = """Ты — опытный Senior / Principal Software Engineer и архитектор программного обеспечения.
Твоя задача — проводить глубокий, бескомпромиссный и полезный код-ревью, обучать разработчика лучшим практикам и архитектурному мышлению.

Когда ты анализируешь код:
1. Архитектура и баги: Ищи скрытые ошибки, race conditions, утечки памяти, неочевидные краевые случаи (edge cases) и проблемы с типами.
2. Алгоритмическая сложность: ВСЕГДА явно оценивай временную O(...) и пространственную O(...) сложность алгоритма. Предлагай варианты оптимизации.
3. Чистота и идиоматичность: Приводи код в соответствие с каноническими соглашениями конкретного языка (PEP8, Go style, Clean Architecture, SOLID, DRY).
4. Форматирование: Все фрагменты кода оборачивай в тройные бэктики с указанием языка (например, ```python ... ```).
5. Тон: Профессиональный, конструктивный, глубокий, без лишней "воды"."""

# Кэш последнего отправленного кода: {user_id: code_str}
last_code_cache = {}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".java", ".kt", ".cs", ".php", ".rb", ".sql", ".sh",
    ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt"
}


def get_action_keyboard():
    """Интерактивные кнопки действий под кодом."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🔍 Ревью и баги", callback_data="act_review")
    builder.button(text="⚡ Оценка O(N) и скорость", callback_data="act_complexity")
    builder.button(text="🧪 Написать Unit-тесты", callback_data="act_tests")
    builder.button(text="📝 Документация / Docstrings", callback_data="act_docs")
    builder.button(text="💡 Рефакторинг по SOLID", callback_data="act_refactor")
    builder.adjust(2, 2, 1)
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


async def ask_senior_ai(user_prompt: str) -> tuple[str, str]:
    """Запрос к Nemotron-120B с извлечением мыслей (reasoning) и основного ответа."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response_stream = await client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=True,
    )

    full_reasoning = ""
    full_content = ""

    async for chunk in response_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Извлечение мыслей модели
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            full_reasoning += reasoning

        if delta.content is not None:
            full_content += delta.content

    return full_reasoning.strip(), full_content.strip()


async def send_ai_response(chat_id: int, reasoning: str, content: str):
    """Красиво отправляет цепочку рассуждений под спойлером и основной ответ."""
    # Отправляем ход мыслей под спойлером
    if reasoning:
        escaped_reasoning = html.escape(reasoning)
        spoiler_text = f"🧠 <b>Ход мыслей Senior AI:</b>\n<tg-spoiler>{escaped_reasoning}</tg-spoiler>"
        for chunk in split_text(spoiler_text, max_chars=3500):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            except Exception:
                await bot.send_message(chat_id=chat_id, text=f"🧠 Ход мыслей Senior AI:\n{reasoning}")

    # Основной ответ
    if content:
        for chunk in split_text(content, max_chars=3900):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await bot.send_message(chat_id=chat_id, text=chunk)
    else:
        await bot.send_message(chat_id=chat_id, text="⚠️ Ответ модели пуст.")


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 <b>Привет! Я твой персональный Senior AI-Ментор.</b>\n\n"
        "Я работаю на базе <code>NVIDIA Nemotron 3 Super 120B</code> с цепочкой рассуждений (thinking).\n\n"
        "<b>Чем могу помочь:</b>\n"
        "• 🔍 Найти неочевидные баги, race conditions и краевые случаи\n"
        "• ⚡ Оценить сложность алгоритма O(N) и узкие места\n"
        "• 🧪 Сгенерировать качественные Unit-тесты\n"
        "• 💡 Сделать рефакторинг по принципам SOLID и Clean Code\n"
        "• 📝 Написать подробные docstrings и документацию\n\n"
        "<b>Как отправить код:</b>\n"
        "1. Отправь файл с исходным кодом (<code>.py</code>, <code>.js</code>, <code>.go</code>, <code>.cpp</code> и т.д.).\n"
        "2. Либо просто пришли фрагмент кода текстом в чат.",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    user_id = message.from_user.id
    last_code_cache.pop(user_id, None)
    await message.answer("🧹 Буфер последнего кода очищен.")


@dp.message(F.document)
async def handle_document(message: types.Message):
    """Обработка файлов с кодом."""
    file_name = message.document.file_name or ""
    _, ext = os.path.splitext(file_name)

    if ext.lower() not in SUPPORTED_EXTENSIONS:
        await message.answer(f"⚠️ Неподдерживаемый формат: <code>{html.escape(ext)}</code>. Пожалуйста, отправьте файл с исходным кодом.", parse_mode=ParseMode.HTML)
        return

    if message.document.file_size and message.document.file_size > 1024 * 1024:
        await message.answer("⚠️ Файл слишком большой. Максимальный размер — 1 МБ.")
        return

    status_msg = await message.answer(f"📥 Загружаю и разбираю <code>{html.escape(file_name)}</code>...", parse_mode=ParseMode.HTML)

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
        f"📄 Файл <b>{html.escape(file_name)}</b> ({line_count} строк) готов к анализу.\n"
        "Выберите действие:",
        reply_markup=get_action_keyboard(),
        parse_mode=ParseMode.HTML,
    )


@dp.message(F.text)
async def handle_text(message: types.Message):
    """Обработка текстового кода или вопроса по программированию."""
    user_id = message.from_user.id
    text = message.text

    # Проверка, похож ли текст на блок кода
    is_code = (
        ("```" in text)
        or ("def " in text)
        or ("class " in text)
        or ("function" in text)
        or ("import " in text)
        or ("{" in text and "}" in text and ";" in text)
    )

    if is_code and len(text.strip().splitlines()) >= 3:
        last_code_cache[user_id] = text
        await message.answer(
            "💻 Код принят в буфер! Выберите задачу:",
            reply_markup=get_action_keyboard(),
        )
        return

    # Если это обычный вопрос / консультация
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    try:
        reasoning, content = await ask_senior_ai(text)
        await send_ai_response(message.chat.id, reasoning, content)
    except Exception as e:
        await message.answer(f"⚠️ Ошибка генерации: {html.escape(str(e))}")


@dp.callback_query(F.data.startswith("act_"))
async def handle_action(callback: types.CallbackQuery):
    """Обработка кнопок действий над кодом."""
    user_id = callback.from_user.id
    code = last_code_cache.get(user_id)

    if not code:
        await callback.answer("⚠️ Код не найден в памяти. Отправьте файл или код заново.", show_alert=True)
        return

    action = callback.data.replace("act_", "")
    prompts = {
        "review": "Проведи подробный Senior Code Review следующего кода. Укажи на скрытые баги, уязвимости, проблемы параллелизма, краевые случаи и предложи конкретные исправления:\n\n```\n" + code + "\n```",
        "complexity": "Оцени временную и пространственную сложность следующего кода (Big O). Подробно объясни каждый шаг, найди узкие места (bottlenecks) и предложи, как его ускорить:\n\n```\n" + code + "\n```",
        "tests": "Напиши полный комплект надежных Unit-тестов для следующего кода. Протестируй happy path, краевые случаи (границы, пустые коллекции, None/null) и исключения:\n\n```\n" + code + "\n```",
        "docs": "Напиши исчерпывающую документацию для этого кода: docstrings/JSDoc для всех методов, описание входных/выходных параметров и пример использования:\n\n```\n" + code + "\n```",
        "refactor": "Выполни глубокий рефакторинг этого кода в соответствии с принципами SOLID, DRY и чистой архитектуры. Покажи обновленный код и объясни, почему новые решения лучше:\n\n```\n" + code + "\n```",
    }

    prompt = prompts.get(action)
    if not prompt:
        return

    await callback.answer()
    status_msg = await callback.message.answer("⚡ Senior AI анализирует код и продумывает решение...")
    await bot.send_chat_action(chat_id=callback.message.chat.id, action=ChatAction.TYPING)

    try:
        reasoning, content = await ask_senior_ai(prompt)
        try:
            await status_msg.delete()
        except Exception:
            pass
        await send_ai_response(callback.message.chat.id, reasoning, content)
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Произошла ошибка: {html.escape(str(e))}")


async def main():
    if not BOT_TOKEN or BOT_TOKEN.startswith("ВСТАВЬТЕ"):
        print("❌ ОШИБКА: Пожалуйста, вставьте валидный BOT_TOKEN в файл .env!")
        return

    print("🚀 Senior AI Code Companion запускается...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except TelegramNetworkError as e:
        print("\n" + "=" * 60)
        print("❌ ОШИБКА СЕТИ TELEGRAM (TelegramNetworkError):")
        print("Ваш интернет-провайдер блокирует прямое подключение к api.telegram.org:443.")
        print("\nКАК ИСПРАВИТЬ:")
        print("1. Включите VPN на компьютере (например, Amnezia, V2Ray, WireGuard и др.).")
        print("   ИЛИ")
        print("2. Если у вас запущен локальный прокси (Clash / V2Ray / Shadowsocks),")
        print("   откройте файл .env и раскомментируйте / укажите строку:")
        print("   TELEGRAM_PROXY=http://127.0.0.1:7890   (или ваш порт, например 7897, 10808)")
        print("   Поддерживаются форматы: http://..., socks5://...")
        print("=" * 60 + "\n")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
