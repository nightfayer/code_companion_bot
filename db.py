import aiosqlite
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "bot.db")


async def init_db():
    """Инициализация таблиц базы данных SQLite."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                mode TEXT DEFAULT 'mentor',
                show_thinking INTEGER DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_code (
                user_id INTEGER PRIMARY KEY,
                filename TEXT,
                code_content TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def is_user_exists(user_id: int) -> bool:
    """Проверяет, зарегистрирован ли уже пользователь в базе данных."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)) as cursor:
            return bool(await cursor.fetchone())


async def get_total_stats() -> dict:
    """Возвращает общую статистику по пользователям и сообщениям для админа."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as c1:
            total_users = (await c1.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM messages") as c2:
            total_messages = (await c2.fetchone())[0]
        return {
            "total_users": total_users,
            "total_messages": total_messages,
        }


async def get_user_settings(user_id: int) -> tuple[str, bool]:
    """Получает текущий режим и настройку thinking для пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT mode, show_thinking FROM users WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0], bool(row[1])
            # Создаем пользователя по умолчанию
            await db.execute("INSERT OR IGNORE INTO users (user_id, mode, show_thinking) VALUES (?, 'mentor', 0)", (user_id,))
            await db.commit()
            return "mentor", False


async def set_user_mode(user_id: int, mode: str):
    """Обновляет режим пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO users (user_id, mode) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET mode = excluded.mode
        """, (user_id, mode))
        await db.commit()


async def toggle_user_thinking(user_id: int) -> bool:
    """Переключает статус отображения мыслей модели."""
    async with aiosqlite.connect(DB_PATH) as db:
        mode, current = await get_user_settings(user_id)
        new_state = 0 if current else 1
        await db.execute("""
            INSERT INTO users (user_id, mode, show_thinking) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET show_thinking = excluded.show_thinking
        """, (user_id, mode, new_state))
        await db.commit()
        return bool(new_state)


async def add_message(user_id: int, role: str, content: str):
    """Добавляет сообщение в историю диалога пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        # Ограничиваем историю пользователя последними 20 сообщениями
        await db.execute("""
            DELETE FROM messages 
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 20
            )
        """, (user_id, user_id))
        await db.commit()


async def get_history(user_id: int, limit: int = 10) -> list[dict]:
    """Возвращает историю сообщений пользователя в формате OpenAI."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
        """, (user_id, limit)) as cursor:
            rows = await cursor.fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]


async def clear_history(user_id: int):
    """Очищает историю диалога пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await db.commit()


async def save_code(user_id: int, filename: str, code: str):
    """Сохраняет последний код и имя файла пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO user_code (user_id, filename, code_content) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET filename = excluded.filename, code_content = excluded.code_content, updated_at = CURRENT_TIMESTAMP
        """, (user_id, filename, code))
        await db.commit()


async def get_code(user_id: int) -> tuple[str, str]:
    """Возвращает (filename, code_content) пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT filename, code_content FROM user_code WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0] or "code.py", row[1] or ""
            return "", ""


async def clear_code(user_id: int):
    """Удаляет сохраненный код пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM user_code WHERE user_id = ?", (user_id,))
        await db.commit()
