#!/usr/bin/env python3
"""mcp_common — общее ядро MCP-обёрток над CLI базы знаний (stdlib-only).

Выделено в фазе 3 аудита памяти из tools/skills_search_mcp.py и
session-search/mcp_server.py (реализация взята из session-версии как более
свежая: stderr=DEVNULL, чтение stdout потоком, принудительный kill).

Windows-паттерн (shutdown-hang): дочерний CLI-процесс после отработки main
иногда «зависает на выходе» (единственный поток idle, exit не наступает),
поэтому ждём не exit-код, а валидный JSON в stdout; получив его —
принудительно kill. Детали:
- `-u` (unbuffered) — JSON допечатывается в pipe сразу, а не удерживается
  буфером stdout до выхода процесса (который может не наступить);
- `stdin=DEVNULL` — отсекает наследуемый от MCP-клиента pipe;
- `stderr=DEVNULL` — прогресс/предупреждения CLI пишутся в stderr, читать
  их некому: заполненный pipe заблокировал бы дочерний процесс до таймаута.

Подключение из MCP-серверов:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from mcp_common import run_cli, detect_venv_python, MCP_TIMEOUT
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Полный гибридный прогон (BM25 + vector + rerank) на холодном кэше
# укладывается в 150с; 120с бывало впритык (синхронизировано в фазе 1).
MCP_TIMEOUT = 150


def detect_venv_python(skills_root) -> Path:
    """Python для CLI базы знаний (sqlite-vec, httpx, snowballstemmer).

    Ищет venv рядом с корнем: Windows → .venv/Scripts/python.exe,
    иначе → .venv/bin/python. Если venv нет (у нас зависимости стоят
    в базовом Python 3.14) — фолбэк на интерпретатор, которым запущен
    сам сервер.
    """
    venv = Path(skills_root) / ".venv"
    for rel in ("Scripts/python.exe", "bin/python"):
        py = venv / rel
        if py.exists():
            return py
    return Path(sys.executable)


async def read_stdout_json(proc: asyncio.subprocess.Process) -> bytes:
    """Читать stdout построчно, пока не получится валидный JSON.

    Полагаемся только на поток stdout, не на exit-код: дочерний процесс
    может не завершаться (shutdown-hang на Windows).
    """
    buf = bytearray()
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        buf.extend(line)
        try:
            json.loads(buf.decode("utf-8", "replace"))
            return bytes(buf)
        except Exception:
            continue
    return bytes(buf)


async def terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """Принудительно завершить дочерний процесс (сам он может не выйти)."""
    proc.kill()
    try:
        await proc.wait()
    except Exception:
        pass


def merge_origin_results(rows_a: list, origin_a: str, rows_b: list,
                         origin_b: str, top_k: int) -> list:
    """Слияние выдач двух индексов в один ранжированный список.

    Используется инструментом `search_skills`: глобальная память (origin_a)
    и проектные скилы (origin_b) ищутся одним вызовом, проектные получают
    лёгкий приоритет: при равном скоре первыми идут строки origin_b.

    - внутри каждого origin скоры нормализуются на максимум своего origin
      (иначе большой глобальный индекс задавит 19 проектных скилов);
    - при равном скоре первыми идут строки origin_b;
    - каждая строка получает поле "origin".

    Числовой boost к проектным скорам сознательно не применяется: оба CLI
    отдают скоры в сравнимом диапазоне (~0..1), а подъём всех проектных
    строк гарантированно вытеснял глобальные результаты даже при явно
    лучшем глобальном матче.

    Строки без поля score считаются нулевыми. Пустой список origin просто
    не участвует в слиянии.
    """
    out: list[dict] = []
    for rows, origin, is_b in ((rows_a, origin_a, False),
                               (rows_b, origin_b, True)):
        if not rows:
            continue
        peak = max((float(r.get("score") or 0.0) for r in rows), default=0.0)
        for r in rows:
            item = dict(r)
            item["score"] = round(float(item.get("score") or 0.0) / peak, 4) \
                if peak > 0 else 0.0
            item["origin"] = origin
            # стабильный порядок внутри origin: проектные выше при равенстве
            item["_origin_rank"] = 0 if is_b else 1
            out.append(item)
    out.sort(key=lambda r: (-r["score"], r["_origin_rank"]))
    for item in out:
        item.pop("_origin_rank", None)
    return out[:max(1, int(top_k))]


async def run_cli(py, script, args, *, timeout: int = MCP_TIMEOUT,
                  cwd=None, marker: str = "{", proc_name: str = "CLI") -> str:
    """Запустить CLI базы знаний в её venv и вернуть JSON из stdout.

    py/script — интерпретатор venv и путь к CLI-скрипту; args — список
    аргументов командной строки. marker — ожидаемый первый символ ответа
    ("{" для JSON-объекта, "[" для массива): диагностические строки перед
    JSON обрезаются. Возвращает текст JSON либо JSON с "error".
    """
    cmd = [str(py), "-u", str(script), *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
    except OSError:
        return json.dumps(
            {"error": f"cannot run {py}", "cmd": cmd}, ensure_ascii=False
        )

    try:
        out = await asyncio.wait_for(read_stdout_json(proc), timeout=timeout)
    except asyncio.TimeoutError:
        await terminate_proc(proc)
        return json.dumps(
            {"error": f"{proc_name} timed out after {timeout}s"},
            ensure_ascii=False,
        )
    await terminate_proc(proc)

    text = out.decode("utf-8", "replace").strip()
    idx = text.find(marker)
    if idx > 0:
        text = text[idx:]
    if not text.startswith(marker):
        # CLI напечатал валидный JSON другого типа (например, объект ошибки
        # вместо ожидаемого массива) — вернуть как есть: это диагностический
        # ответ, а не мусор (фаза 1 аудита р3)
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        return json.dumps(
            {"error": f"no valid JSON from {proc_name}", "cmd": cmd},
            ensure_ascii=False,
        )
    return text
