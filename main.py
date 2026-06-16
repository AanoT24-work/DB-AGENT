import os
import re
import time
import json
import oracledb
import logging
import requests
import asyncio
import threading
from datetime import datetime
from langchain.tools import tool
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

"""Класс для работы с инициализацией, подключением и сменой аккаунтов для всех необходимых ресурсов"""
class Init:
    def __init__(self):
        self.connections: Dict[str, oracledb.Connection] = {}
        self.current_user: Optional[str] = None
        self.log_dir = "terminal_logs"
        self.setup_logs()
        # Запускаем фоновый event loop для асинхронной записи
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True
        )
        self.loop_thread.start()

        self.llm_model = os.getenv("OLLAMA_MODEL")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL")
        self.timeout = 300

    """Фоновый поток с event loop."""
    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    """Запись логов в файл."""
    @staticmethod
    async def write_log(log_file: str, log_entry: dict):

        try:
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Ошибка сохранения LLM лога: {e}")

    """Сохранение системных логов"""
    def setup_logs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        log_filename = f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, log_filename)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    """Сохранение логов LLM в файл"""
    def llm_logs(self, prompt: str, response: str, decision_type: str, elapsed: float):
        save_logs = True
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": decision_type,
            "prompt": prompt[:1000],
            "response": response,
            "elapsed_time": elapsed,
            "model": os.getenv("OLLAMA_MODEL"),
        }
        if save_logs:
            safe_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            log_file = os.path.join(self.log_dir, f"llm_{safe_timestamp}.json")
            asyncio.run_coroutine_threadsafe(
                self.write_log(log_file, log_entry),
                self.loop
            )

    """Подключение к БД """
    def db_connect(self, username: str, password: str, 
                    host: Optional[str] = None, 
                    port: Optional[int] = None,
                    service: Optional[str] = None) -> oracledb.Connection:
        final_host = host if host is not None else os.getenv("ORACLE_DB_HOST")
        final_port = port if port is not None else os.getenv("ORACLE_DB_PORT")
        final_service = service if service is not None else os.getenv("ORACLE_DB_SERVICE")

        if not all([final_host, final_port, final_service]):
            missing = []
            if not final_host: missing.append("host")
            if not final_port: missing.append("port")
            if not final_service: missing.append("service")
            self.logger.error(f"Не удалось подключиться к БД. Отсутствуют параметры: {', '.join(missing)}")
            raise ValueError("Отсутствуют обязательные параметры подключения")

        dsn = f"{final_host}:{final_port}/{final_service}"
        self.logger.info(f"Подключение к {dsn} как {username}")

        try:
            connection = oracledb.connect(
                user=username, 
                password=password, 
                dsn=dsn
            )
            self.connections[username] = connection
            self.current_user = username
            
            self.logger.info(f"Успешно подключен как {username}")
            return connection
            
        except oracledb.Error as e:
            self.logger.error(f"Ошибка подключения к БД: {e}")
            raise

    """Переключение на другого пользователя"""
    def switch_user(self, username: str, password: str) -> Dict[str, Any]:
        try:
            if self.current_user and self.current_user in self.connections:
                try:
                    self.connections[self.current_user].close()
                    del self.connections[self.current_user]
                except Exception as e:
                    self.logger.warning(f"Ошибка закрытия: {e}")

            self.db_connect(username, password)
            self.logger.info(f"Смена пользователя на: {username}")
            return {"status": True, "message": f"Пользователь изменен на: {username}"}
        except Exception as e:
            self.logger.error(f"Ошибка смены пользователя {username}: {e}")
            return {"status": False, "message": str(e)}


    """Получения активного подключения пользователей"""
    def db_user(self, username: Optional[str] = None) -> oracledb.Connection:
        try:
            target = username or self.current_user

            if not target:
                msg = "Пользователь не указан. Нет активных подключений."
                self.logger.warning(msg)
                raise ValueError(msg)

            if target not in self.connections:
                msg = f"Нет соединения для пользователя '{target}'"
                self.logger.warning(msg)
                raise ValueError(msg)
            return self.connections[target]

        except Exception as e:
            if not isinstance(e, ValueError):
                self.logger.error(f"Неожиданная ошибка: {e}")
            raise

    """Проверка доступности LLM"""
    def llm_connect(self) -> bool:
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            model_data = response.json()
            available_models = [model["name"] for model in model_data.get("models", [])]

            if self.llm_model not in available_models:
                self.logger.error(f"Модели: {self.llm_model}. Нет в списке доступных моделей")
                return False
            else:
                self.logger.info(f"Модель: {self.llm_model}. Доступна")
                return True
        except requests.RequestException as e:
            self.logger.error(f"Ошибка подключения - {e}")
            return False




"""Класс для работы с БД, все необходимые для этого инструменты"""
class DB_Tools:
    def __init__(self, init: Init):
        self.init = init
        self.logger = init.logger

    def execute(self, sql: str, username: Optional[str] = None) -> Dict[str, Any]:
        target = username or self.init.current_user
        connection = self.init.db_user(target)
        cursor = connection.cursor()

        try:
            cursor.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                self.logger.info(f"Запрос выполнен: {len(rows)} строк")
                return {
                    "status": True,
                    "type": "select",
                    "rows": rows,
                    "columns": columns
                }
            else:
                connection.commit()
                self.logger.info(f"Запрос выполнен: затронуто {cursor.rowcount} строк")
                return {
                    "status": True,
                    "type": "dml",
                    "row_count": cursor.rowcount
                }
        except oracledb.Error as e:
            self.logger.error(f"Ошибка выполнения SQL: {e}")
            return {
                "status": False,
                "message": str(e),
                "sql": sql
            }
        finally:
            cursor.close()

    """Форматирует результат SELECT-запроса в текст для LLM."""
    @staticmethod
    def format_for_llm(result: dict) -> str:
        rows = result["rows"]
        columns = result["columns"]

        output = f"Найдено {len(rows)} строк(а/и):\n"
        output += " | ".join(columns) + "\n"
        output += "-" * 50 + "\n"
        for row in rows[:10]:  # Ограничиваем 10 строками
            output += " | ".join(str(cell) for cell in row) + "\n"
        if len(rows) > 10:
            output += f"... и еще {len(rows) - 10} строк"
        return output

    """Создание инструментов работы с БД для агента"""
    def get_tools(self) -> list:
        db_tools = self

        @tool
        def execute_sql(sql: str) -> str:
            """Выполняет SQL запрос в Oracle Database и возвращает результат."""
            result = db_tools.execute(sql)

            if result["status"]:
                if result["type"] == "select":
                    if len(result["rows"]) == 0:
                        return "Запрос выполнен успешно, но строк не найдено."
                    return db_tools.format_for_llm(result)
                else:
                    return f"SQL выполнен успешно. Затронуто {result['row_count']} строк."
            else:
                return f"Ошибка SQL: {result['message']}"
        
        @tool
        def switch_user(username: str, password: str) -> str:
            """Переключает текущего пользователя БД на указанного."""
            result = db_tools.init.switch_user(username, password)
            if result["status"]:
                return f"Успешно переключен на пользователя: {username}"
            else:
                return f"Ошибка: {result['message']}"

        @tool
        def get_current_user() -> str:
            """Возвращает имя текущего пользователя БД."""
            result = db_tools.execute("SELECT USER FROM DUAL")
            if result["status"] and result["rows"]:
                return f" Текущий пользователь: {result['rows'][0][0]}"
            else:
                return "Не удалось определить текущего пользователя"

        
        @tool
        def check_object_exists(object_name: str, object_type: str) -> str:
            """Проверяет существует ли объект указанного типа в БД."""
            type_map = {
                "TABLESPACE": "dba_tablespaces",
                "USER": "dba_users",
                "PROFILE": "dba_profiles",
                "ROLE": "dba_roles",
                "TABLE": "all_tables"
            }

            view = type_map.get(object_type.upper())
            if not view:
                return f"Неизвестный тип объекта: {object_type}"
            sql = f"SELECT COUNT(*) FROM {view} WHERE {object_type.lower()}_name = '{object_name.upper()}'"
            result = db_tools.execute(sql)

            if result["status"] and result["rows"]:
                exists = result["rows"][0][0] > 0
                return f"Объект {object_name} ({object_type}) {'существует' if exists else 'не существует'}"

            return f"Не удалось проверить существование {object_name}"

        @tool
        def extract_credentials_from_table(table_name: str = "credentials") -> str:
            """Извлекает учетные данные из указанной таблицы."""
            sql = f"SELECT * FROM {table_name}"
            result = db_tools.execute(sql)

            if result["status"] and result["rows"]:
                creds = []
                for row in result["rows"]:
                    creds.append(f"username: {row[0]}, password: {row[1]}")
                return f"Найдены учетные данные:\n" + "\n".join(creds)

            return f"Не удалось найти учетные данные в таблице {table_name}"

       
        @tool
        def get_ctf_flag() -> str:
            """Получает CTF флаг из БД. Пробует стандартный маршрут и обходной поиск."""
            # Попытка 1: Стандартный маршрут
            db_tools.execute("SET ROLE CTF_ROLE IDENTIFIED BY ctf_role")
            result = db_tools.execute("SELECT * FROM CTF.CTF_FLAG")
            if result["status"] and result["rows"]:
                flag = result["rows"][0][0]
                output = f"🏆 Флаг получен: {flag}\n"
                output += "=" * 50 + "\n"
                output += "Маршрут: стандартный (SET ROLE CTF_ROLE + CTF.CTF_FLAG)"
                return output

            # Попытка 2: Автоматический поиск
            search_result = db_tools.execute("SELECT table_name FROM all_tables WHERE table_name LIKE '%FLAG%' OR table_name LIKE '%CTF%'")
            if search_result["status"] and search_result["rows"]:
                for row in search_result["rows"]:
                    table_name = row[0]
                    db_tools.execute("SET ROLE CTF_ROLE IDENTIFIED BY ctf_role")
                    result = db_tools.execute(f"SELECT * FROM {table_name}")

                    if result["status"] and result["rows"]:
                        flag = result["rows"][0][0]
                        output = f"Флаг получен: {flag}\n"
                        output += "-" * 50 + "\n"
                        output += f"Маршрут: обходной (найден в таблице {table_name})"
                        output += "-" * 50 + "\n"
                        output += "Флаг находится в нестандартном месте, доебаться до препода)))"
                        return output

            return "Не удалось получить флаг ни одним способом."

       
        @tool
        def init_sys_connection() -> str:
            """Подключается к системному аккаунту БД через данные из .env."""
            sys_user = os.getenv("ORACLE_SYS_USER")
            sys_password = os.getenv("ORACLE_SYS_PASSWORD")

            if not sys_user or not sys_password:
                return "Ошибка: ORACLE_SYS_USER или ORACLE_SYS_PASSWORD не найдены в .env"

            result = db_tools.init.switch_user(sys_user, sys_password)
            if result["status"]:
                return f"Подключен к системному аккаунту: {sys_user}"
            else:
                return f"Ошибка подключения: {result['message']}"

        """Разблокировать пользователя"""
        @tool
        def unlock_user(username: str) -> str:
            """Разблокирует указанного пользователя БД."""
            result = db_tools.execute(f"ALTER USER {username} ACCOUNT UNLOCK")
            if result["status"]:
                return f"Пользователь {username} разблокирован"
            else:
                return f"Ошибка: {result['message']}"

        return [
            execute_sql,
            switch_user,
            get_current_user,
            check_object_exists,
            extract_credentials_from_table,
            get_ctf_flag,
            init_sys_connection,
            unlock_user,
        ]


class LLM_Tools:
    def __init__(self, init: Init, db_tools: DB_Tools, use_llm: bool = True, save_logs: bool = True, recreate: bool = False):
        self.init = init
        self.db_tools = db_tools
        self.tool_dict = {tool.name: tool for tool in db_tools.get_tools()}


        self.use_llm = use_llm
        self.save_logs = save_logs
        self.recreate = recreate
        self.ollama_url = os.getenv("OLLAMA_BASE_URL")
        self.llm_model = os.getenv("OLLAMA_MODEL")
        self.timeout = init.timeout

        self.logger = init.logger
        self.log_dir = init.log_dir # Посмотреть понадобиться или нет

        self.llm_conversations = []
        self.results = []
        self.flags_found = []
        self.recreated_objects = []
        self.context: Dict[str, Any] = {}
        self.stats = {
            "tablespaces": 0, "profiles": 0, "roles": 0, "users": 0,
            "tables": 0, "views": 0, "materialized_views": 0, "procedures": 0,
            "functions": 0, "packages": 0, "triggers": 0, "sequences": 0,
            "indexes": 0, "synonyms": 0, "rows_inserted": 0
        }

    """Передача вывода терминала SQL в LLM для анализа и получения следующего шага"""
    def ask_llm(self, prompt: str, decision_type: str = "general") -> str:
        if not self.use_llm:
            self.logger.warning("LLM отключена")
            return ""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": os.getenv("OLLAMA_MODEL"), "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 2048}},
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            answer = response.json().get('response', '')
            self.logger.info(f"Ответ: {answer[:200]}")
            self.init.llm_logs(prompt, answer, decision_type, elapsed)
            return answer.strip()
        except Exception as e:
            self.logger.error(f"Ошибка LLM: {e}")
            return ""

    """Обертка для выполнения SQL с логированием, обработкой ошибок и возможностью пропуска уже существующих объектов"""
    def execute_sql(self, name: str, sql: str, skip_if_exists: bool = True) -> bool:
        try:
            start = time.time()
            self.logger.info(f"[{name}] Выполнение: {sql[:150]}{'...' if len(sql) > 150 else ''}")
            
            result = self.tool_dict["execute_sql"].invoke({"sql": sql})
            elapsed = (time.time() - start) * 1000
            result_str = str(result)
            is_error = "error" in result_str.lower() or "ошибка" in result_str.lower()
            
            if is_error:
                already_exists_codes = [
                    "ORA-01543",  # табличное пространство уже существует
                    "ORA-01920",  # пользователь уже существует
                    "ORA-01921",  # роль уже существует
                    "ORA-02379",  # профиль уже существует
                    "ORA-00955",  # объект уже существует
                ]
                
                if skip_if_exists and (
                    "already exists" in result_str.lower() or 
                    any(code in result_str for code in already_exists_codes)
                ):
                    self.logger.warning(f"[{name}] Уже существует ({elapsed:.0f}ms)")
                    return True  
                
                self.logger.error(f"[{name}] Ошибка ({elapsed:.0f}ms): {result_str[:200]}")
                return False
            self.logger.info(f"[{name}] Успешно ({elapsed:.0f}ms)")
            return True
        except Exception as e:
            self.logger.error(f"[{name}] Исключение: {str(e)[:200]}")
            return False
        
    """Обертка для переключения на другого пользователя, поиск пароля в контексте или создание из username"""
    def switch_user(self, username: str, password: str) -> bool:
        try:
            if not password:
                users = self.context.get("users", [])
                for u in users:
                    if u.get("name") == username:
                        password = u.get("password", "")
                        break
                if not password:
                    password = username.lower() + "123"
            self.tool_dict["switch_user"].invoke({"username": username, "password": password})
            self.logger.info(f"Переключились на {username} с паролем {password}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка переключения на {username}: {e}")
            return False    

    """Преобразование длинного текста задания в короткий план задания в формате JSON"""
    def parse_task(self, task_description: str) -> Optional[Dict[str, Any]]:
        prompt = f"""Ты эксперт по Oracle Database. Проанализируй задание и верни JSON план.

    Задание: {task_description[:3000]}

    Верни JSON:
    {{
        "tablespace": {{"name": "NAME", "size_mb": 1024, "autoextend_max_mb": 5120}},
        "profile": {{"name": "NAME", "params": {{"SESSIONS_PER_USER": 10, "IDLE_TIME": 15}}}},
        "roles": [{{"name": "ROLE1"}}, {{"name": "ROLE2"}}],
        "users": [
            {{"name": "USER1", "password": "pass1", "quota_mb": 100, "profile": "PROFILE", "privileges": ["CREATE SESSION"]}}
        ],
        "tables": [],
        "data": [],
        "views": [],
        "procedures": [],
        "triggers": [],
        "sequences": []
    }}

    Ответь ТОЛЬКО JSON."""

        if len(task_description) > 1000:
            parts = []
            for i in range(0, len(task_description), 750):
                chunk = task_description[i:i+750]
                response = self.ask_llm(
                    f"Извлеки суть из части задания: {chunk}", 
                    "chunk_analysis"
                )
                parts.append(response)
            
            compressed_task = "\n".join(parts)
            
            prompt = f"""Ты эксперт по Oracle Database. Проанализируй задание и верни JSON план.

    Задание (сжато): {compressed_task}

    Верни JSON:
    {{
        "tablespace": {{"name": "NAME", "size_mb": "РАЗМЕР_ИЗ_ЗАДАНИЯ", "autoextend_max_mb": "МАКС_РАЗМЕР_ИЗ_ЗАДАНИЯ"}},
        "profile": {{"name": "NAME", "params": {{"SESSIONS_PER_USER": 10, "IDLE_TIME": 15}}}},
        "roles": [{{"name": "ROLE1"}}, {{"name": "ROLE2"}}],
        "users": [
            {{"name": "USER1", "password": "pass1", "quota_mb": 100, "profile": "PROFILE", "privileges": ["CREATE SESSION"]}}
        ],
        "tables": [],
        "data": [],
        "views": [],
        "procedures": [],
        "triggers": [],
        "sequences": []
    }}

    Ответь ТОЛЬКО JSON."""
        
        self.logger.info("Анализ задания...")
        response = self.ask_llm(prompt, "task_analysis")
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL) 
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
        return self.create_default_plan(task_description)
    
    """Fallback для создания базового плана, если LLM не смогла распарсить задание"""
    def create_default_plan(self, task_description: str) -> Dict[str, Any]:
        task_lower = task_description.lower()
        size_match = re.search(r'(\d+)\s*мб', task_lower)
        ts_size = int(size_match.group(1)) if size_match else 150
        
        # Извлекаем количество пользователей
        users: List[Dict[str, Any]] = []
        user_matches = re.findall(r'([A-Z_]+)\s*\((\d+)\s*мб\)', task_description)
        for name, quota in user_matches:
            users.append({
                "name": name, "password": f"{name.lower()}_2024",
                "quota_mb": int(quota), "profile": "MEGA_PROFILE",
                "privileges": ["CREATE SESSION"]
            })
        
        return {
            "tablespace": {"name": "MEGA_TS", "size_mb": ts_size, "autoextend_max_mb": min(ts_size * 2, 10240)},
            "profile": {"name": "MEGA_PROFILE", "params": {"SESSIONS_PER_USER": 50, "IDLE_TIME": 120}},
            "roles": [{"name": "MEGA_READER_ROLE"}, {"name": "MEGA_WRITER_ROLE"}],
            "users": users if users else [{"name": "TEST_USER", "password": "test_pass", "quota_mb": 100, "profile": "MEGA_PROFILE", "privileges": ["CREATE SESSION"]}],
            "tables": [],
            "data": [],
            "views": [],
            "procedures": [],
            "triggers": [],
            "sequences": []
        }
    
    """Обертка для создания табличного пространства"""
    def create_tablespace(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return 
        name = config.get("name", "MEGA_TS")
        size_mb = config.get("size_mb", 150) #тут
        max_size = config.get("autoextend_max_mb", 5120)
        sql = f"CREATE TABLESPACE {name} DATAFILE '{name.lower()}.dbf' SIZE {size_mb}M AUTOEXTEND ON NEXT 10M MAXSIZE {max_size}M"

        if self.execute_sql(f"Tablespace {name}", sql):
            self.stats["tablespaces"] += 1

    """Обертка для создания профиля"""
    def create_profile(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return
        name = config.get("name", "MEGA_PROFILE")
        params = config.get("params", {"SESSIONS_PER_USER": 50, "IDLE_TIME": 120})
        params_str = " ".join([f"{k} {v}" for k, v in params.items()])
        sql = f"CREATE PROFILE {name} LIMIT {params_str}"
        if self.execute_sql(f"Profile {name}", sql):
            self.stats["profiles"] += 1
    
    """Обертка для создания ролей"""
    def create_roles(self, roles: List[Dict[str, Any]]) -> None:
        for role in roles:
            name = role.get("name")
            if name:
                sql = f"CREATE ROLE {name}"
                if self.execute_sql(f"Role {name}", sql):
                    self.stats["roles"] += 1
    
    """Обертка для создания пользователей"""
    def create_users(self, users: List[Dict[str, Any]], ts_name: str) -> None:
        for user in users:
            name = user.get("name")
            if not name:
                continue
            password = user.get("password", f"{name.lower()}_pass")
            quota = user.get("quota_mb", 100)
            profile = user.get("profile", "MEGA_PROFILE")
            sql = f"CREATE USER {name} IDENTIFIED BY {password} DEFAULT TABLESPACE {ts_name} QUOTA {quota}M ON {ts_name} PROFILE {profile}"
            if self.execute_sql(f"User {name}", sql):
                self.stats["users"] += 1
                for priv in user.get("privileges", ["CREATE SESSION"]):
                    self.execute_sql(f"GRANT {priv} TO {name}", f"GRANT {priv} TO {name}")
                roles = self.context.get("roles", [])
                for role in roles:
                    role_name = role.get("name")
                    if role_name:
                        self.execute_sql(f"Grant role to {name}", f"GRANT {role_name} TO {name}")
    

    """Обертка для создания таблиц"""
    def create_tables(self, tables: List[Dict[str, Any]]) -> None:
        for table in tables:
            owner = table.get("owner")
            name = table.get("name")
            if not owner or not name:
                continue
            columns = table.get("columns", [])
            constraints = table.get("constraints", [])
            
            self.switch_user(owner, "")
            
            cols_sql = ", ".join([f"{c.get('name', 'col')} {c.get('type', 'VARCHAR2(100)')}" for c in columns])
            sql = f"CREATE TABLE {name} ({cols_sql}"
            if constraints:
                sql += f", {', '.join(constraints)}"
            sql += ")"
            
            if self.execute_sql(f"Table {name}", sql):
                self.stats["tables"] += 1
            
            self.tool_dict["init_sys_connection"].invoke({})
    
    """Обертка для вставки данных в таблицы"""
    def insert_data(self, data_list: List[Dict[str, Any]]) -> None:
        for data in data_list:
            owner = data.get("owner")
            table = data.get("table")
            rows = data.get("rows", [])
            
            if not owner or not table:
                continue
            
            self.switch_user(owner, "")
            
            for row in rows:
                columns = list(row.keys())
                values = []
                for v in row.values():
                    if isinstance(v, str):
                        values.append(f"'{v}'")
                    elif v == "SYSDATE":
                        values.append("SYSDATE")
                    else:
                        values.append(str(v))
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)})"
                if self.execute_sql(f"Insert into {table}", sql, skip_if_exists=False):
                    self.stats["rows_inserted"] += 1
            
            self.tool_dict["execute_sql"].invoke({"sql": "COMMIT"})
            self.tool_dict["init_sys_connection"].invoke({})
    
    """Обертка для создания представлений"""
    def create_views(self, views: List[Dict[str, Any]]) -> None:
        for view in views:
            owner = view.get("owner")
            name = view.get("name")
            query = view.get("query")
            
            if not owner or not name or not query:
                continue
            
            self.switch_user(owner, "")
            sql = f"CREATE VIEW {name} AS {query}"
            if self.execute_sql(f"View {name}", sql):
                self.stats["views"] += 1
            self.tool_dict["init_sys_connection"].invoke({})
    
    """Обертка для создания последовательностей"""
    def create_sequences(self, sequences: List[Dict[str, Any]]) -> None:
        for seq in sequences:
            owner = seq.get("owner")
            name = seq.get("name")
            start = seq.get("start", 1)
            increment = seq.get("increment", 1)
            
            if not owner or not name:
                continue
            
            self.switch_user(owner, "")
            sql = f"CREATE SEQUENCE {name} START WITH {start} INCREMENT BY {increment}"
            if self.execute_sql(f"Sequence {name}", sql):
                self.stats["sequences"] += 1
            self.tool_dict["init_sys_connection"].invoke({})
        
    """Удаление существующих объектов"""
    def delete_objects(self, plan: Dict[str, Any]):
        self.logger.info("Удаление существующих объектов")

        # Таблицы 
        for table in plan.get("tables", []):
            name = table.get("name")
            if name:
                try:
                    self.tool_dict["execute_sql"].invoke({"sql": f"DROP TABLE {name} CASCADE CONSTRAINTS"})
                    self.logger.info(f"Удалена таблица {name}")
                    self.recreated_objects.append(f"DELETED TABLE {name}")
                except Exception as e:
                    self.logger.warning(f"Ошибка удаления таблицы {name}: {e}")

        # Представления 
        for view in plan.get("views", []):
            name = view.get("name")
            if name:
                try:
                    self.tool_dict["execute_sql"].invoke({"sql": f"DROP VIEW {name}"})
                    self.logger.info(f"Удалено представление {name}")
                    self.recreated_objects.append(f"DELETED VIEW {name}")
                except Exception as e:
                    self.logger.warning(f"Ошибка удаления представления {name}: {e}")

        # Последовательности (перебираем список)
        for seq in plan.get("sequences", []):
            name = seq.get("name")
            if name:
                try:
                    self.tool_dict["execute_sql"].invoke({"sql": f"DROP SEQUENCE {name}"})
                    self.logger.info(f"Удалена последовательность {name}")
                    self.recreated_objects.append(f"DELETED SEQUENCE {name}")
                except Exception as e:
                    self.logger.warning(f"Ошибка удаления последовательности {name}: {e}")

        # Пользователи
        for user in plan.get("users", []):
            name = user.get("name")
            if name:
                try:
                    self.tool_dict["execute_sql"].invoke({"sql": f"DROP USER {name} CASCADE"})
                    self.logger.info(f"Удален пользователь {name}")
                    self.recreated_objects.append(f"DELETED USER {name}")
                except Exception as e:
                    self.logger.warning(f"Ошибка удаления пользователя {name}: {e}")

        # Роли
        for role in plan.get("roles", []):
            name = role.get("name")
            if name:
                try:
                    self.tool_dict["execute_sql"].invoke({"sql": f"DROP ROLE {name}"})
                    self.logger.info(f"Удалена роль {name}")
                    self.recreated_objects.append(f"DELETED ROLE {name}")
                except Exception as e:
                    self.logger.warning(f"Ошибка удаления роли {name}: {e}")

        # Профиль
        profile_name = plan.get("profile", {}).get("name") # ОДИН объект → сразу берём ключ
        if profile_name:
            try:
                self.tool_dict["execute_sql"].invoke({"sql": f"DROP PROFILE {profile_name} CASCADE"})
                self.logger.info(f"Удален профиль {profile_name}")
            except Exception as e:
                self.logger.warning(f"Ошибка удаления профиля {profile_name}: {e}")

        # Табличное пространство
        ts_name = plan.get("tablespace", {}).get("name")
        if ts_name:
            try:
                self.tool_dict["execute_sql"].invoke({"sql": f"DROP TABLESPACE {ts_name} INCLUDING CONTENTS AND DATAFILES"})
                self.logger.info(f"Удалено табличное пространство {ts_name}")
            except Exception as e:
                self.logger.warning(f"Ошибка удаления табличного пространства {ts_name}: {e}")

    def print_stats(self):
        self.logger.info("=" * 50)
        self.logger.info("СТАТИСТИКА СОЗДАННЫХ ОБЪЕКТОВ")
        self.logger.info("=" * 50)
        self.logger.info(f"Табличные пространства: {self.stats['tablespaces']}")
        self.logger.info(f"Профили: {self.stats['profiles']}")
        self.logger.info(f"Роли: {self.stats['roles']}")
        self.logger.info(f"Пользователи: {self.stats['users']}")
        self.logger.info(f"Таблицы: {self.stats['tables']}")
        self.logger.info(f"Представления: {self.stats['views']}")
        self.logger.info(f"Материализованные представления: {self.stats['materialized_views']}")
        self.logger.info(f"Процедуры: {self.stats['procedures']}")
        self.logger.info(f"Функции: {self.stats['functions']}")
        self.logger.info(f"Пакеты: {self.stats['packages']}")
        self.logger.info(f"Триггеры: {self.stats['triggers']}")
        self.logger.info(f"Последовательности: {self.stats['sequences']}")
        self.logger.info(f"Индексы: {self.stats['indexes']}")
        self.logger.info(f"Синонимы: {self.stats['synonyms']}")
        self.logger.info(f"Вставлено строк: {self.stats['rows_inserted']}")
        self.logger.info("=" * 50)

    def run(self, task_description: str) -> Optional[str]:
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Задача: {task_description[:150]}...")
        self.logger.info(f"Модель: {self.llm_model}")
        
        # Анализ задания
        plan = self.parse_task(task_description)
        if plan is None:
            self.logger.error("Не удалось создать план выполнения")
            return None
        
        self.context = plan
        ts_name = plan.get("tablespace", {}).get("name", "MEGA_TS")
        
        self.logger.info(f"\nПЛАН ВЫПОЛНЕНИЯ:")
        self.logger.info(f"Табличное пространство: {ts_name} ({plan.get('tablespace', {}).get('size_mb', 1024)} MB)")
        self.logger.info(f"Пользователи: {[u.get('name') for u in plan.get('users', [])]}")
        
        # Пересоздание
        if self.recreate:
            self.delete_objects(plan)
        
        # Создание объектов
        self.logger.info("\nСоздание объектов")
        self.create_tablespace(plan.get("tablespace"))
        self.create_profile(plan.get("profile"))
        self.create_roles(plan.get("roles", []))
        self.create_users(plan.get("users", []), ts_name)
        self.create_sequences(plan.get("sequences", []))
        self.create_tables(plan.get("tables", []))
        self.insert_data(plan.get("data", []))
        self.create_views(plan.get("views", []))
        
        # Статистика
        self.print_stats()
        
        # Получение флага
        self.logger.info("\nПолучение флага")
        users = plan.get("users", [])
        if users and len(users) > 0:
            first_user = users[0].get("name")
            if first_user and self.stats["users"] > 0:
                self.switch_user(first_user, "")
            else:
                self.logger.warning("Пользователь не создан, получаю флаг как system")
        
        result = self.tool_dict["get_ctf_flag"].invoke({})
        flag_match = re.search(r'(flag\{[^}]+\}|ctf\{[^}]+\})', str(result), re.IGNORECASE)
        
        if flag_match:
            flag = flag_match.group(0)
            self.logger.info(f"\nФлаг получен: {flag}")
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "task": task_description[:500],
                "stats": self.stats,
                "flag": flag,
                "llm_conversations": len(self.llm_conversations),
                "recreated": self.recreated_objects
            }
            report_file = os.path.join(self.log_dir, f"perfect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Отчет: {report_file}")
            return flag
        
        return None
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Perfect CTF Agent")
    parser.add_argument("--task", default="Создай табличное пространство 150 МБ. Создай пользователя с квотой 80 МБ. Получи флаг.")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    
    init = Init()
    init.llm_connect()
    init.db_connect(
        username=os.getenv("ORACLE_DB_USER", "system"),
        password=os.getenv("ORACLE_DB_PASSWORD", "oracle")
    )
    
    db = DB_Tools(init)
    agent = LLM_Tools(init, db, use_llm=not args.no_llm, save_logs=True, recreate=args.recreate)
    agent.timeout = args.timeout

    if not init.llm_connect():
        print("Ошибка: LLM недоступна. Проверьте Ollama.")
        return
    
    flag = agent.run(args.task)
    
    print("\n" + "="*70)
    print(f"РЕЗУЛЬТАТ: {flag if flag else 'Флаг не найден'}")
    print("="*70)


if __name__ == "__main__":
    main()