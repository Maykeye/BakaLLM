import sqlite3
from typing import Optional
import os
import uuid
from datetime import datetime, timezone

class InsertBuilder:
    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        self.columns = []
        self.values = []

    def __call__(self, column_name, value):
        self.values.append(value)
        self.columns.append(column_name)

    def insert(self, cur: sqlite3.Cursor, returning=None):
        v = ("?," * len(self.columns)).removesuffix(",")
        if isinstance(returning, list):
            returning=",".join(returning)
        elif returning is not None and not isinstance(returning, str):
            raise ValueError(f"Returning has unexpected type {type(returning)}")
        returning = f"RETURNING ({returning})" if returning else ""
        sql = f"INSERT INTO {self.table_name}({','.join(self.columns)}) VALUES ({v}) {returning}"
        cur.execute(sql, self.values)
        self.columns = []
        self.values = []
        return cur.fetchone()

        



class FloatLogger:
    instance: Optional["FloatLogger"] = None
    def __init__(self, buf_sz=16, data_path=None):
        assert FloatLogger.instance is None, "Flogger already initialized"
        FloatLogger.instance = self
        self.buffer_size = buf_sz
        self.data_path: str = data_path or os.environ.get("XDG_DATA_PATH", os.path.expanduser("~/.local/share"))
        if not self.data_path.endswith("/"):
            self.data_path += "/"
        self.enabled = os.path.exists(self.data_path)
        if not self.enabled: 
            print(f"{self.data_path} doesn't exist; flogging disabled")
        self.data_path += "/flogger/"
        os.makedirs(self.data_path, exist_ok=True)
        
        self.project = None
        self.session_id = None
        self.connection: Optional[sqlite3.Connection] = None
        self.known_columns = set()
        self.buffer = []

    def project_path(self, project:str):
        assert all(x.isalnum() or x in '_-' for x in project), f"invalid project id `{project}`"
        return f"{self.data_path}flog_{project}.dat"
    
    def open_project(self, project: str):
        self.project = project
        self.connection = sqlite3.connect(self.project_path(project))
        self.session_id = self.gen_session()
        self.gen_data()
        hl_color = "\x1b[1;32m"
        reset_color = "\x1b[0m"
        print(f"Flogger session id: {hl_color}{self.session_id}{reset_color}, db: {hl_color}{os.path.realpath(self.project_path(project))}{reset_color}")

    def gen_data(self):
        assert self.connection, "Not connected to project"
        with self.connection as con:
            c = con.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS Data(
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL
            );""")
            r = c.execute("PRAGMA table_info(Data);")
            columns = r.fetchall()
            columns = [column[1] for column in columns] #1 -- column name
            self.known_columns.update(columns)

        

    def gen_session(self):
        assert self.connection, "Not connected to project"
        with self.connection:
            c = self.connection.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS Sessions(
                id INTEGER PRIMARY KEY, 
                uuid UUIDv4 NOT NULL UNIQUE,
                start_timestamp TIMESTAMP NOT NULL,
                end_timestamp TIMESTAMP NULL
            );""")
            
            ib = InsertBuilder("Sessions")
            ib("uuid", str(uuid.uuid4()))
            ib("start_timestamp", datetime.now(timezone.utc))
            id, = ib.insert(c, returning="id")

        return id

    def log(self, **kwargs):
        if not self.enabled:
            return
        if not self.project:
            self.open_project(kwargs.pop("project"))
        elif self.project and "project" in kwargs:
            proj = kwargs.pop("project")
            assert proj == self.project, "project is different"
        assert "timestamp" not in kwargs
        kwargs["timestamp"] = datetime.now(timezone.utc)
        self.buffer.append(kwargs)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            self.buffer = []

    def flush(self):
        if not self.connection:
            return
        new_keys = set(k 
                       for kwargs in self.buffer 
                       for k in kwargs.keys()
                       if k not in self.known_columns)
        for k in new_keys:            
            with self.connection as con:
                con.execute(f"ALTER TABLE Data ADD COLUMN {k} USERDATA")
            self.known_columns.update(new_keys)

        with self.connection as con:
            c = con.cursor()
            for row in self.buffer:
                ib = InsertBuilder("Data")
                for k, v in row.items():
                    ib(k, v)
                ib("session_id", self.session_id)
                ib.insert(c)
        self.connection.close()
        assert self.project
        self.connection = sqlite3.connect(self.project_path(self.project))
        self.buffer = []

    def __del__(self):
        self.flush()


    @staticmethod
    def get() -> "FloatLogger":
        if not FloatLogger.instance:
            return FloatLogger()
        return FloatLogger.instance
        

def flog(**kwargs):
    if not FloatLogger.instance:
        FloatLogger()
    FloatLogger.get().log(**kwargs)

if __name__ == "__main__":
    w = FloatLogger(data_path=".")
    flog(project="test", run="zz", loss=4.0)
    flog(project="test", run="zz", loss=5.0)
    print(w.gen_session())

