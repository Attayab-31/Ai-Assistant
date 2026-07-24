import os, json, sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path('d:/AI ASSISTANT/.env'))
db_url = os.getenv('DATABASE_URL')
print('DB_URL_OK', bool(db_url))
if not db_url:
    sys.exit(0)
engine = create_engine(db_url)
with engine.connect() as conn:
    row = conn.execute(text("SELECT value FROM system_settings WHERE key='screening_questions' LIMIT 1")).fetchone()
    if not row:
        print('NO_ROW')
        sys.exit(0)
    raw = row[0]
    print('RAW_TYPE', type(raw).__name__)
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, list):
        print('NOT_LIST')
        sys.exit(0)
    print('COUNT', len(data))
    for q in data:
        print(json.dumps({k:q.get(k) for k in ['state','question','required','requires_confirmation','answer_type','conditional','require_all_extract_fields','extract_fields']}, ensure_ascii=False))
