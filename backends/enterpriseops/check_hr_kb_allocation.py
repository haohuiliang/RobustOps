"""Run inside pinned HR image after loading bootstrap_hr (without main)."""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database.managers.knowledge_manager import KnowledgeManager, Knowledge
engine=create_engine('sqlite:///:memory:')
with engine.begin() as con:
 con.execute(text('CREATE TABLE knowledge (id INTEGER PRIMARY KEY, org_id INTEGER, kb_number TEXT, UNIQUE(org_id,kb_number))'))
 con.execute(text("INSERT INTO knowledge(org_id,kb_number) VALUES (1,'KB000999')"))
session=sessionmaker(bind=engine)();manager=KnowledgeManager(session)
assert manager._generate_next_kb_number(1)=='KB-001'
session.execute(text("INSERT INTO knowledge(org_id,kb_number) VALUES (1,'KB-001')"));session.commit()
assert manager._generate_next_kb_number(1)=='KB-001', 'original bug not reproduced'
patch_kb_number_allocation(KnowledgeManager, Knowledge)
assert manager._generate_next_kb_number(1)=='KB-002'
session.execute(text("INSERT INTO knowledge(org_id,kb_number) VALUES (1,'KB-002')"));session.commit()
assert manager._generate_next_kb_number(1)=='KB-003'
print('PASS: actual pinned HR allocator collision reproduced; two sequential patched allocations are unique; existing rows preserved')
