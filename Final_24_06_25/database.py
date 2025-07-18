import sqlite3
from datetime import datetime
import threading
from contextlib import contextmanager

# Global connection pool for better performance
_db_lock = threading.Lock()
_connection_pool = {}

@contextmanager
def get_db_connection():
    """Context manager for database connections with pooling"""
    thread_id = threading.get_ident()
    
    with _db_lock:
        if thread_id not in _connection_pool:
            _connection_pool[thread_id] = sqlite3.connect('wheel_inspection.db')
            # Enable WAL mode for better concurrency
            _connection_pool[thread_id].execute("PRAGMA journal_mode=WAL")
            _connection_pool[thread_id].execute("PRAGMA synchronous=NORMAL")
            _connection_pool[thread_id].execute("PRAGMA cache_size=10000")
            _connection_pool[thread_id].execute("PRAGMA temp_store=MEMORY")
    
    try:
        yield _connection_pool[thread_id]
    except Exception:
        # Close connection on error and remove from pool
        _connection_pool[thread_id].close()
        del _connection_pool[thread_id]
        raise

def init_db():
    """Initialize database with optimized settings"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Create inspection table - simplified without test_result
        c.execute('''CREATE TABLE IF NOT EXISTS inspections
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      part_no TEXT,
                      model_type TEXT,
                      timestamp DATETIME,
                      diameter_mm REAL,
                      height_mm REAL,
                      image_path_top TEXT,
                      image_path_side TEXT)''')
        
        # Create indexes for better query performance
        c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_model_type ON inspections(model_type)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_date ON inspections(date(timestamp))''')
        
        conn.commit()

# def add_inspection(part_no, model_type, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side):
#     conn = sqlite3.connect('wheel_inspection.db')
#     c = conn.cursor()
    
#     c.execute('''INSERT INTO inspections 
#                  (part_no, model_type, timestamp, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side)
#                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
#               (part_no, model_type, datetime.now(), diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side))
    
#     conn.commit()
#     conn.close()

def add_inspection(part_no, model_type, diameter_mm, height_mm, image_path_top, image_path_side):
    """Add inspection record with optimized connection handling"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO inspections 
                     (part_no, model_type, timestamp, diameter_mm, height_mm, image_path_top, image_path_side)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (part_no, model_type, datetime.now(), diameter_mm, height_mm, image_path_top, image_path_side))
        conn.commit()

def get_daily_report(date):
    """Get daily report with optimized queries"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Get daily counts by model type
        c.execute('''SELECT 
                     model_type, COUNT(*) as count
                     FROM inspections 
                     WHERE date(timestamp) = ?
                     GROUP BY model_type''', (date,))
        model_counts = c.fetchall()
        
        # Get total count
        c.execute('''SELECT COUNT(*) as total FROM inspections WHERE date(timestamp) = ?''', (date,))
        total_count = c.fetchone()[0]
        
        # Get all inspections for the day
        c.execute('''SELECT * FROM inspections 
                     WHERE date(timestamp) = ?
                     ORDER BY timestamp DESC''', (date,))
        inspections = c.fetchall()
        
        return total_count, model_counts, inspections

def get_monthly_report(year, month):
    """Get monthly report with optimized queries"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Get monthly counts by model type
        c.execute('''SELECT 
                     model_type, COUNT(*) as count
                     FROM inspections 
                     WHERE strftime('%Y', timestamp) = ? 
                     AND strftime('%m', timestamp) = ?
                     GROUP BY model_type''', (year, month))
        model_counts = c.fetchall()
        
        # Get total count
        c.execute('''SELECT COUNT(*) as total 
                     FROM inspections 
                     WHERE strftime('%Y', timestamp) = ? 
                     AND strftime('%m', timestamp) = ?''', (year, month))
        total_count = c.fetchone()[0]
        
        return total_count, model_counts

def get_date_range_report(start_date, end_date):
    """Get date range report with optimized queries"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Get model type counts for the date range
        c.execute('''SELECT 
                     model_type, COUNT(*) as count
                     FROM inspections 
                     WHERE date(timestamp) BETWEEN ? AND ?
                     GROUP BY model_type''', (start_date, end_date))
        model_counts = c.fetchall()
        
        # Get inspection details
        c.execute('''SELECT 
                     id, part_no, model_type, timestamp, 
                     diameter_mm, height_mm
                     FROM inspections 
                     WHERE date(timestamp) BETWEEN ? AND ?
                     ORDER BY timestamp DESC''', (start_date, end_date))
        inspections = c.fetchall()
        
        return model_counts, inspections

def cleanup_connections():
    """Clean up database connections (call on application shutdown)"""
    with _db_lock:
        for conn in _connection_pool.values():
            try:
                conn.close()
            except:
                pass
        _connection_pool.clear()