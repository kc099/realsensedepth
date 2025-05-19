import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('wheel_inspection.db')
    c = conn.cursor()
    
    # Create inspection table
    c.execute('''CREATE TABLE IF NOT EXISTS inspections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  part_no TEXT,
                  model_type TEXT,
                  timestamp DATETIME,
                  diameter_mm REAL,
                  thickness_mm REAL,
                  height_mm REAL,
                  test_result TEXT,
                  image_path_top TEXT,
                  image_path_side TEXT)''')
    
    conn.commit()
    conn.close()

# def add_inspection(part_no, model_type, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side):
#     conn = sqlite3.connect('wheel_inspection.db')
#     c = conn.cursor()
    
#     c.execute('''INSERT INTO inspections 
#                  (part_no, model_type, timestamp, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side)
#                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
#               (part_no, model_type, datetime.now(), diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side))
    
#     conn.commit()
#     conn.close()

# In your database operations, use context managers:
def add_inspection(part_no, model_type, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side):
    with sqlite3.connect('wheel_inspection.db') as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO inspections 
                     (part_no, model_type, timestamp, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (part_no, model_type, datetime.now(), diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side))
        conn.commit()

def get_daily_report(date):
    conn = sqlite3.connect('wheel_inspection.db')
    c = conn.cursor()
    
    # Get daily counts
    c.execute('''SELECT 
                 COUNT(*) as total,
                 SUM(CASE WHEN test_result = "OK" THEN 1 ELSE 0 END) as ok_count,
                 SUM(CASE WHEN test_result = "NOT OK" THEN 1 ELSE 0 END) as nok_count
                 FROM inspections 
                 WHERE date(timestamp) = ?''', (date,))
    counts = c.fetchone()
    
    # Get all inspections for the day
    c.execute('''SELECT * FROM inspections 
                 WHERE date(timestamp) = ?
                 ORDER BY timestamp DESC''', (date,))
    inspections = c.fetchall()
    
    conn.close()
    return counts, inspections

def get_monthly_report(year, month):
    conn = sqlite3.connect('wheel_inspection.db')
    c = conn.cursor()
    
    # Get monthly counts
    c.execute('''SELECT 
                 COUNT(*) as total,
                 SUM(CASE WHEN test_result = "OK" THEN 1 ELSE 0 END) as ok_count,
                 SUM(CASE WHEN test_result = "NOT OK" THEN 1 ELSE 0 END) as nok_count
                 FROM inspections 
                 WHERE strftime('%Y', timestamp) = ? 
                 AND strftime('%m', timestamp) = ?''', (year, month))
    counts = c.fetchone()
    
    conn.close()
    return counts

def get_date_range_report(start_date, end_date):
    conn = sqlite3.connect('wheel_inspection.db')
    c = conn.cursor()
    
    c.execute('''SELECT 
                 id, part_no, model_type, timestamp, 
                 diameter_mm, thickness_mm, height_mm, test_result
                 FROM inspections 
                 WHERE date(timestamp) BETWEEN ? AND ?
                 ORDER BY timestamp DESC''', (start_date, end_date))
    inspections = c.fetchall()
    
    conn.close()
    return inspections