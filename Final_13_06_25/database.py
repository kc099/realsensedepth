import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('wheel_inspection.db')
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
def add_inspection(part_no, model_type, diameter_mm, height_mm, image_path_top, image_path_side):
    with sqlite3.connect('wheel_inspection.db') as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO inspections 
                     (part_no, model_type, timestamp, diameter_mm, height_mm, image_path_top, image_path_side)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (part_no, model_type, datetime.now(), diameter_mm, height_mm, image_path_top, image_path_side))
        conn.commit()

def get_daily_report(date):
    conn = sqlite3.connect('wheel_inspection.db')
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
    
    conn.close()
    return total_count, model_counts, inspections

def get_monthly_report(year, month):
    conn = sqlite3.connect('wheel_inspection.db')
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
    
    conn.close()
    return total_count, model_counts

def get_date_range_report(start_date, end_date):
    conn = sqlite3.connect('wheel_inspection.db')
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
    
    conn.close()
    return model_counts, inspections