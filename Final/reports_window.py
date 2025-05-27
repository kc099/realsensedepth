import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
import sqlite3
from datetime import datetime
import pandas as pd
from database import get_daily_report, get_monthly_report, get_date_range_report

# Import app icon module
try:
    from app_icon import set_app_icon
except ImportError:
    def set_app_icon(window):
        pass  # Fallback if module not available

# Colors from main application
OK_COLOR = "#2ECC71"  # Green for OK status
NOK_COLOR = "#E74C3C"  # Red for NOK status
BG_COLOR = "#FFFFFF"  # White background
TEXT_COLOR = "#000000"  # Black text

def show_report_window(root):
    report_window = tk.Toplevel(root)
    report_window.title("Inspection Reports")
    report_window.geometry("1000x700")
    
    # Set custom icon for reports window
    set_app_icon(report_window)
    
    # Date selection
    date_frame = ttk.Frame(report_window)
    date_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(date_frame, text="From:").pack(side=tk.LEFT, padx=5)
    start_date = DateEntry(date_frame)
    start_date.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(date_frame, text="To:").pack(side=tk.LEFT, padx=5)
    end_date = DateEntry(date_frame)
    end_date.pack(side=tk.LEFT, padx=5)
    
    # Statistics frame
    stats_frame = ttk.LabelFrame(report_window, text="Statistics")
    stats_frame.pack(fill=tk.X, padx=10, pady=5)
    
    daily_stats_frame = ttk.Frame(stats_frame)
    daily_stats_frame.pack(side=tk.LEFT, padx=10, pady=5)
    
    ttk.Label(daily_stats_frame, text="Today's Count").pack()
    today = datetime.now().strftime("%Y-%m-%d")
    total_count, model_counts, daily_inspections = get_daily_report(today)
    
    ttk.Label(daily_stats_frame, text=f"Total: {total_count}").pack()
    
    # Display model-specific counts
    for model, count in model_counts:
        ttk.Label(daily_stats_frame, text=f"{model}: {count}").pack()
    
    monthly_stats_frame = ttk.Frame(stats_frame)
    monthly_stats_frame.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Get current month name
    current_month = datetime.now().strftime("%B")  # This will give full month name (e.g., "May")
    ttk.Label(monthly_stats_frame, text=f"{current_month} Count").pack()
    
    current_month_date = datetime.now().strftime("%Y-%m")
    total_count, model_counts = get_monthly_report(current_month_date[:4], current_month_date[5:7])
    
    ttk.Label(monthly_stats_frame, text=f"Total: {total_count}").pack()
    
    # Display model-specific counts
    for model, count in model_counts:
        ttk.Label(monthly_stats_frame, text=f"{model}: {count}").pack()
    
    # Report treeview (single instance)
    report_tree_frame = ttk.Frame(report_window)
    report_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    scrollbar = ttk.Scrollbar(report_tree_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    report_tree = ttk.Treeview(report_tree_frame, columns=(
        "ID", "Part No", "Model", "Date", "Diameter", "Height"
    ), yscrollcommand=scrollbar.set)
    
    report_tree.heading("#0", text="")
    report_tree.heading("ID", text="ID")
    report_tree.heading("Part No", text="Part No")
    report_tree.heading("Model", text="Model")
    report_tree.heading("Date", text="Date")
    report_tree.heading("Diameter", text="Diameter (mm)")
    report_tree.heading("Height", text="Height (mm)")
    
    report_tree.column("#0", width=0, stretch=tk.NO)
    report_tree.column("ID", width=50, anchor=tk.CENTER)
    report_tree.column("Part No", width=120, anchor=tk.CENTER)
    report_tree.column("Model", width=100, anchor=tk.CENTER)
    report_tree.column("Date", width=120, anchor=tk.CENTER)
    report_tree.column("Diameter", width=120, anchor=tk.CENTER)
    report_tree.column("Height", width=100, anchor=tk.CENTER)
    
    report_tree.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=report_tree.yview)
    
    # Add buttons to date frame
    generate_button = ttk.Button(date_frame, text="Generate Report", 
                               command=lambda: generate_report(start_date.get(), end_date.get(), report_tree))
    generate_button.pack(side=tk.LEFT, padx=10)
    
    export_button = ttk.Button(date_frame, text="Export to Excel", 
                             command=lambda: export_to_excel(start_date.get(), end_date.get()))
    export_button.pack(side=tk.LEFT, padx=5)
    
    # Load initial data (today's report)
    generate_report(today, today, report_tree)
    return report_window

def generate_report(start_date, end_date, report_tree):
    """Generate and display report for the selected date range."""
    model_counts, inspections = get_date_range_report(start_date, end_date)
    
    # Clear existing data
    for item in report_tree.get_children():
        report_tree.delete(item)
    
    if not inspections:
        return
    
    for inspection in inspections:
        report_tree.insert("", "end", values=(
            inspection[0],  # ID
            inspection[1],   # Part No
            inspection[2],   # Model
            inspection[3],  # Date
            f"{inspection[4]:.2f}" if inspection[4] else "N/A",  # Diameter
            f"{inspection[5]:.2f}" if inspection[5] else "N/A"   # Height
        ))

def export_to_excel(start_date, end_date):
    """Export inspection data to Excel file."""
    model_counts, inspections = get_date_range_report(start_date, end_date)
    if not inspections:
        messagebox.showwarning("No Data", "No inspection data found for the selected date range")
        return
    
    df = pd.DataFrame(inspections, columns=[
        "ID", "Part No", "Model", "Timestamp", "Diameter (mm)", "Height (mm)"
    ])
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    
    if save_path:
        try:
            df.to_excel(save_path, index=False)
            messagebox.showinfo("Success", f"Report exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")

# def show_dashboard(root, DB_FILE):
#     dashboard = tk.Toplevel(root)
#     dashboard.title("Inspection Dashboard")
#     dashboard.geometry("1200x800")
    
#     # Create a modern dashboard layout with frames for each component
#     header_frame = ttk.Frame(dashboard)
#     header_frame.pack(fill=tk.X, padx=10, pady=10)
    
#     ttk.Label(header_frame, text="AI BASED WHEEL INSPECTION SYSTEM", 
#              font=('Helvetica', 16, 'bold')).pack()
    
#     # Navigation menu
#     nav_frame = ttk.Frame(dashboard)
#     nav_frame.pack(fill=tk.X, padx=10, pady=5)
    
#     buttons = ["Home", "History", "Integration", "Server Status", "Profile", "Settings", "Logout"]
#     for btn_text in buttons:
#         ttk.Button(nav_frame, text=btn_text).pack(side=tk.LEFT, padx=5)
    
#     # Main content area
#     content_frame = ttk.Frame(dashboard)
#     content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
#     # Last inspection status
#     status_frame = ttk.LabelFrame(content_frame, text="Last Inspection Status")
#     status_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    
#     # Get last inspection from database
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("SELECT * FROM inspections ORDER BY timestamp DESC LIMIT 1")
#     last_inspection = c.fetchone()
#     conn.close()
    
#     if last_inspection:
#         ttk.Label(status_frame, text=f"Part No: {last_inspection[1]}").pack(anchor=tk.W)
#         ttk.Label(status_frame, text=f"Status: {last_inspection[7]}", 
#                  foreground=OK_COLOR if last_inspection[7] == "OK" else NOK_COLOR).pack(anchor=tk.W)
#         ttk.Label(status_frame, text=f"Date & Time: {last_inspection[3]}").pack(anchor=tk.W)
#     else:
#         ttk.Label(status_frame, text="No inspections yet").pack()
    
#     # Device health status
#     health_frame = ttk.LabelFrame(content_frame, text="Device Health Status")
#     health_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    
#     devices = ["Camera RH", "EBP", "PLC", "Storage", "MV Light", "Printer", "Scanner"]
#     for device in devices:
#         ttk.Label(health_frame, text=f"{device}: ●", foreground=OK_COLOR).pack(anchor=tk.W)
    
#     # Daily and monthly counts
#     counts_frame = ttk.Frame(content_frame)
#     counts_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
    
#     daily_frame = ttk.LabelFrame(counts_frame, text="Daily Inspection Count")
#     daily_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
#     today = datetime.now().strftime("%Y-%m-%d")
#     daily_counts = get_daily_report(today)[0]
#     total_daily = daily_counts[0] or 1  # Avoid division by zero
    
#     ttk.Label(daily_frame, text=f"OK: {daily_counts[1]}").pack()
#     ttk.Label(daily_frame, text=f"NOT OK: {daily_counts[2]}").pack()
    
#     # Simple pie chart representation
#     canvas = tk.Canvas(daily_frame, width=150, height=150)
#     canvas.pack(pady=10)
    
#     # Draw pie chart
#     ok_angle = 360 * (daily_counts[1] / total_daily)
#     canvas.create_arc(10, 10, 140, 140, start=0, extent=ok_angle, fill=OK_COLOR, outline="")
#     canvas.create_arc(10, 10, 140, 140, start=ok_angle, extent=360-ok_angle, fill=NOK_COLOR, outline="")
    
#     monthly_frame = ttk.LabelFrame(counts_frame, text="Monthly Inspection Count")
#     monthly_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
#     current_month = datetime.now().strftime("%Y-%m")
#     monthly_counts = get_monthly_report(current_month[:4], current_month[5:7])
#     total_monthly = monthly_counts[0] or 1
    
#     ttk.Label(monthly_frame, text=f"OK: {monthly_counts[1]}").pack()
#     ttk.Label(monthly_frame, text=f"NOT OK: {monthly_counts[2]}").pack()
    
#     # Monthly pie chart
#     canvas = tk.Canvas(monthly_frame, width=150, height=150)
#     canvas.pack(pady=10)
    
#     ok_angle = 360 * (monthly_counts[1] / total_monthly)
#     canvas.create_arc(10, 10, 140, 140, start=0, extent=ok_angle, fill=OK_COLOR, outline="")
#     canvas.create_arc(10, 10, 140, 140, start=ok_angle, extent=360-ok_angle, fill=NOK_COLOR, outline="")
    
#     # Configure grid weights
#     content_frame.grid_columnconfigure(0, weight=1)
#     content_frame.grid_columnconfigure(1, weight=1)
#     content_frame.grid_rowconfigure(0, weight=1)
#     content_frame.grid_rowconfigure(1, weight=1)
def show_dashboard(parent, db_file):
    """Show dashboard with key metrics and performance indicators."""
    dashboard = tk.Toplevel(parent)
    dashboard.title("Inspection Dashboard")
    dashboard.geometry("800x600")
    
    # Connect to database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create frame for metrics
    metrics_frame = ttk.Frame(dashboard)
    metrics_frame.pack(fill=tk.X, padx=20, pady=10)
    
    # Calculate daily statistics
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE date(timestamp) = date('now')")
    total_daily = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE date(timestamp) = date('now') AND result = 'OK'")
    daily_ok = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE date(timestamp) = date('now') AND result = 'NOK'")
    daily_nok = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    # Get last 7 days statistics
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE timestamp >= datetime('now', '-7 days')")
    total_week = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE timestamp >= datetime('now', '-7 days') AND result = 'OK'")
    weekly_ok = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE timestamp >= datetime('now', '-7 days') AND result = 'NOK'")
    weekly_nok = cursor.fetchone()[0] or 0  # Default to 0 if None
    
    # Calculate percentages (safely)
    daily_ok_pct = (daily_ok / total_daily * 100) if total_daily > 0 else 0
    daily_nok_pct = (daily_nok / total_daily * 100) if total_daily > 0 else 0
    weekly_ok_pct = (weekly_ok / total_week * 100) if total_week > 0 else 0
    weekly_nok_pct = (weekly_nok / total_week * 100) if total_week > 0 else 0
    
    # Create metric cards
    metrics = [
        {"title": "Today's Inspections", "value": total_daily, "unit": ""},
        {"title": "Today's Pass Rate", "value": daily_ok_pct, "unit": "%"},
        {"title": "Weekly Inspections", "value": total_week, "unit": ""},
        {"title": "Weekly Pass Rate", "value": weekly_ok_pct, "unit": "%"}
    ]
    
    for i, metric in enumerate(metrics):
        frame = ttk.Frame(metrics_frame, borderwidth=2, relief="raised")
        frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(frame, text=metric["title"], font=("Helvetica", 12)).pack(pady=(10, 5))
        ttk.Label(frame, text=f"{metric['value']:.1f}{metric['unit']}", 
                 font=("Helvetica", 18, "bold")).pack(pady=5)
    
    metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
    
    # Create frame for charts
    charts_frame = ttk.Frame(dashboard)
    charts_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    charts_frame.grid_columnconfigure(0, weight=1)
    charts_frame.grid_columnconfigure(1, weight=1)
    charts_frame.grid_rowconfigure(0, weight=1)
    
    # Create canvas for pie chart
    pie_frame = ttk.LabelFrame(charts_frame, text="Today's Results")
    pie_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    pie_canvas = tk.Canvas(pie_frame, width=300, height=300, bg="white")
    pie_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Draw pie chart
    center_x, center_y = 150, 150
    radius = 100
    
    # Only draw pie segments if there's data
    if total_daily > 0:
        # OK portion
        ok_angle = 360 * (daily_ok / total_daily)
        pie_canvas.create_arc(center_x-radius, center_y-radius, 
                             center_x+radius, center_y+radius, 
                             start=0, extent=ok_angle, fill="#2ECC71")
        
        # NOK portion
        nok_angle = 360 * (daily_nok / total_daily)
        pie_canvas.create_arc(center_x-radius, center_y-radius, 
                             center_x+radius, center_y+radius, 
                             start=ok_angle, extent=nok_angle, fill="#E74C3C")
        
        # Legend
        pie_canvas.create_rectangle(center_x+120, center_y-30, center_x+140, center_y-10, fill="#2ECC71")
        pie_canvas.create_text(center_x+145, center_y-20, anchor="w", 
                              text=f"OK ({daily_ok_pct:.1f}%)")
        
        pie_canvas.create_rectangle(center_x+120, center_y+10, center_x+140, center_y+30, fill="#E74C3C")
        pie_canvas.create_text(center_x+145, center_y+20, anchor="w", 
                              text=f"NOK ({daily_nok_pct:.1f}%)")
    else:
        # Show "No Data" message if there's no data
        pie_canvas.create_text(center_x, center_y, text="No data for today", 
                             font=("Helvetica", 14), fill="gray")
    
    # Get trend data for the past 7 days
    cursor.execute("""
        SELECT date(timestamp) as date, 
               COUNT(*) as total,
               SUM(CASE WHEN result = 'OK' THEN 1 ELSE 0 END) as ok_count
        FROM inspections
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY date(timestamp)
        ORDER BY date
    """)
    
    trend_data = cursor.fetchall()
    trend_dates = [row[0] for row in trend_data]
    trend_counts = [row[1] for row in trend_data]
    trend_ok = [row[2] or 0 for row in trend_data]  # Handle None values
    trend_rates = [(ok / total * 100) if total > 0 else 0 for ok, total in zip(trend_ok, trend_counts)]
    
    # Create trend chart
    trend_frame = ttk.LabelFrame(charts_frame, text="7-Day Trend")
    trend_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    
    trend_canvas = tk.Canvas(trend_frame, width=400, height=300, bg="white")
    trend_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Draw trend chart
    if trend_data:
        # Chart dimensions
        chart_left = 50
        chart_right = 350
        chart_top = 50
        chart_bottom = 250
        chart_width = chart_right - chart_left
        chart_height = chart_bottom - chart_top
        
        # X-axis
        trend_canvas.create_line(chart_left, chart_bottom, chart_right, chart_bottom)
        
        # Y-axis
        trend_canvas.create_line(chart_left, chart_top, chart_left, chart_bottom)
        
        # Y-axis labels
        for i in range(6):
            y = chart_bottom - (i * chart_height / 5)
            trend_canvas.create_text(chart_left-10, y, text=f"{i*20}%", anchor="e")
            trend_canvas.create_line(chart_left-5, y, chart_left, y)
        
        # Determine x positions for data points
        num_points = len(trend_dates)
        x_spacing = chart_width / (num_points-1) if num_points > 1 else chart_width
        
        # Plot pass rate points and connect with lines
        for i in range(num_points):
            x = chart_left + (i * x_spacing)
            y = chart_bottom - (trend_rates[i] / 100 * chart_height)
            
            # Plot point
            trend_canvas.create_oval(x-3, y-3, x+3, y+3, fill="#3498DB", outline="")
            
            # Connect with line
            if i > 0:
                prev_x = chart_left + ((i-1) * x_spacing)
                prev_y = chart_bottom - (trend_rates[i-1] / 100 * chart_height)
                trend_canvas.create_line(prev_x, prev_y, x, y, fill="#3498DB", width=2)
            
            # Draw vertical reference line
            trend_canvas.create_line(x, chart_bottom, x, chart_bottom+5)
            
            # X-axis labels (rotate date for better fit)
            date_label = trend_dates[i].split("-")[-1]  # Just show day portion
            trend_canvas.create_text(x, chart_bottom+15, text=date_label, anchor="n")
        
        trend_canvas.create_text(chart_width/2 + chart_left, chart_bottom+30, 
                               text="Date", font=("Helvetica", 10))
        trend_canvas.create_text(chart_left-30, chart_height/2 + chart_top, 
                               text="Pass Rate (%)", font=("Helvetica", 10), angle=90)
                               
    else:
        # Show "No Data" message if there's no data
        trend_canvas.create_text(200, 150, text="No trend data available", 
                               font=("Helvetica", 14), fill="gray")
    
    # Create data table below charts
    table_frame = ttk.LabelFrame(dashboard, text="Recent Inspections")
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
    
    # Recent inspections table
    columns = ("id", "part_no", "model", "diameter", "thickness", "result", "timestamp")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    
    # Define column headers
    tree.heading("id", text="ID")
    tree.heading("part_no", text="Part Number")
    tree.heading("model", text="Model")
    tree.heading("diameter", text="Diameter (mm)")
    tree.heading("thickness", text="Thickness (mm)")
    tree.heading("result", text="Result")
    tree.heading("timestamp", text="Timestamp")
    
    # Set column widths
    tree.column("id", width=50)
    tree.column("part_no", width=120)
    tree.column("model", width=100)
    tree.column("diameter", width=90)
    tree.column("thickness", width=90)
    tree.column("result", width=70)
    tree.column("timestamp", width=150)
    
    # Get recent inspections
    cursor.execute("""
        SELECT id, part_no, model, diameter, thickness, result, timestamp
        FROM inspections
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    # Populate table
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)
    
    conn.close()
    return dashboard