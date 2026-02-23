import sqlite3
import json

def count_extracted_emails(db_path="tasks.db"):
    """
    Connects to the SQLite database and counts how many tasks
    successfully extracted a non-empty email address.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query all successfully completed tasks
        cursor.execute("SELECT task_id, result_data FROM taskrecord WHERE status = 'SUCCESS'")
        tasks = cursor.fetchall()
        
        total_success = len(tasks)
        tasks_with_email = 0
        tasks_missing_email = []
        
        print(f"Total SUCCESS tasks in database: {total_success}")
        print("-" * 30)
        
        for task_id, result_json in tasks:
            if not result_json:
                continue
                
            try:
                # result_data is stored as a JSON string in SQLite
                data = json.loads(result_json) if isinstance(result_json, str) else result_json
                
                # Navigate down the nested JSON structure
                poe_name = data.get("poe_name", "Unknown")
                poe_info = data.get("poe_info", {})
                
                if poe_info:
                    email = poe_info.get("Email", "").strip()
                    if email:
                        tasks_with_email += 1
                        # print(f"Found on {task_id}: {email}")
                    else:
                        tasks_missing_email.append((task_id, poe_name))
                else:
                    tasks_missing_email.append((task_id, poe_name))
                        
            except Exception as e:
                print(f"Error parsing JSON for task {task_id}: {e}")
                
        if tasks_missing_email:
            print("\nTasks that SUCCESSFULLY scraped but FAILED to find an email:")
            for tid, name in tasks_missing_email:
                print(f" - [{tid}] {name}")
                
        print("-" * 30)
        print(f"Tasks with valid emails: {tasks_with_email} out of {total_success}")
        
        if total_success > 0:
            percentage = (tasks_with_email / total_success) * 100
            print(f"Current Email Hit Rate: {percentage:.1f}%")
            
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"Database Error: {e}")
        print("Are you sure 'tasks.db' exists in this directory?")

if __name__ == "__main__":
    count_extracted_emails()
