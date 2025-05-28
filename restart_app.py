#!/usr/bin/env python3
"""
Restart Script for AET-RAG Flask Application
Safely stops any running instances and starts a new one.
"""

import os
import signal
import subprocess
import time
import psutil

def find_flask_processes():
    """Find running Flask processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('main.py' in arg for arg in proc.info['cmdline']):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def stop_flask_app():
    """Stop any running Flask app instances"""
    processes = find_flask_processes()
    if processes:
        print(f"🛑 Found {len(processes)} running Flask process(es)")
        for proc in processes:
            try:
                print(f"   Stopping PID {proc.pid}")
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        print("✓ Stopped all Flask processes")
    else:
        print("ℹ️  No running Flask processes found")

def start_flask_app():
    """Start the Flask application"""
    print("🚀 Starting Flask application...")
    try:
        # Start the app in the background
        process = subprocess.Popen(
            ['python', 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Wait a moment and check if it started successfully
        time.sleep(2)
        if process.poll() is None:
            print("✓ Flask application started successfully")
            print("🌐 Application should be available at: http://localhost:8080")
            print("📋 To view logs, run: tail -f aetna_rag_system.log")
            return process
        else:
            print("❌ Flask application failed to start")
            output, _ = process.communicate()
            print(f"Error output: {output}")
            return None
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return None

def main():
    print("🔄 Restarting AET-RAG Flask Application")
    print("=" * 50)
    
    # Stop any running instances
    stop_flask_app()
    
    # Wait a moment
    time.sleep(1)
    
    # Start new instance
    process = start_flask_app()
    
    if process:
        print("\n✅ Restart completed successfully!")
        print("\n📝 Useful commands:")
        print("   - View logs: tail -f aetna_rag_system.log")
        print("   - Stop app: pkill -f 'python main.py'")
        print("   - Test auth: python test_auth.py")
    else:
        print("\n❌ Restart failed!")

if __name__ == "__main__":
    main() 