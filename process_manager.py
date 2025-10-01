#!/usr/bin/env python3
"""
Process Manager for Fedtalk Analysis
Helps manage and distribute running processes across terminals
"""

import subprocess
import os
import time
import psutil
from datetime import datetime

class ProcessManager:
    def __init__(self):
        self.working_dir = "/Users/atishaykasliwal/untitled folder 2/fedtalk-openai-analysis-main"
        
    def get_running_processes(self):
        """Get all running Python processes related to the project"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if any(keyword in cmdline.lower() for keyword in ['run_chatgpt', 'bert', 'parallel', 'analysis']):
                        processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline,
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def show_status(self):
        """Show current process status"""
        print("🔍 Current Process Status")
        print("=" * 50)
        
        processes = self.get_running_processes()
        if not processes:
            print("❌ No relevant processes currently running")
            return
        
        for proc in processes:
            print(f"PID: {proc['pid']}")
            print(f"Command: {proc['cmdline']}")
            print(f"CPU: {proc['cpu_percent']:.1f}% | Memory: {proc['memory_mb']:.1f} MB")
            print("-" * 50)
    
    def stop_all_processes(self):
        """Stop all running project processes"""
        processes = self.get_running_processes()
        if not processes:
            print("❌ No processes to stop")
            return
        
        print(f"🛑 Stopping {len(processes)} processes...")
        for proc in processes:
            try:
                psutil.Process(proc['pid']).terminate()
                print(f"✅ Stopped PID {proc['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"❌ Could not stop PID {proc['pid']}")
    
    def create_terminal_commands(self):
        """Generate commands for different terminals"""
        commands = {
            "ChatGPT Predictions": [
                "python3 run_single_interval_chatgpt.py --interval 1",
                "python3 run_single_interval_chatgpt.py --interval 5", 
                "python3 run_single_interval_chatgpt.py --interval 10",
                "python3 run_single_interval_chatgpt.py --interval 15",
                "python3 run_single_interval_chatgpt.py --interval 20",
                "python3 run_single_interval_chatgpt.py --interval 25",
                "python3 run_single_interval_chatgpt.py --interval 30"
            ],
            "BERT Predictions": [
                "python3 corrected_cross_interval_bert_predictor.py",
                "python3 final_corrected_bert_predictor.py",
                "python3 advanced_bert_financial_predictor.py"
            ],
            "Parallel Processing": [
                "python3 parallel_interval_runner_independent.py",
                "python3 parallel_interval_runner.py",
                "python3 clean_parallel_runner.py"
            ],
            "Analysis Suite": [
                "python3 run_statements_news_analysis.py",
                "python3 standalone_similarity_analysis.py",
                "python3 comprehensive_predictions.py"
            ]
        }
        
        print("🚀 Terminal Commands")
        print("=" * 50)
        
        terminal_num = 1
        for category, cmds in commands.items():
            print(f"\n📋 {category}:")
            for cmd in cmds:
                print(f"Terminal {terminal_num}: {cmd}")
                terminal_num += 1
    
    def launch_in_terminal(self, command):
        """Launch a command in a new terminal (macOS)"""
        try:
            script = f'''
            tell application "Terminal"
                do script "cd '{self.working_dir}' && {command}"
            end tell
            '''
            subprocess.run(['osascript', '-e', script], check=True)
            print(f"✅ Launched in new terminal: {command}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch terminal: {e}")
    
    def interactive_menu(self):
        """Interactive menu for process management"""
        while True:
            print("\n🎯 Process Manager Menu")
            print("=" * 30)
            print("1. Show current status")
            print("2. Stop all processes")
            print("3. Show terminal commands")
            print("4. Launch ChatGPT predictions")
            print("5. Launch BERT predictions")
            print("6. Launch parallel processing")
            print("7. Launch analysis suite")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                self.show_status()
            elif choice == '2':
                self.stop_all_processes()
            elif choice == '3':
                self.create_terminal_commands()
            elif choice == '4':
                print("🚀 Launching ChatGPT predictions...")
                for interval in [1, 5, 10, 15, 20, 25, 30]:
                    self.launch_in_terminal(f"python3 run_single_interval_chatgpt.py --interval {interval}")
                    time.sleep(1)
            elif choice == '5':
                print("🤖 Launching BERT predictions...")
                bert_commands = [
                    "python3 corrected_cross_interval_bert_predictor.py",
                    "python3 final_corrected_bert_predictor.py",
                    "python3 advanced_bert_financial_predictor.py"
                ]
                for cmd in bert_commands:
                    self.launch_in_terminal(cmd)
                    time.sleep(1)
            elif choice == '6':
                print("📈 Launching parallel processing...")
                parallel_commands = [
                    "python3 parallel_interval_runner_independent.py",
                    "python3 parallel_interval_runner.py",
                    "python3 clean_parallel_runner.py"
                ]
                for cmd in parallel_commands:
                    self.launch_in_terminal(cmd)
                    time.sleep(1)
            elif choice == '7':
                print("🔍 Launching analysis suite...")
                analysis_commands = [
                    "python3 run_statements_news_analysis.py",
                    "python3 standalone_similarity_analysis.py",
                    "python3 comprehensive_predictions.py"
                ]
                for cmd in analysis_commands:
                    self.launch_in_terminal(cmd)
                    time.sleep(1)
            elif choice == '8':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

if __name__ == "__main__":
    manager = ProcessManager()
    manager.interactive_menu()
