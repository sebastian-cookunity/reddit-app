"""
Test script to verify import of process_analysis function
"""

try:
    from components.main_flow import process_analysis

    print("Successfully imported process_analysis function")
except ImportError as e:
    print(f"Import error: {e}")

print("Python path:")
import sys

print(sys.path)
