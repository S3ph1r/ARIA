import sys
import json

def test():
    results = {"lxc_detected": None, "error": None}
    try:
        from google import genai
        results["lxc_detected"] = "MODERN (google.genai)"
    except ImportError:
        try:
            import google.generativeai as genai
            results["lxc_detected"] = "LEGACY (google.generativeai)"
        except ImportError:
            results["lxc_detected"] = "NONE"
    
    print(json.dumps(results))

if __name__ == "__main__":
    test()
